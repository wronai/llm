"""
LoRA (Low-Rank Adaptation) management for WronAI models.
"""

import json
import os
import warnings
from typing import Dict, List, Optional, Union, Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ..utils.logging import get_logger

logger = get_logger(__name__)

try:
    from peft import (
        LoraConfig, PeftModel, get_peft_model,
        prepare_model_for_kbit_training, TaskType
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT not available. Install with: pip install peft")

class LoRAModel:
    """
    Enhanced LoRA model wrapper with Polish language optimizations.
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        lora_config: Optional[Dict[str, Any]] = None,
        adapter_name: str = "default"
    ):
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library required for LoRA functionality")

        self.base_model = base_model
        self.adapter_name = adapter_name
        self.lora_config = lora_config or self._get_default_config()

        # LoRA model will be set after applying adapters
        self.peft_model = None
        self.active_adapters = {}
        self.adapter_configs = {}

        # Polish-specific LoRA optimizations
        self.polish_target_modules = self._get_polish_target_modules()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default LoRA configuration optimized for Polish."""
        return {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        }

    def _get_polish_target_modules(self) -> List[str]:
        """Get target modules optimized for Polish language modeling."""
        # These modules are most important for Polish morphology and syntax
        return [
            "q_proj",      # Query projection - important for attention
            "v_proj",      # Value projection - important for content representation
            "k_proj",      # Key projection - important for context understanding
            "o_proj",      # Output projection - important for final representation
            "gate_proj",   # Gate projection - controls information flow
            "up_proj",     # Up projection - expands representation
            "down_proj"    # Down projection - compresses representation
        ]

    def apply_lora(
        self,
        config: Optional[Dict[str, Any]] = None,
        adapter_name: str = None,
        quantized: bool = False
    ) -> PeftModel:
        """
        Apply LoRA adapters to the base model.

        Args:
            config: LoRA configuration
            adapter_name: Name for the adapter
            quantized: Whether the base model is quantized

        Returns:
            PEFT model with LoRA adapters
        """
        adapter_name = adapter_name or self.adapter_name
        config = config or self.lora_config

        logger.info(f"Applying LoRA adapter '{adapter_name}' with r={config['r']}")

        # Prepare model for training if quantized
        if quantized:
            model = prepare_model_for_kbit_training(self.base_model)
        else:
            model = self.base_model

        # Create LoRA config
        peft_config = LoraConfig(
            r=config["r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias=config["bias"],
            task_type=TaskType.CAUSAL_LM,
            target_modules=config["target_modules"]
        )

        # Apply LoRA
        self.peft_model = get_peft_model(model, peft_config, adapter_name=adapter_name)

        # Store configuration
        self.active_adapters[adapter_name] = True
        self.adapter_configs[adapter_name] = config

        # Print trainable parameters info
        self.peft_model.print_trainable_parameters()

        return self.peft_model

    def add_adapter(
        self,
        adapter_name: str,
        config: Optional[Dict[str, Any]] = None,
        set_as_active: bool = True
    ):
        """
        Add a new LoRA adapter to existing PEFT model.

        Args:
            adapter_name: Name for the new adapter
            config: LoRA configuration for the adapter
            set_as_active: Whether to set as active adapter
        """
        if self.peft_model is None:
            raise ValueError("No PEFT model available. Apply LoRA first.")

        config = config or self.lora_config

        peft_config = LoraConfig(
            r=config["r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias=config["bias"],
            task_type=TaskType.CAUSAL_LM,
            target_modules=config["target_modules"]
        )

        self.peft_model.add_adapter(adapter_name, peft_config)
        self.adapter_configs[adapter_name] = config

        if set_as_active:
            self.set_active_adapter(adapter_name)

        logger.info(f"Added LoRA adapter '{adapter_name}'")

    def set_active_adapter(self, adapter_name: str):
        """Set active adapter."""
        if self.peft_model is None:
            raise ValueError("No PEFT model available.")

        if adapter_name not in self.adapter_configs:
            raise ValueError(f"Adapter '{adapter_name}' not found")

        self.peft_model.set_adapter(adapter_name)

        # Update active status
        for name in self.active_adapters:
            self.active_adapters[name] = (name == adapter_name)

        logger.info(f"Set active adapter to '{adapter_name}'")

    def merge_and_unload(self, adapter_name: Optional[str] = None) -> PreTrainedModel:
        """
        Merge LoRA weights with base model and return merged model.

        Args:
            adapter_name: Specific adapter to merge (if None, merges active)

        Returns:
            Merged model
        """
        if self.peft_model is None:
            raise ValueError("No PEFT model available.")

        if adapter_name:
            self.set_active_adapter(adapter_name)

        logger.info("Merging LoRA weights with base model...")
        merged_model = self.peft_model.merge_and_unload()

        return merged_model

    def save_adapters(
        self,
        save_directory: str,
        adapter_name: Optional[str] = None,
        save_config: bool = True
    ):
        """
        Save LoRA adapters to directory.

        Args:
            save_directory: Directory to save adapters
            adapter_name: Specific adapter to save (if None, saves all)
            save_config: Whether to save configuration
        """
        if self.peft_model is None:
            raise ValueError("No PEFT model available.")

        os.makedirs(save_directory, exist_ok=True)

        if adapter_name:
            # Save specific adapter
            self.peft_model.save_pretrained(
                save_directory,
                selected_adapters=[adapter_name]
            )
            adapters_to_save = [adapter_name]
        else:
            # Save all adapters
            self.peft_model.save_pretrained(save_directory)
            adapters_to_save = list(self.adapter_configs.keys())

        # Save configuration
        if save_config:
            config_data = {
                "adapter_configs": {
                    name: self.adapter_configs[name]
                    for name in adapters_to_save
                },
                "active_adapters": {
                    name: self.active_adapters[name]
                    for name in adapters_to_save
                },
                "polish_optimized": True
            }

            config_path = os.path.join(save_directory, "wronai_lora_config.json")
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)

        logger.info(f"Saved LoRA adapters to {save_directory}")

    def load_adapters(
        self,
        adapter_directory: str,
        adapter_name: str = "default",
        set_as_active: bool = True
    ):
        """
        Load LoRA adapters from directory.

        Args:
            adapter_directory: Directory containing adapters
            adapter_name: Name for the loaded adapter
            set_as_active: Whether to set as active
        """
        if self.peft_model is None:
            # Create PEFT model if not exists
            self.apply_lora()

        # Load adapter
        self.peft_model.load_adapter(adapter_directory, adapter_name)

        # Load configuration if available
        config_path = os.path.join(adapter_directory, "wronai_lora_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            if adapter_name in config_data.get("adapter_configs", {}):
                self.adapter_configs[adapter_name] = config_data["adapter_configs"][adapter_name]

        if set_as_active:
            self.set_active_adapter(adapter_name)

        logger.info(f"Loaded LoRA adapter '{adapter_name}' from {adapter_directory}")

    def get_adapter_info(self) -> Dict[str, Any]:
        """Get information about current adapters."""
        if self.peft_model is None:
            return {"adapters": [], "active_adapter": None}

        return {
            "adapters": list(self.adapter_configs.keys()),
            "active_adapter": next(
                (name for name, active in self.active_adapters.items() if active),
                None
            ),
            "adapter_configs": self.adapter_configs,
            "trainable_parameters": self.get_trainable_parameters_info()
        }

    def get_trainable_parameters_info(self) -> Dict[str, int]:
        """Get information about trainable parameters."""
        if self.peft_model is None:
            return {}

        total_params = sum(p.numel() for p in self.peft_model.parameters())
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": (trainable_params / total_params) * 100 if total_params > 0 else 0
        }

    def benchmark_adapter(
        self,
        input_ids: torch.Tensor,
        num_runs: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark LoRA adapter performance.

        Args:
            input_ids: Input tensor for benchmarking
            num_runs: Number of runs for averaging

        Returns:
            Performance metrics
        """
        if self.peft_model is None:
            raise ValueError("No PEFT model available.")

        self.peft_model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = self.peft_model(input_ids)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    start_event.record()
                    _ = self.peft_model(input_ids)
                    end_event.record()

                    torch.cuda.synchronize()
                    times.append(start_event.elapsed_time(end_event))
                else:
                    import time
                    start = time.time()
                    _ = self.peft_model(input_ids)
                    end = time.time()
                    times.append((end - start) * 1000)  # Convert to ms

        return {
            "average_time_ms": sum(times) / len(times),
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "std_time_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
        }

def apply_lora_adapters(
    model: PreTrainedModel,
    config: Optional[Dict[str, Any]] = None,
    adapter_name: str = "default",
    quantized: bool = False
) -> LoRAModel:
    """
    High-level function to apply LoRA adapters.

    Args:
        model: Base model
        config: LoRA configuration
        adapter_name: Adapter name
        quantized: Whether model is quantized

    Returns:
        LoRA model wrapper
    """
    lora_model = LoRAModel(model, config, adapter_name)
    lora_model.apply_lora(config, adapter_name, quantized)
    return lora_model

def create_polish_lora_config(
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create LoRA configuration optimized for Polish language.

    Args:
        rank: LoRA rank
        alpha: LoRA alpha parameter
        dropout: LoRA dropout
        target_modules: Target modules to adapt

    Returns:
        LoRA configuration
    """
    if target_modules is None:
        # Optimized for Polish morphology and syntax
        target_modules = [
            "q_proj", "v_proj", "k_proj", "o_proj",  # Attention layers
            "gate_proj", "up_proj", "down_proj"      # MLP layers
        ]

    return {
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": target_modules,
        "polish_optimized": True
    }

def compare_lora_configs(
    model: PreTrainedModel,
    configs: Dict[str, Dict[str, Any]],
    test_input: torch.Tensor,
    quantized: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Compare different LoRA configurations.

    Args:
        model: Base model
        configs: Dictionary of config_name -> config
        test_input: Test input for benchmarking
        quantized: Whether model is quantized

    Returns:
        Comparison results
    """
    results = {}

    for config_name, config in configs.items():
        logger.info(f"Testing LoRA configuration: {config_name}")

        try:
            # Apply LoRA
            lora_model = LoRAModel(model, config)
            lora_model.apply_lora(config, config_name, quantized)

            # Get parameter info
            param_info = lora_model.get_trainable_parameters_info()

            # Benchmark performance
            perf_metrics = lora_model.benchmark_adapter(test_input)

            results[config_name] = {
                "config": config,
                "parameters": param_info,
                "performance": perf_metrics,
                "success": True
            }

        except Exception as e:
            logger.error(f"Failed to test config {config_name}: {e}")
            results[config_name] = {
                "config": config,
                "error": str(e),
                "success": False
            }

    return results

def optimize_lora_for_polish(
    model: PreTrainedModel,
    sample_polish_texts: List[str],
    target_performance: float = 0.9
) -> Dict[str, Any]:
    """
    Optimize LoRA configuration for Polish language performance.

    Args:
        model: Base model
        sample_polish_texts: Sample Polish texts for optimization
        target_performance: Target performance ratio

    Returns:
        Optimized LoRA configuration
    """
    # Test different configurations
    test_configs = {
        "minimal": create_polish_lora_config(rank=8, alpha=16),
        "balanced": create_polish_lora_config(rank=16, alpha=32),
        "high_capacity": create_polish_lora_config(rank=32, alpha=64),
        "attention_focused": create_polish_lora_config(
            rank=16, alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        ),
        "mlp_focused": create_polish_lora_config(
            rank=16, alpha=32,
            target_modules=["gate_proj", "up_proj", "down_proj"]
        )
    }

    # Create test input (simplified)
    tokenizer = getattr(model, 'tokenizer', None)
    if tokenizer and sample_polish_texts:
        test_text = sample_polish_texts[0]
        test_input = tokenizer(test_text, return_tensors="pt")["input_ids"]
    else:
        # Fallback to random input
        test_input = torch.randint(0, 1000, (1, 50))

    # Compare configurations
    comparison_results = compare_lora_configs(model, test_configs, test_input)

    # Find best configuration based on efficiency
    best_config = None
    best_score = 0

    for config_name, results in comparison_results.items():
        if not results.get("success", False):
            continue

        # Score based on trainable parameters and performance
        trainable_percent = results["parameters"]["trainable_percentage"]
        avg_time = results["performance"]["average_time_ms"]

        # Lower trainable percentage and time is better
        efficiency_score = (100 - trainable_percent) / 100 * 0.6 + (1000 - min(avg_time, 1000)) / 1000 * 0.4

        if efficiency_score > best_score:
            best_score = efficiency_score
            best_config = results["config"]

    return {
        "recommended_config": best_config,
        "optimization_score": best_score,
        "all_results": comparison_results,
        "optimization_rationale": "Balanced efficiency and performance for Polish language modeling"
    }