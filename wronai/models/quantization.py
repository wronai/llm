"""
quantization.py
"""
"""
Advanced quantization utilities for WronAI models.
"""

import os
import warnings
from typing import Dict, Optional, Union, Any

import torch
import torch.nn as nn
from transformers import BitsAndBytesConfig

from ..utils.logging import get_logger

logger = get_logger(__name__)


class QuantizationConfig:
    """Configuration for model quantization."""

    def __init__(
            self,
            method: str = "bitsandbytes",
            bits: int = 4,
            quant_type: str = "nf4",
            compute_dtype: str = "bfloat16",
            double_quant: bool = True,
            quant_storage: str = "uint8",
            bnb_4bit_quant_storage: str = "uint8"
    ):
        self.method = method
        self.bits = bits
        self.quant_type = quant_type
        self.compute_dtype = compute_dtype
        self.double_quant = double_quant
        self.quant_storage = quant_storage
        self.bnb_4bit_quant_storage = bnb_4bit_quant_storage

        self._validate_config()

    def _validate_config(self):
        """Validate quantization configuration."""
        valid_methods = ["bitsandbytes", "gptq", "awq", "ggml"]
        if self.method not in valid_methods:
            raise ValueError(f"Unsupported quantization method: {self.method}")

        valid_bits = [1, 2, 3, 4, 8, 16]
        if self.bits not in valid_bits:
            raise ValueError(f"Unsupported bit width: {self.bits}")

        valid_quant_types = ["fp4", "nf4", "int4", "int8"]
        if self.quant_type not in valid_quant_types:
            raise ValueError(f"Unsupported quantization type: {self.quant_type}")


class QuantizedModel:
    """Wrapper for quantized models with Polish language optimizations."""

    def __init__(
            self,
            model: nn.Module,
            quantization_config: QuantizationConfig,
            original_dtype: torch.dtype = torch.float16
    ):
        self.model = model
        self.quantization_config = quantization_config
        self.original_dtype = original_dtype
        self._is_quantized = True

        # Store quantization metadata
        self._quantization_info = {
            "method": quantization_config.method,
            "bits": quantization_config.bits,
            "quant_type": quantization_config.quant_type,
            "compute_dtype": quantization_config.compute_dtype,
            "memory_saved_percent": self._estimate_memory_savings()
        }

    def _estimate_memory_savings(self) -> float:
        """Estimate memory savings from quantization."""
        original_bits = 16 if self.original_dtype == torch.float16 else 32
        quantized_bits = self.quantization_config.bits

        if self.quantization_config.double_quant:
            # Double quantization provides additional savings
            savings = 1 - (quantized_bits * 0.8) / original_bits
        else:
            savings = 1 - quantized_bits / original_bits

        return max(0, min(100, savings * 100))

    def get_quantization_info(self) -> Dict[str, Any]:
        """Get quantization information."""
        return self._quantization_info.copy()

    def get_memory_footprint(self) -> Dict[str, float]:
        """Get model memory footprint in GB."""
        try:
            model_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
            model_size_gb = model_size / (1024 ** 3)

            # Estimate original size
            original_size_gb = model_size_gb / (1 - self._quantization_info["memory_saved_percent"] / 100)

            return {
                "quantized_size_gb": model_size_gb,
                "original_size_gb": original_size_gb,
                "memory_saved_gb": original_size_gb - model_size_gb,
                "memory_saved_percent": self._quantization_info["memory_saved_percent"]
            }
        except Exception as e:
            logger.warning(f"Could not calculate memory footprint: {e}")
            return {"error": str(e)}

    def benchmark_inference_speed(self, input_ids: torch.Tensor, num_runs: int = 10) -> Dict[str, float]:
        """Benchmark inference speed."""
        self.model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(input_ids)

        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                if torch.cuda.is_available():
                    start_time.record()
                    _ = self.model(input_ids)
                    end_time.record()
                    torch.cuda.synchronize()
                    times.append(start_time.elapsed_time(end_time))
                else:
                    import time
                    start = time.time()
                    _ = self.model(input_ids)
                    end = time.time()
                    times.append((end - start) * 1000)  # Convert to ms

        avg_time = sum(times) / len(times)
        return {
            "average_time_ms": avg_time,
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "std_time_ms": (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        }


def create_bitsandbytes_config(
        bits: int = 4,
        quant_type: str = "nf4",
        compute_dtype: str = "bfloat16",
        double_quant: bool = True
) -> BitsAndBytesConfig:
    """Create BitsAndBytesConfig for quantization."""

    # Map string dtype to torch dtype
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }

    compute_dtype_torch = dtype_mapping.get(compute_dtype, torch.bfloat16)

    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_compute_dtype=compute_dtype_torch,
            bnb_4bit_use_double_quant=double_quant
        )
    elif bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
    else:
        raise ValueError(f"BitsAndBytes only supports 4-bit and 8-bit quantization, got {bits}")


def load_quantized_model(
        model: nn.Module,
        quantization_config: Optional[QuantizationConfig] = None
) -> QuantizedModel:
    """
    Load model with quantization.

    Args:
        model: Model to quantize
        quantization_config: Quantization configuration

    Returns:
        Quantized model wrapper
    """
    if quantization_config is None:
        quantization_config = QuantizationConfig()

    logger.info(f"Applying {quantization_config.method} quantization ({quantization_config.bits}-bit)")

    try:
        if quantization_config.method == "bitsandbytes":
            # BitsAndBytes quantization is applied during model loading
            # This function wraps an already quantized model
            quantized_model = QuantizedModel(model, quantization_config)

        elif quantization_config.method == "gptq":
            quantized_model = _apply_gptq_quantization(model, quantization_config)

        elif quantization_config.method == "awq":
            quantized_model = _apply_awq_quantization(model, quantization_config)

        else:
            raise NotImplementedError(f"Quantization method {quantization_config.method} not implemented")

        memory_info = quantized_model.get_memory_footprint()
        logger.info(f"Quantization complete. Memory saved: {memory_info.get('memory_saved_percent', 0):.1f}%")

        return quantized_model

    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        raise


def _apply_gptq_quantization(model: nn.Module, config: QuantizationConfig) -> QuantizedModel:
    """Apply GPTQ quantization (placeholder implementation)."""
    try:
        # Try to import auto-gptq
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

        quantize_config = BaseQuantizeConfig(
            bits=config.bits,
            group_size=128,
            desc_act=False,
            damp_percent=0.1
        )

        # Note: This is a simplified implementation
        # In practice, GPTQ requires calibration data
        logger.warning("GPTQ quantization requires calibration data. Using placeholder implementation.")

        return QuantizedModel(model, config)

    except ImportError:
        logger.error("auto-gptq not installed. Install with: pip install auto-gptq")
        raise
    except Exception as e:
        logger.error(f"GPTQ quantization failed: {e}")
        raise


def _apply_awq_quantization(model: nn.Module, config: QuantizationConfig) -> QuantizedModel:
    """Apply AWQ quantization (placeholder implementation)."""
    try:
        # AWQ quantization would be implemented here
        logger.warning("AWQ quantization not fully implemented. Using placeholder.")

        return QuantizedModel(model, config)

    except Exception as e:
        logger.error(f"AWQ quantization failed: {e}")
        raise


def benchmark_quantization_methods(
        model: nn.Module,
        sample_input: torch.Tensor,
        methods: list = None
) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark different quantization methods.

    Args:
        model: Model to benchmark
        sample_input: Sample input tensor
        methods: List of quantization methods to test

    Returns:
        Benchmark results for each method
    """
    if methods is None:
        methods = ["bitsandbytes_4bit", "bitsandbytes_8bit"]

    results = {}
    original_model = model

    # Benchmark original model
    logger.info("Benchmarking original model...")
    original_times = []
    with torch.no_grad():
        for _ in range(10):
            start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

            if torch.cuda.is_available():
                start.record()
                _ = original_model(sample_input)
                end.record()
                torch.cuda.synchronize()
                original_times.append(start.elapsed_time(end))
            else:
                import time
                start_time = time.time()
                _ = original_model(sample_input)
                end_time = time.time()
                original_times.append((end_time - start_time) * 1000)

    results["original"] = {
        "average_time_ms": sum(original_times) / len(original_times),
        "memory_usage_gb": "N/A",
        "quantization_info": {"method": "none", "bits": 16}
    }

    # Benchmark quantized models
    for method in methods:
        logger.info(f"Benchmarking {method}...")

        try:
            if method == "bitsandbytes_4bit":
                config = QuantizationConfig(method="bitsandbytes", bits=4)
            elif method == "bitsandbytes_8bit":
                config = QuantizationConfig(method="bitsandbytes", bits=8)
            else:
                logger.warning(f"Unknown method: {method}")
                continue

            # Create quantized model wrapper (model already quantized)
            quantized_model = QuantizedModel(model, config)

            # Benchmark
            benchmark_results = quantized_model.benchmark_inference_speed(sample_input)
            memory_info = quantized_model.get_memory_footprint()

            results[method] = {
                **benchmark_results,
                "memory_usage_gb": memory_info.get("quantized_size_gb", "N/A"),
                "memory_saved_percent": memory_info.get("memory_saved_percent", 0),
                "quantization_info": quantized_model.get_quantization_info()
            }

        except Exception as e:
            logger.error(f"Benchmarking {method} failed: {e}")
            results[method] = {"error": str(e)}

    return results


def optimize_quantization_for_polish(
        model: nn.Module,
        polish_sample_texts: list = None
) -> QuantizationConfig:
    """
    Optimize quantization configuration for Polish language.

    Args:
        model: Model to optimize
        polish_sample_texts: Sample Polish texts for calibration

    Returns:
        Optimized quantization configuration
    """
    if polish_sample_texts is None:
        polish_sample_texts = [
            "Witaj świecie! Jak się masz?",
            "Sztuczna inteligencja to fascynująca dziedzina.",
            "Polska to piękny kraj w Europie Środkowej.",
            "Tradycyjne polskie potrawy: pierogi, bigos, kotlet schabowy.",
            "Język polski ma bogatą morfologię i fleksję."
        ]

    logger.info("Optimizing quantization for Polish language...")

    # Test different configurations
    configs_to_test = [
        QuantizationConfig(bits=4, quant_type="nf4", double_quant=True),
        QuantizationConfig(bits=4, quant_type="fp4", double_quant=True),
        QuantizationConfig(bits=8, quant_type="int8", double_quant=False)
    ]

    best_config = configs_to_test[0]  # Default
    best_score = float('inf')

    for config in configs_to_test:
        try:
            # Create quantized model wrapper
            quantized_model = QuantizedModel(model, config)

            # Simple scoring based on memory efficiency
            memory_info = quantized_model.get_memory_footprint()
            memory_saved = memory_info.get("memory_saved_percent", 0)

            # Score: higher memory savings is better
            score = 100 - memory_saved

            if score < best_score:
                best_score = score
                best_config = config

            logger.info(f"Config {config.bits}-bit {config.quant_type}: {memory_saved:.1f}% memory saved")

        except Exception as e:
            logger.warning(f"Failed to test config {config.bits}-bit {config.quant_type}: {e}")

    logger.info(f"Best configuration: {best_config.bits}-bit {best_config.quant_type}")
    return best_config


def export_quantized_model(
        quantized_model: QuantizedModel,
        export_path: str,
        format: str = "gguf"
) -> str:
    """
    Export quantized model to specified format.

    Args:
        quantized_model: Quantized model to export
        export_path: Path to save exported model
        format: Export format (gguf, onnx, etc.)

    Returns:
        Path to exported model
    """
    logger.info(f"Exporting quantized model to {format} format...")

    try:
        if format.lower() == "gguf":
            return _export_to_gguf(quantized_model, export_path)
        elif format.lower() == "onnx":
            return _export_to_onnx(quantized_model, export_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise


def _export_to_gguf(quantized_model: QuantizedModel, export_path: str) -> str:
    """Export to GGUF format (placeholder)."""
    # This would require llama.cpp integration
    logger.warning("GGUF export not fully implemented. Would require llama.cpp integration.")

    # Save model state dict as placeholder
    torch.save({
        'model_state_dict': quantized_model.model.state_dict(),
        'quantization_info': quantized_model.get_quantization_info()
    }, export_path)

    return export_path


def _export_to_onnx(quantized_model: QuantizedModel, export_path: str) -> str:
    """Export to ONNX format (placeholder)."""
    logger.warning("ONNX export for quantized models not fully implemented.")

    # This would require careful handling of quantized operations
    # torch.onnx.export(quantized_model.model, dummy_input, export_path)

    return export_path