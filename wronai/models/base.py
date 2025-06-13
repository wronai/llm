"""
Base model classes for WronAI.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """Configuration for WronAI models."""

    # Model architecture
    model_name: str = "mistralai/Mistral-7B-v0.1"
    model_type: str = "causal_lm"

    # Polish language specific
    polish_vocab_size: Optional[int] = None
    polish_tokens: List[str] = None
    morphological_analysis: bool = True

    # Quantization
    quantization_enabled: bool = True
    quantization_bits: int = 4
    quantization_type: str = "nf4"

    # LoRA configuration
    lora_enabled: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None

    # Training
    max_sequence_length: int = 2048
    gradient_checkpointing: bool = True

    # Device and precision
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True

    def __post_init__(self):
        """Post-initialization validation."""
        if self.polish_tokens is None:
            self.polish_tokens = [
                "<polish>",
                "</polish>",
                "<formal>",
                "</formal>",
                "<informal>",
                "</informal>",
                "<question>",
                "</question>",
                "<answer>",
                "</answer>",
            ]

        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]


class WronAIModel(nn.Module, ABC):
    """
    Base class for WronAI models.

    Provides common functionality for Polish language models
    with quantization and LoRA support.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = None
        self.tokenizer = None
        self._is_quantized = False
        self._has_lora = False

    @classmethod
    def from_pretrained(
        cls, model_name: str, config: Optional[ModelConfig] = None, **kwargs
    ):
        """Load model from pretrained weights."""
        if config is None:
            config = ModelConfig(model_name=model_name)
        else:
            config.model_name = model_name

        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        instance = cls(config)
        instance.load_model()
        instance.load_tokenizer()

        return instance

    def load_model(self):
        """Load the underlying transformer model."""
        logger.info(f"Loading model: {self.config.model_name}")

        # Determine model class based on type
        if self.config.model_type == "causal_lm":
            model_class = AutoModelForCausalLM
        else:
            model_class = AutoModel

        # Load model
        self.model = model_class.from_pretrained(
            self.config.model_name,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code,
        )

        # Enable gradient checkpointing if requested
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        logger.info(
            f"Model loaded successfully. Parameters: {self.get_parameter_count():,}"
        )

    def load_tokenizer(self):
        """Load and configure tokenizer."""
        logger.info(f"Loading tokenizer: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=self.config.trust_remote_code
        )

        # Add Polish-specific tokens
        if self.config.polish_tokens:
            num_added = self.tokenizer.add_tokens(self.config.polish_tokens)
            if num_added > 0:
                logger.info(f"Added {num_added} Polish tokens to tokenizer")
                # Resize model embeddings
                self.model.resize_token_embeddings(len(self.tokenizer))

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")

    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Generate text using the model."""
        return self.model.generate(*args, **kwargs)

    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())

    def get_trainable_parameter_count(self) -> int:
        """Get number of trainable parameters."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def print_parameter_info(self):
        """Print parameter information."""
        total = self.get_parameter_count()
        trainable = self.get_trainable_parameter_count()

        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Trainable %: {100 * trainable / total:.2f}%")

        if self._is_quantized:
            print("Model is quantized (4-bit)")
        if self._has_lora:
            print("Model has LoRA adapters")

    def save_pretrained(self, save_directory: str):
        """Save model and tokenizer."""
        logger.info(f"Saving model to: {save_directory}")

        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(save_directory)

        if self.tokenizer:
            self.tokenizer.save_pretrained(save_directory)

        # Save config
        import json
        import os

        config_path = os.path.join(save_directory, "wronai_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

        logger.info("Model saved successfully")

    def to(self, device):
        """Move model to device."""
        if self.model:
            self.model = self.model.to(device)
        return self

    def eval(self):
        """Set model to evaluation mode."""
        if self.model:
            self.model.eval()
        return self

    def train(self, mode=True):
        """Set model to training mode."""
        if self.model:
            self.model.train(mode)
        return self

    @property
    def device(self):
        """Get model device."""
        if self.model and hasattr(self.model, "device"):
            return self.model.device
        return torch.device("cpu")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage information."""
        if not torch.cuda.is_available():
            return {"gpu_memory_used": 0, "gpu_memory_total": 0}

        torch.cuda.synchronize()
        memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB

        return {
            "gpu_memory_used": memory_used,
            "gpu_memory_total": memory_total,
            "gpu_memory_percent": 100 * memory_used / memory_total,
        }

    @abstractmethod
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for Polish language specifics."""
        pass

    @abstractmethod
    def postprocess_text(self, text: str) -> str:
        """Postprocess generated text."""
        pass
