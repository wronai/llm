"""
WronAI Models Module

Model architectures and utilities for Polish language models.
"""

from .base import WronAIModel, ModelConfig
from .mistral import WronAIMistral
from .llama import WronAILlama
from .quantization import QuantizedModel, load_quantized_model
from .lora import LoRAModel, apply_lora_adapters


def load_model(
    model_name: str, config: dict = None, quantize: bool = True, device: str = "auto"
):
    """
    Load WronAI model with automatic architecture detection.

    Args:
        model_name: Path to model or HuggingFace model name
        config: Optional model configuration
        quantize: Whether to apply quantization
        device: Device placement strategy

    Returns:
        Loaded WronAI model
    """
    if "mistral" in model_name.lower():
        model_class = WronAIMistral
    elif "llama" in model_name.lower():
        model_class = WronAILlama
    else:
        # Default to base model
        model_class = WronAIModel

    model = model_class.from_pretrained(model_name, config=config, device_map=device)

    if quantize:
        model = load_quantized_model(model)

    return model


__all__ = [
    "WronAIModel",
    "ModelConfig",
    "WronAIMistral",
    "WronAILlama",
    "QuantizedModel",
    "LoRAModel",
    "load_model",
    "load_quantized_model",
    "apply_lora_adapters",
]
