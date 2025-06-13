"""
WronAI Inference Module

Inference engines and utilities for Polish language generation.
"""

from .engine import InferenceEngine, InferenceConfig
from .pipeline import TextGenerationPipeline, PolishGenerationPipeline
from .chat import ChatBot, ConversationManager
from .api import create_api_server, APIConfig


def generate_text(
    model,
    prompt: str,
    max_length: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    **kwargs,
):
    """
    High-level text generation function.

    Args:
        model: WronAI model for generation
        prompt: Input prompt
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        **kwargs: Additional generation parameters

    Returns:
        Generated text string
    """
    engine = InferenceEngine(model)
    return engine.generate(
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        **kwargs,
    )


__all__ = [
    "InferenceEngine",
    "InferenceConfig",
    "TextGenerationPipeline",
    "PolishGenerationPipeline",
    "ChatBot",
    "ConversationManager",
    "create_api_server",
    "APIConfig",
    "generate_text",
]
