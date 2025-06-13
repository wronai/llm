"""
WronAI - Polski Model Językowy

Demokratyzacja sztucznej inteligencji dla języka polskiego.
Efektywny trening i inferencja na sprzęcie konsumenckim.
"""

from .data import PolishDataset, PolishTokenizer, prepare_polish_data
from .inference import ChatBot, InferenceEngine, generate_text

# Core components
from .models import WronAIModel, load_model, load_quantized_model
from .training import TrainingConfig, WronAITrainer, train_model

# Utilities
from .utils import get_device_info, memory_monitor, setup_logging
from .version import __version__

__all__ = [
    "__version__",
    # Models
    "WronAIModel",
    "load_model",
    "load_quantized_model",
    # Training
    "WronAITrainer",
    "TrainingConfig",
    "train_model",
    # Inference
    "InferenceEngine",
    "ChatBot",
    "generate_text",
    # Data
    "PolishDataset",
    "PolishTokenizer",
    "prepare_polish_data",
    # Utils
    "setup_logging",
    "get_device_info",
    "memory_monitor"
]

# Package metadata
__author__ = "Tom Sapletta"
__email__ = "info@softreck.dev"
__license__ = "Apache 2.0"
__description__ = "Polski model językowy - demokratyzacja AI"