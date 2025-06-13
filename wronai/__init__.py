"""
WronAI - Polski Model Językowy

Demokratyzacja sztucznej inteligencji dla języka polskiego.
Efektywny trening i inferencja na sprzęcie konsumenckim.
"""

from .version import __version__

# Core components
from .models import (
    WronAIModel,
    load_model,
    load_quantized_model
)

from .training import (
    WronAITrainer,
    TrainingConfig,
    train_model
)

from .inference import (
    InferenceEngine,
    ChatBot,
    generate_text
)

from .data import (
    PolishDataset,
    PolishTokenizer,
    prepare_polish_data
)

# Utilities
from .utils import (
    setup_logging,
    get_device_info,
    memory_monitor
)

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
__author__ = Tom Sapletta
__email__ = "wronai@softreck.dev"
__license__ = "Apache 2.0"
__description__ = "Polski model językowy - demokratyzacja AI"