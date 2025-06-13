"""
WronAI Training Module

Training utilities and trainers for Polish language models.
"""

from .callbacks import MemoryMonitorCallback, PolishEvaluationCallback
from .optimizer import get_optimizer, get_scheduler
from .trainer import TrainingConfig, WronAITrainer
from .utils import prepare_model_for_training, setup_training


def train_model(
    model, dataset, config: TrainingConfig, output_dir: str = "./checkpoints", **kwargs
):
    """
    High-level function to train a WronAI model.

    Args:
        model: WronAI model to train
        dataset: Training dataset
        config: Training configuration
        output_dir: Output directory for checkpoints
        **kwargs: Additional arguments for trainer

    Returns:
        Trained model and training results
    """
    trainer = WronAITrainer(model=model, config=config, output_dir=output_dir, **kwargs)

    results = trainer.train(dataset)
    return model, results


__all__ = [
    "WronAITrainer",
    "TrainingConfig",
    "PolishEvaluationCallback",
    "MemoryMonitorCallback",
    "get_optimizer",
    "get_scheduler",
    "setup_training",
    "prepare_model_for_training",
    "train_model",
]
