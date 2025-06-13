"""
Training utilities and trainer class for WronAI models.
"""

import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from datasets import Dataset

from ..models.base import WronAIModel
from ..utils.logging import get_logger
from ..utils.memory import memory_monitor, clear_cache
from .callbacks import PolishEvaluationCallback, MemoryMonitorCallback

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for WronAI training."""

    # Basic training parameters
    output_dir: str = "./checkpoints/wronai"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16

    # Optimization
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_32bit"
    max_grad_norm: float = 1.0

    # Precision and performance
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    dataloader_pin_memory: bool = False
    dataloader_num_workers: int = 2

    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"

    # Evaluation
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

    # Data processing
    max_seq_length: int = 2048
    group_by_length: bool = True
    remove_unused_columns: bool = False

    # Polish specific
    enable_polish_evaluation: bool = True
    polish_eval_samples: int = 100

    # Advanced
    deepspeed: Optional[str] = None
    report_to: List[str] = None
    run_name: Optional[str] = None

    # Seeds for reproducibility
    seed: int = 42
    data_seed: int = 42

    def __post_init__(self):
        """Post-initialization setup."""
        if self.report_to is None:
            self.report_to = ["tensorboard"]

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Set run name if not provided
        if self.run_name is None:
            timestamp = int(time.time())
            self.run_name = f"wronai-training-{timestamp}"

    def to_training_arguments(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments."""
        args_dict = asdict(self)

        # Remove custom fields not supported by TrainingArguments
        custom_fields = {
            "enable_polish_evaluation",
            "polish_eval_samples",
            "max_seq_length",
            "early_stopping_patience",
            "early_stopping_threshold",
        }

        for field in custom_fields:
            args_dict.pop(field, None)

        return TrainingArguments(**args_dict)


class WronAITrainer:
    """
    Custom trainer for WronAI models with Polish language optimizations.
    """

    def __init__(
        self,
        model: WronAIModel,
        config: TrainingConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        data_collator: Optional[Any] = None,
        callbacks: Optional[List[Any]] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # Setup data collator
        if data_collator is None:
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=model.tokenizer, mlm=False, pad_to_multiple_of=8
            )
        else:
            self.data_collator = data_collator

        # Setup callbacks
        self.callbacks = callbacks or []
        self._setup_default_callbacks()

        # Initialize trainer
        self.trainer = None
        self._setup_trainer()

        logger.info(f"WronAI trainer initialized for {model.config.model_name}")

    def _setup_default_callbacks(self):
        """Setup default training callbacks."""
        # Early stopping
        if self.config.early_stopping_patience > 0:
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=self.config.early_stopping_patience,
                early_stopping_threshold=self.config.early_stopping_threshold,
            )
            self.callbacks.append(early_stopping)

        # Memory monitoring
        if torch.cuda.is_available():
            memory_callback = MemoryMonitorCallback()
            self.callbacks.append(memory_callback)

        # Polish evaluation
        if self.config.enable_polish_evaluation and self.eval_dataset:
            polish_callback = PolishEvaluationCallback(
                eval_samples=self.config.polish_eval_samples
            )
            self.callbacks.append(polish_callback)

    def _setup_trainer(self):
        """Setup HuggingFace trainer."""
        training_args = self.config.to_training_arguments()

        self.trainer = Trainer(
            model=self.model.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.model.tokenizer,
            data_collator=self.data_collator,
            callbacks=self.callbacks,
            compute_metrics=self._compute_metrics if self.eval_dataset else None,
        )

    def train(
        self,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_dataset: Training dataset (overrides initialization)
            eval_dataset: Evaluation dataset (overrides initialization)
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training results
        """
        logger.info("Starting WronAI training...")

        # Update datasets if provided
        if train_dataset:
            self.train_dataset = train_dataset
            self.trainer.train_dataset = train_dataset

        if eval_dataset:
            self.eval_dataset = eval_dataset
            self.trainer.eval_dataset = eval_dataset

        # Verify we have training data
        if not self.train_dataset:
            raise ValueError("No training dataset provided")

        # Log training info
        self._log_training_info()

        try:
            # Clear cache before training
            clear_cache()

            # Start training
            train_result = self.trainer.train(
                resume_from_checkpoint=resume_from_checkpoint
            )

            # Save final model
            self.save_model()

            # Log final results
            self._log_training_results(train_result)

            return train_result

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Cleanup
            clear_cache()

    def evaluate(
        self, eval_dataset: Optional[Dataset] = None, metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        """
        Evaluate the model.

        Args:
            eval_dataset: Evaluation dataset
            metric_key_prefix: Prefix for metric names

        Returns:
            Evaluation metrics
        """
        if eval_dataset:
            self.trainer.eval_dataset = eval_dataset

        if not self.trainer.eval_dataset:
            raise ValueError("No evaluation dataset provided")

        logger.info("Starting evaluation...")

        try:
            eval_result = self.trainer.evaluate(metric_key_prefix=metric_key_prefix)

            logger.info(f"Evaluation completed: {eval_result}")
            return eval_result

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def predict(self, test_dataset: Dataset, metric_key_prefix: str = "test") -> Any:
        """
        Generate predictions on test dataset.

        Args:
            test_dataset: Test dataset
            metric_key_prefix: Prefix for metric names

        Returns:
            Prediction results
        """
        logger.info("Starting prediction...")

        try:
            predictions = self.trainer.predict(
                test_dataset=test_dataset, metric_key_prefix=metric_key_prefix
            )

            logger.info("Prediction completed")
            return predictions

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def save_model(self, output_dir: Optional[str] = None):
        """
        Save the trained model.

        Args:
            output_dir: Output directory (defaults to config output_dir)
        """
        save_dir = output_dir or self.config.output_dir

        logger.info(f"Saving model to {save_dir}")

        # Save model using WronAI method
        self.model.save_pretrained(save_dir)

        # Also save trainer state
        self.trainer.save_model(save_dir)

        # Save training config
        config_path = os.path.join(save_dir, "training_config.json")
        import json

        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        logger.info("Model saved successfully")

    def _compute_metrics(self, eval_preds) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            eval_preds: Evaluation predictions

        Returns:
            Computed metrics
        """
        predictions, labels = eval_preds

        # Basic perplexity calculation
        import numpy as np

        # Flatten predictions and labels
        predictions = predictions.reshape(-1, predictions.shape[-1])
        labels = labels.reshape(-1)

        # Filter out ignored tokens (usually -100)
        mask = labels != -100
        predictions = predictions[mask]
        labels = labels[mask]

        # Calculate perplexity
        if len(labels) > 0:
            # Convert logits to probabilities
            probs = torch.softmax(torch.tensor(predictions), dim=-1)

            # Get probabilities for actual tokens
            selected_probs = probs[range(len(labels)), labels]

            # Calculate perplexity
            log_probs = torch.log(selected_probs + 1e-10)
            perplexity = torch.exp(-log_probs.mean()).item()
        else:
            perplexity = float("inf")

        return {"perplexity": perplexity, "num_tokens": len(labels)}

    def _log_training_info(self):
        """Log training information."""
        logger.info("=== Training Information ===")
        logger.info(f"Model: {self.model.config.model_name}")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info(
            f"Training samples: {len(self.train_dataset) if self.train_dataset else 0}"
        )
        logger.info(
            f"Evaluation samples: {len(self.eval_dataset) if self.eval_dataset else 0}"
        )
        logger.info(f"Epochs: {self.config.num_train_epochs}")
        logger.info(f"Batch size: {self.config.per_device_train_batch_size}")
        logger.info(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"Learning rate: {self.config.learning_rate}")

        # Model info
        if hasattr(self.model, "print_parameter_info"):
            self.model.print_parameter_info()

        # Memory info
        if hasattr(self.model, "get_memory_usage"):
            memory_info = self.model.get_memory_usage()
            logger.info(
                f"GPU memory usage: {memory_info.get('gpu_memory_percent', 0):.1f}%"
            )

    def _log_training_results(self, train_result):
        """Log training results."""
        logger.info("=== Training Results ===")

        if hasattr(train_result, "training_loss"):
            logger.info(f"Final training loss: {train_result.training_loss:.4f}")

        if hasattr(train_result, "metrics"):
            for key, value in train_result.metrics.items():
                logger.info(f"{key}: {value}")

        # Log final evaluation if available
        if self.eval_dataset:
            try:
                final_eval = self.evaluate()
                logger.info("Final evaluation results:")
                for key, value in final_eval.items():
                    logger.info(f"  {key}: {value:.4f}")
            except Exception as e:
                logger.warning(f"Final evaluation failed: {e}")

    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state."""
        if not self.trainer.state:
            return {}

        return {
            "epoch": self.trainer.state.epoch,
            "global_step": self.trainer.state.global_step,
            "max_steps": self.trainer.state.max_steps,
            "num_train_epochs": self.trainer.state.num_train_epochs,
            "train_batch_size": self.trainer.state.train_batch_size,
            "eval_batch_size": self.trainer.state.eval_batch_size,
            "logging_steps": self.trainer.state.logging_steps,
            "save_steps": self.trainer.state.save_steps,
            "eval_steps": self.trainer.state.eval_steps,
            "best_metric": self.trainer.state.best_metric,
            "best_model_checkpoint": self.trainer.state.best_model_checkpoint,
            "log_history": (
                self.trainer.state.log_history[-5:]
                if self.trainer.state.log_history
                else []
            ),
        }

    def resume_training(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Resume training from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Training results
        """
        logger.info(f"Resuming training from {checkpoint_path}")

        return self.train(resume_from_checkpoint=checkpoint_path)

    def hyperparameter_search(
        self,
        hp_space: Dict[str, Any],
        compute_objective: Optional[callable] = None,
        n_trials: int = 10,
        direction: str = "minimize",
    ) -> Any:
        """
        Perform hyperparameter search.

        Args:
            hp_space: Hyperparameter search space
            compute_objective: Function to compute objective
            n_trials: Number of trials
            direction: Optimization direction

        Returns:
            Best hyperparameters
        """
        try:
            import optuna
        except ImportError:
            raise ImportError("optuna is required for hyperparameter search")

        logger.info(f"Starting hyperparameter search with {n_trials} trials")

        def objective(trial):
            # Sample hyperparameters
            trial_config = TrainingConfig(**asdict(self.config))

            for param, values in hp_space.items():
                if isinstance(values, list):
                    value = trial.suggest_categorical(param, values)
                elif isinstance(values, tuple) and len(values) == 2:
                    if isinstance(values[0], float):
                        value = trial.suggest_float(param, values[0], values[1])
                    else:
                        value = trial.suggest_int(param, values[0], values[1])
                else:
                    continue

                setattr(trial_config, param, value)

            # Update trainer config
            self.config = trial_config
            self._setup_trainer()

            # Train and evaluate
            train_result = self.train()

            if compute_objective:
                return compute_objective(train_result)
            else:
                # Default: minimize validation loss
                eval_result = self.evaluate()
                return eval_result.get("eval_loss", float("inf"))

        # Create study
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)

        logger.info(f"Best hyperparameters: {study.best_params}")
        logger.info(f"Best value: {study.best_value}")

        return study.best_params

    def create_model_card(self) -> str:
        """Create a model card for the trained model."""
        model_card = f"""
# WronAI Model Card

## Model Details
- **Model Name**: {self.model.config.model_name}
- **Model Type**: Polish Language Model
- **Architecture**: {getattr(self.model, 'model_type', 'WronAI')}
- **Parameters**: {self.model.get_parameter_count():,}
- **Trainable Parameters**: {self.model.get_trainable_parameter_count():,}

## Training Details
- **Training Data**: Polish language instruction dataset
- **Training Epochs**: {self.config.num_train_epochs}
- **Batch Size**: {self.config.per_device_train_batch_size}
- **Learning Rate**: {self.config.learning_rate}
- **Optimizer**: {self.config.optim}

## Performance
- **Memory Usage**: {getattr(self.model, '_is_quantized', False) and '4-bit quantized' or 'Full precision'}
- **LoRA**: {getattr(self.model, '_has_lora', False) and 'Enabled' or 'Disabled'}

## Usage
```python
from wronai import load_model
model = load_model("{self.config.output_dir}")
response = model.generate("Jak się masz?")
```

## Limitations
- Optimized for Polish language
- May require specific prompt formatting
- Performance depends on hardware configuration
"""

        return model_card.strip()
