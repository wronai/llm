"""
Training callbacks for WronAI models.
"""

import time
from typing import Dict, List, Optional, Any

import torch
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

from ..utils.logging import get_logger
from ..utils.memory import get_memory_usage

logger = get_logger(__name__)

class MemoryMonitorCallback(TrainerCallback):
    """
    Callback to monitor GPU memory usage during training.
    """

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.step_count = 0
        self.memory_history = []

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Monitor memory at the beginning of each step."""
        self.step_count += 1

        if self.step_count % self.log_interval == 0:
            memory_info = get_memory_usage()
            self.memory_history.append({
                "step": state.global_step,
                "memory_used_gb": memory_info.get("memory_used_gb", 0),
                "memory_total_gb": memory_info.get("memory_total_gb", 0),
                "memory_percent": memory_info.get("memory_percent", 0)
            })

            logger.info(
                f"Step {state.global_step}: GPU Memory: "
                f"{memory_info.get('memory_used_gb', 0):.1f}GB / "
                f"{memory_info.get('memory_total_gb', 0):.1f}GB "
                f"({memory_info.get('memory_percent', 0):.1f}%)"
            )

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Log memory summary at the end of training."""
        if self.memory_history:
            max_memory = max(item["memory_percent"] for item in self.memory_history)
            avg_memory = sum(item["memory_percent"] for item in self.memory_history) / len(self.memory_history)

            logger.info(f"Memory usage summary - Max: {max_memory:.1f}%, Average: {avg_memory:.1f}%")

class PolishEvaluationCallback(TrainerCallback):
    """
    Callback for Polish-specific evaluation during training.
    """

    def __init__(
        self,
        eval_samples: int = 100,
        polish_prompts: Optional[List[str]] = None,
        eval_interval: int = 500
    ):
        self.eval_samples = eval_samples
        self.eval_interval = eval_interval

        # Default Polish evaluation prompts
        if polish_prompts is None:
            self.polish_prompts = [
                "Opowiedz o Polsce:",
                "Jakie są tradycyjne polskie potrawy?",
                "Wyjaśnij pojęcie sztucznej inteligencji:",
                "Przetłumacz na angielski: 'Dziękuję bardzo'",
                "Napisz krótki wiersz o jesieni:"
            ]
        else:
            self.polish_prompts = polish_prompts

        self.evaluation_results = []

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        """Run Polish evaluation during regular evaluation."""
        try:
            # Simple generation test
            tokenizer = kwargs.get('tokenizer')
            if tokenizer and hasattr(model, 'generate'):

                results = []
                for prompt in self.polish_prompts[:3]:  # Test with first 3 prompts
                    try:
                        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                        with torch.no_grad():
                            outputs = model.generate(
                                inputs.input_ids,
                                max_length=inputs.input_ids.shape[1] + 50,
                                temperature=0.7,
                                do_sample=True,
                                pad_token_id=tokenizer.eos_token_id
                            )

                        generated = tokenizer.decode(
                            outputs[0][inputs.input_ids.shape[1]:],
                            skip_special_tokens=True
                        )

                        results.append({
                            "prompt": prompt,
                            "generated": generated[:100],  # First 100 chars
                            "length": len(generated.split())
                        })

                    except Exception as e:
                        logger.warning(f"Polish evaluation failed for prompt '{prompt}': {e}")

                if results:
                    avg_length = sum(r["length"] for r in results) / len(results)
                    logger.info(f"Polish generation test - Average length: {avg_length:.1f} words")

                    self.evaluation_results.append({
                        "step": state.global_step,
                        "average_length": avg_length,
                        "samples": results
                    })

        except Exception as e:
            logger.warning(f"Polish evaluation callback failed: {e}")

class PerformanceMonitorCallback(TrainerCallback):
    """
    Callback to monitor training performance metrics.
    """

    def __init__(self):
        self.step_times = []
        self.last_step_time = None
        self.tokens_per_second_history = []

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Record step start time."""
        self.last_step_time = time.time()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Calculate and log step performance."""
        if self.last_step_time is not None:
            step_time = time.time() - self.last_step_time
            self.step_times.append(step_time)

            # Estimate tokens per second
            batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
            if torch.distributed.is_initialized():
                batch_size *= torch.distributed.get_world_size()

            # Rough estimate assuming average sequence length
            estimated_tokens = batch_size * 1024  # Assume 1024 tokens per sample
            tokens_per_second = estimated_tokens / step_time
            self.tokens_per_second_history.append(tokens_per_second)

            # Log every 50 steps
            if state.global_step % 50 == 0:
                avg_step_time = sum(self.step_times[-50:]) / min(50, len(self.step_times))
                avg_tokens_per_sec = sum(self.tokens_per_second_history[-50:]) / min(50, len(self.tokens_per_second_history))

                logger.info(
                    f"Performance - Step time: {avg_step_time:.2f}s, "
                    f"Tokens/sec: {avg_tokens_per_sec:.0f}"
                )

class LossMonitorCallback(TrainerCallback):
    """
    Callback to monitor training loss and detect issues.
    """

    def __init__(
        self,
        loss_spike_threshold: float = 2.0,
        loss_window_size: int = 10
    ):
        self.loss_spike_threshold = loss_spike_threshold
        self.loss_window_size = loss_window_size
        self.loss_history = []
        self.loss_spikes = []

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, float] = None, **kwargs):
        """Monitor training loss."""
        if logs and "train_loss" in logs:
            current_loss = logs["train_loss"]
            self.loss_history.append(current_loss)

            # Check for loss spikes
            if len(self.loss_history) >= self.loss_window_size:
                recent_losses = self.loss_history[-self.loss_window_size:]
                avg_loss = sum(recent_losses[:-1]) / (len(recent_losses) - 1)

                if current_loss > avg_loss * self.loss_spike_threshold:
                    spike_info = {
                        "step": state.global_step,
                        "loss": current_loss,
                        "avg_loss": avg_loss,
                        "spike_ratio": current_loss / avg_loss
                    }
                    self.loss_spikes.append(spike_info)

                    logger.warning(
                        f"Loss spike detected at step {state.global_step}: "
                        f"{current_loss:.4f} (avg: {avg_loss:.4f}, "
                        f"ratio: {current_loss/avg_loss:.2f}x)"
                    )

            # Log loss trends every 100 steps
            if state.global_step % 100 == 0 and len(self.loss_history) >= 20:
                recent_avg = sum(self.loss_history[-20:]) / 20
                older_avg = sum(self.loss_history[-40:-20]) / 20 if len(self.loss_history) >= 40 else recent_avg

                trend = "improving" if recent_avg < older_avg else "worsening"
                logger.info(
                    f"Loss trend: {trend} (recent: {recent_avg:.4f}, "
                    f"older: {older_avg:.4f})"
                )

class CheckpointCleanupCallback(TrainerCallback):
    """
    Callback to clean up old checkpoints and manage disk space.
    """

    def __init__(self, keep_best_n: int = 3, cleanup_interval: int = 1000):
        self.keep_best_n = keep_best_n
        self.cleanup_interval = cleanup_interval
        self.checkpoint_scores = []

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Track checkpoint quality and clean up old ones."""
        # Record checkpoint info
        checkpoint_info = {
            "path": f"{args.output_dir}/checkpoint-{state.global_step}",
            "step": state.global_step,
            "best_metric": state.best_metric
        }
        self.checkpoint_scores.append(checkpoint_info)

        # Clean up old checkpoints periodically
        if state.global_step % self.cleanup_interval == 0:
            self._cleanup_old_checkpoints(args)

    def _cleanup_old_checkpoints(self, args: TrainingArguments):
        """Remove old checkpoints keeping only the best ones."""
        import os
        import shutil

        if len(self.checkpoint_scores) <= self.keep_best_n:
            return

        # Sort by best metric (lower is better for loss)
        sorted_checkpoints = sorted(
            self.checkpoint_scores,
            key=lambda x: x["best_metric"] if x["best_metric"] is not None else float('inf')
        )

        # Keep the best N checkpoints
        to_keep = sorted_checkpoints[:self.keep_best_n]
        to_remove = sorted_checkpoints[self.keep_best_n:]

        for checkpoint in to_remove:
            checkpoint_path = checkpoint["path"]
            if os.path.exists(checkpoint_path):
                try:
                    shutil.rmtree(checkpoint_path)
                    logger.info(f"Removed old checkpoint: {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint_path}: {e}")

        # Update checkpoint list
        self.checkpoint_scores = to_keep

class WandBLoggingCallback(TrainerCallback):
    """
    Enhanced Weights & Biases logging callback.
    """

    def __init__(self, log_model_details: bool = True):
        self.log_model_details = log_model_details
        self.wandb_available = False

        try:
            import wandb
            self.wandb = wandb
            self.wandb_available = True
        except ImportError:
            logger.warning("wandb not available. Install with: pip install wandb")

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model, **kwargs):
        """Log model architecture and training setup to wandb."""
        if not self.wandb_available or not self.wandb.run:
            return

        try:
            # Log model details
            if self.log_model_details and hasattr(model, 'get_parameter_count'):
                self.wandb.log({
                    "model/total_parameters": model.get_parameter_count(),
                    "model/trainable_parameters": getattr(model, 'get_trainable_parameter_count', lambda: 0)(),
                })

            # Log training configuration
            config_dict = {
                f"training/{k}": v for k, v in args.__dict__.items()
                if isinstance(v, (int, float, str, bool))
            }
            self.wandb.log(config_dict)

        except Exception as e:
            logger.warning(f"WandB logging failed: {e}")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, float] = None, **kwargs):
        """Enhanced logging with custom metrics."""
        if not self.wandb_available or not self.wandb.run or not logs:
            return

        try:
            # Add memory usage if available
            memory_info = get_memory_usage()
            enhanced_logs = {**logs}

            if memory_info:
                enhanced_logs.update({
                    "system/gpu_memory_percent": memory_info.get("memory_percent", 0),
                    "system/gpu_memory_used_gb": memory_info.get("memory_used_gb", 0)
                })

            self.wandb.log(enhanced_logs, step=state.global_step)

        except Exception as e:
            logger.warning(f"WandB logging failed: {e}")