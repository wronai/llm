"""
Optimizer module for WronAI training.

Provides specialized optimizers and learning rate schedulers for efficient
training of large language models, with specific optimizations for Polish language.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Callable

import torch
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizerConfig:
    """Configuration for optimizers and schedulers."""
    
    # Optimizer settings
    optimizer_type: str = "adamw"  # adamw, adafactor, lion, sophia
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Scheduler settings
    lr_scheduler_type: str = "cosine"  # linear, cosine, polynomial, constant, plateau
    warmup_ratio: float = 0.03
    warmup_steps: int = 0  # Will use warmup_ratio if this is 0
    num_training_steps: int = 1000  # Total training steps
    
    # Advanced settings
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Polish-specific optimization
    polish_token_lr_multiplier: float = 1.2  # Higher learning rate for Polish tokens
    
    # Experimental features
    use_8bit_optimizer: bool = False
    use_paged_optimizer: bool = False
    use_fused_optimizer: bool = False


def get_optimizer(
    model: torch.nn.Module,
    config: OptimizerConfig
) -> Optimizer:
    """
    Create an optimizer based on the provided configuration.
    
    Args:
        model: The model to optimize
        config: Optimizer configuration
        
    Returns:
        Configured optimizer
    """
    # Get model parameters that require gradients
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    
    # Apply different learning rates to Polish tokens if needed
    if config.polish_token_lr_multiplier != 1.0 and hasattr(model, "get_input_embeddings"):
        try:
            # Get token embeddings
            embeddings = model.get_input_embeddings()
            
            # Identify Polish-specific tokens (simplified approach)
            polish_chars = "ąćęłńóśźż"
            polish_tokens = []
            
            if hasattr(model, "tokenizer") and hasattr(model.tokenizer, "encode"):
                # Get token IDs for Polish characters
                for char in polish_chars:
                    token_ids = model.tokenizer.encode(char, add_special_tokens=False)
                    polish_tokens.extend(token_ids)
                
                # Create parameter groups with different learning rates
                polish_params = []
                other_params = []
                
                for p in params:
                    if p is embeddings.weight:
                        # Split embedding parameters
                        polish_mask = torch.zeros_like(p, dtype=torch.bool)
                        polish_mask[polish_tokens] = True
                        
                        # Use parameter groups for different learning rates
                        polish_params.append({
                            "params": [p[polish_mask]],
                            "lr": config.learning_rate * config.polish_token_lr_multiplier
                        })
                        other_params.append({
                            "params": [p[~polish_mask]]
                        })
                    else:
                        other_params.append({"params": [p]})
                
                params = polish_params + other_params
                logger.info(f"Applied Polish token LR multiplier: {config.polish_token_lr_multiplier}")
        except Exception as e:
            logger.warning(f"Failed to apply Polish token LR multiplier: {e}")
            # Fall back to standard parameters
            params = [{"params": params}]
    else:
        # Standard parameter group
        params = [{"params": params}]
    
    # Create optimizer based on type
    if config.optimizer_type.lower() == "adamw":
        # Use 8-bit AdamW if requested (requires bitsandbytes)
        if config.use_8bit_optimizer:
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(
                    params,
                    lr=config.learning_rate,
                    betas=(config.adam_beta1, config.adam_beta2),
                    eps=config.adam_epsilon,
                    weight_decay=config.weight_decay
                )
                logger.info("Using 8-bit AdamW optimizer")
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to standard AdamW")
                optimizer = AdamW(
                    params,
                    lr=config.learning_rate,
                    betas=(config.adam_beta1, config.adam_beta2),
                    eps=config.adam_epsilon,
                    weight_decay=config.weight_decay
                )
        # Use fused AdamW if requested (requires PyTorch with CUDA)
        elif config.use_fused_optimizer and torch.cuda.is_available():
            try:
                from torch.optim.adamw import FusedAdamW
                optimizer = FusedAdamW(
                    params,
                    lr=config.learning_rate,
                    betas=(config.adam_beta1, config.adam_beta2),
                    eps=config.adam_epsilon,
                    weight_decay=config.weight_decay
                )
                logger.info("Using fused AdamW optimizer")
            except (ImportError, AttributeError):
                logger.warning("FusedAdamW not available, falling back to standard AdamW")
                optimizer = AdamW(
                    params,
                    lr=config.learning_rate,
                    betas=(config.adam_beta1, config.adam_beta2),
                    eps=config.adam_epsilon,
                    weight_decay=config.weight_decay
                )
        else:
            # Standard AdamW
            optimizer = AdamW(
                params,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon,
                weight_decay=config.weight_decay
            )
    elif config.optimizer_type.lower() == "adafactor":
        try:
            from transformers.optimization import Adafactor
            optimizer = Adafactor(
                params,
                lr=config.learning_rate,
                eps=(1e-30, 1e-3),
                clip_threshold=1.0,
                decay_rate=-0.8,
                beta1=config.adam_beta1,
                weight_decay=config.weight_decay,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False
            )
            logger.info("Using Adafactor optimizer")
        except ImportError:
            logger.warning("Adafactor not available, falling back to AdamW")
            optimizer = AdamW(
                params,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon,
                weight_decay=config.weight_decay
            )
    elif config.optimizer_type.lower() == "lion":
        try:
            from lion_pytorch import Lion
            optimizer = Lion(
                params,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                weight_decay=config.weight_decay
            )
            logger.info("Using Lion optimizer")
        except ImportError:
            logger.warning("Lion optimizer not available, falling back to AdamW")
            optimizer = AdamW(
                params,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon,
                weight_decay=config.weight_decay
            )
    elif config.optimizer_type.lower() == "sophia":
        try:
            from sophia import SophiaG
            optimizer = SophiaG(
                params,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                weight_decay=config.weight_decay
            )
            logger.info("Using Sophia optimizer")
        except ImportError:
            logger.warning("Sophia optimizer not available, falling back to AdamW")
            optimizer = AdamW(
                params,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon,
                weight_decay=config.weight_decay
            )
    else:
        logger.warning(f"Unknown optimizer type: {config.optimizer_type}, using AdamW")
        optimizer = AdamW(
            params,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay
        )
    
    return optimizer


def get_scheduler(
    optimizer: Optimizer,
    config: OptimizerConfig
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create a learning rate scheduler based on the provided configuration.
    
    Args:
        optimizer: The optimizer
        config: Optimizer configuration
        
    Returns:
        Configured scheduler
    """
    # Calculate warmup steps if using ratio
    if config.warmup_steps == 0 and config.warmup_ratio > 0:
        config.warmup_steps = int(config.num_training_steps * config.warmup_ratio)
    
    logger.info(f"Using {config.warmup_steps} warmup steps with {config.num_training_steps} total steps")
    
    # Create scheduler based on type
    if config.lr_scheduler_type == "linear":
        def lr_lambda(current_step: int):
            if current_step < config.warmup_steps:
                return float(current_step) / float(max(1, config.warmup_steps))
            return max(
                0.0,
                float(config.num_training_steps - current_step) / 
                float(max(1, config.num_training_steps - config.warmup_steps))
            )
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
    elif config.lr_scheduler_type == "cosine":
        def lr_lambda(current_step: int):
            if current_step < config.warmup_steps:
                return float(current_step) / float(max(1, config.warmup_steps))
            progress = float(current_step - config.warmup_steps) / float(
                max(1, config.num_training_steps - config.warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
    elif config.lr_scheduler_type == "polynomial":
        def lr_lambda(current_step: int):
            if current_step < config.warmup_steps:
                return float(current_step) / float(max(1, config.warmup_steps))
            progress = float(current_step - config.warmup_steps) / float(
                max(1, config.num_training_steps - config.warmup_steps)
            )
            return max(0.0, (1.0 - progress) ** 0.5)
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
    elif config.lr_scheduler_type == "constant":
        def lr_lambda(current_step: int):
            if current_step < config.warmup_steps:
                return float(current_step) / float(max(1, config.warmup_steps))
            return 1.0
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
    elif config.lr_scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True
        )
        logger.info("Using ReduceLROnPlateau scheduler - requires manual step() calls with metrics")
    
    else:
        logger.warning(f"Unknown scheduler type: {config.lr_scheduler_type}, using constant")
        scheduler = LambdaLR(optimizer, lambda _: 1.0)
    
    return scheduler


def create_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: OptimizerConfig
) -> tuple:
    """
    Create both optimizer and scheduler in one call.
    
    Args:
        model: The model to optimize
        config: Optimizer configuration
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    return optimizer, scheduler


def get_grouped_params(
    model: torch.nn.Module,
    weight_decay: float = 0.01,
    no_decay_name_list: List[str] = ["bias", "LayerNorm.weight"]
) -> List[Dict]:
    """
    Get grouped parameters for different weight decay settings.
    
    Args:
        model: Model to get parameters from
        weight_decay: Weight decay value
        no_decay_name_list: Parameter names that should not have weight decay
        
    Returns:
        List of parameter dictionaries
    """
    params_with_decay = []
    params_without_decay = []
    
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
            
        if any(nd in n for nd in no_decay_name_list):
            params_without_decay.append(p)
        else:
            params_with_decay.append(p)
            
    return [
        {"params": params_with_decay, "weight_decay": weight_decay},
        {"params": params_without_decay, "weight_decay": 0.0},
    ]


class GradientAccumulator:
    """
    Helper class for gradient accumulation.
    """
    
    def __init__(self, steps: int = 1):
        """
        Initialize gradient accumulator.
        
        Args:
            steps: Number of accumulation steps
        """
        self.steps = max(1, steps)
        self.current_step = 0
        
    def should_update(self) -> bool:
        """
        Check if optimizer should update weights.
        
        Returns:
            True if optimizer should update
        """
        self.current_step += 1
        if self.current_step >= self.steps:
            self.current_step = 0
            return True
        return False
