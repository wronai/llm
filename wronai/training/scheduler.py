"""
Zaawansowane schedulery treningowe dla modeli WronAI.

Moduł zawiera specjalizowane schedulery do treningu modeli językowych,
w tym schedulery dostosowane do specyfiki języka polskiego.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Callable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SchedulerConfig:
    """Konfiguracja dla schedulerów treningowych."""
    
    # Podstawowe ustawienia
    scheduler_type: str = "cosine"  # cosine, linear, polynomial, constant, warmup_cosine, plateau
    num_training_steps: int = 1000
    num_warmup_steps: int = 0
    warmup_ratio: float = 0.03  # Używane jeśli num_warmup_steps = 0
    
    # Zaawansowane ustawienia
    min_lr_ratio: float = 0.0  # Minimalny współczynnik LR (jako część początkowego LR)
    cooldown_steps: int = 0  # Liczba kroków z stałym LR po zakończeniu głównego schedulera
    
    # Ustawienia dla plateau
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    plateau_threshold: float = 1e-4
    plateau_threshold_mode: str = "rel"
    plateau_cooldown: int = 0
    
    # Ustawienia dla polynomial
    polynomial_power: float = 1.0
    
    # Ustawienia dla języka polskiego
    polish_boost_steps: List[int] = field(default_factory=list)  # Kroki, w których zwiększamy LR dla lepszego uczenia polskich tokenów
    polish_boost_factor: float = 1.2  # Współczynnik zwiększenia LR w polish_boost_steps
    
    def __post_init__(self):
        # Oblicz kroki rozgrzewki na podstawie współczynnika, jeśli nie podano bezpośrednio
        if self.num_warmup_steps == 0 and self.warmup_ratio > 0:
            self.num_warmup_steps = int(self.num_training_steps * self.warmup_ratio)
            logger.info(f"Ustawiono {self.num_warmup_steps} kroków rozgrzewki na podstawie współczynnika {self.warmup_ratio}")


class WarmupCosineScheduler(_LRScheduler):
    """Scheduler z liniową rozgrzewką i schładzaniem cosinusowym."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        min_lr_ratio: float = 0.0,
        cooldown_steps: int = 0,
        last_epoch: int = -1,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        self.min_lr_ratio = min_lr_ratio
        self.cooldown_steps = cooldown_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            # Liniowa rozgrzewka
            return [base_lr * self.last_epoch / max(1, self.num_warmup_steps) for base_lr in self.base_lrs]
        elif self.last_epoch > (self.num_training_steps - self.cooldown_steps):
            # Okres cooldown - stały minimalny LR
            return [base_lr * self.min_lr_ratio for base_lr in self.base_lrs]
        else:
            # Schładzanie cosinusowe
            progress = float(self.last_epoch - self.num_warmup_steps) / \
                      float(max(1, self.num_training_steps - self.num_warmup_steps - self.cooldown_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * self.num_cycles * 2.0 * progress))
            decayed_lr_ratio = self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_decay
            return [base_lr * decayed_lr_ratio for base_lr in self.base_lrs]


class PolishAwareScheduler(_LRScheduler):
    """Scheduler świadomy specyfiki języka polskiego.
    
    Zwiększa learning rate w określonych krokach treningu, aby lepiej uczyć
    reprezentacje specyficzne dla języka polskiego.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        base_scheduler: _LRScheduler,
        boost_steps: List[int],
        boost_factor: float = 1.2,
        last_epoch: int = -1,
    ):
        self.base_scheduler = base_scheduler
        self.boost_steps = sorted(boost_steps)
        self.boost_factor = boost_factor
        self.boost_active = False
        self.current_boost_idx = 0
        self.boost_duration = 50  # Domyślna długość boostu (w krokach)
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        # Najpierw pobierz bazowe LR z głównego schedulera
        base_lrs = self.base_scheduler.get_lr()
        
        # Sprawdź, czy jesteśmy w kroku boost
        if self.current_boost_idx < len(self.boost_steps):
            boost_start = self.boost_steps[self.current_boost_idx]
            boost_end = boost_start + self.boost_duration
            
            if self.last_epoch == boost_start:
                logger.info(f"Aktywowano boost LR dla języka polskiego w kroku {self.last_epoch}")
                self.boost_active = True
            elif self.last_epoch == boost_end:
                logger.info(f"Deaktywowano boost LR dla języka polskiego w kroku {self.last_epoch}")
                self.boost_active = False
                self.current_boost_idx += 1
        
        # Jeśli boost jest aktywny, zwiększ LR
        if self.boost_active:
            return [lr * self.boost_factor for lr in base_lrs]
        else:
            return base_lrs
    
    def step(self, epoch=None):
        # Najpierw wykonaj krok bazowego schedulera
        self.base_scheduler.step(epoch)
        super().step(epoch)


class CurriculumLearningScheduler(_LRScheduler):
    """Scheduler implementujący curriculum learning.
    
    Dostosowuje learning rate w zależności od trudności próbek treningowych.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        base_scheduler: _LRScheduler,
        difficulty_fn: Callable[[int], float],
        max_difficulty_factor: float = 1.5,
        last_epoch: int = -1,
    ):
        self.base_scheduler = base_scheduler
        self.difficulty_fn = difficulty_fn  # Funkcja zwracająca trudność (0-1) dla danego kroku
        self.max_difficulty_factor = max_difficulty_factor
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        # Pobierz bazowe LR
        base_lrs = self.base_scheduler.get_lr()
        
        # Oblicz współczynnik trudności (0-1)
        difficulty = self.difficulty_fn(self.last_epoch)
        
        # Dostosuj LR w zależności od trudności
        # Trudniejsze próbki = niższy LR dla stabilności
        difficulty_factor = 1.0 - (difficulty * (1.0 - 1.0/self.max_difficulty_factor))
        
        return [lr * difficulty_factor for lr in base_lrs]
    
    def step(self, epoch=None):
        # Najpierw wykonaj krok bazowego schedulera
        self.base_scheduler.step(epoch)
        super().step(epoch)


def create_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig
) -> _LRScheduler:
    """Tworzy scheduler na podstawie konfiguracji.
    
    Args:
        optimizer: Optymizator
        config: Konfiguracja schedulera
        
    Returns:
        Skonfigurowany scheduler
    """
    logger.info(f"Tworzenie schedulera typu: {config.scheduler_type}")
    logger.info(f"Kroki treningu: {config.num_training_steps}, kroki rozgrzewki: {config.num_warmup_steps}")
    
    # Utwórz bazowy scheduler
    if config.scheduler_type == "cosine":
        scheduler = WarmupCosineScheduler(
            optimizer,
            num_warmup_steps=config.num_warmup_steps,
            num_training_steps=config.num_training_steps,
            min_lr_ratio=config.min_lr_ratio,
            cooldown_steps=config.cooldown_steps
        )
    elif config.scheduler_type == "linear":
        def lr_lambda(current_step):
            if current_step < config.num_warmup_steps:
                return float(current_step) / float(max(1, config.num_warmup_steps))
            return max(
                config.min_lr_ratio,
                float(config.num_training_steps - current_step) / 
                float(max(1, config.num_training_steps - config.num_warmup_steps))
            )
        scheduler = LambdaLR(optimizer, lr_lambda)
    elif config.scheduler_type == "polynomial":
        def lr_lambda(current_step):
            if current_step < config.num_warmup_steps:
                return float(current_step) / float(max(1, config.num_warmup_steps))
            progress = float(current_step - config.num_warmup_steps) / float(
                max(1, config.num_training_steps - config.num_warmup_steps)
            )
            remaining = max(0.0, (1.0 - progress) ** config.polynomial_power)
            return max(config.min_lr_ratio, remaining)
        scheduler = LambdaLR(optimizer, lr_lambda)
    elif config.scheduler_type == "constant":
        def lr_lambda(current_step):
            if current_step < config.num_warmup_steps:
                return float(current_step) / float(max(1, config.num_warmup_steps))
            return 1.0
        scheduler = LambdaLR(optimizer, lr_lambda)
    elif config.scheduler_type == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.plateau_factor,
            patience=config.plateau_patience,
            threshold=config.plateau_threshold,
            threshold_mode=config.plateau_threshold_mode,
            cooldown=config.plateau_cooldown,
            min_lr=config.min_lr_ratio * optimizer.param_groups[0]["lr"],
            verbose=True
        )
        logger.info("Utworzono scheduler ReduceLROnPlateau - wymaga ręcznych wywołań step() z metrykami")
        return scheduler  # Zwróć bezpośrednio, nie opakowuj w PolishAwareScheduler
    else:
        warnings.warn(f"Nieznany typ schedulera: {config.scheduler_type}, używam stałego LR")
        scheduler = LambdaLR(optimizer, lambda _: 1.0)
    
    # Jeśli zdefiniowano kroki boost dla języka polskiego, opakuj w PolishAwareScheduler
    if config.polish_boost_steps:
        logger.info(f"Dodaję Polish-aware scheduling z krokami boost: {config.polish_boost_steps}")
        scheduler = PolishAwareScheduler(
            optimizer,
            scheduler,
            boost_steps=config.polish_boost_steps,
            boost_factor=config.polish_boost_factor
        )
    
    return scheduler


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr_ratio: float = 0.0,
    last_epoch: int = -1
) -> _LRScheduler:
    """Tworzy scheduler z liniową rozgrzewką i schładzaniem cosinusowym.
    
    Args:
        optimizer: Optymizator
        num_warmup_steps: Liczba kroków rozgrzewki
        num_training_steps: Całkowita liczba kroków treningu
        num_cycles: Liczba cykli cosinusowych
        min_lr_ratio: Minimalny współczynnik LR (jako część początkowego LR)
        last_epoch: Ostatni wykonany epoch
        
    Returns:
        Scheduler
    """
    return WarmupCosineScheduler(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_ratio=min_lr_ratio,
        last_epoch=last_epoch
    )


def get_polynomial_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    power: float = 1.0,
    min_lr_ratio: float = 0.0,
    last_epoch: int = -1
) -> LambdaLR:
    """Tworzy scheduler z liniową rozgrzewką i schładzaniem wielomianowym.
    
    Args:
        optimizer: Optymizator
        num_warmup_steps: Liczba kroków rozgrzewki
        num_training_steps: Całkowita liczba kroków treningu
        power: Wykładnik wielomianu
        min_lr_ratio: Minimalny współczynnik LR (jako część początkowego LR)
        last_epoch: Ostatni wykonany epoch
        
    Returns:
        Scheduler
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        remaining = max(0.0, (1.0 - progress) ** power)
        return max(min_lr_ratio, remaining)
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)
