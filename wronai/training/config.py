"""
Konfiguracja treningu dla modeli WronAI.

Ten moduł zawiera klasy konfiguracyjne dla różnych aspektów treningu modeli WronAI,
w tym konfigurację danych, modelu, treningu i ewaluacji.
"""

import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Union, Any

from ..utils.logging import get_logger
from .optimizer import OptimizerConfig
from .scheduler import SchedulerConfig

logger = get_logger(__name__)


@dataclass
class DataConfig:
    """Konfiguracja danych treningowych."""

    # Ścieżki do danych
    train_data_path: str = "data/processed/train.jsonl"
    eval_data_path: Optional[str] = "data/processed/eval.jsonl"
    test_data_path: Optional[str] = "data/processed/test.jsonl"

    # Parametry przetwarzania danych
    max_seq_length: int = 2048
    truncation: bool = True
    padding: str = "max_length"  # "max_length", "longest", "do_not_pad"
    pad_to_multiple_of: int = 8

    # Parametry ładowania danych
    batch_size: int = 1
    eval_batch_size: int = 1
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True

    # Parametry dla języka polskiego
    polish_data_ratio: float = 0.7  # Proporcja polskich danych w każdym batchu
    polish_data_paths: List[str] = field(default_factory=list)
    english_data_paths: List[str] = field(default_factory=list)

    # Parametry augmentacji danych
    enable_augmentation: bool = False
    augmentation_probability: float = 0.1
    augmentation_types: List[str] = field(default_factory=lambda: ["synonym", "backtranslation"])

    # Parametry dla curriculum learning
    enable_curriculum: bool = False
    curriculum_strategy: str = "length"  # "length", "perplexity", "custom"
    curriculum_steps: List[int] = field(default_factory=list)  # Kroki, w których zmieniamy trudność

    def __post_init__(self):
        """Walidacja i inicjalizacja po utworzeniu instancji."""
        # Sprawdź, czy pliki danych istnieją
        for path in [self.train_data_path, self.eval_data_path, self.test_data_path]:
            if path and not os.path.exists(path):
                logger.warning(f"Plik danych nie istnieje: {path}")

        # Ustaw domyślne ścieżki dla polskich i angielskich danych, jeśli nie podano
        if not self.polish_data_paths and os.path.exists("data/processed/polish"):
            self.polish_data_paths = [
                os.path.join("data/processed/polish", f)
                for f in os.listdir("data/processed/polish")
                if f.endswith(".jsonl")
            ]

        if not self.english_data_paths and os.path.exists("data/processed/english"):
            self.english_data_paths = [
                os.path.join("data/processed/english", f)
                for f in os.listdir("data/processed/english")
                if f.endswith(".jsonl")
            ]


@dataclass
class ModelConfig:
    """Konfiguracja modelu."""

    # Podstawowe parametry modelu
    model_name_or_path: str = "mistralai/Mistral-7B-v0.1"
    model_type: str = "mistral"  # "mistral", "llama", "phi", "gemma"
    tokenizer_name_or_path: Optional[str] = None  # Jeśli None, używamy model_name_or_path

    # Parametry kwantyzacji
    quantization: Optional[str] = None  # None, "4bit", "8bit"
    quantization_type: str = "nf4"  # "nf4", "fp4", "int8"

    # Parametry LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Parametry dla języka polskiego
    polish_token_ids: List[int] = field(default_factory=list)
    polish_token_upweight: float = 1.2  # Waga dla polskich tokenów w treningu

    # Parametry wydajnościowe
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True
    bf16: bool = True
    fp16: bool = False

    # Parametry zaawansowane
    trust_remote_code: bool = False
    use_auth_token: bool = False
    revision: str = "main"
    attn_implementation: Optional[str] = "flash_attention_2"  # "eager", "sdpa", "flash_attention_2"

    def __post_init__(self):
        """Walidacja i inicjalizacja po utworzeniu instancji."""
        # Ustaw tokenizer_name_or_path jeśli nie podano
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path

        # Sprawdź zgodność kwantyzacji i precyzji
        if self.quantization and self.bf16 and self.fp16:
            logger.warning("Używanie kwantyzacji z bf16 i fp16 jednocześnie może powodować problemy.")

        # Sprawdź zgodność flash attention z typem modelu
        if self.use_flash_attention and self.model_type not in ["mistral", "llama"]:
            logger.warning(f"Flash Attention może nie być wspierane dla modelu typu {self.model_type}.")


@dataclass
class TrainingConfig:
    """Główna konfiguracja treningu."""

    # Podstawowe parametry treningu
    output_dir: str = "checkpoints/wronai"
    num_train_epochs: int = 3
    max_steps: int = -1  # -1 oznacza trenowanie przez wszystkie epoki
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16

    # Parametry optymalizacji
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Parametry logowania i zapisywania
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"  # "steps", "epoch", "no"
    save_strategy: str = "steps"  # "steps", "epoch", "no"

    # Parametry ewaluacji
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Parametry early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

    # Parametry dla języka polskiego
    enable_polish_evaluation: bool = True
    polish_eval_samples: int = 100

    # Parametry zaawansowane
    deepspeed: Optional[str] = None
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    run_name: Optional[str] = None
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None

    # Parametry reprodukowalności
    seed: int = 42
    data_seed: int = 42

    def __post_init__(self):
        """Walidacja i inicjalizacja po utworzeniu instancji."""
        # Utwórz katalog wyjściowy
        os.makedirs(self.output_dir, exist_ok=True)

        # Ustaw nazwę uruchomienia, jeśli nie podano
        if self.run_name is None:
            import time
            timestamp = int(time.time())
            self.run_name = f"wronai-training-{timestamp}"

        # Ustaw hub_model_id, jeśli push_to_hub=True i nie podano hub_model_id
        if self.push_to_hub and not self.hub_model_id:
            self.hub_model_id = f"wronai/llm-{self.run_name}"


@dataclass
class EvaluationConfig:
    """Konfiguracja ewaluacji modelu."""

    # Podstawowe parametry ewaluacji
    metrics: List[str] = field(default_factory=lambda: ["perplexity", "accuracy"])
    eval_batch_size: int = 1
    max_eval_samples: Optional[int] = None

    # Parametry generacji
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 256
    num_beams: int = 1
    repetition_penalty: float = 1.1

    # Parametry dla języka polskiego
    polish_benchmarks: List[str] = field(
        default_factory=lambda: ["klej", "poleval", "polish_qa"]
    )
    evaluate_polish_grammar: bool = True
    evaluate_polish_coherence: bool = True

    # Parametry dla ewaluacji ludzkiej
    human_eval_samples: int = 10
    human_eval_criteria: List[str] = field(
        default_factory=lambda: ["correctness", "fluency", "coherence", "relevance"]
    )


@dataclass
class WronAITrainingConfig:
    """Główna konfiguracja treningu WronAI, zawierająca wszystkie podkonfiguracje."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Metadane
    version: str = "1.0.0"
    description: str = "Konfiguracja treningu modelu WronAI"
    author: str = "WronAI Team"

    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje konfigurację do słownika."""
        return {
            "data": asdict(self.data),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "evaluation": asdict(self.evaluation),
            "version": self.version,
            "description": self.description,
            "author": self.author,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "WronAITrainingConfig":
        """Tworzy konfigurację ze słownika."""
        data_config = DataConfig(**config_dict.get("data", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        
        # Obsługa zagnieżdżonych konfiguracji
        training_dict = config_dict.get("training", {})
        if "optimizer" in training_dict:
            training_dict["optimizer"] = OptimizerConfig(**training_dict["optimizer"])
        if "scheduler" in training_dict:
            training_dict["scheduler"] = SchedulerConfig(**training_dict["scheduler"])
        training_config = TrainingConfig(**training_dict)
        
        evaluation_config = EvaluationConfig(**config_dict.get("evaluation", {}))
        
        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            evaluation=evaluation_config,
            version=config_dict.get("version", "1.0.0"),
            description=config_dict.get("description", ""),
            author=config_dict.get("author", ""),
        )

    def save(self, path: str) -> None:
        """Zapisuje konfigurację do pliku YAML."""
        import yaml
        
        config_dict = self.to_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Zapisano konfigurację do {path}")

    @classmethod
    def load(cls, path: str) -> "WronAITrainingConfig":
        """Ładuje konfigurację z pliku YAML."""
        import yaml
        
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        config = cls.from_dict(config_dict)
        logger.info(f"Załadowano konfigurację z {path}")
        return config


def get_default_config() -> WronAITrainingConfig:
    """Zwraca domyślną konfigurację treningu WronAI."""
    return WronAITrainingConfig()


def merge_configs(base_config: WronAITrainingConfig, override_config: Dict[str, Any]) -> WronAITrainingConfig:
    """Łączy bazową konfigurację z nadpisującymi wartościami."""
    from copy import deepcopy
    import yaml
    
    # Konwertuj bazową konfigurację do słownika
    base_dict = base_config.to_dict()
    
    # Funkcja rekurencyjna do łączenia słowników
    def deep_merge(base: Dict, override: Dict) -> Dict:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    # Wykonaj głębokie łączenie
    merged_dict = deep_merge(deepcopy(base_dict), override_config)
    
    # Utwórz nową konfigurację z połączonego słownika
    return WronAITrainingConfig.from_dict(merged_dict)


def load_config_from_yaml(path: str) -> Dict[str, Any]:
    """Ładuje konfigurację z pliku YAML jako słownik."""
    import yaml
    
    with open(path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    
    return config_dict
