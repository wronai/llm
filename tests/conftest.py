"""
Pytest configuration and fixtures for WronAI tests.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
import torch
from transformers import AutoTokenizer

from wronai.models import ModelConfig
from wronai.utils.logging import setup_logging


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Setup logging for tests."""
    setup_logging(level="WARNING", log_file=None)


@pytest.fixture(scope="session")
def device() -> str:
    """Get the best available device for testing."""
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_polish_texts():
    """Sample Polish texts for testing."""
    return [
        "Witaj świecie! Jak się masz?",
        "To jest test polskiego modelu językowego.",
        "Sztuczna inteligencja może pomóc w rozwiązywaniu problemów.",
        "Polska to piękny kraj w Europie Środkowej.",
        "Machine learning i deep learning to fascynujące dziedziny."
    ]


@pytest.fixture
def sample_instruction_data():
    """Sample instruction dataset for testing."""
    return [
        {
            "instruction": "Przetłumacz na język angielski: 'Miło Cię poznać'",
            "response": "'Miło Cię poznać' w języku angielskim to 'Nice to meet you'."
        },
        {
            "instruction": "Wyjaśnij czym jest sztuczna inteligencja",
            "response": "Sztuczna inteligencja to dziedzina informatyki zajmująca się tworzeniem systemów zdolnych do wykonywania zadań wymagających ludzkiej inteligencji."
        },
        {
            "instruction": "Podaj przykład polskiego przysłowia",
            "response": "Przykładem polskiego przysłowia jest: 'Co ma wisieć, nie utonie'."
        }
    ]


@pytest.fixture
def small_tokenizer():
    """Small tokenizer for testing."""
    return AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")


@pytest.fixture
def test_model_config():
    """Test model configuration."""
    return ModelConfig(
        model_name="microsoft/DialoGPT-small",
        max_sequence_length=128,
        lora_r=8,
        lora_alpha=16,
        quantization_enabled=False,  # Disable for testing
        polish_tokens=["<test>", "</test>"]
    )


@pytest.fixture
def mock_training_data(temp_dir, sample_instruction_data):
    """Create mock training data file."""
    import json

    data_file = temp_dir / "training_data.json"
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(sample_instruction_data, f, ensure_ascii=False, indent=2)

    return str(data_file)


@pytest.fixture
def mock_polish_corpus(temp_dir, sample_polish_texts):
    """Create mock Polish corpus file."""
    import json

    corpus_file = temp_dir / "polish_corpus.jsonl"
    with open(corpus_file, 'w', encoding='utf-8') as f:
        for text in sample_polish_texts:
            json.dump({"text": text, "source": "test"}, f, ensure_ascii=False)
            f.write('\n')

    return str(corpus_file)


@pytest.fixture
def skip_if_no_gpu():
    """Skip test if GPU is not available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")


@pytest.fixture
def skip_if_no_transformers():
    """Skip test if transformers is not available."""
    try:
        import transformers
    except ImportError:
        pytest.skip("transformers not available")


@pytest.fixture(scope="session")
def test_model_path(tmp_path_factory):
    """Path for test model storage."""
    return tmp_path_factory.mktemp("test_models")


@pytest.fixture
def minimal_training_config():
    """Minimal training configuration for tests."""
    return {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-3,
        "logging_steps": 1,
        "save_steps": 10,
        "eval_steps": 10,
        "max_steps": 5,  # Very short training
        "warmup_ratio": 0.0,
        "weight_decay": 0.0
    }


@pytest.fixture
def polish_test_prompts():
    """Polish test prompts for generation testing."""
    return [
        "Opowiedz o Polsce:",
        "Jakie są tradycyjne polskie potrawy?",
        "Wyjaśnij pojęcie sztucznej inteligencji:",
        "Przetłumacz na angielski: 'Dziękuję bardzo'",
        "Napisz krótki wiersz o jesieni:"
    ]


# Pytest markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )
    config.addinivalue_line(
        "markers", "polish: mark test as Polish language specific"
    )


# Auto-use fixtures for cleanup
@pytest.fixture(autouse=True)
def cleanup_cuda_cache():
    """Clean CUDA cache after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def set_test_environment():
    """Set environment variables for testing."""
    # Disable wandb logging during tests
    os.environ["WANDB_MODE"] = "disabled"
    # Set test mode
    os.environ["WRONAI_TEST_MODE"] = "true"
    # Disable tokenizers parallelism warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    yield

    # Cleanup
    for key in ["WANDB_MODE", "WRONAI_TEST_MODE"]:
        os.environ.pop(key, None)