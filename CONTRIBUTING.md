# Contributing to WronAI 🤝

Dziękujemy za zainteresowanie współpracą z projektem WronAI! Twój wkład pomoże w rozwoju polskiej sztucznej inteligencji.

## 🎯 Jak możesz pomóc

### 🐛 Zgłaszanie błędów
- Sprawdź czy błąd nie został już zgłoszony w [Issues](https://github.com/twoje-repo/WronAI/issues)
- Użyj template dla bug reportów
- Dołącz informacje o systemie i reprodukcję błędu

### 💡 Propozycje funkcji
- Otwórz [Feature Request](https://github.com/twoje-repo/WronAI/issues/new)
- Opisz przypadek użycia i korzyści
- Zaproponuj implementację jeśli możliwe

### 📝 Poprawa dokumentacji
- Poprawki literówek i błędów
- Dodawanie przykładów użycia
- Tłumaczenia dokumentacji

### 🔧 Kod i implementacja
- Nowe features zgodne z roadmap
- Optymalizacje wydajności
- Dodawanie testów

## 🚀 Proces współpracy

### 1. Setup środowiska

```bash
# Fork repozytorium na GitHub
git clone https://github.com/twoje-username/WronAI.git
cd WronAI

# Stwórz virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Zainstaluj dependencies
pip install -e ".[dev]"

# Zainstaluj pre-commit hooks
pre-commit install
```

### 2. Tworzenie branch

```bash
# Stwórz nowy branch dla swojej funkcji
git checkout -b feature/nazwa-funkcji

# Lub dla bugfix
git checkout -b bugfix/opis-bledu

# Lub dla dokumentacji
git checkout -b docs/opis-zmiany
```

### 3. Development

#### Code Style
- Używamy **Black** dla formatowania kodu
- **Flake8** dla linting
- **mypy** dla type checking
- Maksymalna długość linii: 88 znaków

```bash
# Format kodu
black .

# Sprawdź style
flake8 .

# Type checking
mypy scripts/ models/
```

#### Testy
- Pisz testy dla nowych funkcji
- Uruchamiaj testy przed commitem

```bash
# Uruchom wszystkie testy
pytest

# Testy z coverage
pytest --cov=wronai --cov-report=html
```

#### Commitowanie
- Używaj [Conventional Commits](https://www.conventionalcommits.org/)
- Podnaj zmiany w małych, logicznych commitach

```bash
# Przykłady commitów
git commit -m "feat: add Polish BERT tokenizer support"
git commit -m "fix: resolve memory leak in training loop"
git commit -m "docs: update installation instructions"
git commit -m "test: add unit tests for data preprocessing"
git commit -m "perf: optimize model inference speed"
git commit -m "refactor: simplify configuration loading"
```

### 4. Pull Request

#### Przed wysłaniem PR
- [ ] Kod jest sformatowany (black, flake8)
- [ ] Testy przechodzą
- [ ] Dokumentacja jest zaktualizowana
- [ ] CHANGELOG.md jest zaktualizowany
- [ ] Commit messages są prawidłowe

#### Template PR
```markdown
## Opis zmian
Krótki opis tego co zostało zmienione i dlaczego.

## Typ zmiany
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change
- [ ] Documentation update

## Testy
- [ ] Dodano nowe testy
- [ ] Wszystkie testy przechodzą
- [ ] Manual testing przeprowadzony

## Checklist
- [ ] Self-review kodu
- [ ] Komentarze w trudnych fragmentach
- [ ] Dokumentacja zaktualizowana
- [ ] Brak conflictów z main branch
```

## 📋 Standardy kodu

### Python Code Style
```python
# Dobre praktyki
def process_polish_text(text: str, normalize: bool = True) -> str:
    """
    Process Polish text for model training.
    
    Args:
        text: Input Polish text
        normalize: Whether to normalize whitespace
        
    Returns:
        Processed text ready for tokenization
    """
    if not text or not isinstance(text, str):
        raise ValueError("Text must be a non-empty string")
    
    # Implementation details...
    return processed_text

# Type hints wszędzie gdzie możliwe
from typing import List, Dict, Optional, Union

def train_model(
    config: Dict[str, Any],
    dataset: Optional[Dataset] = None
) -> Tuple[AutoModel, float]:
    """Train WronAI model with given configuration."""
    pass
```

### Dokumentacja docstring
```python
def fine_tune_model(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    learning_rate: float = 2e-4,
    epochs: int = 3
) -> str:
    """
    Fine-tune Polish language model using QLoRA.
    
    This function implements efficient fine-tuning for Polish LLMs
    using 4-bit quantization and LoRA adapters to minimize
    memory requirements while maintaining model quality.
    
    Args:
        model_name: Name of base model (e.g., 'mistralai/Mistral-7B-v0.1')
        dataset_path: Path to Polish instruction dataset
        output_dir: Directory to save fine-tuned model
        learning_rate: Learning rate for training
        epochs: Number of training epochs
        
    Returns:
        Path to saved fine-tuned model
        
    Raises:
        ValueError: If model_name is not supported
        FileNotFoundError: If dataset_path doesn't exist
        
    Example:
        >>> model_path = fine_tune_model(
        ...     "mistralai/Mistral-7B-v0.1",
        ...     "data/polish_instructions.json",
        ...     "checkpoints/wronai-7b",
        ...     learning_rate=1e-4,
        ...     epochs=5
        ... )
        >>> print(f"Model saved to: {model_path}")
    """
```

## 🗂️ Struktura projektu

Dodając nowe pliki, zachowaj organizację:

```
WronAI/
├── wronai/                    # Main package
│   ├── __init__.py
│   ├── models/                # Model architectures
│   ├── training/              # Training utilities
│   ├── data/                  # Data processing
│   └── utils/                 # Common utilities
├── scripts/                   # CLI scripts
├── configs/                   # Configuration files
├── tests/                     # Test files
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── fixtures/              # Test data
├── docs/                      # Documentation
├── notebooks/                 # Jupyter examples
└── tools/                     # Development tools
```

## 🧪 Testowanie

### Unit Tests
```python
# tests/unit/test_tokenizer.py
import pytest
from wronai.data.tokenizer import PolishTokenizer

class TestPolishTokenizer:
    def test_tokenize_polish_text(self):
        tokenizer = PolishTokenizer()
        text = "Witaj świecie! Jak się masz?"
        tokens = tokenizer.tokenize(text)
        
        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)
    
    def test_empty_text_handling(self):
        tokenizer = PolishTokenizer()
        
        with pytest.raises(ValueError):
            tokenizer.tokenize("")
```

### Integration Tests
```python
# tests/integration/test_training.py
def test_full_training_pipeline():
    """Test complete training workflow with minimal data."""
    config = {
        "model": {"name": "microsoft/DialoGPT-small"},
        "training": {"num_train_epochs": 1}
    }
    
    # Run minimal training
    result = train_model(config)
    
    assert result.success
    assert os.path.exists(result.model_path)
```

## 📊 Performance Guidelines

### Memory Optimization
- Używaj gradient checkpointing dla dużych modeli
- Implementuj data streaming dla ogromnych datasetów
- Monitoruj memory usage w testach

```python
# Przykład memory-efficient data loading
def load_dataset_streaming(path: str, batch_size: int = 1000):
    """Load large dataset in streaming mode."""
    for chunk in pd.read_csv(path, chunksize=batch_size):
        yield process_chunk(chunk)
```

### Training Best Practices
- Zapisuj checkpoints regularnie
- Używaj mixed precision (fp16/bf16)
- Implementuj early stopping
- Loguj metryki do wandb/tensorboard

## 🌍 Internationalization

### Polskie komentarze w kodzie
```python
# Preferujemy angielskie komentarze w kodzie
# ale polskie są OK w dokumentacji użytkownika

# Dobrze: English technical comments
def preprocess_text(text: str) -> str:
    # Remove extra whitespace and normalize Polish diacritics
    return normalize_polish_chars(text.strip())

# OK: Polish comments for domain-specific logic
def detect_polish_sentiment(text: str) -> float:
    # Analiza sentymentu dla języka polskiego
    # uwzględniająca specyfikę fleksji
    pass
```

## 🏆 Recognition

### Contributors
Wszyscy contributors będą dodani do:
- README.md Contributors section
- CONTRIBUTORS.md z opisem wkładu
- Release notes dla znaczących zmian

### Levels of Contribution
- 🥉 **Bronze**: 1-5 merged PRs
- 🥈 **Silver**: 6-15 merged PRs + significant feature
- 🥇 **Gold**: 16+ merged PRs + major contribution
- 💎 **Diamond**: Core team member

## 📞 Komunikacja

### Channels
- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, showcases
- **Discord**: Real-time chat [Join here](https://discord.gg/wronai)
- **Email**: wronai@example.com dla prywatnych spraw

### Response Times
- Bug reports: 24-48 godzin
- Feature requests: 3-7 dni
- PRs: 2-5 dni roboczych
- Security issues: Natychmiast

## 🚫 Co nie jest akceptowane

- Kod naruszający prawa autorskie
- Modele trenowane na danych bez odpowiednich licencji
- Implementacje które promują hate speech
- Kod bez testów dla krytycznych funkcji
- PR bez opisu zmian
- Commitsy bez opisowych messages

## ❓ FAQ

**Q: Czy mogę dodać support dla innego języka słowiańskiego?**
A: Tak! Chętnie przyjmiemy wsparcie dla czeskiego, słowackiego, ukraińskiego itp.

**Q: Jak długo trwa review PR?**
A: Zwykle 2-5 dni roboczych, zależnie od złożoności.

**Q: Czy mogę używać WronAI do celów komercyjnych?**
A: Tak, projekt jest na licencji Apache 2.0.

**Q: Jak zostać core contributor?**
A: Aktywność w community, wysokiej jakości contributions, pomoc innym.

---

Dziękujemy za wkład w rozwój polskiej AI! 🇵🇱🤖

**Happy coding!** 💻✨