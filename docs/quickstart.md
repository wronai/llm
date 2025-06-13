# 🚀 WronAI Quick Start Guide

Witaj w WronAI! Ten przewodnik pomoże Ci szybko rozpocząć pracę z polskim modelem językowym.

## 📋 Wymagania systemowe

### Minimalne wymagania
- **Python**: 3.8+
- **GPU**: NVIDIA z 8GB VRAM (RTX 3070/4060 Ti lub lepszy)
- **RAM**: 16GB DDR4
- **Storage**: 50GB wolnego miejsca
- **OS**: Linux, Windows, macOS z CUDA 11.8+

### Zalecane wymagania
- **GPU**: NVIDIA RTX 4080/4090 (16GB+ VRAM)
- **RAM**: 32GB DDR4/DDR5
- **Storage**: 500GB NVMe SSD

## 🛠️ Instalacja

### 1. Klonowanie repozytorium

```bash
git clone https://github.com/wronai/llm.git
cd WronAI
```

### 2. Środowisko wirtualne

```bash
# Tworzenie środowiska
python -m venv venv

# Aktywacja (Linux/Mac)
source venv/bin/activate

# Aktywacja (Windows)
venv\Scripts\activate
```

### 3. Instalacja zależności

```bash
# Podstawowa instalacja
pip install -e .

# Instalacja z dodatkami
pip install -e ".[dev,inference,polish]"

# Sprawdzenie CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 4. Instalacja modelu spaCy (opcjonalnie)

```bash
python -m spacy download pl_core_news_sm
```

## ⚡ Szybki start

### 1-minutowy test

```python
from wronai import load_model, generate_text

# Ładowanie modelu (pierwsze uruchomienie może potrwać kilka minut)
model = load_model("mistralai/Mistral-7B-v0.1", quantize=True)

# Generacja tekstu
response = generate_text(
    model, 
    "Opowiedz o Polsce:", 
    max_length=200,
    temperature=0.7
)

print(f"🐦‍⬛ WronAI: {response}")
```

### Makefile commands

```bash
# Wszystko w jednym - quickstart
make quickstart

# Lub krok po kroku:
make install-dev          # Instalacja zależności
make prepare-data-minimal # Przygotowanie danych
make train-quick          # Szybki trening testowy
make inference            # Test inferencji
```

## 💬 Przykłady użycia

### Podstawowa generacja tekstu

```python
from wronai import load_model
from wronai.inference import InferenceEngine, InferenceConfig

# Ładowanie modelu
model = load_model("mistralai/Mistral-7B-v0.1")

# Konfiguracja inferencji
config = InferenceConfig(
    max_length=256,
    temperature=0.7,
    top_p=0.9,
    use_polish_formatting=True
)

# Silnik inferencji
engine = InferenceEngine(model, config)

# Generacja
prompts = [
    "Jakie są tradycyjne polskie potrawy?",
    "Wyjaśnij pojęcie sztucznej inteligencji:",
    "Przetłumacz na angielski: 'Miło Cię poznać'"
]

for prompt in prompts:
    response = engine.generate(prompt)
    print(f"Prompt: {prompt}")
    print(f"Odpowiedź: {response}\n")
```

### Chat z historią konwersacji

```python
from wronai.inference import InferenceEngine

# Konwersacja z historią
conversation_history = []

while True:
    user_input = input("Ty: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    
    # Generacja odpowiedzi
    response = engine.chat(user_input, conversation_history)
    print(f"🐦‍⬛ WronAI: {response}")
    
    # Aktualizacja historii
    conversation_history.append({
        "user": user_input,
        "assistant": response
    })
    
    # Zachowaj tylko ostatnie 5 wymian
    if len(conversation_history) > 5:
        conversation_history = conversation_history[-5:]
```

### Batch processing

```python
from wronai.data import prepare_polish_data, PolishDataset

# Przygotowanie danych
dataset = prepare_polish_data(
    data_path="data/polish_texts.json",
    tokenizer_name="mistralai/Mistral-7B-v0.1",
    instruction_format=True
)

# Batch generacja
prompts = [
    "Opisz Kraków:",
    "Co to jest pierogi?",
    "Jaka jest stolica Polski?"
]

responses = engine.generate(prompts)
for prompt, response in zip(prompts, responses):
    print(f"Q: {prompt}")
    print(f"A: {response}\n")
```

## 🏋️ Trening własnego modelu

### Przygotowanie danych

```bash
# Pobieranie i przygotowanie polskich danych
python scripts/prepare_data.py --all --output-dir data/processed
```

### Konfiguracja treningu

```python
from wronai.training import WronAITrainer, TrainingConfig
from wronai.data import create_polish_dataset

# Konfiguracja treningu
config = TrainingConfig(
    output_dir="./checkpoints/moj-model",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    save_steps=500,
    eval_steps=500,
    logging_steps=10
)

# Przygotowanie danych
train_dataset = create_polish_dataset(
    dataset_type="instruction",
    data_path="data/processed/polish_instructions.json",
    tokenizer=model.tokenizer
)

# Trening
trainer = WronAITrainer(model, config)
results = trainer.train(train_dataset)

print("Trening zakończony!")
print(f"Final loss: {results.training_loss:.4f}")
```

### Szybki trening testowy

```bash
# Trening na małym datasecie (5 minut)
python scripts/train.py --config configs/quick_test.yaml

# Pełny trening (kilka godzin)
python scripts/train.py --config configs/default.yaml
```

## 🌐 Web Interface

### Uruchomienie aplikacji Streamlit

```bash
# Uruchomienie web interface
streamlit run wronai/web/app.py

# Aplikacja dostępna na: http://localhost:8501
```

### REST API Server

```bash
# Uruchomienie API server
python scripts/serve.py --host 0.0.0.0 --port 8000 --model mistralai/Mistral-7B-v0.1

# API dostępne na: http://localhost:8000
# Dokumentacja: http://localhost:8000/docs
```

#### Przykład użycia API

```python
import requests

# Generacja tekstu przez API
response = requests.post("http://localhost:8000/generate", 
    json={
        "prompt": "Opowiedz o Polsce:",
        "max_length": 200,
        "temperature": 0.7
    }
)

result = response.json()
print(result["generated_text"])
```

## 🐳 Docker

### Uruchomienie z Docker

```bash
# Build image
docker build -t wronai:latest .

# Uruchomienie treningu
docker-compose up wronai-training

# Uruchomienie API server
docker-compose up wronai-inference

# Web interface + API + monitoring
docker-compose up
```

### Docker - pojedyncze komendy

```bash
# Tylko inferencja
docker run --gpus all -p 8000:8000 wronai:latest python scripts/serve.py

# Interaktywny tryb
docker run --gpus all -it wronai:latest python scripts/inference.py --chat
```

## 🔧 Konfiguracja zaawansowana

### Optymalizacja pamięci

```python
from wronai.models import ModelConfig

# Konfiguracja dla GPU 8GB
config = ModelConfig(
    model_name="mistralai/Mistral-7B-v0.1",
    quantization_enabled=True,
    quantization_bits=4,
    lora_enabled=True,
    lora_r=16,
    gradient_checkpointing=True,
    max_sequence_length=1024  # Zmniejszone dla oszczędności pamięci
)

model = WronAIMistral(config)
```

### Monitoring pamięci

```python
from wronai.utils.memory import memory_monitor, get_memory_usage

# Real-time monitoring
with memory_monitor() as monitor:
    response = model.generate_polish_text("Test prompt")
    peak_usage = monitor.get_peak_usage()
    print(f"Peak GPU memory: {peak_usage.get('gpu_memory_percent', 0):.1f}%")

# Jednorazowy check
memory_info = get_memory_usage()
print(f"Current GPU usage: {memory_info['gpu_memory_percent']:.1f}%")
```

## 🐛 Troubleshooting

### Częste problemy

#### Problem: CUDA out of memory
```bash
# Rozwiązania:
# 1. Zmniejsz batch size
per_device_train_batch_size=1

# 2. Zwiększ gradient accumulation
gradient_accumulation_steps=32

# 3. Użyj gradient checkpointing
gradient_checkpointing=true

# 4. Zmniejsz max_length
max_seq_length=1024
```

#### Problem: Model nie ładuje się
```python
# Sprawdź dostępność modelu
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    print("Model dostępny")
except Exception as e:
    print(f"Błąd: {e}")
```

#### Problem: Wolna generacja
```python
# Optymalizacje dla szybszej inferencji
config = InferenceConfig(
    max_length=128,      # Krótsza generacja
    do_sample=False,     # Greedy decoding
    use_cache=True,      # Cache attention
    batch_size=1         # Pojedyncze sample
)
```

### Diagnostyka systemu

```bash
# Sprawdzenie wymagań
make health-check

# Informacje o systemie
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} ({props.total_memory//1024**3}GB)')
"
```

## 📚 Następne kroki

1. **Eksploruj przykłady**: `notebooks/` zawiera Jupyter notebooks z zaawansowanymi przykładami
2. **Przeczytaj dokumentację**: Szczegółowe informacje w `docs/`
3. **Dołącz do community**: [Discord](https://discord.gg/wronai) i [GitHub Discussions](https://github.com/wronai/llm/discussions)
4. **Contribute**: Zobacz `CONTRIBUTING.md` jak pomóc w rozwoju

## 🎯 Przykładowe projekty

- **Chatbot**: Zbuduj polskiego chatbota dla swojej strony
- **Tłumacz**: System tłumaczenia PL-EN z kontekstem
- **Content generator**: Generator artykułów i postów blog
- **Code assistant**: Asystent programisty z polskim interfejsem
- **Q&A system**: System odpowiadający na pytania o Twoje dokumenty

---

**Potrzebujesz pomocy?** 
- 📖 [Pełna dokumentacja](docs/)
- 💬 [Discord Community](https://discord.gg/wronai) 
- 🐛 [GitHub Issues](https://github.com/wronai/llm/issues)
- 📧 [Email](mailto:wronai@example.com)

**Happy coding z WronAI!** 🐦‍⬛🇵🇱