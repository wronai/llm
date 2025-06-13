# Instrukcja trenowania modeli WronAI

Ten dokument zawiera szczegółowe instrukcje dotyczące trenowania modeli WronAI przy użyciu dedykowanych skryptów w folderze `train/`.

## Spis treści

- [Wymagania](#wymagania)
- [Przygotowanie środowiska](#przygotowanie-środowiska)
- [Podstawowe użycie](#podstawowe-użycie)
- [Zaawansowane opcje treningu](#zaawansowane-opcje-treningu)
- [Monitorowanie treningu](#monitorowanie-treningu)
- [Konfiguracja Weights & Biases](#konfiguracja-weights--biases)
- [Optymalizacja sprzętowa](#optymalizacja-sprzętowa)
- [Zarządzanie plikami](#zarządzanie-plikami)
- [Rozwiązywanie problemów](#rozwiązywanie-problemów)

## Wymagania

Przed rozpoczęciem treningu upewnij się, że:

1. Masz zainstalowane wszystkie zależności (patrz [README.md](../README.md))
2. Masz dostęp do GPU z co najmniej 8GB VRAM (dla pełnego treningu)
3. Przygotowałeś dane treningowe

## Przygotowanie środowiska

1. Aktywuj środowisko wirtualne:

```bash
cd ..  # przejdź do głównego katalogu projektu
source venv/bin/activate  # lub odpowiednia komenda dla Twojego środowiska
cd train  # wróć do katalogu train
```

2. Przygotuj dane treningowe:

```bash
make prepare-data
```

## Podstawowe użycie

### Trening z domyślną konfiguracją

```bash
make train
```

Ta komenda uruchomi trening z konfiguracją z pliku `configs/default.yaml`.

### Szybki trening testowy

```bash
make train-quick
```

Ta komenda uruchomi szybki trening testowy z konfiguracją z pliku `configs/quick_test.yaml`.

### Trening z niestandardową konfiguracją

```bash
make train-custom CONFIG=../configs/moja_konfiguracja.yaml
```

Ta komenda pozwala na użycie własnego pliku konfiguracyjnego.

## Zaawansowane opcje treningu

### Struktura pliku konfiguracyjnego

Plik konfiguracyjny YAML powinien zawierać następujące sekcje:

```yaml
# Model Configuration
model:
  name: "facebook/opt-1.3b"  # nazwa modelu bazowego
  trust_remote_code: true
  torch_dtype: "bfloat16"
  device_map: "auto"

# LoRA Configuration
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  # ...

# Quantization Configuration
quantization:
  load_in_4bit: true
  # ...

# Training Configuration
training:
  output_dir: "./checkpoints/wronai-7b"
  num_train_epochs: 3
  # ...

# Data Configuration
data:
  dataset_name: "wronai/polish-instruct"
  # ...
```

### Trening na CPU (bez GPU)

```bash
make train-cpu-only
```

### Trening na określonym GPU

```bash
make train-specific-gpu GPU=0  # używa tylko GPU 0
```

### Trening na wielu GPU

```bash
make train-multi-gpu
```

## Monitorowanie treningu

### TensorBoard

Aby uruchomić TensorBoard do monitorowania treningu:

```bash
make tensorboard
```

Następnie otwórz przeglądarkę pod adresem: http://localhost:6006

### Weights & Biases

Aby skonfigurować Weights & Biases do śledzenia eksperymentów:

```bash
make setup-wandb
```

Aby skonfigurować lokalny serwer Weights & Biases (bez wysyłania danych do chmury):

```bash
make setup-wandb-local
```

## Optymalizacja sprzętowa

### Sprawdzenie dostępności GPU

```bash
make check-gpu
```

## Zarządzanie plikami

### Czyszczenie punktów kontrolnych

```bash
make clean-checkpoints
```

### Czyszczenie logów

```bash
make clean-logs
```

### Czyszczenie wszystkich wygenerowanych plików

```bash
make clean-all
```

## Rozwiązywanie problemów

### Problem: Brak dostępu do modelu bazowego (gated model)

Jeśli podczas treningu pojawia się błąd związany z dostępem do modelu bazowego (np. Mistral-7B), możesz:

1. Zalogować się do Hugging Face Hub:

```bash
huggingface-cli login
```

2. Lub zmienić model bazowy na otwarty model w pliku konfiguracyjnym:

```yaml
model:
  name: "facebook/opt-1.3b"  # otwarty model
```

### Problem: Błędy pamięci GPU

Jeśli pojawiają się błędy związane z pamięcią GPU, możesz:

1. Zmniejszyć rozmiar batcha w pliku konfiguracyjnym:

```yaml
training:
  per_device_train_batch_size: 1  # zmniejsz z 4 na 1
```

2. Włączyć gradient checkpointing:

```yaml
training:
  gradient_checkpointing: true
```

3. Użyć kwantyzacji 4-bitowej:

```yaml
quantization:
  load_in_4bit: true
```

### Problem: Wolny trening

Jeśli trening jest zbyt wolny:

1. Upewnij się, że używasz GPU
2. Sprawdź, czy kwantyzacja jest włączona
3. Rozważ zmniejszenie rozmiaru modelu

### Problem: Błędy z biblioteką streamlit

Jeśli występują problemy z uruchomieniem interfejsu Streamlit:

```bash
pip install streamlit
```

lub użyj odpowiedniej wersji Pythona:

```bash
pyenv global 3.12.0  # lub inna wersja z zainstalowanym streamlit
```
