# Instrukcja trenowania modeli WronAI

Ten dokument zawiera szczegółowe instrukcje dotyczące trenowania modeli WronAI przy użyciu dedykowanego pliku `Makefile.train`.

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
source wronai-env/bin/activate  # lub odpowiednia komenda dla Twojego środowiska
```

2. Przygotuj dane treningowe:

```bash
make -f Makefile.train prepare-data
```

## Podstawowe użycie

### Trening z domyślną konfiguracją

```bash
make -f Makefile.train train
```

Ta komenda uruchomi trening z konfiguracją z pliku `configs/default.yaml`.

### Szybki trening testowy

```bash
make -f Makefile.train train-quick
```

Ta komenda uruchomi szybki trening testowy z konfiguracją z pliku `configs/quick_test.yaml`.

### Trening z niestandardową konfiguracją

```bash
make -f Makefile.train train-custom CONFIG=configs/moja_konfiguracja.yaml
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

# Logging and Monitoring
logging:
  report_to: ["tensorboard"]
  # ...
```

Pełny przykład konfiguracji znajdziesz w pliku [configs/default.yaml](../configs/default.yaml).

### Tworzenie własnej konfiguracji

1. Skopiuj istniejący plik konfiguracyjny:

```bash
cp configs/default.yaml configs/moja_konfiguracja.yaml
```

2. Dostosuj parametry według potrzeb
3. Uruchom trening z nową konfiguracją:

```bash
make -f Makefile.train train-custom CONFIG=configs/moja_konfiguracja.yaml
```

## Monitorowanie treningu

### TensorBoard

Uruchom TensorBoard, aby monitorować postęp treningu:

```bash
make -f Makefile.train tensorboard
```

Następnie otwórz przeglądarkę pod adresem http://localhost:6006

### Weights & Biases

Jeśli masz skonfigurowany Weights & Biases, możesz monitorować trening przez ich interfejs webowy.

## Konfiguracja Weights & Biases

### Konfiguracja standardowa

```bash
make -f Makefile.train setup-wandb
```

Ta komenda zainstaluje wandb i przeprowadzi przez proces logowania.

### Konfiguracja lokalnego serwera W&B

```bash
make -f Makefile.train setup-wandb-local
```

Ta komenda utworzy konfigurację Docker Compose dla lokalnego serwera W&B. Po wykonaniu komendy otrzymasz instrukcje jak uruchomić serwer:

1. Przejdź do katalogu wandb-local:

```bash
cd wandb-local
```

2. Uruchom serwer:

```bash
docker-compose up -d
```

3. Skonfiguruj trening do używania lokalnego serwera:

```bash
export WANDB_BASE_URL=http://localhost:8080
export WANDB_API_KEY=admin
```

4. Zmodyfikuj plik konfiguracyjny, aby włączyć raportowanie do wandb:

```yaml
logging:
  report_to: ["tensorboard", "wandb"]
```

## Optymalizacja sprzętowa

### Sprawdzenie dostępności GPU

```bash
make -f Makefile.train check-gpu
```

### Trening tylko na CPU

```bash
make -f Makefile.train train-cpu-only
```

### Trening na określonym GPU

```bash
make -f Makefile.train train-specific-gpu GPU=0
```

Gdzie `0` to indeks GPU, który chcesz użyć.

### Trening na wielu GPU

```bash
make -f Makefile.train train-multi-gpu
```

## Zarządzanie plikami

### Usuwanie punktów kontrolnych

```bash
make -f Makefile.train clean-checkpoints
```

### Usuwanie logów

```bash
make -f Makefile.train clean-logs
```

### Usuwanie wszystkich wygenerowanych plików

```bash
make -f Makefile.train clean-all
```

## Rozwiązywanie problemów

### Błąd dostępu do modelu

Jeśli napotkasz błąd dostępu do modelu bazowego (np. 401 Unauthorized), możesz:

1. Zalogować się do Hugging Face:

```bash
huggingface-cli login
```

2. Lub zmienić model bazowy na otwarty model w pliku konfiguracyjnym:

```yaml
model:
  name: "facebook/opt-1.3b"  # otwarty model
```

### Problemy z pamięcią GPU

Jeśli napotkasz problemy z pamięcią GPU (OOM - Out of Memory), możesz:

1. Zmniejszyć rozmiar batcha:

```yaml
training:
  per_device_train_batch_size: 1
```

2. Zwiększyć kroki akumulacji gradientu:

```yaml
training:
  gradient_accumulation_steps: 16
```

3. Włączyć gradient checkpointing:

```yaml
training:
  gradient_checkpointing: true
```

4. Użyć bardziej agresywnej kwantyzacji:

```yaml
quantization:
  load_in_4bit: true
  bnb_4bit_use_double_quant: true
```

### Wolny trening

Jeśli trening jest zbyt wolny, sprawdź:

1. Czy używasz GPU
2. Czy dane są odpowiednio przygotowane
3. Czy nie używasz zbyt dużej liczby kroków ewaluacji

## Dodatkowe zasoby

- [Dokumentacja treningu](training.md) - Szczegółowy opis procesu treningu
- [Dokumentacja potoku](pipeline.md) - Techniczny opis potoku treningu
- [Architektura modelu](model_architecture.md) - Opis architektury modelu
- [Indeks dokumentacji treningu](training_index.md) - Centralny punkt nawigacji po dokumentacji treningu
