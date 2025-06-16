# WronAI Data Collection Pipeline

System do pobierania i przetwarzania polskich danych treningowych dla modeli językowych.

## Opis projektu

WronAI Data Collection Pipeline to kompleksowe narzędzie do zbierania, czyszczenia i przetwarzania polskojęzycznych danych tekstowych z różnych źródeł. System został zaprojektowany do tworzenia wysokiej jakości korpusu treningowego dla modeli językowych w języku polskim.

## Źródła danych

Pipeline zbiera dane z następujących źródeł:

1. **Wysokiej jakości**:
   - Polska Wikipedia
   - Wolne Lektury (polskie książki w domenie publicznej)
   - Artykuły akademickie

2. **Średniej jakości**:
   - OSCAR Polish (filtrowany Common Crawl)
   - Common Crawl (wybrane polskie domeny)

## Funkcjonalności

- Pobieranie danych z wielu źródeł
- Czyszczenie i normalizacja tekstu
- Deduplikacja na poziomie dokumentów
- Identyfikacja języka (filtrowanie niepolskich tekstów)
- Tworzenie podziałów na zbiory treningowe, walidacyjne i testowe
- Generowanie metadanych

## Wymagania systemowe

- Python 3.8 lub nowszy
- Dostęp do internetu
- Min. 16GB RAM (zalecane)
- Przestrzeń dyskowa: min. 100GB (zależnie od docelowego rozmiaru korpusu)

## Instalacja

```bash
# Sklonuj repozytorium
git clone https://github.com/wronai/llm.git
cd llm/wronai_data

# Uruchom skrypt instalacyjny
bash setup.sh
```

Skrypt `setup.sh` automatycznie:
1. Tworzy wirtualne środowisko Python
2. Instaluje wszystkie wymagane zależności z pliku requirements.txt
3. Pobiera model FastText do identyfikacji języka

## Autorzy

Zespół WronAI

## 📋 **Przegląd Skryptów WronAI**

1. **🔍 `collect_wronai_data.py`** - Główny skrypt zbierania danych
   - Pobiera polską Wikipedię, OSCAR, Wolne Lektury i inne źródła
   - Obsługuje różne formaty i źródła danych
   - Pełna implementacja pipeline'u zbierania danych

2. **🔍 `collect_wronai_data_fixed.py`** - Naprawiona wersja skryptu zbierania danych
   - Używa dostępnych źródeł danych bez wymagania specjalnego dostępu
   - Zawiera mechanizmy fallback dla niedostępnych źródeł
   - Lepsze zarządzanie błędami i obsługa wyjątków

3. **🔍 `quick_data_collection.py`** - Uproszczona wersja zbierania danych
   - Szybkie uruchomienie bez problemów z dostępem do danych
   - Minimalna wersja bez zewnętrznych zależności
   - Domyślnie zbiera 500MB danych

4. **⚙️ `processor.py`** - Przetwarzanie danych
   - Czyszczenie i filtrowanie tekstów
   - Tokenizacja i chunking
   - Deduplikacja i train/val/test split
   - Przygotowanie danych do formatu treningowego

5. **🏋️ `trainer.py`** - Trening modelu
   - QLoRA + 4-bit quantization
   - Optimized dla GPU/CPU
   - Early stopping i monitoring treningu
   - Obsługa różnych modeli bazowych

6. **🤖 `inference.py`** - Inferowanie i ewaluacja
   - Generowanie tekstów z wytrenowanego modelu
   - Gradio interface dla interaktywnego testowania
   - Ewaluacja jakości modelu
   - Custom stopping criteria dla polskiego tekstu

7. **🎯 `pipeline.py`** - Master orchestrator
   - Zarządza całym pipeline'm od zbierania danych do inferowania
   - CLI interface z różnymi komendami
   - State management i śledzenie postępu
   - Automatyczne tworzenie wymaganych katalogów i plików

8. **🧪 `test_data_collection.py`** - Testy zbierania danych
   - Szybkie testy funkcjonalności zbierania danych
   - Mniejszy zestaw danych (100MB)
   - Weryfikacja poprawności działania pipeline'u

## 📋 **Szczegółowy opis skryptów**

### 1. **collect_wronai_data.py**
Główny skrypt zbierania danych dla WronAI. Implementuje pełny pipeline pobierania danych z różnych źródeł polskich tekstów.

```python
# Główne funkcje:
- WronAIDataCollector - klasa zarządzająca całym procesem zbierania
- collect_wikipedia_polish() - pobiera artykuły z polskiej Wikipedii
- collect_oscar_polish() - pobiera teksty z korpusu OSCAR
- collect_wolne_lektury() - pobiera książki z Wolnych Lektur
- collect_github_corpora() - pobiera korpusy z GitHub
- run_collection_pipeline() - uruchamia cały proces zbierania danych
```

### 2. **collect_wronai_data_fixed.py**
Naprawiona wersja skryptu zbierania danych, która rozwiązuje problemy z dostępem do źródeł i zawiera lepsze mechanizmy obsługi błędów.

```python
# Główne funkcje:
- WronAICollector - uproszczona klasa do zbierania danych
- collect_wikipedia_polish() - z mechanizmem fallback
- collect_wikipedia_fallback() - alternatywna metoda pobierania z mC4
- collect_wolne_lektury() - z lepszym error handling
- clean_text() - czyszczenie i normalizacja tekstu
- run() - uruchamia cały proces zbierania danych
```

### 3. **quick_data_collection.py**
Uproszczona wersja skryptu zbierania danych, działająca z minimalnymi zależnościami i zaprojektowana do szybkiego uruchomienia.

```python
# Główne funkcje:
- SimpleWronAICollector - minimalistyczna klasa do zbierania danych
- collect_wikipedia_simple() - uproszczone pobieranie z Wikipedii
- collect_wolne_lektury_simple() - uproszczone pobieranie z Wolnych Lektur
- generate_synthetic_data() - generowanie syntetycznych danych jako fallback
- is_polish_simple() - prosta detekcja języka polskiego
- run_quick_collection() - szybki proces zbierania danych
```

### 4. **processor.py**
Skrypt do przetwarzania zebranych danych tekstowych, przygotowujący je do treningu modelu.

```python
# Główne funkcje:
- WronAIDataProcessor - klasa do przetwarzania danych
- load_raw_data() - ładowanie surowych danych z plików JSONL
- clean_and_filter() - czyszczenie i filtrowanie tekstów
- tokenize_and_chunk() - tokenizacja i dzielenie na chunki
- deduplicate() - usuwanie duplikatów
- create_train_val_test_split() - podział na zbiory treningowe/walidacyjne/testowe
- save_processed_data() - zapisywanie przetworzonych danych
- run_processing_pipeline() - uruchamia cały proces przetwarzania
```

### 5. **trainer.py**
Skrypt do treningu modelu językowego z wykorzystaniem technik QLoRA i 4-bit quantization.

```python
# Główne funkcje:
- WronAITrainer - klasa do treningu modelu
- load_datasets() - ładowanie przetworzonych datasetów
- setup_model() - konfiguracja modelu z kwantyzacją i LoRA
- train_model() - trening modelu
- evaluate_model() - ewaluacja wytrenowanego modelu
- generate_sample_text() - generowanie przykładowego tekstu
- run_full_training_pipeline() - uruchamia cały proces treningu
```

### 6. **inference.py**
Skrypt do inferowania z wytrenowanego modelu, zawierający interfejs Gradio do interaktywnego testowania.

```python
# Główne funkcje:
- WronAIInference - klasa do inferowania z modelu
- PolishStoppingCriteria - niestandardowe kryteria zatrzymania dla polskiego tekstu
- generate_text() - generowanie tekstu na podstawie promptu
- chat() - konwersacja z modelem
- evaluate_polish_capabilities() - ewaluacja zdolności modelu w języku polskim
- benchmark_performance() - benchmark wydajności modelu
```

### 7. **pipeline.py**
Master orchestrator zarządzający całym procesem od zbierania danych do inferowania.

```python
# Główne funkcje:
- WronAIMasterPipeline - klasa zarządzająca całym pipeline'm
- run_data_collection() - uruchamia zbieranie danych
- run_data_processing() - uruchamia przetwarzanie danych
- run_model_training() - uruchamia trening modelu
- run_inference() - uruchamia inferowanie
- run_full_pipeline() - uruchamia cały proces od początku do końca
- create_requirements_file() - tworzy plik requirements.txt
```

### 8. **test_data_collection.py**
Skrypt do testowania funkcjonalności zbierania danych na małym zestawie danych.

```python
# Główne funkcje:
- test_small_collection() - test zbierania danych z małym rozmiarem (100MB)
```

## 🚀 **Jak uruchomić poszczególne skrypty:**

### Zbieranie danych:
```bash
# Pełny pipeline zbierania danych
python3 scripts/collect_wronai_data.py

# Naprawiona wersja z lepszą obsługą błędów
python3 scripts/collect_wronai_data_fixed.py

# Szybka wersja z minimalnymi zależnościami
python3 scripts/quick_data_collection.py

# Test zbierania danych (100MB)
python3 scripts/test_data_collection.py
```

### Przetwarzanie danych:
```bash
# Przetwarzanie zebranych danych
python3 scripts/processor.py --input_dir ./data --output_dir ./processed
```

### Trening modelu:
```bash
# Trening modelu z domyślnymi parametrami
python3 scripts/trainer.py

# Trening z niestandardowymi parametrami
python3 scripts/trainer.py --model_name "microsoft/DialoGPT-medium" --data_dir ./processed --output_dir ./model
```

### Inferowanie:
```bash
# Uruchomienie interfejsu Gradio
python3 scripts/inference.py --model_path ./model
```

### Pełny pipeline:
```bash
# Uruchomienie całego pipeline'u
python3 scripts/pipeline.py full --size 1000 --model "microsoft/DialoGPT-medium"

# Tylko zbieranie danych
python3 scripts/pipeline.py collect --size 2000

# Tylko trening
python3 scripts/pipeline.py train --model "microsoft/DialoGPT-medium"

# Tylko inferowanie
python3 scripts/pipeline.py infer
```

## 🔧 **Dostępne komendy pipeline:**

```bash
# Sprawdź status pipeline'u
python3 scripts/pipeline.py status

# Tylko zbieranie danych (2GB)
python3 scripts/pipeline.py collect --size 2000

# Tylko trening (jeśli dane gotowe)
python3 scripts/pipeline.py train --model "microsoft/DialoGPT-medium"

# Pełny pipeline
python3 scripts/pipeline.py full --size 1000 --model "microsoft/DialoGPT-medium"

# Uruchom inferowanie
python3 scripts/pipeline.py infer

# Wyczyść workspace
python3 scripts/pipeline.py clean
```

## 📊 **Funkcje systemu:**

### ✅ **Smart Data Collection**
- Automatyczne fallback na dostępne źródła
- Graceful error handling
- Progress tracking
- Obsługa wielu formatów i źródeł danych

### ✅ **Optimized Training**
- QLoRA dla efektywności pamięci
- 4-bit quantization
- Auto GPU/CPU detection
- Early stopping

### ✅ **Interactive Interface**
- Gradio web UI
- Chat interface
- Real-time generation
- Parameter tuning

### ✅ **Comprehensive Evaluation**
- Polish language capabilities testing
- Performance benchmarking
- Quality metrics
- Generation samples

### ✅ **Production Ready**
- State management
- Error recovery
- Logging & monitoring
- Modular architecture

## 🎯 **Pipeline Flow:**

```
📥 Data Collection → ⚙️ Processing → 🏋️ Training → 🤖 Inference
     ↓                    ↓              ↓           ↓
  500MB+ Polish       Tokenized      WronAI      Gradio UI
  Text Corpus        Chunks         Model       + Chat
```

## 📈 **Expected Results:**

Po zakończeniu pipeline'u otrzymasz:

### 📂 **Workspace Structure:**
```
wronai_workspace/
├── data/              # Surowe dane (Wikipedia, lektury)
├── processed/         # Przetworzone chunki + tokeny
├── model/            # Wytrenowany model WronAI
├── logs/             # Logi treningu
└── pipeline_state.json
```

### 🤖 **Trained Model:**
- **Bazowy**: DialoGPT-medium (350M parametrów)
- **LoRA adapters**: ~16M trenowalnych parametrów  
- **Język**: Dostosowany do polskiego
- **Rozmiar**: ~700MB (z quantization)

## 🛠️ **Troubleshooting:**

### **Problem: Brak dostępu do danych Wikipedia**
```bash
# Skrypt automatycznie użyje alternatywnych źródeł
python3 run_scripts.py fixed
```

### **Problem: Brak GPU**
```bash
# Pipeline automatycznie przełączy na CPU
# Trening będzie wolniejszy ale zadziała
python3 scripts/pipeline.py full --size 100  # Mniejszy dataset
```

### **Problem: Mało pamięci**
```bash
# Zmniejsz batch size w scripts/trainer.py
# Edytuj parametry:
batch_size = 1  # zamiast 4
gradient_accumulation_steps = 32  # zamiast 8
```

### **Problem: Dane się nie pobierają**
```bash
# Sprawdź czy masz internet i uruchom tylko data collection
python3 scripts/pipeline.py collect --size 500
```

## 🔮 **Dalszy rozwój:**

### **Immediate improvements:**
1. **Więcej danych**: Zwiększ `--size` do 5000+ MB
2. **Lepszy model bazowy**: Użyj `mistralai/Mistral-7B-v0.1`
3. **Domain adaptation**: Dodaj specjalistyczne korpusy

### **Advanced features:**
1. **RLHF**: Reinforcement Learning from Human Feedback
2. **Multimodal**: Dodaj obsługę obrazów
3. **RAG**: Retrieval Augmented Generation
4. **Fine-tuning**: Task-specific adapters

## 📚 **Następne kroki:**

1. **Uruchom pipeline** - `python3 scripts/pipeline.py full`
2. **Przetestuj model** w interfejsie Gradio
3. **Oceń wyniki** przez generation samples
4. **Iteruj** - dostrajaj parametry i zwiększaj dane
5. **Deploy** - stwórz API dla aplikacji
