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

# WronAI Data Collection - Naprawiona Wersja

## Przegląd

Ta wersja naprawia problemy z oryginalnym skryptem:

✅ **Naprawione problemy:**
- Używa dostępnej wersji Wikipedii (20220301.pl)
- Lepszy error handling dla Wolnych Lektur
- Alternatywne źródła zamiast niedostępnego OSCAR
- Prosta deduplikacja hash-based
- Graceful degradation gdy źródła są niedostępne

## Szybki start

```bash
# Setup
bash setup.sh
source venv/bin/activate

# Test (100MB)
python3 scripts/test_data_collection.py

# Pełne zbieranie (5GB)
python3 scripts/collect_wronai_data_fixed.py
```

## Dostępne źródła

1. **Wikipedia PL** (20220301) - ~1.5M artykułów
2. **Wolne Lektury** - literatura klasyczna
3. **CC-100 PL** - korpus internetowy (jeśli dostępny)
4. **Fallback sources** - dodatkowe źródła

## Struktura danych

```
data/
├── wikipedia_pl.jsonl      # Artykuły Wikipedia
├── wolne_lektury.jsonl     # Książki
└── cc100_pl.jsonl          # Teksty internetowe
```

Format JSONL:
```json
{
  "id": "wiki_12345",
  "title": "Tytuł artykułu",
  "text": "Treść...",
  "source": "wikipedia"
}
```

## Monitoring

- Logi w konsoli z progress bars
- Statystyki na końcu wykonania
- Graceful stop przy osiągnięciu limitu rozmiaru

## Rozwiązywanie problemów

**Problem**: Błąd dostępu do datasetu
**Rozwiązanie**: Skrypt automatycznie przechodzi do następnego źródła

**Problem**: Timeout przy pobieraniu
**Rozwiązanie**: Zwiększone timeout'y i retry logic

**Problem**: Brak polskich znaków
**Rozwiązanie**: Prostsza heurystyka detekcji języka


## Użycie

### Uruchamianie skryptów

WronAI Data Collection Pipeline zawiera kilka skryptów do zbierania danych. Możesz je łatwo uruchamiać za pomocą skryptu `run_scripts.py`:

```bash
# Aktywuj środowisko wirtualne
source venv/bin/activate

# Wyświetl dostępne skrypty
python3 run_scripts.py list

# Uruchom pełny pipeline zbierania danych
python3 run_scripts.py collect

# Uruchom szybką wersję zbierania danych (mniej zależności)
python3 run_scripts.py quick

# Uruchom testy
python3 run_scripts.py test
```

### Parametry konfiguracyjne

Możesz dostosować parametry uruchomienia:

```python
collector = WronAIDataCollector(
    output_dir="./output_data",  # Katalog wyjściowy
    target_size_gb=50            # Docelowy rozmiar korpusu w GB
)
collector.run_collection_pipeline()
```

## Struktura katalogów

```
wronai_data/
├── raw_data/                # Surowe dane z różnych źródeł
│   ├── high_quality/        # Źródła wysokiej jakości
│   │   ├── wikipedia_pl/
│   │   ├── wolne_lektury/
│   │   └── academic_papers/
│   └── medium_quality/      # Źródła średniej jakości
│       ├── oscar_pl/
│       └── common_crawl/
├── processed_data/          # Przetworzone dane
│   ├── filtered/            # Po filtrowaniu języka
│   ├── deduplicated/        # Po deduplikacji
│   └── tokenized/           # Po tokenizacji
├── splits/                  # Podziały na zbiory
│   ├── train/
│   ├── validation/
│   └── test/
├── metadata/                # Metadane korpusu
└── logs/                    # Logi procesu
```
{źródło}_{jakość}_{domena}_{wersja}_{split}_{shard}.{format}
```

**Przykłady:**
- `wikipedia_high_general_v2.1_train_00001.jsonl`
- `oscar_medium_web_v23.01_val_00042.parquet`
- `nkjp_high_literature_v1.5_test_00001.jsonl`


## Algorytm pobierania w kolejności priorytetowej

### Strategia Quality-First z balansowaniem domen

**Faza 1: Fundament wysokiej jakości (15GB)**
1. **Wikipedia Polish** (4GB) - pobierz najnowszy dump, wyodrębnij czyste teksty
2. **Wolne Lektury** (3GB) - użyj API do pobrania wszystkich dostępnych dzieł
3. **NKJP zbalansowany** (5GB) - pobierz dostępne podkorpusy
4. **CLARIN-PL akademicki** (2GB) - zbierz wysokiej jakości zbiory specjalistyczne
5. **Parliamentary Corpus** (1GB) - oficjalne transkrypcje

**Faza 2: Główny korpus internetowy (30GB)**
1. **OSCAR Polish deduplicated** (30GB) - pobierz najnowszą wersję 23.01
2. Zastosuj dodatkowe filtrowanie jakości (perplexity < 500)
3. Wykonaj deduplikację na poziomie dokumentów i zdań

**Faza 3: Uzupełnienie i balansowanie (5GB)**
1. **Common Crawl CC-100** - dla zwiększenia różnorodności internetowej
2. **Conversational data** - SpokesBiz i DiaBiz dla języka mówionego
3. **Specialized domains** - techniczne, naukowe, biznesowe teksty

### Strategie deduplikacji i czyszczenia

**Multi-poziomowa deduplikacja:**
- **Dokument-level**: MinHash LSH z progiem 0.8 (usuwa 70-80% duplikatów)
- **Zdanie-level**: Usuwanie powtarzających się fraz
- **Line-level**: Deduplikacja w ramach losowych bucket'ów

**Pipeline czyszczenia:**
1. **Encoding normalization**: Konwersja do UTF-8, naprawa ftfy
2. **Language detection**: FastText z 95% confidence threshold
3. **Content filtering**: Usuwanie HTML, boilerplate, offensive content
4. **Quality scoring**: Perplexity-based + heuristic features
5. **Length filtering**: Min 100 znaków, max 50K znaków na dokument

### Balansowanie domen językowych

**Docelowy rozkład 50GB datasetu:**
- **OSCAR (web content)**: 35GB (70%) - różnorodność internetowa
- **Wikipedia**: 4GB (8%) - wysoka jakość, encyklopedyczność
- **NKJP**: 5GB (10%) - zbalansowana reprezentacja gatunków
- **Wolne Lektury**: 3GB (6%) - język literacki, kultura
- **CLARIN-PL**: 2GB (4%) - domeny specjalistyczne
- **Parliamentary**: 1GB (2%) - język formalny, polityczny

## Konkretny skrypt do automatycznego pobierania


### Skrypt Docker dla reprodukowalnego środowiska


## Szacunkowe rozmiary i wymagania

### Rozmiary źródeł danych

**Źródła pierwszego priorytetu (15GB):**
- **Wikipedia Polish**: 4GB nieskompresowany tekst
- **Wolne Lektury**: 3GB literatury wysokiej jakości  
- **NKJP zbalansowany**: 5GB zbalansowanego korpusu
- **CLARIN-PL**: 2GB zbiorów akademickich
- **Parliamentary Corpus**: 1GB oficjalnych transkrypcji

**Główny korpus (30GB):**
- **OSCAR Polish deduplicated**: 49GB dostępny, 30GB po dodatkowym filtrowaniu
- **Common Crawl CC-100**: 12GB skompresowany, uzupełnienie różnorodności

**Uzupełnienie (5GB):**
- **Conversational datasets**: 1GB rozmów i dialogów
- **Specialized domains**: 4GB technicznych i naukowych tekstów

### Wymagania infrastrukturalne

## Rozwiązywanie problemów

### Brak dostępu do OSCAR

Jeśli napotkasz problemy z dostępem do datasetu OSCAR, upewnij się, że używasz publicznie dostępnej wersji lub masz odpowiednie uprawnienia.

### Problemy z pamięcią

Dla dużych korpusów zalecane jest uruchamianie skryptu na maszynie z co najmniej 16GB RAM. Możesz zmniejszyć parametr `target_size_gb` aby zmniejszyć wymagania pamięciowe.

## Licencja

Ten projekt jest udostępniany na licencji Apache 2.0. Zobacz plik `LICENSE` w repozytorium.

## Autorzy

Zespół WronAI



### 📋 **Przegląd Skryptów**

1. **🔍 `quick_start_wronai.py`** - Zbieranie danych
   - Pobiera polską Wikipedię, Wolne Lektury
   - Fallback na syntetyczne dane
   - Domyślnie 500MB danych

2. **⚙️ `processing.py`** - Przetwarzanie danych  
   - Czyszczenie i filtrowanie tekstów
   - Tokenizacja i chunking
   - Deduplikacja i train/val/test split

3. **🏋️ `training.py`** - Trening modelu
   - QLoRA + 4-bit quantization
   - Optimized dla GPU/CPU
   - Tensorboard logging

4. **🤖 `inference.py`** - Inferowanie i ewaluacja
   - Generowanie tekstów
   - Gradio interface  
   - Performance benchmarking

5. **🎯 `pipeline.py`** - Master orchestrator
   - Zarządza całym pipeline'm
   - CLI interface
   - State management

---

## 🚀 **Jak uruchomić (3 kroki):**

### **Krok 1: Setup**
```bash
python pipeline.py setup
pip install -r requirements.txt
```

### **Krok 2: Pełny Pipeline**
```bash
python pipeline.py full --size 1000 --model microsoft/DialoGPT-medium
```

### **Krok 3: Uruchom model**
```bash
python scripts/pipeline.py infer
```

---

## 🔧 **Dostępne komendy:**

```bash
# Sprawdź status
python scripts/pipeline.py status

# Tylko zbieranie danych (2GB)
python scripts/pipeline.py collect --size 2000

# Tylko trening (jeśli dane gotowe)
python scripts/pipeline.py train --model "microsoft/DialoGPT-medium"

# Wyczyść workspace
python scripts/pipeline.py clean
```

---

## 📊 **Funkcje systemu:**

### ✅ **Smart Data Collection**
- Automatyczne fallback na dostępne źródła
- Graceful error handling
- Progress tracking

### ✅ **Optimized Training**
- QLoRA dla efektywności pamięci
- 4-bit quantization
- Auto GPU/CPU detection
- Early stopping

### ✅ **Interactive Interface**
- Gradio web UI
- Chat interface
- Real-time generation

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

---

## 🎯 **Pipeline Flow:**

```
📥 Data Collection → ⚙️ Processing → 🏋️ Training → 🤖 Inference
     ↓                    ↓              ↓           ↓
  500MB+ Polish       Tokenized      WronAI      Gradio UI
  Text Corpus        Chunks         Model       + Chat
```

---

## 📈 **Expected Results:**

Po zakończeniu pipeline'u otrzymasz:

### 📂 **Workspace Structure:**
```
wronai_workspace/
├── data/              # Surowe dane (Wikipedia, lektury)
├── processed/         # Przetworzone chunki + tokeny
├── model/            # Wytrenowany model WronAI
├── logs/             # Logi treningu
└── final_pipeline_report.json
```

### 🤖 **Trained Model:**
- **Bazowy**: DialoGPT-medium (350M parametrów)
- **LoRA adapters**: ~16M trenowalnych parametrów  
- **Język**: Dostosowany do polskiego
- **Rozmiar**: ~700MB (z quantization)

### 📊 **Performance Metrics:**
- **Perplexity**: ~15-25 (im niższy tym lepiej)
- **Generation speed**: 10-50 tokenów/s (GPU dependent)
- **Polish capability**: 70-85% success rate na testach

---

## 🛠️ **Troubleshooting:**

### **Problem: Brak GPU**
```bash
# Pipeline automatycznie przełączy na CPU
# Trening będzie wolniejszy ale zadziała
python scripts/pipeline.py full --size 100  # Mniejszy dataset
```

### **Problem: Mało pamięci**
```bash
# Zmniejsz batch size w wronai_training.py
batch_size = 1  # zamiast 4
gradient_accumulation_steps = 32  # zamiast 8
```

### **Problem: Dane się nie pobierają**
```bash
# Sprawdź czy masz internet i uruchom tylko data collection
python scripts/pipeline.py collect --size 500
```

### **Problem: Model nie generuje dobrze**
```bash
# Dostraj parametry w interfejsie:
# - Temperature: 0.7-0.9 (kreatywność)
# - Top-p: 0.8-0.95 (różnorodność)
# - Max length: 100-300 (długość)
```

---

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

### **Production deployment:**
1. **API server**: FastAPI + uvicorn
2. **Docker**: Containerization
3. **Monitoring**: MLflow + Prometheus
4. **Scaling**: Kubernetes deployment

---

## 📚 **Następne kroki:**

1. **Uruchom pipeline** - `python scripts/pipeline.py full`
2. **Przetestuj model** w interfejsie Gradio
3. **Oceń wyniki** przez generation samples
4. **Iteruj** - dostrajaj parametry i zwiększaj dane
5. **Deploy** - stwórz API dla aplikacji

---

## 🏆 **Sukces oznacza:**

✅ **Model generuje sensowne polskie teksty**  
✅ **Rozumie kontekst i gramatykę**  
✅ **Interface działa płynnie**  
✅ **Performance jest akceptowalny**  
✅ **System jest skalowalny i rozszerzalny**



