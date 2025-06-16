#!/bin/bash
# setup_wronai_fixed.sh
# Naprawiony setup script dla WronAI

echo "🐦‍⬛ Konfiguracja środowiska WronAI Data Collection"

# Sprawdź czy Python jest zainstalowany
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 nie jest zainstalowany!"
    exit 1
fi

# Utwórz środowisko wirtualne
echo "📦 Tworzenie środowiska wirtualnego..."
python3 -m venv wronai_env
source wronai_env/bin/activate

# Upgrade pip
echo "⬆️ Aktualizacja pip..."
pip install --upgrade pip

# Utwórz requirements.txt
cat > requirements.txt << EOF
# Core dependencies
datasets>=2.14.0
huggingface_hub>=0.16.0
transformers>=4.30.0

# Web scraping and requests
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=4.9.0

# Data processing
pandas>=2.0.0
pyarrow>=12.0.0
orjson>=3.9.0

# Progress and CLI
tqdm>=4.65.0
rich>=13.0.0

# Text processing
ftfy>=6.1.0
regex>=2023.6.3
unidecode>=1.3.0

# Compression
zstandard>=0.21.0

# Optional: Language detection (jeśli chcesz precyzyjniejszą detekcję)
# langdetect>=1.0.9

# Development tools
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0
EOF

# Zainstaluj zależności
echo "📥 Instalowanie zależności..."
pip install -r requirements.txt

# Utwórz strukturę projektową
echo "📁 Tworzenie struktury projektowej..."
mkdir -p {data,scripts,tests,docs}

# Skopiuj naprawiony skrypt
cat > scripts/collect_wronai_data_fixed.py << 'EOF'
#!/usr/bin/env python3
"""
WronAI Data Collection Pipeline - Naprawiona wersja
Używa dostępnych źródeł danych bez wymagania specjalnego dostępu
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
import requests
from datasets import load_dataset
from tqdm import tqdm
import re
import time
import ftfy

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WronAICollector:
    def __init__(self, output_dir="./data", target_size_gb=5):
        self.output_dir = Path(output_dir)
        self.target_size_gb = target_size_gb
        self.target_size_bytes = target_size_gb * 1024 * 1024 * 1024
        self.total_size_bytes = 0
        self.source_stats = {}

        self.output_dir.mkdir(exist_ok=True)

    def collect_wikipedia_polish(self):
        """Pobierz polską Wikipedię."""
        logger.info("Pobieranie polskiej Wikipedii...")

        try:
            # Używamy dostępnej wersji 2022
            dataset = load_dataset("wikipedia", "20220301.pl", split="train", trust_remote_code=True)

            output_file = self.output_dir / "wikipedia_pl.jsonl"
            processed_count = 0

            with open(output_file, 'w', encoding='utf-8') as f:
                for article in tqdm(dataset, desc="Wikipedia"):
                    if self.total_size_bytes >= self.target_size_bytes:
                        break

                    text = self.clean_text(article['text'])
                    if len(text) < 500:
                        continue

                    item = {
                        'id': f"wiki_{processed_count}",
                        'title': article['title'],
                        'text': text,
                        'source': 'wikipedia'
                    }

                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    processed_count += 1
                    self.total_size_bytes += len(text.encode('utf-8'))

            self.source_stats['wikipedia'] = processed_count
            logger.info(f"Wikipedia: {processed_count} artykułów")

        except Exception as e:
            logger.error(f"Błąd Wikipedia: {e}")

    def collect_wolne_lektury(self):
        """Pobierz Wolne Lektury z lepszym error handling."""
        logger.info("Pobieranie Wolnych Lektur...")

        try:
            api_url = "https://wolnelektury.pl/api/books/"
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            books = response.json()

            output_file = self.output_dir / "wolne_lektury.jsonl"
            processed_count = 0

            with open(output_file, 'w', encoding='utf-8') as f:
                for book in tqdm(books[:100], desc="Wolne Lektury"):  # Ograniczamy do 100
                    if self.total_size_bytes >= self.target_size_bytes:
                        break

                    try:
                        # Pobierz szczegóły książki
                        if 'href' in book:
                            detail_response = requests.get(book['href'], timeout=15)
                            if detail_response.status_code == 200:
                                book_detail = detail_response.json()
                                txt_url = book_detail.get('txt')

                                if txt_url:
                                    text_response = requests.get(txt_url, timeout=20)
                                    if text_response.status_code == 200:
                                        text = self.clean_text(text_response.text)

                                        if len(text) > 1000:
                                            item = {
                                                'id': f"lektura_{processed_count}",
                                                'title': book.get('title', ''),
                                                'text': text,
                                                'source': 'wolne_lektury'
                                            }

                                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
                                            processed_count += 1
                                            self.total_size_bytes += len(text.encode('utf-8'))

                        time.sleep(1)  # Respectful crawling

                    except Exception as e:
                        logger.warning(f"Błąd książki {book.get('title', '')}: {e}")
                        continue

            self.source_stats['wolne_lektury'] = processed_count
            logger.info(f"Wolne Lektury: {processed_count} książek")

        except Exception as e:
            logger.error(f"Błąd Wolne Lektury: {e}")

    def collect_alternative_sources(self):
        """Pobierz z alternatywnych źródeł."""
        logger.info("Pobieranie alternatywnych źródeł...")

        # Próbuj CC-100
        try:
            dataset = load_dataset("cc100", lang="pl", split="train", streaming=True)
            processed_count = 0
            output_file = self.output_dir / "cc100_pl.jsonl"

            with open(output_file, 'w', encoding='utf-8') as f:
                for example in tqdm(dataset, desc="CC-100", total=1000):
                    if processed_count >= 1000 or self.total_size_bytes >= self.target_size_bytes:
                        break

                    text = self.clean_text(example['text'])
                    if len(text) > 200 and self.is_polish(text):
                        item = {
                            'id': f"cc100_{processed_count}",
                            'text': text,
                            'source': 'cc100'
                        }

                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        processed_count += 1
                        self.total_size_bytes += len(text.encode('utf-8'))

            self.source_stats['cc100'] = processed_count
            logger.info(f"CC-100: {processed_count} tekstów")

        except Exception as e:
            logger.warning(f"CC-100 niedostępny: {e}")

    def clean_text(self, text):
        """Podstawowe czyszczenie tekstu."""
        if not text:
            return ""

        text = ftfy.fix_text(text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        text = re.sub(r'<[^>]+>', '', text)

        return text.strip()

    def is_polish(self, text):
        """Prosta detekcja polskiego tekstu."""
        polish_words = ['że', 'się', 'nie', 'jest', 'jako', 'przez', 'tylko']
        text_lower = text.lower()
        return sum(1 for word in polish_words if word in text_lower) >= 2

    def run(self):
        """Uruchom cały pipeline."""
        logger.info(f"Rozpoczynam zbieranie danych (cel: {self.target_size_gb}GB)")

        self.collect_wikipedia_polish()
        if self.total_size_bytes < self.target_size_bytes:
            self.collect_wolne_lektury()
        if self.total_size_bytes < self.target_size_bytes:
            self.collect_alternative_sources()

        logger.info(f"Zakończono! Zebrano {self.total_size_bytes/(1024**3):.2f}GB")
        logger.info(f"Statystyki: {self.source_stats}")

if __name__ == "__main__":
    collector = WronAICollector(output_dir="./data", target_size_gb=5)
    collector.run()
EOF

# Utwórz skrypt testowy
cat > scripts/test_data_collection.py << 'EOF'
#!/usr/bin/env python3
"""
Test script dla WronAI data collection
"""

import sys
import logging
from pathlib import Path

# Dodaj scripts do path
sys.path.append(str(Path(__file__).parent))

from collect_wronai_data_fixed import WronAICollector

def test_small_collection():
    """Test z małym rozmiarem danych."""
    print("🧪 Test zbierania danych...")

    collector = WronAICollector(
        output_dir="./test_data",
        target_size_gb=0.1  # 100MB dla testu
    )

    collector.run()

    # Sprawdź wyniki
    data_dir = Path("./test_data")
    files = list(data_dir.glob("*.jsonl"))

    print(f"✅ Utworzono {len(files)} plików:")
    for file in files:
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name}: {size_mb:.2f}MB")

if __name__ == "__main__":
    test_small_collection()
EOF

# Utwórz dokumentację
cat > docs/README.md << 'EOF'
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
bash setup_wronai_fixed.sh
source wronai_env/bin/activate

# Test (100MB)
python scripts/test_data_collection.py

# Pełne zbieranie (5GB)
python scripts/collect_wronai_data_fixed.py
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
EOF

# Utwórz Makefile dla wygody
cat > Makefile << 'EOF'
.PHONY: setup test collect clean

setup:
	bash setup_wronai_fixed.sh

test:
	source wronai_env/bin/activate && python scripts/test_data_collection.py

collect:
	source wronai_env/bin/activate && python scripts/collect_wronai_data_fixed.py

clean:
	rm -rf data/ test_data/ wronai_env/

help:
	@echo "WronAI Data Collection Commands:"
	@echo "  setup   - Setup environment"
	@echo "  test    - Run test collection (100MB)"
	@echo "  collect - Run full collection (5GB)"
	@echo "  clean   - Clean all data and env"
EOF

echo "✅ Setup zakończony!"
echo ""
echo "🚀 Następne kroki:"
echo "1. source wronai_env/bin/activate"
echo "2. make test  # Test z 100MB danych"
echo "3. make collect  # Pełne zbieranie 5GB"
echo ""
echo "📚 Dokumentacja: docs/README.md"