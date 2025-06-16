#!/usr/bin/env python3
"""
WronAI Data Collection Pipeline
Kompletny system pobierania i przetwarzania polskich danych treningowych
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Iterator, Dict, Any
import requests
from datasets import load_dataset
from tqdm import tqdm
import fasttext
from datasketch import MinHash, MinHashLSH
import re
import ftfy

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wronai_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WronAIDataCollector:
    def __init__(self, output_dir: str = "./wronai_data", target_size_gb: int = 50):
        self.output_dir = Path(output_dir)
        self.target_size_gb = target_size_gb
        self.target_size_bytes = target_size_gb * 1024 * 1024 * 1024

        # Utwórz strukturę katalogów
        self.setup_directories()

        # Inicjalizuj komponenty
        self.text_cleaner = PolishTextCleaner()
        self.deduplicator = TextDeduplicator()
        self.progress_tracker = ProgressTracker()

        # Ładuj model identyfikacji języka
        self.lang_model = fasttext.load_model('lid.176.bin')

        # Statystyki
        self.total_processed = 0
        self.total_size_bytes = 0
        self.source_stats = {}

    def setup_directories(self):
        """Utwórz strukturę katalogów."""
        dirs = [
            "raw_data/high_quality/wikipedia_pl",
            "raw_data/high_quality/wolne_lektury",
            "raw_data/high_quality/academic_papers",
            "raw_data/medium_quality/oscar_pl",
            "raw_data/medium_quality/common_crawl",
            "processed_data/filtered",
            "processed_data/deduplicated",
            "processed_data/tokenized",
            "splits/train",
            "splits/validation",
            "splits/test",
            "metadata",
            "logs"
        ]

        for dir_path in dirs:
            (self.output_dir / dir_path).mkdir(parents=True, exist_ok=True)

        logger.info(f"Struktura katalogów utworzona w {self.output_dir}")

    def collect_wikipedia_polish(self) -> int:
        """Pobierz i przetworz polską Wikipedię."""
        logger.info("Rozpoczynam pobieranie polskiej Wikipedii...")

        try:
            # Pobierz dataset Wikipedii - używamy dostępnej wersji
            dataset = load_dataset("wikipedia", "20220301.pl", split="train", trust_remote_code=True)

            output_file = self.output_dir / "raw_data/high_quality/wikipedia_pl/articles.jsonl"
            processed_count = 0

            with open(output_file, 'w', encoding='utf-8') as f:
                for article in tqdm(dataset, desc="Przetwarzanie Wikipedii"):
                    if self.total_size_bytes >= self.target_size_bytes:
                        break

                    cleaned_text = self.text_cleaner.clean_text(article['text'])

                    if len(cleaned_text) < 500:  # Pomiń bardzo krótkie artykuły
                        continue

                    if self.is_polish_text(cleaned_text):
                        processed_item = {
                            'id': f"wiki_pl_{processed_count}",
                            'title': article['title'],
                            'text': cleaned_text,
                            'source': 'wikipedia_pl',
                            'url': article.get('url', ''),
                            'timestamp': datetime.now().isoformat()
                        }

                        f.write(json.dumps(processed_item, ensure_ascii=False) + '\n')
                        processed_count += 1
                        self.total_size_bytes += len(cleaned_text.encode('utf-8'))

            self.source_stats['wikipedia_pl'] = processed_count
            logger.info(f"Wikipedia: przetworzono {processed_count} artykułów")
            return processed_count

        except Exception as e:
            logger.error(f"Błąd podczas pobierania Wikipedii: {e}")
            return 0

    def collect_oscar_polish(self) -> int:
        """Pobierz i przetworz OSCAR Polish."""
        logger.info("Rozpoczynam pobieranie OSCAR Polish...")

        try:
            # Załaduj OSCAR dataset - używamy publicznie dostępnej wersji
            dataset = load_dataset(
                "oscar",
                "unshuffled_deduplicated_pl",
                split="train",
                streaming=True,
                trust_remote_code=True
            )

            output_file = self.output_dir / "raw_data/medium_quality/oscar_pl/texts.jsonl"
            processed_count = 0

            with open(output_file, 'w', encoding='utf-8') as f:
                for example in tqdm(dataset, desc="Przetwarzanie OSCAR"):
                    if self.total_size_bytes >= self.target_size_bytes:
                        break

                    cleaned_text = self.text_cleaner.clean_text(example['text'])

                    if len(cleaned_text) < 100:
                        continue

                    # Sprawdź duplikaty
                    if not self.deduplicator.find_duplicates(cleaned_text):
                        doc_id = f"oscar_pl_{processed_count}"
                        self.deduplicator.add_document(doc_id, cleaned_text)

                        processed_item = {
                            'id': doc_id,
                            'text': cleaned_text,
                            'source': 'oscar_pl',
                            'timestamp': datetime.now().isoformat()
                        }

                        f.write(json.dumps(processed_item, ensure_ascii=False) + '\n')
                        processed_count += 1
                        self.total_size_bytes += len(cleaned_text.encode('utf-8'))

            self.source_stats['oscar_pl'] = processed_count
            logger.info(f"OSCAR: przetworzono {processed_count} tekstów")
            return processed_count

        except Exception as e:
            logger.error(f"Błąd podczas pobierania OSCAR: {e}")
            return 0

    def collect_wolne_lektury(self) -> int:
        """Pobierz teksty z Wolnych Lektur."""
        logger.info("Rozpoczynam pobieranie Wolnych Lektur...")

        try:
            base_url = "https://wolnelektury.pl"
            api_url = f"{base_url}/api/books/"

            # Pobierz listę książek
            response = requests.get(api_url)
            response.raise_for_status()
            books = response.json()

            output_file = self.output_dir / "raw_data/high_quality/wolne_lektury/books.jsonl"
            processed_count = 0

            with open(output_file, 'w', encoding='utf-8') as f:
                for book in tqdm(books[:1000], desc="Pobieranie Wolnych Lektur"):
                    if self.total_size_bytes >= self.target_size_bytes:
                        break

                    try:
                        # Pobierz tekst książki
                        txt_url = book.get('txt')
                        if not txt_url:
                            continue

                        text_response = requests.get(txt_url)
                        text_response.raise_for_status()

                        cleaned_text = self.text_cleaner.clean_text(text_response.text)

                        if len(cleaned_text) < 1000:  # Pomiń bardzo krótkie teksty
                            continue

                        processed_item = {
                            'id': f"wolne_lektury_{processed_count}",
                            'title': book.get('title', ''),
                            'author': ', '.join([a['name'] for a in book.get('authors', [])]),
                            'text': cleaned_text,
                            'source': 'wolne_lektury',
                            'url': book.get('url', ''),
                            'timestamp': datetime.now().isoformat()
                        }

                        f.write(json.dumps(processed_item, ensure_ascii=False) + '\n')
                        processed_count += 1
                        self.total_size_bytes += len(cleaned_text.encode('utf-8'))

                    except Exception as e:
                        logger.warning(f"Błąd pobierania książki {book.get('title', 'unknown')}: {e}")
                        continue

            self.source_stats['wolne_lektury'] = processed_count
            logger.info(f"Wolne Lektury: przetworzono {processed_count} książek")
            return processed_count

        except Exception as e:
            logger.error(f"Błąd podczas pobierania Wolnych Lektur: {e}")
            return 0

    def is_polish_text(self, text: str) -> bool:
        """Sprawdź czy tekst jest w języku polskim."""
        try:
            predictions = self.lang_model.predict(text.replace('\n', ' '), k=1)
            return predictions[0][0] == '__label__pl' and predictions[1][0] > 0.9
        except:
            return False

    def create_splits(self):
        """Utwórz podziały train/validation/test."""
        logger.info("Tworzenie podziałów datasetu...")

        # Zbierz wszystkie przetworzone pliki
        all_files = []
        for source_dir in (self.output_dir / "raw_data").rglob("*.jsonl"):
            all_files.append(source_dir)

        # Wczytaj wszystkie dane
        all_data = []
        for file_path in all_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        all_data.append(item)
                    except json.JSONDecodeError:
                        continue

        # Pomieszaj dane
        import random
        random.shuffle(all_data)

        # Podziel na train/val/test (80/10/10)
        total = len(all_data)
        train_size = int(0.8 * total)
        val_size = int(0.1 * total)

        train_data = all_data[:train_size]
        val_data = all_data[train_size:train_size + val_size]
        test_data = all_data[train_size + val_size:]

        # Zapisz podziały
        splits = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }

        for split_name, split_data in splits.items():
            output_file = self.output_dir / f"splits/{split_name}/data.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in split_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

        logger.info(f"Utworzono podziały: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    def generate_metadata(self):
        """Wygeneruj metadane datasetu."""
        metadata = {
            "dataset_info": {
                "name": "wronai_training_v1.0",
                "version": "1.0.0",
                "created_date": datetime.now().isoformat(),
                "total_samples": sum(self.source_stats.values()),
                "total_size_gb": round(self.total_size_bytes / (1024 ** 3), 2),
                "target_size_gb": self.target_size_gb
            },
            "source_stats": self.source_stats,
            "source_attribution": {
                "primary_sources": list(self.source_stats.keys()),
                "licenses": ["CC0", "CC-BY-SA", "Public Domain"],
                "created_by": "WronAI Data Collection Pipeline"
            },
            "processing_info": {
                "deduplication_performed": True,
                "language_filtering": True,
                "quality_filtering": True,
                "min_text_length": 100
            }
        }

        metadata_file = self.output_dir / "metadata/dataset_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Metadane zapisane w {metadata_file}")

    def run_collection_pipeline(self):
        """Uruchom kompletny pipeline zbierania danych."""
        logger.info(f"Rozpoczynam zbieranie danych dla WronAI (cel: {self.target_size_gb}GB)")

        # Faza 1: Wysokiej jakości źródła
        self.collect_wikipedia_polish()
        self.collect_wolne_lektury()

        # Faza 2: Główne korpusy internetowe
        if self.total_size_bytes < self.target_size_bytes:
            self.collect_oscar_polish()

        # Faza 3: Przetwarzanie końcowe
        self.create_splits()
        self.generate_metadata()

        logger.info(f"Pipeline zakończony! Zebrano {self.total_size_bytes / (1024 ** 3):.2f}GB danych")
        logger.info(f"Statystyki źródeł: {self.source_stats}")


class PolishTextCleaner:
    """Klasa do czyszczenia polskich tekstów."""

    def __init__(self):
        self.polish_chars = set('ąćęłńóśźżĄĆĘŁŃÓŚŹŻ')

    def clean_text(self, text: str) -> str:
        """Wyczyść i znormalizuj polski tekst."""
        if not text:
            return ""

        # Napraw encoding
        text = ftfy.fix_text(text)

        # Usuń nadmiar białych znaków
        text = re.sub(r'\s+', ' ', text)

        # Usuń znaki kontrolne
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

        # Znormalizuj polskie cudzysłowy
        text = re.sub(r'[„""]', '"', text)
        text = re.sub(r'[\']', "'", text)

        # Usuń URLe
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        return text.strip()


class TextDeduplicator:
    """Klasa do deduplikacji tekstów."""

    def __init__(self, threshold=0.8, num_perm=128):
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.processed_count = 0

    def get_minhash(self, text: str) -> MinHash:
        """Utwórz MinHash dla tekstu."""
        minhash = MinHash(num_perm=self.num_perm)

        # Utwórz shingle (3-gramy słów)
        words = text.lower().split()
        for i in range(len(words) - 2):
            shingle = ' '.join(words[i:i + 3])
            minhash.update(shingle.encode('utf-8'))

        return minhash

    def add_document(self, doc_id: str, text: str):
        """Dodaj dokument do indeksu deduplikacji."""
        minhash = self.get_minhash(text)
        self.lsh.insert(doc_id, minhash)
        self.processed_count += 1

    def find_duplicates(self, text: str) -> list:
        """Znajdź podobne dokumenty."""
        minhash = self.get_minhash(text)
        return self.lsh.query(minhash)


class ProgressTracker:
    """Klasa do śledzenia postępu."""

    def __init__(self):
        self.start_time = datetime.now()

    def log_progress(self, processed: int, total: int, source: str):
        """Zaloguj postęp przetwarzania."""
        percentage = (processed / total) * 100 if total > 0 else 0
        elapsed = datetime.now() - self.start_time
        logger.info(f"{source}: {processed}/{total} ({percentage:.1f}%) - czas: {elapsed}")


# Punkt wejścia
if __name__ == "__main__":
    collector = WronAIDataCollector(
        output_dir="./",
        target_size_gb=50
    )

    collector.run_collection_pipeline()