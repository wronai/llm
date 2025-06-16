#!/usr/bin/env python3
"""
WronAI Data Collection Pipeline - Naprawiona wersja
Kompletny system pobierania i przetwarzania polskich danych treningowych
"""

import os
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import Iterator, Dict, Any, List
import requests
from datasets import load_dataset
from tqdm import tqdm
import re
import time
import ftfy
from urllib.parse import urljoin
import gzip
import bz2

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
        self.simple_deduplicator = SimpleDeduplicator()

        # Statystyki
        self.total_processed = 0
        self.total_size_bytes = 0
        self.source_stats = {}

        # Session dla requestów
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'WronAI Data Collector (Educational/Research Purpose)'
        })

    def setup_directories(self):
        """Utwórz strukturę katalogów."""
        dirs = [
            "raw_data/high_quality/wikipedia_pl",
            "raw_data/high_quality/wolne_lektury",
            "raw_data/high_quality/academic_papers",
            "raw_data/medium_quality/oscar_pl",
            "raw_data/medium_quality/common_crawl",
            "raw_data/medium_quality/news_portals",
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
        """Pobierz i przetworz polską Wikipedię - naprawiona wersja."""
        logger.info("Rozpoczynam pobieranie polskiej Wikipedii...")

        try:
            # Używamy dostępnej wersji
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

                    if self.is_polish_text_simple(cleaned_text):
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

                        if processed_count % 1000 == 0:
                            logger.info(f"Wikipedia: przetworzono {processed_count} artykułów")

            self.source_stats['wikipedia_pl'] = processed_count
            logger.info(f"Wikipedia: przetworzono {processed_count} artykułów")
            return processed_count

        except Exception as e:
            logger.error(f"Błąd podczas pobierania Wikipedii: {e}")
            return 0

    def collect_wolne_lektury(self) -> int:
        """Pobierz teksty z Wolnych Lektur - ulepszona wersja."""
        logger.info("Rozpoczynam pobieranie Wolnych Lektur...")

        try:
            base_url = "https://wolnelektury.pl"
            api_url = f"{base_url}/api/books/"

            # Pobierz listę książek z limitami
            response = self.session.get(api_url, timeout=30)
            response.raise_for_status()
            books = response.json()

            output_file = self.output_dir / "raw_data/high_quality/wolne_lektury/books.jsonl"
            processed_count = 0
            failed_count = 0

            with open(output_file, 'w', encoding='utf-8') as f:
                for book in tqdm(books[:500], desc="Pobieranie Wolnych Lektur"):  # Ograniczamy do 500
                    if self.total_size_bytes >= self.target_size_bytes:
                        break

                    try:
                        # Sprawdź czy istnieje link do tekstu
                        txt_url = book.get('txt')
                        if not txt_url:
                            # Spróbuj alternatywnych formatów
                            if 'href' in book:
                                book_detail = self.session.get(book['href'], timeout=15)
                                book_detail.raise_for_status()
                                book_data = book_detail.json()
                                txt_url = book_data.get('txt')

                        if not txt_url:
                            failed_count += 1
                            continue

                        # Pobierz tekst książki
                        text_response = self.session.get(txt_url, timeout=30)
                        text_response.raise_for_status()

                        if text_response.status_code != 200:
                            failed_count += 1
                            continue

                        cleaned_text = self.text_cleaner.clean_text(text_response.text)

                        if len(cleaned_text) < 1000:  # Pomiń bardzo krótkie teksty
                            continue

                        # Sprawdź czy nie jest duplikatem
                        if not self.simple_deduplicator.is_duplicate(cleaned_text[:1000]):
                            self.simple_deduplicator.add_text(cleaned_text[:1000])

                            processed_item = {
                                'id': f"wolne_lektury_{processed_count}",
                                'title': book.get('title', ''),
                                'author': ', '.join([a.get('name', '') for a in book.get('authors', [])]),
                                'text': cleaned_text,
                                'source': 'wolne_lektury',
                                'url': book.get('url', ''),
                                'timestamp': datetime.now().isoformat()
                            }

                            f.write(json.dumps(processed_item, ensure_ascii=False) + '\n')
                            processed_count += 1
                            self.total_size_bytes += len(cleaned_text.encode('utf-8'))

                        # Pauza między requestami
                        time.sleep(0.5)

                    except Exception as e:
                        failed_count += 1
                        logger.warning(f"Błąd pobierania książki {book.get('title', 'unknown')}: {e}")
                        continue

            self.source_stats['wolne_lektury'] = processed_count
            logger.info(f"Wolne Lektury: przetworzono {processed_count} książek, błędów: {failed_count}")
            return processed_count

        except Exception as e:
            logger.error(f"Błąd podczas pobierania Wolnych Lektur: {e}")
            return 0

    def collect_oscar_alternative(self) -> int:
        """Pobierz dane z alternatywnych źródeł zamiast OSCAR."""
        logger.info("OSCAR niedostępny - używam alternatywnych źródeł polskiego tekstu...")

        try:
            # CC-100 Polish jako alternatywa
            try:
                dataset = load_dataset("cc100", lang="pl", split="train", streaming=True)
                return self.process_streaming_dataset(dataset, "cc100_pl", 10000)
            except:
                logger.warning("CC-100 niedostępny, próbuję inne źródła...")

            # mC4 Polish
            try:
                dataset = load_dataset("mc4", "pl", split="train", streaming=True)
                return self.process_streaming_dataset(dataset, "mc4_pl", 5000)
            except:
                logger.warning("mC4 niedostępny, próbuję crawling...")

            # Fallback: crawling polskich stron
            return self.collect_polish_web_content()

        except Exception as e:
            logger.error(f"Błąd podczas pobierania alternatywnych źródeł: {e}")
            return 0

    def process_streaming_dataset(self, dataset, source_name: str, max_samples: int) -> int:
        """Przetwórz streaming dataset."""
        output_file = self.output_dir / f"raw_data/medium_quality/{source_name}/texts.jsonl"
        processed_count = 0

        with open(output_file, 'w', encoding='utf-8') as f:
            for example in tqdm(dataset, desc=f"Przetwarzanie {source_name}", total=max_samples):
                if processed_count >= max_samples or self.total_size_bytes >= self.target_size_bytes:
                    break

                text = example.get('text', '')
                cleaned_text = self.text_cleaner.clean_text(text)

                if len(cleaned_text) < 100:
                    continue

                if self.is_polish_text_simple(cleaned_text) and not self.simple_deduplicator.is_duplicate(
                        cleaned_text[:500]):
                    self.simple_deduplicator.add_text(cleaned_text[:500])

                    processed_item = {
                        'id': f"{source_name}_{processed_count}",
                        'text': cleaned_text,
                        'source': source_name,
                        'timestamp': datetime.now().isoformat()
                    }

                    f.write(json.dumps(processed_item, ensure_ascii=False) + '\n')
                    processed_count += 1
                    self.total_size_bytes += len(cleaned_text.encode('utf-8'))

        self.source_stats[source_name] = processed_count
        return processed_count

    def collect_polish_web_content(self) -> int:
        """Pobierz content z polskich stron internetowych."""
        logger.info("Pobieranie zawartości z polskich portali...")

        polish_sources = [
            "https://www.gov.pl/web/",
            "https://www.sejm.gov.pl/",
            "https://www.nask.pl/",
            "https://www.uw.edu.pl/",
            "https://www.pw.edu.pl/",
        ]

        processed_count = 0
        output_file = self.output_dir / "raw_data/medium_quality/polish_web/content.jsonl"

        # To jest uproszczona implementacja - w prawdziwym scenariuszu
        # użylibyśmy właściwego scrapera z Scrapy lub Beautiful Soup
        logger.warning("Web crawling wymagałby bardziej zaawansowanej implementacji")

        return processed_count

    def collect_additional_sources(self) -> int:
        """Pobierz z dodatkowych publicznie dostępnych źródeł."""
        logger.info("Pobieranie z dodatkowych źródeł...")

        total_collected = 0

        # 1. Wikiquote po polsku
        total_collected += self.collect_wikiquote_pl()

        # 2. Wolne podręczniki
        total_collected += self.collect_wikibooks_pl()

        # 3. Publiczne korpusy z GitHub
        total_collected += self.collect_github_corpora()

        return total_collected

    def collect_wikiquote_pl(self) -> int:
        """Pobierz polskie cytaty z Wikiquote."""
        try:
            dataset = load_dataset("wikiquote", "pl", split="train", trust_remote_code=True)
            return self.process_wiki_dataset(dataset, "wikiquote_pl")
        except Exception as e:
            logger.warning(f"Wikiquote niedostępny: {e}")
            return 0

    def collect_wikibooks_pl(self) -> int:
        """Pobierz polskie podręczniki z Wikibooks."""
        try:
            # Wikibooks może być dostępny przez API Wikipedii
            api_url = "https://pl.wikibooks.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'allpages',
                'aplimit': 500
            }

            response = self.session.get(api_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            processed_count = 0
            output_file = self.output_dir / "raw_data/high_quality/wikibooks_pl/books.jsonl"

            # Uproszczona implementacja - wymagałaby pełnego parsowania API
            logger.info("Wikibooks wymagałby bardziej szczegółowej implementacji API")

            return processed_count

        except Exception as e:
            logger.warning(f"Wikibooks niedostępny: {e}")
            return 0

    def collect_github_corpora(self) -> int:
        """Pobierz publicznie dostępne korpusy z GitHub."""
        logger.info("Sprawdzanie publicznych korpusów na GitHub...")

        # Lista znanych publicznych korpusów polskich
        github_corpora = [
            "https://raw.githubusercontent.com/ipipan/corpus-sample/master/sample.txt",
            # Dodaj więcej gdy będą dostępne
        ]

        processed_count = 0
        output_dir = self.output_dir / "raw_data/medium_quality/github_corpora"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "texts.jsonl"

        with open(output_file, 'w', encoding='utf-8') as f:
            for url in github_corpora:
                try:
                    response = self.session.get(url, timeout=30)
                    response.raise_for_status()

                    cleaned_text = self.text_cleaner.clean_text(response.text)

                    if len(cleaned_text) > 1000 and self.is_polish_text_simple(cleaned_text):
                        processed_item = {
                            'id': f"github_corpus_{processed_count}",
                            'text': cleaned_text,
                            'source': 'github_corpora',
                            'url': url,
                            'timestamp': datetime.now().isoformat()
                        }

                        f.write(json.dumps(processed_item, ensure_ascii=False) + '\n')
                        processed_count += 1
                        self.total_size_bytes += len(cleaned_text.encode('utf-8'))

                except Exception as e:
                    logger.warning(f"Błąd pobierania z {url}: {e}")
                    continue

        self.source_stats['github_corpora'] = processed_count
        return processed_count

    def process_wiki_dataset(self, dataset, source_name: str) -> int:
        """Pomocnicza funkcja do przetwarzania datasetów wiki."""
        processed_count = 0
        output_file = self.output_dir / f"raw_data/high_quality/{source_name}/content.jsonl"

        with open(output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(dataset, desc=f"Przetwarzanie {source_name}"):
                if self.total_size_bytes >= self.target_size_bytes:
                    break

                text = item.get('text', item.get('content', ''))
                cleaned_text = self.text_cleaner.clean_text(text)

                if len(cleaned_text) > 200:
                    processed_item = {
                        'id': f"{source_name}_{processed_count}",
                        'title': item.get('title', ''),
                        'text': cleaned_text,
                        'source': source_name,
                        'timestamp': datetime.now().isoformat()
                    }

                    f.write(json.dumps(processed_item, ensure_ascii=False) + '\n')
                    processed_count += 1
                    self.total_size_bytes += len(cleaned_text.encode('utf-8'))

        self.source_stats[source_name] = processed_count
        return processed_count

    def is_polish_text_simple(self, text: str) -> bool:
        """Prosta heurystyka sprawdzania czy tekst jest polski."""
        if len(text) < 50:
            return False

        # Sprawdź polskie znaki
        polish_chars = 'ąćęłńóśźżĄĆĘŁŃÓŚŹŻ'
        polish_char_count = sum(1 for char in text if char in polish_chars)

        # Sprawdź polskie słowa
        polish_words = ['że', 'się', 'nie', 'jest', 'jako', 'oraz', 'przez', 'tylko', 'może', 'będzie']
        text_lower = text.lower()
        polish_word_count = sum(1 for word in polish_words if word in text_lower)

        # Sprawdź czy wygląda na polski
        return polish_char_count > 5 or polish_word_count > 2

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
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            all_data.append(item)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.warning(f"Błąd czytania pliku {file_path}: {e}")
                continue

        if not all_data:
            logger.warning("Brak danych do podziału!")
            return

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
                "licenses": ["CC-BY-SA", "Public Domain", "CC0"],
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

        # Faza 2: Główne korpusy internetowe (zastąpienie OSCAR)
        if self.total_size_bytes < self.target_size_bytes:
            self.collect_oscar_alternative()

        # Faza 3: Dodatkowe źródła
        if self.total_size_bytes < self.target_size_bytes:
            self.collect_additional_sources()

        # Faza 4: Przetwarzanie końcowe
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

        # Usuń HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        return text.strip()


class SimpleDeduplicator:
    """Prosta klasa do deduplikacji tekstów."""

    def __init__(self):
        self.seen_hashes = set()

    def is_duplicate(self, text: str) -> bool:
        """Sprawdź czy tekst jest duplikatem."""
        text_hash = hash(text.lower().strip())
        if text_hash in self.seen_hashes:
            return True
        return False

    def add_text(self, text: str):
        """Dodaj tekst do indeksu."""
        text_hash = hash(text.lower().strip())
        self.seen_hashes.add(text_hash)


# Punkt wejścia
if __name__ == "__main__":
    collector = WronAIDataCollector(
        output_dir="./wronai_data",
        target_size_gb=50
    )

    collector.run_collection_pipeline()