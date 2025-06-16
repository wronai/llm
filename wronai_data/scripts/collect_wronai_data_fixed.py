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
            # Używamy alternatywnego podejścia - bezpośrednio przez HuggingFace
            dataset = load_dataset(
                "wikipedia", 
                language="pl", 
                date="20220301", 
                split="train", 
                trust_remote_code=True
            )

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
            # Fallback - użyj alternatywnego źródła
            self.collect_wikipedia_fallback()

    def collect_wikipedia_fallback(self):
        """Alternatywna metoda pobierania danych z Wikipedii."""
        logger.info("Próba alternatywnego pobierania Wikipedii...")
        
        try:
            # Użyj mC4 jako alternatywy - zawiera dużo tekstów z Wikipedii
            dataset = load_dataset(
                "mc4", 
                "pl", 
                split="train", 
                streaming=True,
                trust_remote_code=True
            )
            
            output_file = self.output_dir / "wikipedia_pl.jsonl"
            processed_count = 0
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in tqdm(dataset.take(5000), desc="Wikipedia (fallback)"):
                    if self.total_size_bytes >= self.target_size_bytes:
                        break
                    
                    # Filtruj tylko teksty, które wyglądają jak z Wikipedii
                    if "wikipedia" in item.get('url', '').lower():
                        text = self.clean_text(item['text'])
                        if len(text) < 500:
                            continue
                        
                        doc = {
                            'id': f"wiki_fallback_{processed_count}",
                            'title': item.get('url', '').split('/')[-1],
                            'text': text,
                            'source': 'wikipedia_fallback'
                        }
                        
                        f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                        processed_count += 1
                        self.total_size_bytes += len(text.encode('utf-8'))
            
            self.source_stats['wikipedia_fallback'] = processed_count
            logger.info(f"Wikipedia (fallback): {processed_count} artykułów")
            
        except Exception as e:
            logger.error(f"Błąd Wikipedia fallback: {e}")
            logger.info("Pomijam źródło Wikipedia - kontynuuję z innymi źródłami")

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
