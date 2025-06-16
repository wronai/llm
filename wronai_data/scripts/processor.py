#!/usr/bin/env python3
"""
WronAI Data Processing Pipeline
Przetwarzanie zebranych danych do formatu treningowego
"""

import json
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import random
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WronAIDataProcessor:
    """Processor do przygotowania danych treningowych."""

    def __init__(self,
                 input_dir: str = "./wronai_simple_data",
                 output_dir: str = "./wronai_processed",
                 tokenizer_name: str = "microsoft/DialoGPT-medium"):

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Załaduj tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Załadowano tokenizer: {tokenizer_name}")
        except Exception as e:
            logger.error(f"Błąd ładowania tokenizer: {e}")
            raise

        self.stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'total_tokens': 0,
            'filtered_out': 0,
            'duplicates_removed': 0
        }

        # Parametry przetwarzania
        self.min_length = 50  # Minimalna długość tekstu
        self.max_length = 1024  # Maksymalna długość chunka
        self.chunk_overlap = 100  # Overlap między chunkami
        self.quality_threshold = 0.7  # Próg jakości tekstu

    def load_raw_data(self) -> List[Dict[str, Any]]:
        """Załaduj wszystkie surowe dane."""
        logger.info("📥 Ładowanie surowych danych...")

        all_data = []

        # Znajdź wszystkie pliki JSONL
        jsonl_files = list(self.input_dir.glob("*.jsonl"))

        if not jsonl_files:
            raise FileNotFoundError(f"Brak plików JSONL w {self.input_dir}")

        for file_path in jsonl_files:
            logger.info(f"Ładowanie {file_path.name}...")

            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line)
                        item['file_source'] = file_path.name
                        item['line_number'] = line_num
                        all_data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Błąd JSON w {file_path.name}:{line_num}: {e}")
                        continue

        self.stats['total_documents'] = len(all_data)
        logger.info(f"✅ Załadowano {len(all_data)} dokumentów z {len(jsonl_files)} plików")

        return all_data

    def clean_and_filter_text(self, text: str) -> str:
        """Wyczyść i przefiltruj tekst."""
        if not text or not isinstance(text, str):
            return ""

        # Podstawowe czyszczenie
        text = text.strip()

        # Usuń nadmierne powtórzenia
        text = re.sub(r'(.)\1{4,}', r'\1\1\1', text)  # Max 3 powtórzenia znaku
        text = re.sub(r'(\w+\s+)\1{3,}', r'\1', text)  # Max 3 powtórzenia słowa

        # Usuń bardzo długie linie (prawdopodobnie spam/błędy)
        lines = text.split('\n')
        filtered_lines = [line for line in lines if len(line) < 500]
        text = '\n'.join(filtered_lines)

        # Usuń nadmiar białych znaków
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Max 2 \n z rzędu

        return text.strip()

    def assess_text_quality(self, text: str) -> float:
        """Oceń jakość tekstu (0-1)."""
        if not text:
            return 0.0

        score = 1.0

        # Sprawdź długość
        if len(text) < self.min_length:
            return 0.0

        # Sprawdź stosunek liter do znaków specjalnych
        letter_ratio = sum(c.isalpha() for c in text) / len(text)
        if letter_ratio < 0.6:
            score *= 0.5

        # Sprawdź czy nie ma za dużo cyfr
        digit_ratio = sum(c.isdigit() for c in text) / len(text)
        if digit_ratio > 0.3:
            score *= 0.7

        # Sprawdź polskie znaki
        polish_chars = 'ąćęłńóśźżĄĆĘŁŃÓŚŹŻ'
        has_polish = any(c in polish_chars for c in text)
        if has_polish:
            score *= 1.1  # Bonus za polskie znaki

        # Sprawdź polskie słowa
        polish_words = ['że', 'się', 'nie', 'jest', 'jako', 'przez', 'tylko', 'może', 'oraz', 'ale']
        text_lower = text.lower()
        polish_word_count = sum(1 for word in polish_words if word in text_lower)
        if polish_word_count >= 3:
            score *= 1.2  # Bonus za polskie słowa

        return min(score, 1.0)

    def create_text_chunks(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Podziel tekst na chunki odpowiednie do treningu."""
        if not text:
            return []

        chunks = []

        # Tokenizuj tekst
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        # Jeśli tekst jest krótki, zwróć jako jeden chunk
        if len(tokens) <= self.max_length:
            chunk_text = self.tokenizer.decode(tokens)
            chunks.append({
                'text': chunk_text,
                'doc_id': doc_id,
                'chunk_id': f"{doc_id}_0",
                'tokens': len(tokens),
                'chunk_index': 0
            })
            return chunks

        # Podziel na chunki z overlapem
        chunk_index = 0
        start = 0

        while start < len(tokens):
            end = min(start + self.max_length, len(tokens))
            chunk_tokens = tokens[start:end]

            # Dekoduj chunk
            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunks.append({
                'text': chunk_text,
                'doc_id': doc_id,
                'chunk_id': f"{doc_id}_{chunk_index}",
                'tokens': len(chunk_tokens),
                'chunk_index': chunk_index
            })

            chunk_index += 1

            # Następny chunk z overlapem
            if end >= len(tokens):
                break
            start = end - self.chunk_overlap

        return chunks

    def deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Usuń duplikaty na poziomie chunków."""
        logger.info("🔍 Deduplikacja chunków...")

        seen_hashes = set()
        unique_chunks = []
        duplicates = 0

        for chunk in tqdm(chunks, desc="Deduplikacja"):
            # Utwórz hash z tekstu (pierwsze 200 znaków)
            text_sample = chunk['text'][:200].lower().strip()
            text_hash = hashlib.md5(text_sample.encode('utf-8')).hexdigest()

            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_chunks.append(chunk)
            else:
                duplicates += 1

        self.stats['duplicates_removed'] = duplicates
        logger.info(f"🗑️ Usunięto {duplicates} duplikatów")

        return unique_chunks

    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Główna funkcja przetwarzania dokumentów."""
        logger.info("⚙️ Przetwarzanie dokumentów...")

        all_chunks = []
        filtered_out = 0

        for doc in tqdm(documents, desc="Przetwarzanie dokumentów"):
            # Wyciągnij tekst
            text = doc.get('text', '')
            if 'title' in doc and doc['title']:
                text = f"{doc['title']}\n\n{text}"

            # Wyczyść tekst
            cleaned_text = self.clean_and_filter_text(text)

            # Oceń jakość
            quality_score = self.assess_text_quality(cleaned_text)

            if quality_score < self.quality_threshold:
                filtered_out += 1
                continue

            # Utwórz chunki
            doc_id = doc.get('id', f"doc_{len(all_chunks)}")
            chunks = self.create_text_chunks(cleaned_text, doc_id)

            # Dodaj metadane do chunków
            for chunk in chunks:
                chunk.update({
                    'source': doc.get('source', 'unknown'),
                    'quality_score': quality_score,
                    'original_length': len(text),
                    'processed_timestamp': datetime.now().isoformat()
                })

            all_chunks.extend(chunks)

        self.stats['filtered_out'] = filtered_out
        self.stats['total_chunks'] = len(all_chunks)

        logger.info(f"📊 Utworzono {len(all_chunks)} chunków, odfiltrowano {filtered_out} dokumentów")

        return all_chunks

    def create_training_dataset(self, chunks: List[Dict[str, Any]]) -> Dataset:
        """Utwórz dataset do treningu."""
        logger.info("📚 Tworzenie datasetu treningowego...")

        # Przygotuj dane do treningu języka
        training_texts = []

        for chunk in chunks:
            # Format dla language modeling
            text = chunk['text']

            # Dodaj specjalne tokeny
            formatted_text = f"{self.tokenizer.bos_token}{text}{self.tokenizer.eos_token}"
            training_texts.append(formatted_text)

        # Utwórz HuggingFace Dataset
        dataset_dict = {
            'text': training_texts,
            'length': [len(text) for text in training_texts]
        }

        dataset = Dataset.from_dict(dataset_dict)

        # Tokenizuj
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_special_tokens_mask=False
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )

        self.stats['total_tokens'] = sum(len(example['input_ids']) for example in tokenized_dataset)

        logger.info(f"✅ Dataset utworzony: {len(tokenized_dataset)} próbek, {self.stats['total_tokens']} tokenów")

        return tokenized_dataset

    def split_dataset(self, dataset: Dataset,
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.1) -> Dict[str, Dataset]:
        """Podziel dataset na train/val/test."""
        logger.info(f"✂️ Dzielenie datasetu ({train_ratio:.1%}/{val_ratio:.1%}/{test_ratio:.1%})...")

        # Sprawdź proporcje
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Proporcje muszą sumować się do 1.0"

        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size

        # Pomieszaj przed podziałem
        shuffled_dataset = dataset.shuffle(seed=42)

        # Podziel
        train_dataset = shuffled_dataset.select(range(train_size))
        val_dataset = shuffled_dataset.select(range(train_size, train_size + val_size))
        test_dataset = shuffled_dataset.select(range(train_size + val_size, total_size))

        splits = {
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        }

        logger.info(f"📊 Podział: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

        return splits

    def save_processed_data(self, dataset_splits: Dict[str, Dataset], chunks: List[Dict[str, Any]]):
        """Zapisz przetworzone dane."""
        logger.info("💾 Zapisywanie przetworzonych danych...")

        # Zapisz datasety w formacie HuggingFace
        for split_name, split_dataset in dataset_splits.items():
            output_path = self.output_dir / f"{split_name}_dataset"
            split_dataset.save_to_disk(str(output_path))
            logger.info(f"Zapisano {split_name}: {output_path}")

        # Zapisz czyste chunki jako backup
        chunks_file = self.output_dir / "processed_chunks.jsonl"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

        # Zapisz statystyki
        stats_file = self.output_dir / "processing_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)

        # Zapisz konfigurację tokenizera
        tokenizer_dir = self.output_dir / "tokenizer"
        self.tokenizer.save_pretrained(str(tokenizer_dir))

        logger.info(f"✅ Wszystkie dane zapisane w {self.output_dir}")

    def generate_data_report(self):
        """Wygeneruj raport z przetwarzania."""
        report = {
            'processing_date': datetime.now().isoformat(),
            'input_directory': str(self.input_dir),
            'output_directory': str(self.output_dir),
            'tokenizer_used': self.tokenizer.name_or_path,
            'parameters': {
                'min_length': self.min_length,
                'max_length': self.max_length,
                'chunk_overlap': self.chunk_overlap,
                'quality_threshold': self.quality_threshold
            },
            'statistics': self.stats,
            'files_created': [
                'train_dataset/',
                'validation_dataset/',
                'test_dataset/',
                'processed_chunks.jsonl',
                'processing_stats.json',
                'tokenizer/'
            ]
        }

        report_file = self.output_dir / "processing_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return report

    def run_processing_pipeline(self):
        """Uruchom kompletny pipeline przetwarzania."""
        logger.info("🚀 Rozpoczynam przetwarzanie danych WronAI...")

        try:
            # 1. Załaduj surowe dane
            documents = self.load_raw_data()

            # 2. Przetwórz dokumenty na chunki
            chunks = self.process_documents(documents)

            # 3. Deduplikacja
            unique_chunks = self.deduplicate_chunks(chunks)

            # 4. Utwórz dataset treningowy
            dataset = self.create_training_dataset(unique_chunks)

            # 5. Podziel dataset
            splits = self.split_dataset(dataset)

            # 6. Zapisz wszystko
            self.save_processed_data(splits, unique_chunks)

            # 7. Wygeneruj raport
            report = self.generate_data_report()

            logger.info("🎉 Przetwarzanie zakończone pomyślnie!")
            logger.info(f"📊 Finalne statystyki:")
            logger.info(f"  - Dokumenty: {self.stats['total_documents']}")
            logger.info(f"  - Chunki: {self.stats['total_chunks']}")
            logger.info(f"  - Tokeny: {self.stats['total_tokens']:,}")
            logger.info(f"  - Duplikaty usunięte: {self.stats['duplicates_removed']}")
            logger.info(f"  - Odfiltrowane: {self.stats['filtered_out']}")

            return report

        except Exception as e:
            logger.error(f"❌ Błąd podczas przetwarzania: {e}")
            raise


def main():
    """Główna funkcja."""
    print("🐦‍⬛ WronAI Data Processing Pipeline")
    print("=" * 50)

    try:
        processor = WronAIDataProcessor(
            input_dir="./wronai_simple_data",
            output_dir="./wronai_processed",
            tokenizer_name="microsoft/DialoGPT-medium"  # Dobry starter tokenizer
        )

        report = processor.run_processing_pipeline()

        print("\n✅ PRZETWARZANIE ZAKOŃCZONE!")
        print(f"📁 Dane treningowe w: {processor.output_dir}")
        print(f"📊 Tokeny: {report['statistics']['total_tokens']:,}")
        print(f"📚 Chunki: {report['statistics']['total_chunks']:,}")

        print("\nNastępne kroki:")
        print("1. Sprawdź wygenerowane datasety")
        print("2. Uruchom trening modelu WronAI")
        print("3. Ewaluuj model na test set")

    except KeyboardInterrupt:
        print("\n⏹️ Przerwano przez użytkownika")
    except Exception as e:
        print(f"\n❌ Błąd: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()