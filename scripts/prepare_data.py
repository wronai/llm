#!/usr/bin/env python3
"""
WronAI Data Preparation Script
Prepare Polish language datasets for training
"""

import argparse
import json
import logging
import os
import re
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
import spacy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolishDataProcessor:
    def __init__(self, output_dir: str = "data/processed"):
        """Initialize Polish data processor."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load Polish spaCy model for text processing
        try:
            self.nlp = spacy.load("pl_core_news_sm")
        except OSError:
            logger.warning(
                "Polish spaCy model not found. Install with: python -m spacy download pl_core_news_sm"
            )
            self.nlp = None

    def download_polish_wikipedia(self) -> str:
        """Download Polish Wikipedia dataset."""
        logger.info("Downloading Polish Wikipedia dataset...")

        dataset = load_dataset("wikipedia", "20220301.pl", split="train")

        # Save to jsonl format
        wiki_path = self.output_dir / "polish_wikipedia.jsonl"

        with open(wiki_path, "w", encoding="utf-8") as f:
            for article in tqdm(dataset, desc="Processing Wikipedia articles"):
                if len(article["text"].strip()) > 200:  # Filter short articles
                    f.write(
                        json.dumps(
                            {
                                "text": article["text"],
                                "title": article["title"],
                                "source": "wikipedia",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

        logger.info(f"Polish Wikipedia saved to: {wiki_path}")
        return str(wiki_path)

    def download_oscar_polish(self) -> str:
        """Download OSCAR Polish dataset."""
        logger.info("Downloading OSCAR Polish dataset...")

        try:
            # OSCAR is quite large, so we'll take a subset
            dataset = load_dataset(
                "oscar-corpus/OSCAR-2301", "pl", split="train", streaming=True
            )

            oscar_path = self.output_dir / "oscar_polish.jsonl"

            with open(oscar_path, "w", encoding="utf-8") as f:
                count = 0
                for item in tqdm(
                    dataset.take(100000), desc="Processing OSCAR"
                ):  # Take 100k samples
                    text = item["text"].strip()
                    if len(text) > 100 and self.is_good_quality_text(text):
                        f.write(
                            json.dumps(
                                {"text": text, "source": "oscar"}, ensure_ascii=False
                            )
                            + "\n"
                        )
                        count += 1

                        if count >= 50000:  # Limit to manageable size
                            break

            logger.info(f"OSCAR Polish subset saved to: {oscar_path}")
            return str(oscar_path)

        except Exception as e:
            logger.error(f"Error downloading OSCAR: {e}")
            return None

    def create_instruction_dataset(self) -> str:
        """Create Polish instruction-following dataset."""
        logger.info("Creating Polish instruction dataset...")

        # Sample Polish instructions and responses
        polish_instructions = [
            {
                "instruction": "Wyjaśnij czym jest sztuczna inteligencja.",
                "response": "Sztuczna inteligencja (AI) to dziedzina informatyki zajmująca się tworzeniem systemów komputerowych zdolnych do wykonywania zadań, które tradycyjnie wymagają ludzkiej inteligencji, takich jak rozpoznawanie mowy, podejmowanie decyzji, rozwiązywanie problemów czy uczenie się.",
            },
            {
                "instruction": "Napisz krótki wiersz o Polsce.",
                "response": "Nad Wisłą szumią wierzby,\nBiało-czerwona flaga,\nW polskich sercach miłość,\nDo ojczyzny droga jaga.\n\nKraków, Warszawa, Gdańsk,\nHistoria w kamieniu,\nPolska nasza piękna,\nW każdym polskim imieniu.",
            },
            {
                "instruction": "Jakie są tradycyjne polskie potrawy?",
                "response": "Tradycyjne polskie potrawy to między innymi: pierogi z różnymi nadzieniami, bigos (kapusta kiszona z mięsem), kotlet schabowy, żurek, rosół, placki ziemniaczane, kielbasa, gołąbki, sernik i makowiec. Polska kuchnia charakteryzuje się sytnymi, pożywnymi daniami często opartymi na mięsie, kapuście i ziemniakach.",
            },
            {
                "instruction": "Jak się liczy do dziesięciu po polsku?",
                "response": "Liczenie do dziesięciu po polsku: jeden, dwa, trzy, cztery, pięć, sześć, siedem, osiem, dziewięć, dziesięć.",
            },
            {
                "instruction": "Opisz pogodę w Polsce.",
                "response": "Polska ma klimat umiarkowany przejściowy między oceanicznym a kontynentalnym. Charakteryzuje się czterema wyraźnymi porami roku: ciepłe lato (20-25°C), chłodna zima (0 do -5°C), oraz łagodna wiosna i jesień. Opady są rozłożone równomiernie przez cały rok, z nieco większą intensywnością latem.",
            },
        ]

        # Expand with variations and additional examples
        expanded_instructions = []

        # Add base instructions
        expanded_instructions.extend(polish_instructions)

        # Add question-answer variations
        qa_pairs = [
            (
                "Co to jest demokratacja?",
                "Demokracja to ustrój polityczny, w którym władza należy do narodu i jest sprawowana przez wybrane przez niego organy.",
            ),
            (
                "Kiedy Polska odzyskała niepodległość?",
                "Polska odzyskała niepodległość 11 listopada 1918 roku, po 123 latach zaborów.",
            ),
            ("Jak nazywa się stolica Polski?", "Stolicą Polski jest Warszawa."),
            ("Ile województw ma Polska?", "Polska ma 16 województw."),
            (
                "Jakie morze obmywa Polskę?",
                "Polska jest obmywana przez Morze Bałtyckie.",
            ),
            (
                "Kto napisał 'Pana Tadeusza'?",
                "Autorem 'Pana Tadeusza' jest Adam Mickiewicz.",
            ),
            ("Jak się nazywa polska waluta?", "Polską walutą jest złoty (PLN)."),
            ("Ile osób mieszka w Polsce?", "W Polsce mieszka około 38 milionów osób."),
        ]

        for question, answer in qa_pairs:
            expanded_instructions.append({"instruction": question, "response": answer})

        # Add task-based instructions
        task_instructions = [
            {
                "instruction": "Przetłumacz na język angielski: 'Miło Cię poznać'",
                "response": "'Miło Cię poznać' po angielsku to 'Nice to meet you'.",
            },
            {
                "instruction": "Podaj synonimy słowa 'piękny'",
                "response": "Synonimy słowa 'piękny': ładny, przystojny, atrakcyjny, wspaniały, cudowny, śliczny, urczy, przepiękny.",
            },
            {
                "instruction": "Wyjaśnij różnicę między 'może' a 'może być'",
                "response": "'Może' to przysłówek wyrażający możliwość (np. 'może przyjdę'), a 'może być' to wyrażenie oznaczające zgodę lub prawdopodobieństwo (np. 'może być, że tak').",
            },
        ]

        expanded_instructions.extend(task_instructions)

        # Save instruction dataset
        instruct_path = self.output_dir / "polish_instructions.json"

        with open(instruct_path, "w", encoding="utf-8") as f:
            json.dump(expanded_instructions, f, ensure_ascii=False, indent=2)

        logger.info(f"Polish instruction dataset saved to: {instruct_path}")
        return str(instruct_path)

    def is_good_quality_text(self, text: str) -> bool:
        """Check if text is good quality for training."""
        # Basic quality filters
        if len(text) < 50:
            return False

        # Check for excessive repetition
        words = text.lower().split()
        if len(set(words)) / len(words) < 0.3:  # Less than 30% unique words
            return False

        # Check for excessive punctuation or numbers
        non_alpha_ratio = len(
            [c for c in text if not c.isalpha() and not c.isspace()]
        ) / len(text)
        if non_alpha_ratio > 0.3:
            return False

        # Check if text contains Polish characters
        polish_chars = set("ąćęłńóśźż")
        if not any(char in polish_chars for char in text.lower()):
            return False

        return True

    def process_text_file(self, file_path: str) -> List[Dict]:
        """Process a text file and extract sentences/paragraphs."""
        texts = []

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        for paragraph in paragraphs:
            if self.is_good_quality_text(paragraph):
                texts.append({"text": paragraph, "source": Path(file_path).stem})

        return texts

    def create_combined_dataset(self) -> str:
        """Combine all processed datasets into final training format."""
        logger.info("Creating combined dataset...")

        combined_data = []

        # Process instruction data
        instruct_path = self.output_dir / "polish_instructions.json"
        if instruct_path.exists():
            with open(instruct_path, "r", encoding="utf-8") as f:
                instructions = json.load(f)

            for item in instructions:
                combined_data.append(
                    {
                        "text": f"Instrukcja: {item['instruction']}\nOdpowiedź: {item['response']}",
                        "type": "instruction",
                        "source": "manual",
                    }
                )

        # Process other text files
        for file_path in self.output_dir.glob("*.jsonl"):
            logger.info(f"Processing {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if "text" in item and self.is_good_quality_text(item["text"]):
                            combined_data.append(
                                {
                                    "text": item["text"],
                                    "type": "general",
                                    "source": item.get("source", "unknown"),
                                }
                            )
                    except json.JSONDecodeError:
                        continue

        # Create train/validation split
        train_size = int(0.95 * len(combined_data))
        train_data = combined_data[:train_size]
        val_data = combined_data[train_size:]

        # Save datasets
        dataset_dict = DatasetDict(
            {
                "train": Dataset.from_pandas(pd.DataFrame(train_data)),
                "validation": Dataset.from_pandas(pd.DataFrame(val_data)),
            }
        )

        dataset_path = self.output_dir / "wronai_dataset"
        dataset_dict.save_to_disk(str(dataset_path))

        logger.info(f"Combined dataset saved to: {dataset_path}")
        logger.info(
            f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}"
        )

        return str(dataset_path)


def main():
    parser = argparse.ArgumentParser(description="WronAI Data Preparation")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--download-wikipedia", action="store_true", help="Download Polish Wikipedia"
    )
    parser.add_argument(
        "--download-oscar", action="store_true", help="Download OSCAR Polish subset"
    )
    parser.add_argument(
        "--create-instructions", action="store_true", help="Create instruction dataset"
    )
    parser.add_argument(
        "--combine-all", action="store_true", help="Combine all datasets"
    )
    parser.add_argument("--all", action="store_true", help="Run all preparation steps")

    args = parser.parse_args()

    processor = PolishDataProcessor(args.output_dir)

    if args.all:
        args.download_wikipedia = True
        args.download_oscar = True
        args.create_instructions = True
        args.combine_all = True

    if args.create_instructions:
        processor.create_instruction_dataset()

    if args.download_wikipedia:
        processor.download_polish_wikipedia()

    if args.download_oscar:
        processor.download_oscar_polish()

    if args.combine_all:
        processor.create_combined_dataset()

    logger.info("Data preparation completed!")


if __name__ == "__main__":
    main()
