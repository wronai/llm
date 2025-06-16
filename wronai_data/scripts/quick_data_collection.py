#!/usr/bin/env python3
"""
WronAI Quick Start - Minimal working version
Prostą wersja do szybkiego uruchomienia bez problemów z dostępem do danych
"""

import json
import logging
from pathlib import Path
from datetime import datetime
import requests
from datasets import load_dataset
from tqdm import tqdm
import re
import time

# Konfiguracja
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleWronAICollector:
    """Uproszczona wersja collectora bez zewnętrznych zależności."""

    def __init__(self, target_size_mb=500):  # Domyślnie 500MB
        self.output_dir = Path("./wronai_simple_data")
        self.target_size_bytes = target_size_mb * 1024 * 1024
        self.total_size = 0
        self.stats = {}

        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"Cel: {target_size_mb}MB danych w {self.output_dir}")

    def clean_text(self, text):
        """Podstawowe czyszczenie."""
        if not text:
            return ""

        # Usuń nadmiar whitespace
        text = re.sub(r'\s+', ' ', text)
        # Usuń HTML tagi
        text = re.sub(r'<[^>]+>', '', text)
        # Usuń URLe
        text = re.sub(r'http\S+', '', text)

        return text.strip()

    def is_polish_simple(self, text):
        """Bardzo prosta detekcja polskiego."""
        polish_indicators = ['że', 'się', 'nie', 'jest', 'jako', 'przez', 'tylko', 'może', 'oraz']
        text_lower = text.lower()
        score = sum(1 for word in polish_indicators if word in text_lower)
        return score >= 2

    def collect_wikipedia(self):
        """Pobierz polską Wikipedię - wersja która działa."""
        logger.info("📖 Pobieranie Wikipedia...")

        try:
            # Użyj wersji która na pewno istnieje
            dataset = load_dataset("wikipedia", "20220301.pl", split="train", trust_remote_code=True)

            output_file = self.output_dir / "wikipedia.jsonl"
            count = 0

            with open(output_file, 'w', encoding='utf-8') as f:
                for article in tqdm(dataset, desc="Wikipedia articles"):
                    if self.total_size >= self.target_size_bytes:
                        break

                    text = self.clean_text(article['text'])
                    if len(text) < 300:  # Pomiń krótkie
                        continue

                    item = {
                        'id': f"wiki_{count}",
                        'title': article['title'],
                        'text': text,
                        'source': 'wikipedia',
                        'timestamp': datetime.now().isoformat()
                    }

                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    count += 1
                    self.total_size += len(text.encode('utf-8'))

                    if count % 1000 == 0:
                        logger.info(f"Wikipedia: {count} artykułów, {self.total_size / 1024 / 1024:.1f}MB")

            self.stats['wikipedia'] = count
            logger.info(f"✅ Wikipedia: {count} artykułów zebrano")

        except Exception as e:
            logger.error(f"❌ Wikipedia error: {e}")
            self.stats['wikipedia'] = 0

    def collect_synthetic_polish(self):
        """Generuj syntetyczne polskie teksty jako fallback."""
        logger.info("🔧 Generowanie syntetycznych przykładów...")

        # Podstawowe polskie teksty jako starter
        polish_samples = [
            "Język polski należy do grupy zachodniej języków słowiańskich. Jest językiem urzędowym w Polsce.",
            "Historia Polski sięga X wieku, kiedy to powstało pierwsze państwo polskie pod rządami Mieszka I.",
            "Warszawa jest stolicą Polski od 1596 roku. Miasto leży nad Wisłą w województwie mazowieckim.",
            "Literatura polska ma bogate tradycje. Najsłynniejszymi poetami są Adam Mickiewicz i Juliusz Słowacki.",
            "Polskie tradycje kulinarne obejmują pierogi, bigos, kotlet schabowy i wiele innych potraw.",
            "System edukacji w Polsce składa się ze szkoły podstawowej, średniej i wyższej.",
            "Polska przystąpiła do Unii Europejskiej w 2004 roku wraz z dziewięcioma innymi krajami.",
            "Gospodarka Polski opiera się głównie na przemyśle, usługach i rolnictwie.",
            "Polskie miasta mają bogate dziedzictwo architektoniczne, szczególnie Kraków i Gdańsk.",
            "Język polski używa alfabetu łacińskiego rozszerzonego o polskie znaki diakrytyczne."
        ]

        output_file = self.output_dir / "synthetic.jsonl"
        count = 0

        with open(output_file, 'w', encoding='utf-8') as f:
            # Rozszerz próbki przez warianty
            for i, base_text in enumerate(polish_samples):
                if self.total_size >= self.target_size_bytes:
                    break

                # Utwórz kilka wariantów każdego tekstu
                variants = [
                    base_text,
                    f"Według ekspertów, {base_text.lower()}",
                    f"Warto wiedzieć, że {base_text.lower()}",
                    f"Jak pokazują badania, {base_text.lower()}",
                    f"Należy podkreślić, że {base_text.lower()}"
                ]

                for variant in variants:
                    if self.total_size >= self.target_size_bytes:
                        break

                    # Dodaj więcej kontekstu
                    extended_text = f"{variant} To sprawia, że jest to ważny element polskiej kultury i tożsamości narodowej. Wiedza na ten temat jest kluczowa dla zrozumienia współczesnej Polski."

                    item = {
                        'id': f"synthetic_{count}",
                        'text': extended_text,
                        'source': 'synthetic',
                        'timestamp': datetime.now().isoformat()
                    }

                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    count += 1
                    self.total_size += len(extended_text.encode('utf-8'))

        self.stats['synthetic'] = count
        logger.info(f"✅ Synthetic: {count} tekstów wygenerowano")

    def collect_wolne_lektury_safe(self):
        """Bezpieczne pobieranie Wolnych Lektur z timeout'ami."""
        logger.info("📚 Próbuję pobrać Wolne Lektury...")

        try:
            # Krótki timeout - jeśli nie działa, nie blokujemy
            response = requests.get("https://wolnelektury.pl/api/books/", timeout=10)

            if response.status_code != 200:
                logger.warning("Wolne Lektury niedostępne, pomijam")
                return

            books = response.json()[:20]  # Tylko pierwsze 20

            output_file = self.output_dir / "lektury.jsonl"
            count = 0

            with open(output_file, 'w', encoding='utf-8') as f:
                for book in tqdm(books, desc="Wolne Lektury"):
                    if self.total_size >= self.target_size_bytes:
                        break

                    try:
                        if 'href' not in book:
                            continue

                        # Pobierz szczegóły
                        detail_resp = requests.get(book['href'], timeout=5)
                        if detail_resp.status_code == 200:
                            detail = detail_resp.json()
                            txt_url = detail.get('txt')

                            if txt_url:
                                text_resp = requests.get(txt_url, timeout=10)
                                if text_resp.status_code == 200:
                                    text = self.clean_text(text_resp.text)

                                    if len(text) > 1000:  # Tylko dłuższe teksty
                                        item = {
                                            'id': f"lektura_{count}",
                                            'title': book.get('title', ''),
                                            'text': text[:5000],  # Ogranicz rozmiar
                                            'source': 'wolne_lektury',
                                            'timestamp': datetime.now().isoformat()
                                        }

                                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                                        count += 1
                                        self.total_size += len(item['text'].encode('utf-8'))

                        time.sleep(0.5)  # Bądź grzeczny

                    except Exception as e:
                        logger.warning(f"Błąd książki: {e}")
                        continue

            self.stats['wolne_lektury'] = count
            logger.info(f"✅ Wolne Lektury: {count} książek")

        except Exception as e:
            logger.warning(f"Wolne Lektury całkowicie niedostępne: {e}")
            self.stats['wolne_lektury'] = 0

    def create_summary(self):
        """Utwórz podsumowanie zebranych danych."""
        summary = {
            'collection_date': datetime.now().isoformat(),
            'target_size_mb': self.target_size_bytes / 1024 / 1024,
            'actual_size_mb': self.total_size / 1024 / 1024,
            'sources': self.stats,
            'total_items': sum(self.stats.values()),
            'files_created': list(self.output_dir.glob("*.jsonl"))
        }

        # Zapisz podsumowanie
        summary_file = self.output_dir / "collection_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            # Konwertuj Path objects do string dla JSON
            summary['files_created'] = [str(f.name) for f in summary['files_created']]
            json.dump(summary, f, ensure_ascii=False, indent=2)

        return summary

    def run(self):
        """Uruchom kompletny proces zbierania."""
        logger.info("🚀 Rozpoczynam zbieranie danych WronAI...")

        # Faza 1: Wikipedia (najważniejsze)
        if self.total_size < self.target_size_bytes:
            self.collect_wikipedia()

        # Faza 2: Wolne Lektury (jeśli dostępne)
        if self.total_size < self.target_size_bytes:
            self.collect_wolne_lektury_safe()

        # Faza 3: Syntetyczne dane (fallback)
        if self.total_size < self.target_size_bytes:
            self.collect_synthetic_polish()

        # Podsumowanie
        summary = self.create_summary()

        logger.info("🎉 Zbieranie zakończone!")
        logger.info(f"📊 Zebrano: {summary['actual_size_mb']:.1f}MB z {summary['total_items']} elementów")
        logger.info(f"📁 Pliki: {', '.join(summary['files_created'])}")
        logger.info(f"📈 Źródła: {summary['sources']}")

        return summary


def main():
    """Główna funkcja."""
    print("🐦‍⬛ WronAI Data Collection - Quick Start")
    print("=" * 50)

    # Domyślnie zbierz 500MB - wystarczy na testy
    collector = SimpleWronAICollector(target_size_mb=500)

    try:
        summary = collector.run()

        print("\n✅ SUKCES!")
        print(f"Zebrano {summary['actual_size_mb']:.1f}MB polskich danych")
        print(f"Lokalizacja: {collector.output_dir}")
        print("\nNastępne kroki:")
        print("1. Sprawdź pliki .jsonl w katalogu wronai_simple_data/")
        print("2. Użyj danych do treningu modelu WronAI")
        print("3. Zwiększ target_size_mb dla większego datasetu")

    except KeyboardInterrupt:
        print("\n⏹️ Przerwano przez użytkownika")
    except Exception as e:
        print(f"\n❌ Błąd: {e}")
        print("Sprawdź logi powyżej dla szczegółów")


if __name__ == "__main__":
    main()