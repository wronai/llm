#!/usr/bin/env python3
"""
WronAI Master Pipeline
Orchestruje cały proces: zbieranie danych → przetwarzanie → trening → inferowanie
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WronAIMasterPipeline:
    """Główny pipeline do zarządzania całym procesem WronAI."""

    def __init__(self, base_dir: str = "./wronai_workspace"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

        # Ścieżki do poszczególnych etapów
        self.data_dir = self.base_dir / "data"
        self.processed_dir = self.base_dir / "processed"
        self.model_dir = self.base_dir / "model"
        self.logs_dir = self.base_dir / "logs"

        # Utwórz strukturę katalogów
        for dir_path in [self.data_dir, self.processed_dir, self.model_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)

        # Stan pipeline'u
        self.pipeline_state = {
            'data_collection': False,
            'data_processing': False,
            'model_training': False,
            'model_ready': False
        }

        self.load_pipeline_state()

    def load_pipeline_state(self):
        """Załaduj stan pipeline'u."""
        state_file = self.base_dir / "pipeline_state.json"

        if state_file.exists():
            with open(state_file, 'r') as f:
                self.pipeline_state = json.load(f)
            logger.info("📋 Stan pipeline'u załadowany")
        else:
            logger.info("📋 Nowy pipeline - stan resetowany")

    def save_pipeline_state(self):
        """Zapisz stan pipeline'u."""
        self.pipeline_state['last_updated'] = datetime.now().isoformat()

        state_file = self.base_dir / "pipeline_state.json"
        with open(state_file, 'w') as f:
            json.dump(self.pipeline_state, f, indent=2)

    def check_dependencies(self):
        """Sprawdź czy wszystkie zależności są zainstalowane."""
        logger.info("🔍 Sprawdzanie zależności...")

        required_packages = [
            'torch', 'transformers', 'datasets', 'peft',
            'bitsandbytes', 'accelerate', 'gradio'
        ]

        missing = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)

        if missing:
            logger.error(f"❌ Brakujące pakiety: {', '.join(missing)}")
            logger.info("Zainstaluj przez: pip install -r requirements.txt")
            return False

        logger.info("✅ Wszystkie zależności dostępne")
        return True

    def run_data_collection(self, target_size_mb: int = 500):
        """Uruchom zbieranie danych."""
        logger.info("📥 Rozpoczynam zbieranie danych...")

        try:
            # Import i uruchomienie data collection
            from quick_start_wronai import SimpleWronAICollector

            collector = SimpleWronAICollector(target_size_mb=target_size_mb)
            summary = collector.run()

            # Przenieś dane do workspace
            import shutil
            source_dir = Path("./wronai_simple_data")
            if source_dir.exists():
                if self.data_dir.exists():
                    shutil.rmtree(self.data_dir)
                shutil.move(str(source_dir), str(self.data_dir))

            self.pipeline_state['data_collection'] = True
            self.save_pipeline_state()

            logger.info(f"✅ Zbieranie danych zakończone: {summary['actual_size_mb']:.1f}MB")
            return summary

        except Exception as e:
            logger.error(f"❌ Błąd zbierania danych: {e}")
            return None

    def run_data_processing(self):
        """Uruchom przetwarzanie danych."""
        if not self.pipeline_state['data_collection']:
            logger.error("❌ Najpierw zbierz dane!")
            return None

        logger.info("⚙️ Rozpoczynam przetwarzanie danych...")

        try:
            # Import i uruchomienie data processing
            sys.path.append(str(Path(__file__).parent))
            from wronai_data_processing import WronAIDataProcessor

            processor = WronAIDataProcessor(
                input_dir=str(self.data_dir),
                output_dir=str(self.processed_dir)
            )

            report = processor.run_processing_pipeline()

            self.pipeline_state['data_processing'] = True
            self.save_pipeline_state()

            logger.info(f"✅ Przetwarzanie zakończone: {report['statistics']['total_chunks']} chunków")
            return report

        except Exception as e:
            logger.error(f"❌ Błąd przetwarzania: {e}")
            return None

    def run_model_training(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Uruchom trening modelu."""
        if not self.pipeline_state['data_processing']:
            logger.error("❌ Najpierw przetwórz dane!")
            return None

        logger.info("🏋️ Rozpoczynam trening modelu...")

        try:
            from wronai_training import WronAITrainer

            trainer = WronAITrainer(
                model_name=model_name,
                data_dir=str(self.processed_dir),
                output_dir=str(self.model_dir)
            )

            report = trainer.run_full_training_pipeline()

            self.pipeline_state['model_training'] = True
            self.pipeline_state['model_ready'] = True
            self.save_pipeline_state()

            logger.info(f"✅ Trening zakończony: {report['training_stats']['train_loss']:.4f} loss")
            return report

        except Exception as e:
            logger.error(f"❌ Błąd treningu: {e}")
            return None

    def run_model_inference(self):
        """Uruchom inferowanie modelu."""
        if not self.pipeline_state['model_ready']:
            logger.error("❌ Najpierw wytrenuj model!")
            return None

        logger.info("🤖 Uruchamiam inferowanie...")

        try:
            from wronai_inference import WronAIInference

            inference = WronAIInference(model_path=str(self.model_dir))

            # Podstawowa ewaluacja
            eval_results = inference.evaluate_polish_capabilities()
            benchmark_results = inference.benchmark_performance()

            logger.info("✅ Model gotowy do użycia!")
            return inference, eval_results, benchmark_results

        except Exception as e:
            logger.error(f"❌ Błąd inferowania: {e}")
            return None

    def run_full_pipeline(self, target_size_mb: int = 500, model_name: str = "microsoft/DialoGPT-medium"):
        """Uruchom kompletny pipeline."""
        logger.info("🚀 Rozpoczynam pełny pipeline WronAI...")

        pipeline_results = {}

        # 1. Zbieranie danych
        if not self.pipeline_state['data_collection']:
            logger.info("📥 Etap 1/4: Zbieranie danych")
            data_summary = self.run_data_collection(target_size_mb)
            if not data_summary:
                return None
            pipeline_results['data_collection'] = data_summary
        else:
            logger.info("⏭️ Etap 1/4: Dane już zebrane")

        # 2. Przetwarzanie danych
        if not self.pipeline_state['data_processing']:
            logger.info("⚙️ Etap 2/4: Przetwarzanie danych")
            processing_report = self.run_data_processing()
            if not processing_report:
                return None
            pipeline_results['data_processing'] = processing_report
        else:
            logger.info("⏭️ Etap 2/4: Dane już przetworzone")

        # 3. Trening modelu
        if not self.pipeline_state['model_training']:
            logger.info("🏋️ Etap 3/4: Trening modelu")
            training_report = self.run_model_training(model_name)
            if not training_report:
                return None
            pipeline_results['model_training'] = training_report
        else:
            logger.info("⏭️ Etap 3/4: Model już wytrenowany")

        # 4. Inferowanie
        logger.info("🤖 Etap 4/4: Testowanie modelu")
        inference_results = self.run_model_inference()
        if inference_results:
            pipeline_results['inference'] = {
                'evaluation': inference_results[1],
                'benchmark': inference_results[2]
            }

        # Zapisz finalny raport
        final_report = {
            'pipeline_completed': datetime.now().isoformat(),
            'workspace': str(self.base_dir),
            'results': pipeline_results,
            'state': self.pipeline_state
        }

        report_file = self.base_dir / "final_pipeline_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)

        logger.info("🎉 Pipeline WronAI zakończony pomyślnie!")
        logger.info(f"📁 Workspace: {self.base_dir}")
        logger.info(f"📊 Raport: {report_file}")

        return final_report

    def show_status(self):
        """Pokaż status pipeline'u."""
        print("\n🐦‍⬛ Status Pipeline WronAI")
        print("=" * 40)
        print(f"📁 Workspace: {self.base_dir}")
        print(f"📊 Stan:")

        status_icons = {True: "✅", False: "❌"}

        for step, completed in self.pipeline_state.items():
            if step != 'last_updated':
                icon = status_icons[completed]
                step_name = step.replace('_', ' ').title()
                print(f"  {icon} {step_name}")

        if 'last_updated' in self.pipeline_state:
            print(f"🕒 Ostatnia aktualizacja: {self.pipeline_state['last_updated']}")

        print()

    def clean_workspace(self):
        """Wyczyść workspace."""
        import shutil

        response = input(f"⚠️ Czy na pewno usunąć workspace {self.base_dir}? (y/N): ")
        if response.lower() == 'y':
            shutil.rmtree(self.base_dir)
            logger.info("🗑️ Workspace wyczyszczony")
        else:
            logger.info("Anulowano czyszczenie")


def create_requirements_file():
    """Utwórz plik requirements.txt."""
    requirements = """# WronAI - Wymagania systemowe

# Core ML libraries
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
tokenizers>=0.13.0

# Training optimizations
peft>=0.4.0
bitsandbytes>=0.41.0
accelerate>=0.20.0

# Data processing
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Web scraping & HTTP
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=4.9.0

# Text processing
ftfy>=6.1.0
regex>=2023.6.3
unidecode>=1.3.0

# Progress & CLI
tqdm>=4.65.0
rich>=13.0.0

# Interface
gradio>=3.40.0

# Logging & monitoring
tensorboard>=2.13.0
wandb>=0.15.0

# JSON handling
orjson>=3.9.0

# Compression
zstandard>=0.21.0

# Development
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0
"""

    with open("requirements.txt", 'w') as f:
        f.write(requirements)

    logger.info("📝 Utworzono requirements.txt")


def main():
    """Główna funkcja CLI."""
    parser = argparse.ArgumentParser(description="WronAI Master Pipeline")
    parser.add_argument("--workspace", default="./wronai_workspace", help="Ścieżka do workspace")
    parser.add_argument("--size", type=int, default=500, help="Rozmiar danych w MB")
    parser.add_argument("--model", default="microsoft/DialoGPT-medium", help="Model bazowy")

    subparsers = parser.add_subparsers(dest="command", help="Dostępne komendy")

    # Komendy
    subparsers.add_parser("setup", help="Setup środowiska")
    subparsers.add_parser("collect", help="Zbierz dane")
    subparsers.add_parser("process", help="Przetwórz dane")
    subparsers.add_parser("train", help="Trenuj model")
    subparsers.add_parser("infer", help="Uruchom inferowanie")
    subparsers.add_parser("full", help="Pełny pipeline")
    subparsers.add_parser("status", help="Status pipeline")
    subparsers.add_parser("clean", help="Wyczyść workspace")

    args = parser.parse_args()

    # Utwórz pipeline
    pipeline = WronAIMasterPipeline(args.workspace)

    print("🐦‍⬛ WronAI Master Pipeline")
    print("=" * 50)

    if args.command == "setup":
        create_requirements_file()
        print("🔧 Setup zakończony!")
        print("Następne kroki:")
        print("1. pip install -r requirements.txt")
        print("2. python wronai_master_pipeline.py full")

    elif args.command == "collect":
        if pipeline.check_dependencies():
            pipeline.run_data_collection(args.size)

    elif args.command == "process":
        if pipeline.check_dependencies():
            pipeline.run_data_processing()

    elif args.command == "train":
        if pipeline.check_dependencies():
            pipeline.run_model_training(args.model)

    elif args.command == "infer":
        if pipeline.check_dependencies():
            results = pipeline.run_model_inference()
            if results:
                inference, eval_results, benchmark = results

                print(f"\n📊 Wyniki ewaluacji:")
                success_rate = (eval_results['passed_tests'] / eval_results['total_tests']) * 100
                print(f"  - Testy: {success_rate:.1f}% zaliczonych")
                print(f"  - Wydajność: {benchmark['tokens_per_second']:.1f} tokenów/s")

                # Interaktywny tryb
                response = input("\nUruchomić interfejs Gradio? (y/N): ")
                if response.lower() == 'y':
                    interface = inference.create_gradio_interface()
                    interface.launch(share=True)

    elif args.command == "full":
        if pipeline.check_dependencies():
            pipeline.run_full_pipeline(args.size, args.model)

    elif args.command == "status":
        pipeline.show_status()

    elif args.command == "clean":
        pipeline.clean_workspace()

    else:
        print("Użycie:")
        print("  python wronai_master_pipeline.py setup    # Setup środowiska")
        print("  python wronai_master_pipeline.py full     # Pełny pipeline")
        print("  python wronai_master_pipeline.py status   # Status")
        print("  python wronai_master_pipeline.py infer    # Inferowanie")
        print()
        print("Opcje:")
        print("  --workspace DIR    # Katalog roboczy")
        print("  --size MB          # Rozmiar danych")
        print("  --model NAME       # Model bazowy")


if __name__ == "__main__":
    main()