#!/usr/bin/env python3
"""
WronAI Model Training Script
Trening polskiego modelu językowego z wykorzystaniem QLoRA i optymalizacji
"""

import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig, TaskType, get_peft_model,
    prepare_model_for_kbit_training
)
import bitsandbytes as bnb
from accelerate import Accelerator
import wandb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WronAITrainer:
    """Klasa do treningu modelu WronAI."""

    def __init__(self,
                 model_name: str = "microsoft/DialoGPT-medium",
                 data_dir: str = "./wronai_processed",
                 output_dir: str = "./wronai_model",
                 use_lora: bool = True,
                 use_4bit: bool = True):

        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.use_lora = use_lora
        self.use_4bit = use_4bit

        # Sprawdź dostępność GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            logger.info(f"🚀 Używam GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"💾 Pamięć GPU: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")
        else:
            logger.warning("⚠️ GPU niedostępne, używam CPU (będzie wolno!)")

        # Inicjalizuj accelerator
        self.accelerator = Accelerator()

        # Statystyki treningu
        self.training_stats = {}

    def load_datasets(self):
        """Załaduj przetworzone datasety."""
        logger.info("📚 Ładowanie datasetów...")

        try:
            train_dataset = load_from_disk(str(self.data_dir / "train_dataset"))
            val_dataset = load_from_disk(str(self.data_dir / "validation_dataset"))
            test_dataset = load_from_disk(str(self.data_dir / "test_dataset"))

            logger.info(f"✅ Datasety załadowane:")
            logger.info(f"  - Train: {len(train_dataset)} próbek")
            logger.info(f"  - Validation: {len(val_dataset)} próbek")
            logger.info(f"  - Test: {len(test_dataset)} próbek")

            return {
                'train': train_dataset,
                'validation': val_dataset,
                'test': test_dataset
            }

        except Exception as e:
            logger.error(f"❌ Błąd ładowania datasetów: {e}")
            raise

    def setup_tokenizer(self):
        """Skonfiguruj tokenizer."""
        logger.info("🔤 Konfigurowanie tokenizera...")

        try:
            # Załaduj tokenizer z zapisanych danych lub domyślny
            tokenizer_path = self.data_dir / "tokenizer"
            if tokenizer_path.exists():
                self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
                logger.info(f"Załadowano tokenizer z {tokenizer_path}")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                logger.info(f"Załadowano domyślny tokenizer: {self.model_name}")

            # Dodaj padding token jeśli nie ma
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Ustawiono pad_token = eos_token")

            # Sprawdź specjalne tokeny
            logger.info(f"Specjalne tokeny:")
            logger.info(f"  - PAD: {self.tokenizer.pad_token}")
            logger.info(f"  - EOS: {self.tokenizer.eos_token}")
            logger.info(f"  - BOS: {self.tokenizer.bos_token}")
            logger.info(f"  - UNK: {self.tokenizer.unk_token}")

            return self.tokenizer

        except Exception as e:
            logger.error(f"❌ Błąd konfiguracji tokenizera: {e}")
            raise

    def setup_model(self):
        """Skonfiguruj model z opcjonalną kwantyzacją i LoRA."""
        logger.info(f"🤖 Ładowanie modelu: {self.model_name}")

        try:
            # Konfiguracja kwantyzacji 4-bit
            if self.use_4bit and self.device == "cuda":
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )

                logger.info("🔢 Używam kwantyzacji 4-bit")

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )

            # Resize embeddings dla tokenizera
            self.model.resize_token_embeddings(len(self.tokenizer))

            # Przygotuj model do treningu
            if self.use_4bit:
                self.model = prepare_model_for_kbit_training(self.model)

            # Konfiguracja LoRA
            if self.use_lora:
                logger.info("🎯 Konfigurowanie LoRA...")

                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=16,  # Rank - większy = więcej parametrów
                    lora_alpha=32,  # Scaling parameter
                    lora_dropout=0.1,  # Dropout dla regularizacji
                    target_modules=[  # Moduły do adaptacji
                        "q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"
                    ],
                    bias="none",
                    inference_mode=False
                )

                self.model = get_peft_model(self.model, lora_config)

                # Wyświetl info o parametrach
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self.model.parameters())

                logger.info(f"📊 Parametry modelu:")
                logger.info(f"  - Trenowalne: {trainable_params:,}")
                logger.info(f"  - Całkowite: {total_params:,}")
                logger.info(f"  - Procent trenowalnych: {100 * trainable_params / total_params:.2f}%")

            return self.model

        except Exception as e:
            logger.error(f"❌ Błąd ładowania modelu: {e}")
            raise

    def setup_training_arguments(self, num_train_samples: int) -> TrainingArguments:
        """Skonfiguruj argumenty treningu."""
        logger.info("⚙️ Konfigurowanie parametrów treningu...")

        # Oblicz kroki
        batch_size = 4 if self.device == "cuda" else 1
        gradient_accumulation_steps = 8 if self.device == "cuda" else 16

        steps_per_epoch = num_train_samples // (batch_size * gradient_accumulation_steps)
        max_steps = steps_per_epoch * 3  # 3 epoki

        eval_steps = max(steps_per_epoch // 4, 50)  # Ewaluacja 4 razy na epokę
        save_steps = max(steps_per_epoch // 2, 100)  # Zapis 2 razy na epokę

        training_args = TrainingArguments(
            # Podstawowe
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,

            # Trening
            num_train_epochs=3,
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,

            # Optymalizacja
            learning_rate=2e-4,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,

            # Scheduler
            lr_scheduler_type="cosine",
            warmup_steps=100,

            # Ewaluacja i zapis
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            # Logowanie
            logging_steps=50,
            logging_dir=str(self.output_dir / "logs"),
            report_to=["tensorboard"],

            # Optymalizacje GPU
            fp16=True if self.device == "cuda" else False,
            dataloader_pin_memory=True,
            dataloader_num_workers=4 if self.device == "cuda" else 0,

            # Inne
            seed=42,
            data_seed=42,
            remove_unused_columns=False,
            label_names=["labels"]
        )

        logger.info(f"📊 Parametry treningu:")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Gradient accumulation: {gradient_accumulation_steps}")
        logger.info(f"  - Effective batch size: {batch_size * gradient_accumulation_steps}")
        logger.info(f"  - Max steps: {max_steps}")
        logger.info(f"  - Eval steps: {eval_steps}")
        logger.info(f"  - Learning rate: {training_args.learning_rate}")

        return training_args

    def setup_data_collator(self):
        """Skonfiguruj data collator."""
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Nie używamy masked language modeling
            pad_to_multiple_of=8 if self.device == "cuda" else None
        )

        logger.info("✅ Data collator skonfigurowany")
        return data_collator

    def compute_metrics(self, eval_pred):
        """Oblicz metryki dla ewaluacji."""
        predictions, labels = eval_pred

        # Oblicz perplexity
        shift_labels = labels[..., 1:].contiguous()
        shift_logits = predictions[..., :-1, :].contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        perplexity = torch.exp(loss)

        return {
            "perplexity": perplexity.item()
        }

    def setup_callbacks(self):
        """Skonfiguruj callback'i."""
        callbacks = []

        # Early stopping
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.01
        )
        callbacks.append(early_stopping)

        logger.info("✅ Callbacks skonfigurowane")
        return callbacks

    def train_model(self, datasets: Dict[str, Any]):
        """Główna funkcja treningu."""
        logger.info("🏋️ Rozpoczynam trening modelu WronAI...")

        try:
            # Setup wszystkich komponentów
            self.setup_tokenizer()
            self.setup_model()

            # Training arguments
            training_args = self.setup_training_arguments(len(datasets['train']))

            # Data collator
            data_collator = self.setup_data_collator()

            # Callbacks
            callbacks = self.setup_callbacks()

            # Utwórz trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=datasets['train'],
                eval_dataset=datasets['validation'],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
                callbacks=callbacks
            )

            # Wyświetl info przed treningiem
            logger.info(f"🚀 Rozpoczynam trening:")
            logger.info(f"  - Model: {self.model_name}")
            logger.info(f"  - LoRA: {'✅' if self.use_lora else '❌'}")
            logger.info(f"  - 4-bit: {'✅' if self.use_4bit else '❌'}")
            logger.info(f"  - Device: {self.device}")

            # Zapisz stan przed treningiem
            self.save_training_config(training_args, datasets)

            # TRENING!
            train_result = trainer.train()

            # Zapisz model
            trainer.save_model()
            trainer.save_state()

            # Zapisz tokenizer
            self.tokenizer.save_pretrained(str(self.output_dir))

            # Zapisz statystyki
            self.training_stats = {
                'train_runtime': train_result.metrics['train_runtime'],
                'train_samples_per_second': train_result.metrics['train_samples_per_second'],
                'train_steps_per_second': train_result.metrics['train_steps_per_second'],
                'total_flos': train_result.metrics['total_flos'],
                'train_loss': train_result.metrics['train_loss'],
                'epoch': train_result.metrics['epoch']
            }

            logger.info("✅ Trening zakończony pomyślnie!")
            logger.info(f"📊 Statystyki treningu:")
            logger.info(f"  - Czas treningu: {train_result.metrics['train_runtime']:.1f}s")
            logger.info(f"  - Próbek/s: {train_result.metrics['train_samples_per_second']:.2f}")
            logger.info(f"  - Final loss: {train_result.metrics['train_loss']:.4f}")

            return trainer, train_result

        except Exception as e:
            logger.error(f"❌ Błąd podczas treningu: {e}")
            raise

    def evaluate_model(self, trainer, test_dataset):
        """Ewaluuj wytrenowany model."""
        logger.info("📊 Ewaluacja modelu na test set...")

        try:
            # Ewaluacja na test set
            eval_results = trainer.evaluate(test_dataset)

            logger.info("✅ Ewaluacja zakończona:")
            logger.info(f"  - Test Loss: {eval_results['eval_loss']:.4f}")
            logger.info(f"  - Test Perplexity: {eval_results['eval_perplexity']:.2f}")

            return eval_results

        except Exception as e:
            logger.error(f"❌ Błąd ewaluacji: {e}")
            return {}

    def generate_sample_text(self, prompt: str = "Język polski", max_length: int = 200):
        """Wygeneruj przykładowy tekst."""
        logger.info(f"✍️ Generowanie tekstu dla: '{prompt}'")

        try:
            # Przygotuj input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = inputs.to("cuda")

            # Generuj
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Dekoduj
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            logger.info(f"🎯 Wygenerowany tekst:")
            logger.info(f"'{generated_text}'")

            return generated_text

        except Exception as e:
            logger.error(f"❌ Błąd generowania: {e}")
            return ""

    def save_training_config(self, training_args, datasets):
        """Zapisz konfigurację treningu."""
        config = {
            'model_name': self.model_name,
            'data_dir': str(self.data_dir),
            'output_dir': str(self.output_dir),
            'use_lora': self.use_lora,
            'use_4bit': self.use_4bit,
            'device': self.device,
            'training_args': training_args.to_dict(),
            'dataset_sizes': {
                'train': len(datasets['train']),
                'validation': len(datasets['validation']),
                'test': len(datasets['test'])
            },
            'training_date': datetime.now().isoformat()
        }

        config_file = self.output_dir / "training_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 Konfiguracja zapisana: {config_file}")

    def save_final_report(self, eval_results, generated_samples):
        """Zapisz finalny raport z treningu."""
        report = {
            'training_completed': datetime.now().isoformat(),
            'model_name': self.model_name,
            'training_stats': self.training_stats,
            'evaluation_results': eval_results,
            'generated_samples': generated_samples,
            'model_location': str(self.output_dir),
            'files_created': [
                'pytorch_model.bin',
                'config.json',
                'tokenizer.json',
                'training_config.json',
                'training_args.bin'
            ]
        }

        report_file = self.output_dir / "training_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return report

    def run_full_training_pipeline(self):
        """Uruchom kompletny pipeline treningu."""
        logger.info("🐦‍⬛ Rozpoczynam pipeline treningu WronAI...")

        try:
            # 1. Załaduj datasety
            datasets = self.load_datasets()

            # 2. Trenuj model
            trainer, train_result = self.train_model(datasets)

            # 3. Ewaluuj model
            eval_results = self.evaluate_model(trainer, datasets['test'])

            # 4. Wygeneruj przykłady
            sample_prompts = [
                "Język polski",
                "Historia Polski",
                "Warszawa to",
                "Literatura polska"
            ]

            generated_samples = {}
            for prompt in sample_prompts:
                generated_samples[prompt] = self.generate_sample_text(prompt)

            # 5. Zapisz finalny raport
            report = self.save_final_report(eval_results, generated_samples)

            logger.info("🎉 Pipeline treningu zakończony!")
            logger.info(f"📁 Model zapisany w: {self.output_dir}")

            return report

        except Exception as e:
            logger.error(f"❌ Błąd pipeline'u: {e}")
            raise


def main():
    """Główna funkcja."""
    print("🐦‍⬛ WronAI Model Training")
    print("=" * 50)

    # Sprawdź wymagania
    if not torch.cuda.is_available():
        print("⚠️ UWAGA: GPU niedostępne. Trening będzie bardzo wolny!")
        response = input("Kontynuować? (y/N): ")
        if response.lower() != 'y':
            return

    try:
        # Konfiguracja treningu
        trainer = WronAITrainer(
            model_name="microsoft/DialoGPT-medium",  # Dobry starter model
            data_dir="./wronai_processed",
            output_dir="./wronai_model",
            use_lora=True,  # LoRA dla efektywności
            use_4bit=True  # 4-bit dla oszczędności pamięci
        )

        # Uruchom trening
        report = trainer.run_full_training_pipeline()

        print("\n🎉 TRENING ZAKOŃCZONY!")
        print(f"📁 Model w: {trainer.output_dir}")
        print(f"📊 Train loss: {report['training_stats']['train_loss']:.4f}")
        print(f"📊 Test loss: {report['evaluation_results'].get('eval_loss', 'N/A')}")

        print("\n🎯 Przykłady generacji:")
        for prompt, generated in report['generated_samples'].items():
            print(f"Prompt: '{prompt}'")
            print(f"Output: '{generated[:100]}...'")
            print()

        print("Następne kroki:")
        print("1. Sprawdź wygenerowane przykłady")
        print("2. Dostrajaj hyperparametry jeśli potrzeba")
        print("3. Użyj modelu do inferowania")

    except KeyboardInterrupt:
        print("\n⏹️ Trening przerwany przez użytkownika")
    except Exception as e:
        print(f"\n❌ Błąd: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()