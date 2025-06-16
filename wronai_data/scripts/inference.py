#!/usr/bin/env python3
"""
WronAI Inference & Evaluation Script
Uruchamianie wytrenowanego modelu i ewaluacja jego możliwości
"""

import json
import logging
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    GenerationConfig, StoppingCriteria, StoppingCriteriaList
)
from peft import PeftModel
import gradio as gr
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PolishStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria dla polskiego tekstu."""

    def __init__(self, tokenizer, stop_words: List[str] = None):
        self.tokenizer = tokenizer
        self.stop_words = stop_words or [".", "!", "?", "\n\n"]
        self.stop_token_ids = [tokenizer.encode(word, add_special_tokens=False) for word in self.stop_words]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Sprawdź ostatnie tokeny
        last_tokens = input_ids[0][-10:].tolist()

        for stop_ids in self.stop_token_ids:
            if len(stop_ids) > 0 and stop_ids[0] in last_tokens:
                return True

        return False


class WronAIInference:
    """Klasa do inferowania z modelu WronAI."""

    def __init__(self,
                 model_path: str = "./wronai_model",
                 device: str = "auto"):

        self.model_path = Path(model_path)

        # Sprawdź czy model istnieje
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model nie znaleziony w: {model_path}")

        # Konfiguracja urządzenia
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"🤖 Ładowanie modelu WronAI z: {self.model_path}")
        logger.info(f"📱 Urządzenie: {self.device}")

        # Załaduj model i tokenizer
        self.load_model_and_tokenizer()

        # Historia konwersacji
        self.conversation_history = []

        # Statystyki
        self.stats = {
            'generations_count': 0,
            'total_tokens_generated': 0,
            'avg_generation_time': 0.0
        }

    def load_model_and_tokenizer(self):
        """Załaduj model i tokenizer."""
        try:
            # Załaduj tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("✅ Tokenizer załadowany")

            # Sprawdź czy to model z LoRA
            adapter_config = self.model_path / "adapter_config.json"

            if adapter_config.exists():
                # Model z LoRA
                logger.info("🎯 Wykryto model LoRA")

                # Załaduj config aby znaleźć base model
                with open(adapter_config, 'r') as f:
                    config = json.load(f)
                base_model_name = config.get('base_model_name_or_path', 'microsoft/DialoGPT-medium')

                # Załaduj base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )

                # Załaduj LoRA weights
                self.model = PeftModel.from_pretrained(base_model, str(self.model_path))

            else:
                # Zwykły model
                logger.info("📦 Ładowanie standardowego modelu")
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )

            # Ustaw model w trybie ewaluacji
            self.model.eval()

            logger.info("✅ Model załadowany pomyślnie")

            # Wyświetl info o modelu
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"📊 Parametry modelu: {total_params:,}")

        except Exception as e:
            logger.error(f"❌ Błąd ładowania modelu: {e}")
            raise

    def generate_text(self,
                      prompt: str,
                      max_length: int = 200,
                      temperature: float = 0.8,
                      top_p: float = 0.9,
                      top_k: int = 50,
                      repetition_penalty: float = 1.1,
                      do_sample: bool = True,
                      num_return_sequences: int = 1) -> List[str]:
        """Wygeneruj tekst na podstawie prompt'u."""

        start_time = time.time()

        try:
            # Przygotuj input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = inputs.to("cuda")

            # Konfiguracja generacji
            generation_config = GenerationConfig(
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )

            # Stopping criteria
            stopping_criteria = StoppingCriteriaList([
                PolishStoppingCriteria(self.tokenizer)
            ])

            # Generuj
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    generation_config=generation_config,
                    stopping_criteria=stopping_criteria,
                    return_dict_in_generate=True,
                    output_scores=True
                )

            # Dekoduj wyniki
            generated_texts = []
            for sequence in outputs.sequences:
                # Usuń prompt z wyniku
                generated_tokens = sequence[inputs.shape[1]:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                generated_texts.append(generated_text.strip())

            # Aktualizuj statystyki
            generation_time = time.time() - start_time
            self.stats['generations_count'] += 1
            self.stats['total_tokens_generated'] += sum(len(text.split()) for text in generated_texts)
            self.stats['avg_generation_time'] = (
                    (self.stats['avg_generation_time'] * (self.stats['generations_count'] - 1) + generation_time)
                    / self.stats['generations_count']
            )

            logger.info(f"✍️ Wygenerowano tekst w {generation_time:.2f}s")

            return generated_texts

        except Exception as e:
            logger.error(f"❌ Błąd generowania: {e}")
            return ["Błąd generowania tekstu."]

    def chat(self, message: str, max_history: int = 5) -> str:
        """Konwersacja z modelem."""

        # Dodaj wiadomość do historii
        self.conversation_history.append(f"Użytkownik: {message}")

        # Ogranicz historię
        if len(self.conversation_history) > max_history * 2:
            self.conversation_history = self.conversation_history[-max_history * 2:]

        # Utwórz prompt z historią
        conversation_prompt = "\n".join(self.conversation_history)
        conversation_prompt += "\nAsystent:"

        # Wygeneruj odpowiedź
        responses = self.generate_text(
            conversation_prompt,
            max_length=150,
            temperature=0.7,
            top_p=0.9
        )

        response = responses[0] if responses else "Przepraszam, nie mogę odpowiedzieć."

        # Dodaj odpowiedź do historii
        self.conversation_history.append(f"Asystent: {response}")

        return response

    def evaluate_polish_capabilities(self):
        """Ewaluuj zdolności modelu w języku polskim."""
        logger.info("📊 Ewaluacja zdolności polskiego modelu...")

        test_prompts = [
            # Kompletowanie zdań
            {
                'category': 'completion',
                'prompt': 'Język polski należy do grupy',
                'expected_keywords': ['słowiańskich', 'zachodnich', 'indoeuropejskich']
            },
            {
                'category': 'completion',
                'prompt': 'Warszawa jest stolicą',
                'expected_keywords': ['Polski', 'Mazowsza', 'województwa']
            },

            # Wiedza ogólna
            {
                'category': 'knowledge',
                'prompt': 'Adam Mickiewicz to',
                'expected_keywords': ['poeta', 'pisarz', 'Romantyzm', 'Pan Tadeusz']
            },
            {
                'category': 'knowledge',
                'prompt': 'Wisła to',
                'expected_keywords': ['rzeka', 'Polska', 'najdłuższa', 'Kraków']
            },

            # Gramatyka
            {
                'category': 'grammar',
                'prompt': 'Koty łapią',
                'expected_keywords': ['myszy', 'myszki', 'ptaki']
            },
            {
                'category': 'grammar',
                'prompt': 'Dzieci bawią się',
                'expected_keywords': ['zabawkami', 'piłką', 'parku', 'domu']
            },

            # Kultura
            {
                'category': 'culture',
                'prompt': 'Polskie tradycyjne potrawy to',
                'expected_keywords': ['pierogi', 'bigos', 'kotlet', 'rosół']
            }
        ]

        results = {
            'total_tests': len(test_prompts),
            'passed_tests': 0,
            'results_by_category': {},
            'detailed_results': []
        }

        for test in test_prompts:
            logger.info(f"Test: {test['prompt']}")

            # Wygeneruj odpowiedź
            generated = self.generate_text(
                test['prompt'],
                max_length=100,
                temperature=0.7
            )[0]

            # Sprawdź czy zawiera oczekiwane słowa kluczowe
            generated_lower = generated.lower()
            matched_keywords = [
                keyword for keyword in test['expected_keywords']
                if keyword.lower() in generated_lower
            ]

            passed = len(matched_keywords) > 0
            if passed:
                results['passed_tests'] += 1

            # Aktualizuj wyniki per kategoria
            category = test['category']
            if category not in results['results_by_category']:
                results['results_by_category'][category] = {'total': 0, 'passed': 0}

            results['results_by_category'][category]['total'] += 1
            if passed:
                results['results_by_category'][category]['passed'] += 1

            # Szczegółowe wyniki
            test_result = {
                'prompt': test['prompt'],
                'generated': generated,
                'expected_keywords': test['expected_keywords'],
                'matched_keywords': matched_keywords,
                'passed': passed,
                'category': category
            }
            results['detailed_results'].append(test_result)

            logger.info(f"Wynik: {'✅ PASS' if passed else '❌ FAIL'}")
            logger.info(f"Wygenerowano: '{generated[:50]}...'")

        # Oblicz wyniki procentowe
        success_rate = (results['passed_tests'] / results['total_tests']) * 100

        logger.info(f"📊 Wyniki ewaluacji:")
        logger.info(f"  - Ogólnie: {results['passed_tests']}/{results['total_tests']} ({success_rate:.1f}%)")

        for category, cat_results in results['results_by_category'].items():
            cat_rate = (cat_results['passed'] / cat_results['total']) * 100
            logger.info(f"  - {category}: {cat_results['passed']}/{cat_results['total']} ({cat_rate:.1f}%)")

        return results

    def benchmark_performance(self, num_iterations: int = 10):
        """Benchmark wydajności modelu."""
        logger.info(f"⚡ Benchmark wydajności ({num_iterations} iteracji)...")

        test_prompts = [
            "Język polski",
            "Historia Polski to",
            "Warszawa jest",
            "Polscy pisarze",
            "Tradycyjne polskie"
        ]

        times = []
        token_counts = []

        for i in range(num_iterations):
            prompt = test_prompts[i % len(test_prompts)]

            start_time = time.time()
            generated = self.generate_text(prompt, max_length=100)[0]
            end_time = time.time()

            generation_time = end_time - start_time
            token_count = len(self.tokenizer.encode(generated))

            times.append(generation_time)
            token_counts.append(token_count)

            logger.info(f"Iteracja {i + 1}/{num_iterations}: {generation_time:.2f}s, {token_count} tokenów")

        # Oblicz statystyki
        avg_time = sum(times) / len(times)
        avg_tokens = sum(token_counts) / len(token_counts)
        tokens_per_second = avg_tokens / avg_time

        benchmark_results = {
            'num_iterations': num_iterations,
            'avg_generation_time': avg_time,
            'avg_tokens_generated': avg_tokens,
            'tokens_per_second': tokens_per_second,
            'min_time': min(times),
            'max_time': max(times),
            'device': self.device
        }

        logger.info(f"📊 Wyniki benchmark:")
        logger.info(f"  - Średni czas: {avg_time:.2f}s")
        logger.info(f"  - Średnia tokenów: {avg_tokens:.1f}")
        logger.info(f"  - Tokenów na sekundę: {tokens_per_second:.1f}")

        return benchmark_results