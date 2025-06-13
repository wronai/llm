#!/usr/bin/env python3
"""
WronAI Inference Script
Generate text using trained WronAI model
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WronAIInference:
    def __init__(self, model_path: str, quantize: bool = True):
        """Initialize inference engine."""
        self.model_path = model_path
        self.quantize = quantize
        self.tokenizer = None
        self.model = None
        self.pipeline = None

        self.load_model()

    def load_model(self):
        """Load model and tokenizer for inference."""
        logger.info(f"Loading model from: {self.model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config for inference
        if self.quantize:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        else:
            quantization_config = None

        # Load base model
        base_model_name = "mistralai/Mistral-7B-v0.1"  # Should be saved in config

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        # Load LoRA adapters if they exist
        try:
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
            logger.info("LoRA adapters loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load LoRA adapters: {e}")
            # Try loading as full model
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=quantization_config,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
                logger.info("Full model loaded successfully")
            except Exception as e2:
                logger.error(f"Could not load model: {e2}")
                sys.exit(1)

        # Create text generation pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        logger.info("Model loaded successfully!")

    def format_prompt(self, instruction: str, system_prompt: Optional[str] = None) -> str:
        """Format prompt for WronAI model."""
        if system_prompt is None:
            system_prompt = "Jesteś pomocnym asystentem AI specjalizującym się w języku polskim."

        prompt = f"<polish><question>{instruction}</question><answer>"
        return prompt

    def generate(
            self,
            prompt: str,
            max_length: int = 512,
            temperature: float = 0.7,
            top_p: float = 0.9,
            top_k: int = 50,
            repetition_penalty: float = 1.1,
            do_sample: bool = True,
            system_prompt: Optional[str] = None
    ) -> str:
        """Generate text response."""

        # Format prompt
        formatted_prompt = self.format_prompt(prompt, system_prompt)

        # Generate
        outputs = self.pipeline(
            formatted_prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_full_text=False
        )

        # Extract generated text
        generated_text = outputs[0]['generated_text']

        # Clean up output
        if '</answer>' in generated_text:
            generated_text = generated_text.split('</answer>')[0]

        return generated_text.strip()

    def chat(self):
        """Interactive chat mode."""
        print("🐦‍⬛ WronAI - Polski Asystent AI")
        print("Wpisz 'quit' aby zakończyć, 'clear' aby wyczyścić historię")
        print("-" * 50)

        conversation_history = []

        while True:
            try:
                user_input = input("\n👤 Ty: ").strip()

                if user_input.lower() == 'clear':
                    conversation_history = []
                    print("Historia rozmowy wyczyszczona.")
                    continue

                if not user_input:
                    continue

                # Add context from conversation history
                context = ""
                if conversation_history:
                    context = "\n".join([f"Użytkownik: {h['user']}\nAsystent: {h['assistant']}"
                                         for h in conversation_history[-3:]])  # Last 3 exchanges
                    context += f"\nUżytkownik: {user_input}\nAsystent:"
                else:
                    context = user_input

                print("🤖 WronAI myśli...")

                # Generate response
                response = self.generate(
                    context,
                    max_length=256,
                    temperature=0.7,
                    top_p=0.9
                )

                print(f"🐦‍⬛ WronAI: {response}")

                # Add to conversation history
                conversation_history.append({
                    'user': user_input,
                    'assistant': response
                })

                # Keep only last 10 exchanges to manage memory
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-10:]

            except KeyboardInterrupt:
                print("\n\nRozmowa przerwana. Dziękuję! 👋")
                break
            except Exception as e:
                print(f"❌ Błąd: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(description="WronAI Inference Script")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained WronAI model"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt for generation"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable quantization (requires more VRAM)"
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start interactive chat mode"
    )

    args = parser.parse_args()

    # Check if model path exists
    if not Path(args.model).exists():
        logger.error(f"Model path not found: {args.model}")
        sys.exit(1)

    # Initialize inference engine
    inference = WronAIInference(
        model_path=args.model,
        quantize=not args.no_quantize
    )

    if args.chat:
        # Interactive chat mode
        inference.chat()
    elif args.prompt:
        # Single prompt mode
        response = inference.generate(
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p
        )
        print(f"Prompt: {args.prompt}")
        print(f"Response: {response}")
    else:
        print("Please provide either --prompt for single generation or --chat for interactive mode")
        sys.exit(1)


if __name__ == "__main__":
    main()
    input.lower() in ['quit', 'exit', 'q']:
    print("Dziękuję za rozmowę! 👋")
    break

if user_