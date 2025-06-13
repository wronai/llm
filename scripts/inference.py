#!/usr/bin/env python3
"""
WronAI Inference Script
Generate text using trained WronAI model

Note: To use Llama-2 models, you need to authenticate with Hugging Face:
1. Get your access token from https://huggingface.co/settings/tokens
2. Run: huggingface-cli login
3. Enter your access token when prompted
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    logging as hf_logging
)

# Reduce verbosity from the transformers library
hf_logging.set_verbosity_error()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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

        # Model configuration
        # Using OPT-1.3B to match the LoRA checkpoint
        base_model_name = "facebook/opt-1.3b"
        # Fallback to a smaller model if needed
        fallback_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        # Check if we should use the local model directory
        model_path = Path(self.model_path)
        if model_path.exists() and (model_path / "config.json").exists():
            logger.info(f"Found local model at: {self.model_path}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    use_fast=True
                )
                logger.info("Tokenizer loaded from local model directory")
                return
            except Exception as e:
                logger.warning(f"Could not load tokenizer from local model: {e}")
                logger.info("Falling back to base model...")
        
        # Try to load from base model with authentication
        try:
            logger.info(f"Attempting to load tokenizer from: {base_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                use_fast=True
            )
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load tokenizer from {base_model_name}: {e}")
            if "authentication" in str(e).lower():
                logger.error("\n" + "="*80)
                logger.error("AUTHENTICATION REQUIRED")
                logger.error("To use Llama-2 models, you need to authenticate with Hugging Face:")
                logger.error("1. Get your access token from https://huggingface.co/settings/tokens")
                logger.error("2. Run: huggingface-cli login")
                logger.error("3. Enter your access token when prompted")
                logger.error("="*80 + "\n")
            
            # Fallback to TinyLlama
            logger.info(f"Falling back to: {fallback_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                fallback_model_name,
                use_fast=True
            )
            logger.info("Tokenizer loaded from fallback model")
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the model
        try:
            # First try to load from local directory if it exists
            if model_path.exists() and (model_path / "config.json").exists():
                logger.info(f"Loading model from local directory: {self.model_path}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=quantization_config,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
                logger.info("Model loaded from local directory")
                return
                
            # If no local model, try loading base model
            logger.info(f"Loading model: {base_model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Base model loaded successfully")
            
        except Exception as e:
            # If base model fails, try fallback model
            logger.warning(f"Could not load base model {base_model_name}: {e}")
            logger.info(f"Loading fallback model: {fallback_model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                fallback_model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Fallback model loaded successfully")

        # Check if this is a LoRA checkpoint
        if (Path(self.model_path) / "adapter_config.json").exists():
            try:
                logger.info(f"Found LoRA checkpoint at: {self.model_path}")
                if hasattr(self, 'model'):
                    logger.info("Applying LoRA adapters to the base model...")
                    self.model = PeftModel.from_pretrained(self.model, self.model_path)
                    logger.info("LoRA adapters loaded and applied successfully")
                else:
                    logger.error("Base model not loaded. Cannot apply LoRA adapters.")
            except Exception as e:
                logger.error(f"Failed to load LoRA adapters: {e}")
                logger.info("Continuing without LoRA adapters")
        else:
            # Try loading as a full model
            try:
                logger.info(f"Attempting to load as full model from: {self.model_path}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=quantization_config,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
                logger.info("Full model loaded successfully")
            except Exception as e2:
                logger.warning(f"Could not load full model: {e2}")
                logger.info("Continuing with base model only")

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
        """Format prompt for the model."""
        if system_prompt is None:
            system_prompt = (
                "Jesteś pomocnym asystentem AI specjalizującym się w języku polskim. "
                "Odpowiadaj szczegółowo i wyczerpująco na zadane pytania. "
                "Jeśli pytanie dotyczy Polski, uwzględnij kontekst historyczny, kulturalny i społeczny."
            )

        # OPT uses a simple text format without special tokens
        prompt = f"{system_prompt}\n\nPytanie: {instruction}\nOdpowiedź: "
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
            max_length=1024,  # Increased from 512 for more detailed responses
            temperature=0.3,   # Lower temperature for more focused responses
            top_p=0.95,       # Slightly higher for better diversity
            top_k=40,         # Slightly more focused than 50
            repetition_penalty=1.2,  # Increased to reduce repetition
            do_sample=True,
            num_beams=1,     # For more creative responses
            no_repeat_ngram_size=3,  # Prevent repeating n-grams
            length_penalty=1.0,  # Slightly encourage longer responses
            early_stopping=True,
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