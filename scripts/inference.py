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
    def __init__(self, model_path: str, quantize: bool = True, base_model: Optional[str] = None):
        """Initialize WronAI inference.

        Args:
            model_path (str): Path to model directory or huggingface model name
            quantize (bool): Use 4-bit quantization for faster inference with less memory
            base_model (Optional[str]): Override base model to use with LoRA checkpoint
        """
        self.model_path = model_path
        self.quantize = quantize
        self.base_model_override = base_model
        self.tokenizer = None
        self.model = None
        self.pipeline = None

        # Step 1: Load tokenizer and model
        self.base_model_name, self.lora_config = self.detect_model_config()
        self.load_tokenizer()
        self.load_model()
        
        # Step 2: Initialize pipeline
        self.init_pipeline()

        logger.info("Model loaded successfully!")

    def detect_model_config(self):
        """Detect model configuration from checkpoint path or use defaults."""
        # Default models
        base_model_name = "facebook/opt-1.3b"
        fallback_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        lora_config = None
        
        # If user specified a base model, use that
        if self.base_model_override:
            logger.info(f"Using user-specified base model: {self.base_model_override}")
            base_model_name = self.base_model_override
        
        # Check if path contains a LoRA checkpoint
        model_path = Path(self.model_path)
        if (model_path / "adapter_config.json").exists():
            try:
                # Try to load adapter_config to get base model information
                import json
                adapter_config_path = model_path / "adapter_config.json"
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                
                # Store the complete config
                lora_config = adapter_config
                
                # Extract base model name from config
                base_model_from_config = adapter_config.get("base_model_name_or_path")
                if base_model_from_config and not self.base_model_override:
                    logger.info(f"LoRA was trained on base model: {base_model_from_config}")
                    # Update the base model name if we found it in the config
                    base_model_name = base_model_from_config
            except Exception as e:
                logger.warning(f"Could not read adapter_config.json: {e}")
        
        return base_model_name, lora_config
        
    def load_tokenizer(self):
        """Load tokenizer for inference."""
        logger.info(f"Loading tokenizer from: {self.model_path}")
        
        # We'll use the base_model_name detected in __init__ and fallback if needed
        fallback_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        # Check if we should use the local model directory
        model_path = Path(self.model_path)
        
        # If this is a LoRA checkpoint, look for the tokenizer there
        if (model_path / "adapter_config.json").exists():
            logger.info(f"Found LoRA checkpoint at: {self.model_path}")
            try:
                # We already detected the config in __init__, just load tokenizer
                
                # Try to load tokenizer from checkpoint or base model
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_path,
                        use_fast=True
                    )
                    logger.info("Tokenizer loaded from LoRA checkpoint directory")
                    return
                except Exception as e:
                    logger.warning(f"Could not load tokenizer from checkpoint: {e}")
                    logger.info("Trying to load tokenizer from base model...")
                    
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            self.base_model_name,
                            use_fast=True
                        )
                        logger.info(f"Tokenizer loaded from base model: {self.base_model_name}")
                        return
                    except Exception as e2:
                        logger.warning(f"Could not load tokenizer from base model {self.base_model_name}: {e2}")
            except Exception as e:
                logger.warning(f"Error processing LoRA checkpoint: {e}")
        
        # Check for a full model (not LoRA)
        elif model_path.exists() and (model_path / "config.json").exists():
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
        
        # Try to load the tokenizer from the base model
        try:
            logger.info(f"Attempting to load tokenizer from: {base_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
            logger.info("Tokenizer loaded successfully")
            return
        except Exception as e:
            logger.warning(f"Could not load tokenizer from {self.base_model_name}: {e}")
        
        # Finally, try the fallback model
        try:
            logger.info(f" Falling back to {fallback_model_name}. Note that LoRA adapters may not be compatible.")
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model_name, use_fast=True)
            logger.info("Tokenizer loaded from fallback model")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

    def load_model(self):
        """Load model for inference."""
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
            
        # Fallback model if needed
        fallback_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        # Path to model or checkpoint
        model_path = Path(self.model_path)
        
        # Load the model
        try:
            # First check if the path is to a complete model (not LoRA)
            if model_path.exists() and (model_path / "config.json").exists() and not (model_path / "adapter_config.json").exists():
                logger.info(f"Loading full model from local directory: {self.model_path}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=quantization_config,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
                logger.info("Model loaded from local directory")
                return
        except Exception as e:
            logger.warning(f"Could not load from local directory: {e}")
            
        # Try to load from base model or use fallback
        try:
            # Try to load base model
            logger.info(f"Loading model: {self.base_model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Base model loaded successfully")
        except Exception as e:
            logger.error(f" Error: Could not load base model {self.base_model_name}. {e}")
            logger.info(f"Loading fallback model: {fallback_model_name}")
            
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    fallback_model_name,
                    quantization_config=quantization_config,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
                logger.info("Fallback model loaded successfully")
            except Exception as e2:
                logger.error(f"Could not load fallback model: {e2}")
                raise
                
        # Check if this is a LoRA checkpoint
        if (Path(self.model_path) / "adapter_config.json").exists():
            try:
                logger.info(f"Found LoRA checkpoint at: {self.model_path}")
                if hasattr(self, 'model'):
                    logger.info("Applying LoRA adapters to the base model...")
                    try:
                        # Try first without strict loading to handle vocab size mismatches
                        try:
                            self.model = PeftModel.from_pretrained(
                                self.model, self.model_path, 
                                is_trainable=False,
                                strict=False
                            )
                            logger.info("LoRA adapters loaded with non-strict matching (handling vocab size differences)")
                        except Exception as e:
                            logger.warning(f"Non-strict LoRA loading failed: {e}")
                            # Fall back to standard loading
                            self.model = PeftModel.from_pretrained(
                                self.model, self.model_path, is_trainable=False
                            )
                            logger.info("LoRA adapters loaded with strict matching")
                    except Exception as e:
                        logger.error(f"Failed to load LoRA adapters: {e}")
                        logger.info("Continuing without LoRA adapters")
                else:
                    logger.error("Base model not loaded. Cannot apply LoRA adapters.")
            except Exception as e:
                logger.error(f"Failed to load LoRA config/adapters: {e}")
                logger.info("Continuing without LoRA adapters")
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
            logger.info(f"Attempting to load tokenizer from: {self.base_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                use_fast=True
            )
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load tokenizer from {self.base_model_name}: {e}")
            if "authentication" in str(e).lower():
                logger.error("\n" + "="*80)
                logger.error("AUTHENTICATION REQUIRED")
                logger.error("To use Llama-2 models, you need to authenticate with Hugging Face:")
                logger.error("1. Get your access token from https://huggingface.co/settings/tokens")
                logger.error("2. Run: huggingface-cli login")
                logger.error("3. Enter your access token when prompted")
                logger.error("="*80 + "\n")
            
            # Fallback to TinyLlama
            logger.info(f" Falling back to {fallback_model_name}. Note that LoRA adapters may not be compatible.")
            self.tokenizer = AutoTokenizer.from_pretrained(
                fallback_model_name,
                use_fast=True
            )
            logger.info("Tokenizer loaded from fallback model")
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the model
        try:
            # First check if the path is to a complete model (not LoRA)
            if model_path.exists() and (model_path / "config.json").exists() and not (model_path / "adapter_config.json").exists():
                logger.info(f"Loading full model from local directory: {self.model_path}")
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
            logger.info(f"Loading model: {self.base_model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Base model loaded successfully")
            
        except Exception as e:
            # If base model fails, try fallback model
            logger.error(f" Error: Could not load base model {self.base_model_name}. {e}")
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
                
                # Try to load adapter_config to get base model information
                import json
                adapter_config_path = Path(self.model_path) / "adapter_config.json"
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                
                # Extract base model name from config
                base_model_from_config = adapter_config.get("base_model_name_or_path")
                if base_model_from_config:
                    logger.info(f"LoRA was trained on base model: {base_model_from_config}")
                    # Update the base model name if we found it in the config
                    base_model_name = base_model_from_config
                
                # Load tokenizer from LoRA checkpoint rather than base model
                # This is important if the tokenizer was customized during training
                if hasattr(self, 'model'):
                    logger.info("Applying LoRA adapters to the base model...")
                    try:
                        # First try with strict=True
                        self.model = PeftModel.from_pretrained(
                            self.model, self.model_path, is_trainable=False
                        )
                        logger.info("LoRA adapters loaded and applied successfully")
                    except Exception as e:
                        logger.warning(f"This model requires authentication. Please login with `huggingface-cli login` {e}")
                        logger.info("Continuing without LoRA adapters")
                else:
                    logger.error("Base model not loaded. Cannot apply LoRA adapters.")
            except Exception as e:
                logger.error(f"Failed to load LoRA config/adapters: {e}")
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

    def init_pipeline(self):
        """Initialize the text generation pipeline."""
        if not hasattr(self, 'model') or self.model is None:
            logger.error("No model loaded, cannot initialize pipeline")
            return

        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            logger.error("No tokenizer loaded, cannot initialize pipeline")
            return

        logger.info("Initializing pipeline...")
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            logger.info("Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise

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
    """Run inference with WronAI model."""
    parser = argparse.ArgumentParser(description="WronAI Inference Script")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to model directory or huggingface model name"
    )
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--base-model", type=str, default=None,
        help="Override base model to use with LoRA checkpoint"
    )
    parser.add_argument(
        "--max-length", type=int, default=512,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Generation temperature (higher = more creative, lower = more focused)"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9,
        help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--no-quantize", action="store_true", default=False,
        help="Disable 4-bit quantization"
    )
    parser.add_argument(
        "--chat", action="store_true", default=False,
        help="Enter interactive chat mode"
    )

    args = parser.parse_args()

    try:
        inference = WronAIInference(
            model_path=args.model, 
            quantize=not args.no_quantize,
            base_model=args.base_model
        )
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        sys.exit(1)

    # Check if model path exists
    if not Path(args.model).exists() and not args.model.startswith(('http://', 'https://', 'huggingface.co')):
        logger.error(f"Model path not found: {args.model}")
        sys.exit(1)
    
    # Generate response or start chat mode
    try:
        if args.chat:
            inference.chat()
        elif args.prompt:
            print(f"Prompt: {args.prompt}")
            response = inference.generate(
                args.prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p
            )
            print(f"Response: {response}")
        else:
            logger.info("No prompt provided, starting chat mode...")
            inference.chat()
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        sys.exit(1)
    #else:
    #    print("Please provide either --prompt for single generation or --chat for interactive mode")
    #    sys.exit(1)


if __name__ == "__main__":
    main()