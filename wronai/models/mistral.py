"""
Mistral model implementation for WronAI.
"""

import re
from typing import Optional, Dict, Any

from transformers import MistralForCausalLM
from .base import WronAIModel, ModelConfig
from ..utils.logging import get_logger
from ..data.polish import normalize_polish_text, POLISH_DIACRITICS_MAP

logger = get_logger(__name__)


class WronAIMistral(WronAIModel):
    """
    WronAI implementation based on Mistral architecture.

    Optimized for Polish language with QLoRA fine-tuning support.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model_type = "mistral"

    def load_model(self):
        """Load Mistral model with WronAI optimizations."""
        logger.info(f"Loading Mistral model: {self.config.model_name}")

        # Configure quantization if enabled
        quantization_config = None
        if self.config.quantization_enabled:
            from transformers import BitsAndBytesConfig
            import torch

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type=self.config.quantization_type,
                bnb_4bit_use_double_quant=True,
            )
            self._is_quantized = True

        # Load model
        self.model = MistralForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code,
        )

        # Enable gradient checkpointing if requested
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Prepare for LoRA if enabled
        if self.config.lora_enabled:
            self._setup_lora()

        logger.info(f"Mistral model loaded. Parameters: {self.get_parameter_count():,}")

    def _setup_lora(self):
        """Setup LoRA adapters for efficient fine-tuning."""
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            # Prepare model for k-bit training if quantized
            if self._is_quantized:
                self.model = prepare_model_for_kbit_training(self.model)

            # LoRA configuration
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=self.config.lora_target_modules,
            )

            # Apply LoRA
            self.model = get_peft_model(self.model, lora_config)
            self._has_lora = True

            logger.info("LoRA adapters configured successfully")

        except ImportError:
            logger.warning("PEFT not available. LoRA disabled.")
            self.config.lora_enabled = False

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for Polish language specifics.

        Args:
            text: Input text to preprocess

        Returns:
            Preprocessed text
        """
        if not text:
            return text

        # Normalize Polish text
        text = normalize_polish_text(text)

        # Fix common Polish typography issues
        text = self._fix_polish_typography(text)

        # Standardize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _fix_polish_typography(self, text: str) -> str:
        """Fix common Polish typography issues."""
        # Fix quotes
        text = re.sub(r'["„"]([^"]*?)[""]', r'„\1"', text)

        # Fix dashes
        text = text.replace(" - ", " – ")
        text = text.replace("--", "—")

        # Fix ellipsis
        text = re.sub(r"\.{3,}", "…", text)

        # Fix spaces around punctuation
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        text = re.sub(r"([¿¡])\s+", r"\1", text)

        return text

    def postprocess_text(self, text: str) -> str:
        """
        Postprocess generated text.

        Args:
            text: Generated text to postprocess

        Returns:
            Cleaned and formatted text
        """
        if not text:
            return text

        # Remove special tokens
        for token in self.config.polish_tokens:
            text = text.replace(token, "")

        # Clean up extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Fix sentence structure
        text = self._fix_sentence_structure(text)

        # Ensure proper capitalization
        text = self._fix_capitalization(text)

        return text

    def _fix_sentence_structure(self, text: str) -> str:
        """Fix sentence structure and punctuation."""
        # Ensure sentences end with proper punctuation
        sentences = re.split(r"([.!?]+)", text)
        fixed_sentences = []

        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            punct = sentences[i + 1] if i + 1 < len(sentences) else "."

            if sentence:
                # Capitalize first letter
                sentence = (
                    sentence[0].upper() + sentence[1:]
                    if len(sentence) > 1
                    else sentence.upper()
                )
                fixed_sentences.append(sentence + punct)

        return " ".join(fixed_sentences)

    def _fix_capitalization(self, text: str) -> str:
        """Fix Polish capitalization rules."""
        # Capitalize after sentence endings
        text = re.sub(
            r"([.!?]+\s+)([a-ząćęłńóśźż])",
            lambda m: m.group(1) + m.group(2).upper(),
            text,
        )

        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]

        return text

    def generate_polish_text(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """
        Generate Polish text with optimized parameters.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            repetition_penalty: Repetition penalty
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters

        Returns:
            Generated Polish text
        """
        # Preprocess prompt
        formatted_prompt = self.preprocess_text(prompt)

        # Add Polish context markers
        if not any(token in formatted_prompt for token in self.config.polish_tokens):
            formatted_prompt = f"<polish>{formatted_prompt}"

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_sequence_length - max_length,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=inputs.input_ids.shape[1] + max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        # Decode and clean
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        return self.postprocess_text(generated_text)

    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        return {
            "model_type": "mistral",
            "model_name": self.config.model_name,
            "total_parameters": self.get_parameter_count(),
            "trainable_parameters": self.get_trainable_parameter_count(),
            "is_quantized": self._is_quantized,
            "has_lora": self._has_lora,
            "device": str(self.device),
            "memory_usage": self.get_memory_usage(),
            "config": {
                "max_sequence_length": self.config.max_sequence_length,
                "lora_r": self.config.lora_r if self._has_lora else None,
                "quantization_bits": (
                    self.config.quantization_bits if self._is_quantized else None
                ),
            },
        }

    def estimate_memory_usage(
        self, batch_size: int = 1, sequence_length: int = None
    ) -> Dict[str, float]:
        """
        Estimate memory usage for given batch size and sequence length.

        Args:
            batch_size: Batch size for estimation
            sequence_length: Sequence length (defaults to config max)

        Returns:
            Memory usage estimation in GB
        """
        if sequence_length is None:
            sequence_length = self.config.max_sequence_length

        # Base model parameters (in bytes)
        param_count = self.get_parameter_count()

        # Quantization factor
        if self._is_quantized:
            bytes_per_param = 0.5  # 4-bit quantization
        else:
            bytes_per_param = 2  # bfloat16

        model_memory = param_count * bytes_per_param / (1024**3)  # GB

        # Activation memory (rough estimate)
        activation_memory = batch_size * sequence_length * 768 * 4 / (1024**3)  # GB

        # Gradient memory (if training)
        gradient_memory = model_memory if self.training else 0

        # Optimizer states (if training)
        optimizer_memory = model_memory * 2 if self.training else 0

        return {
            "model_memory": model_memory,
            "activation_memory": activation_memory,
            "gradient_memory": gradient_memory,
            "optimizer_memory": optimizer_memory,
            "total_estimated": model_memory
            + activation_memory
            + gradient_memory
            + optimizer_memory,
        }
