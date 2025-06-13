"""
Inference engine for WronAI models.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any

import torch
from transformers import GenerationConfig

from ..models.base import WronAIModel
from ..utils.logging import get_logger
from ..utils.memory import memory_monitor

logger = get_logger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference engine."""

    # Generation parameters
    max_length: int = 256
    max_new_tokens: Optional[int] = None
    min_length: int = 1
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3

    # Sampling
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = True

    # Polish specific
    use_polish_formatting: bool = True
    remove_special_tokens: bool = True

    # Performance
    batch_size: int = 1
    use_cache: bool = True

    # Safety
    max_time: float = 30.0  # Maximum generation time in seconds

    def to_generation_config(self) -> GenerationConfig:
        """Convert to HuggingFace GenerationConfig."""
        config_dict = {
            "max_length": self.max_length,
            "min_length": self.min_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
            "no_repeat_ngram_size": self.no_repeat_ngram_size,
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
            "early_stopping": self.early_stopping,
            "use_cache": self.use_cache,
        }

        if self.max_new_tokens is not None:
            config_dict["max_new_tokens"] = self.max_new_tokens
            config_dict.pop("max_length", None)

        return GenerationConfig(**config_dict)


class InferenceEngine:
    """
    High-performance inference engine for WronAI models.
    """

    def __init__(
        self,
        model: WronAIModel,
        config: Optional[InferenceConfig] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.config = config or InferenceConfig()

        # Ensure model is in eval mode
        self.model.eval()

        # Move to device if specified
        if device:
            self.model.to(device)

        self.device = self.model.device

        # Performance tracking
        self.generation_stats = {
            "total_generations": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "average_tokens_per_second": 0.0,
        }

        logger.info(f"Inference engine initialized on {self.device}")

    def generate(
        self,
        prompt: Union[str, List[str]],
        config: Optional[InferenceConfig] = None,
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Generate text from prompt(s).

        Args:
            prompt: Input prompt or list of prompts
            config: Generation configuration (overrides default)
            **kwargs: Additional generation parameters

        Returns:
            Generated text or list of generated texts
        """
        generation_config = config or self.config

        # Handle single prompt
        if isinstance(prompt, str):
            return self._generate_single(prompt, generation_config, **kwargs)

        # Handle batch of prompts
        return self._generate_batch(prompt, generation_config, **kwargs)

    def _generate_single(self, prompt: str, config: InferenceConfig, **kwargs) -> str:
        """Generate from single prompt."""
        start_time = time.time()

        try:
            # Preprocess prompt
            if hasattr(self.model, "preprocess_text"):
                processed_prompt = self.model.preprocess_text(prompt)
            else:
                processed_prompt = prompt

            # Add Polish formatting if enabled
            if config.use_polish_formatting and not any(
                token in processed_prompt for token in ["<polish>", "<question>"]
            ):
                processed_prompt = f"<polish>{processed_prompt}"

            # Tokenize
            inputs = self.model.tokenizer(
                processed_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.model.config.max_sequence_length
                - (config.max_new_tokens or config.max_length),
            ).to(self.device)

            input_length = inputs.input_ids.shape[1]

            # Generate
            generation_config = config.to_generation_config()

            # Update generation config with kwargs
            for key, value in kwargs.items():
                if hasattr(generation_config, key):
                    setattr(generation_config, key, value)

            with torch.no_grad():
                with (
                    memory_monitor()
                    if torch.cuda.is_available()
                    else torch.inference_mode()
                ):
                    outputs = self.model.model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        generation_config=generation_config,
                        pad_token_id=self.model.tokenizer.eos_token_id,
                        eos_token_id=self.model.tokenizer.eos_token_id,
                    )

            # Decode only the generated part
            generated_tokens = outputs[0][input_length:]
            generated_text = self.model.tokenizer.decode(
                generated_tokens, skip_special_tokens=config.remove_special_tokens
            )

            # Postprocess if available
            if hasattr(self.model, "postprocess_text"):
                generated_text = self.model.postprocess_text(generated_text)

            # Update statistics
            generation_time = time.time() - start_time
            self._update_stats(len(generated_tokens), generation_time)

            return generated_text.strip()

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def _generate_batch(
        self, prompts: List[str], config: InferenceConfig, **kwargs
    ) -> List[str]:
        """Generate from batch of prompts."""
        # For now, process sequentially
        # TODO: Implement true batch processing
        results = []

        for prompt in prompts:
            try:
                result = self._generate_single(prompt, config, **kwargs)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to generate for prompt: {e}")
                results.append("")

        return results

    def chat(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        config: Optional[InferenceConfig] = None,
    ) -> str:
        """
        Generate chat response with conversation context.

        Args:
            message: User message
            conversation_history: Previous conversation turns
            config: Generation configuration

        Returns:
            Model response
        """
        # Build conversation context
        context = self._build_conversation_context(message, conversation_history)

        # Generate response
        response = self.generate(context, config)

        return response

    def _build_conversation_context(
        self, message: str, history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Build conversation context from history."""
        context_parts = ["<polish><dialogue>"]

        # Add conversation history
        if history:
            for turn in history[-5:]:  # Last 5 turns
                role = turn.get("role", "user")
                content = turn.get("content", "")

                if role == "user":
                    context_parts.append(f"<question>{content}</question>")
                else:
                    context_parts.append(f"<answer>{content}</answer>")

        # Add current message
        context_parts.append(f"<question>{message}</question><answer>")

        return "".join(context_parts)

    def _update_stats(self, token_count: int, generation_time: float):
        """Update generation statistics."""
        self.generation_stats["total_generations"] += 1
        self.generation_stats["total_tokens"] += token_count
        self.generation_stats["total_time"] += generation_time

        if self.generation_stats["total_time"] > 0:
            self.generation_stats["average_tokens_per_second"] = (
                self.generation_stats["total_tokens"]
                / self.generation_stats["total_time"]
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return {
            **self.generation_stats,
            "model_info": (
                self.model.get_model_info()
                if hasattr(self.model, "get_model_info")
                else {}
            ),
            "device": str(self.device),
            "memory_usage": (
                self.model.get_memory_usage()
                if hasattr(self.model, "get_memory_usage")
                else {}
            ),
        }

    def reset_stats(self):
        """Reset generation statistics."""
        self.generation_stats = {
            "total_generations": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "average_tokens_per_second": 0.0,
        }

    def benchmark(
        self,
        prompts: List[str],
        config: Optional[InferenceConfig] = None,
        warmup_runs: int = 3,
    ) -> Dict[str, float]:
        """
        Benchmark inference performance.

        Args:
            prompts: List of test prompts
            config: Generation configuration
            warmup_runs: Number of warmup runs

        Returns:
            Benchmark results
        """
        logger.info(f"Starting benchmark with {len(prompts)} prompts")

        # Warmup
        if warmup_runs > 0:
            logger.info(f"Warming up with {warmup_runs} runs")
            for _ in range(warmup_runs):
                self.generate(prompts[0], config)

        # Reset stats
        self.reset_stats()

        # Benchmark
        start_time = time.time()

        for prompt in prompts:
            self.generate(prompt, config)

        total_time = time.time() - start_time

        return {
            "total_time": total_time,
            "prompts_per_second": len(prompts) / total_time,
            "tokens_per_second": self.generation_stats["average_tokens_per_second"],
            "average_generation_time": total_time / len(prompts),
            "total_tokens": self.generation_stats["total_tokens"],
            "average_tokens_per_generation": self.generation_stats["total_tokens"]
            / len(prompts),
        }

    def estimate_generation_time(self, prompt_length: int, target_length: int) -> float:
        """
        Estimate generation time based on current performance.

        Args:
            prompt_length: Length of input prompt in tokens
            target_length: Target generation length in tokens

        Returns:
            Estimated time in seconds
        """
        if self.generation_stats["average_tokens_per_second"] == 0:
            # No stats available, return rough estimate
            return target_length * 0.1  # 10 tokens per second estimate

        return target_length / self.generation_stats["average_tokens_per_second"]

    def set_generation_config(self, **kwargs):
        """Update default generation configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")

    def save_cache(self, path: str):
        """Save inference cache (if implemented)."""
        # TODO: Implement caching mechanism
        pass

    def load_cache(self, path: str):
        """Load inference cache (if implemented)."""
        # TODO: Implement caching mechanism
        pass
