"""
Generation pipelines for WronAI models.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Iterator, AsyncIterator

import torch
from transformers import Pipeline, AutoTokenizer

from .engine import InferenceEngine, InferenceConfig
from ..utils.logging import get_logger
from ..utils.memory import memory_monitor

logger = get_logger(__name__)

@dataclass
class GenerationRequest:
    """Request for text generation."""

    prompt: str
    max_length: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_return_sequences: int = 1
    seed: Optional[int] = None
    stop_sequences: Optional[List[str]] = None

    # Polish-specific options
    use_polish_formatting: bool = True
    formal_register: bool = False

    # Request metadata
    request_id: str = None
    user_id: str = None
    priority: int = 0  # Higher number = higher priority

@dataclass
class GenerationResponse:
    """Response from text generation."""

    generated_text: str
    request_id: str
    generation_time: float
    tokens_generated: int
    finish_reason: str  # "length", "stop", "eos"
    metadata: Dict[str, Any]

class TextGenerationPipeline(ABC):
    """
    Abstract base class for text generation pipelines.
    """

    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        config: Optional[InferenceConfig] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or InferenceConfig()

        # Pipeline statistics
        self.stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_time": 0.0,
            "error_count": 0
        }

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text from request."""
        pass

    @abstractmethod
    async def generate_async(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text asynchronously."""
        pass

    def _update_stats(self, response: GenerationResponse):
        """Update pipeline statistics."""
        self.stats["total_requests"] += 1
        self.stats["total_tokens"] += response.tokens_generated
        self.stats["total_time"] += response.generation_time

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        avg_time = self.stats["total_time"] / max(self.stats["total_requests"], 1)
        avg_tokens_per_sec = self.stats["total_tokens"] / max(self.stats["total_time"], 1)

        return {
            **self.stats,
            "average_time": avg_time,
            "average_tokens_per_second": avg_tokens_per_sec
        }

class PolishGenerationPipeline(TextGenerationPipeline):
    """
    Text generation pipeline optimized for Polish language.
    """

    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        config: Optional[InferenceConfig] = None
    ):
        super().__init__(model, tokenizer, config)

        # Polish-specific patterns
        self.polish_stop_sequences = [
            "<|endoftext|>", "</s>", "<eos>",
            "Użytkownik:", "System:", "Human:", "Assistant:"
        ]

        # Formal/informal patterns
        self.formal_indicators = ["szanowny", "uprzejmie", "proszę", "dziękuję"]
        self.informal_indicators = ["cześć", "siema", "hej", "dzięki"]

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate Polish text from request.

        Args:
            request: Generation request

        Returns:
            Generation response
        """
        start_time = time.time()
        request_id = request.request_id or f"gen_{int(time.time() * 1000)}"

        try:
            # Preprocess prompt for Polish
            processed_prompt = self._preprocess_polish_prompt(request)

            # Set random seed if provided
            if request.seed is not None:
                torch.manual_seed(request.seed)

            # Tokenize input
            inputs = self.tokenizer(
                processed_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length - request.max_length
            ).to(self.model.device)

            input_length = inputs.input_ids.shape[1]

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=input_length + request.max_length,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    repetition_penalty=request.repetition_penalty,
                    do_sample=request.do_sample,
                    num_return_sequences=request.num_return_sequences,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )

            # Decode generated text
            generated_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )

            # Postprocess for Polish
            processed_text = self._postprocess_polish_text(generated_text, request)

            # Determine finish reason
            finish_reason = self._determine_finish_reason(
                generated_tokens, request, processed_text
            )

            generation_time = time.time() - start_time

            response = GenerationResponse(
                generated_text=processed_text,
                request_id=request_id,
                generation_time=generation_time,
                tokens_generated=len(generated_tokens),
                finish_reason=finish_reason,
                metadata={
                    "input_length": input_length,
                    "total_length": outputs[0].shape[1],
                    "prompt": request.prompt,
                    "config_used": {
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "repetition_penalty": request.repetition_penalty
                    }
                }
            )

            self._update_stats(response)
            return response

        except Exception as e:
            self.stats["error_count"] += 1
            logger.error(f"Generation failed for request {request_id}: {e}")

            return GenerationResponse(
                generated_text="",
                request_id=request_id,
                generation_time=time.time() - start_time,
                tokens_generated=0,
                finish_reason="error",
                metadata={"error": str(e)}
            )

    async def generate_async(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text asynchronously."""
        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, request)

    def _preprocess_polish_prompt(self, request: GenerationRequest) -> str:
        """Preprocess prompt for Polish generation."""
        prompt = request.prompt.strip()

        # Add Polish context if enabled
        if request.use_polish_formatting:
            if not prompt.startswith("<polish>"):
                prompt = f"<polish>{prompt}"

        # Handle formal/informal register
        if request.formal_register:
            # Add formal context
            if not any(indicator in prompt.lower() for indicator in self.formal_indicators):
                prompt = f"W oficjalnym tonie: {prompt}"

        # Ensure proper formatting
        prompt = prompt.replace('\n\n\n', '\n\n')

        return prompt

    def _postprocess_polish_text(self, text: str, request: GenerationRequest) -> str:
        """Postprocess generated Polish text."""
        # Remove context markers
        if request.use_polish_formatting:
            text = text.replace("<polish>", "").replace("</polish>", "")

        # Clean up formatting
        text = text.strip()

        # Handle stop sequences
        if request.stop_sequences:
            for stop_seq in request.stop_sequences:
                if stop_seq in text:
                    text = text.split(stop_seq)[0]

        # Default Polish stop sequences
        for stop_seq in self.polish_stop_sequences:
            if stop_seq in text:
                text = text.split(stop_seq)[0]

        # Fix common Polish text issues
        text = self._fix_polish_text_issues(text)

        return text.strip()

    def _fix_polish_text_issues(self, text: str) -> str:
        """Fix common issues in generated Polish text."""
        # Fix spacing
        text = text.replace('  ', ' ')

        # Fix punctuation spacing
        import re
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-ZĄĆĘŁŃÓŚŹŻ])', r'\1 \2', text)

        # Fix capitalization after sentence endings
        text = re.sub(r'([.!?]+\s+)([a-ząćęłńóśźż])',
                     lambda m: m.group(1) + m.group(2).upper(), text)

        # Ensure first letter is capitalized
        if text:
            text = text[0].upper() + text[1:]

        return text

    def _determine_finish_reason(
        self,
        generated_tokens: torch.Tensor,
        request: GenerationRequest,
        text: str
    ) -> str:
        """Determine why generation finished."""
        if len(generated_tokens) >= request.max_length:
            return "length"

        if self.tokenizer.eos_token_id in generated_tokens:
            return "eos"

        if request.stop_sequences:
            for stop_seq in request.stop_sequences:
                if stop_seq in text:
                    return "stop"

        for stop_seq in self.polish_stop_sequences:
            if stop_seq in text:
                return "stop"

        return "unknown"

#class BatchGenerationPipeline(PolishGenerationPipeline):
