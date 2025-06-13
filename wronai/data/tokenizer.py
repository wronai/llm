"""
Polish-optimized tokenizer implementation for WronAI.
"""

import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Union

from transformers import AutoTokenizer, PreTrainedTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
from tokenizers.normalizers import NFD, Lowercase, StripAccents

from ..utils.logging import get_logger
from .polish import POLISH_SPECIAL_TOKENS, POLISH_DIACRITICS_MAP, load_polish_stopwords

logger = get_logger(__name__)

class PolishTokenizer:
    """
    Enhanced tokenizer optimized for Polish language.
    """

    def __init__(
        self,
        base_tokenizer: Union[str, PreTrainedTokenizer],
        vocab_size: int = 32000,
        add_polish_tokens: bool = True,
        preserve_diacritics: bool = True,
        normalize_whitespace: bool = True
    ):
        self.vocab_size = vocab_size
        self.add_polish_tokens = add_polish_tokens
        self.preserve_diacritics = preserve_diacritics
        self.normalize_whitespace = normalize_whitespace

        # Load base tokenizer
        if isinstance(base_tokenizer, str):
            self.base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer)
        else:
            self.base_tokenizer = base_tokenizer

        # Polish language specific settings
        self.polish_stopwords = load_polish_stopwords()
        self.polish_special_tokens = POLISH_SPECIAL_TOKENS.copy()

        # Add custom Polish tokens if requested
        if self.add_polish_tokens:
            self._add_polish_tokens()

        # Statistics
        self.token_stats = {
            "total_tokens": 0,
            "polish_tokens": 0,
            "unknown_tokens": 0,
            "special_tokens": 0
        }

    def _add_polish_tokens(self):
        """Add Polish-specific tokens to tokenizer."""
        # Special tokens for Polish content
        special_tokens = [
            "<polish>", "</polish>",
            "<formal>", "</formal>",
            "<informal>", "</informal>",
            "<question>", "</question>",
            "<answer>", "</answer>",
            "<dialogue>", "</dialogue>",
            "<summary>", "</summary>",
            "<title>", "</title>"
        ]

        # Common Polish words that should be single tokens
        polish_words = [
            "że", "się", "nie", "być", "mieć", "móc", "już", "tylko",
            "bardzo", "gdzie", "kiedy", "dlaczego", "który", "która", "które",
            "ponieważ", "dlatego", "jednak", "również", "więc", "czyli",
            "przede", "wszystkim", "oczywiście", "prawdopodobnie"
        ]

        # Polish morphological endings
        morphological_tokens = [
            "ów", "ach", "ami", "em", "ie", "ę", "ą", "y", "ymi",
            "ość", "anie", "enie", "ość", "ić", "ować", "nąć"
        ]

        all_new_tokens = special_tokens + polish_words + morphological_tokens

        # Add tokens that don't already exist
        existing_vocab = set(self.base_tokenizer.get_vocab().keys())
        new_tokens = [token for token in all_new_tokens if token not in existing_vocab]

        if new_tokens:
            num_added = self.base_tokenizer.add_tokens(new_tokens)
            logger.info(f"Added {num_added} Polish tokens to tokenizer")

            # Update special tokens list
            self.polish_special_tokens.extend(new_tokens)

    def tokenize(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_attention_mask: bool = False,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        truncation: bool = False
    ) -> Union[List[str], Dict[str, List]]:
        """
        Tokenize text with Polish language optimizations.

        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
            return_attention_mask: Whether to return attention mask
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate

        Returns:
            Tokenized output
        """
        # Preprocess for Polish
        processed_text = self._preprocess_polish_text(text)

        # Tokenize
        tokens = self.base_tokenizer.tokenize(processed_text)

        # Post-process tokens
        tokens = self._postprocess_tokens(tokens)

        # Update statistics
        self._update_token_stats(tokens)

        if return_attention_mask or max_length or padding or truncation:
            # Use full encoding
            encoding = self.base_tokenizer(
                processed_text,
                add_special_tokens=add_special_tokens,
                return_attention_mask=return_attention_mask,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=None
            )

            # Replace token IDs with actual tokens for debugging
            if 'input_ids' in encoding:
                encoding['tokens'] = [self.base_tokenizer.decode([token_id]) for token_id in encoding['input_ids']]

            return encoding

        return tokens

    def _preprocess_polish_text(self, text: str) -> str:
        """Preprocess text for Polish language specifics."""
        if not text:
            return text

        # Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()

        # Handle Polish quotation marks
        text = re.sub(r'[""]([^"]*?)[""]', r'„\1"', text)

        # Normalize Polish dashes
        text = text.replace(' - ', ' – ')
        text = text.replace('--', '—')

        # Handle ellipsis
        text = re.sub(r'\.{3,}', '…', text)

        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-ZĄĆĘŁŃÓŚŹŻ])', r'\1 \2', text)

        return text

    def _postprocess_tokens(self, tokens: List[str]) -> List[str]:
        """Post-process tokens for Polish optimizations."""
        processed_tokens = []

        for token in tokens:
            # Handle subword tokens with Polish characters
            if token.startswith('##') and any(char in token for char in 'ąćęłńóśźż'):
                # Keep Polish subwords as-is
                processed_tokens.append(token)
            elif token in self.polish_special_tokens:
                # Mark special tokens
                processed_tokens.append(token)
            else:
                processed_tokens.append(token)

        return processed_tokens

    def _update_token_stats(self, tokens: List[str]):
        """Update tokenization statistics."""
        self.token_stats["total_tokens"] += len(tokens)

        for token in tokens:
            if any(char in token for char in 'ąćęłńóśźż'):
                self.token_stats["polish_tokens"] += 1
            elif token in self.polish_special_tokens:
                self.token_stats["special_tokens"] += 1
            elif token == self.base_tokenizer.unk_token:
                self.token_stats["unknown_tokens"] += 1

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None
    ):
        """Encode text to token IDs."""
        processed_text = self._preprocess_polish_text(text)

        return self.base_tokenizer(
            processed_text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> str:
        """Decode token IDs to text."""
        decoded = self.base_tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )

        # Post-process decoded text for Polish
        return self._postprocess_decoded_text(decoded)

    def _postprocess_decoded_text(self, text: str) -> str:
        """Post-process decoded text for Polish formatting."""
        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text).strip()

        # Fix Polish punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-ZĄĆĘŁŃÓŚŹŻ])', r'\1 \2', text)

        return text

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.base_tokenizer.get_vocab())

    def get_polish_coverage(self, text: str) -> Dict[str, float]:
        """
        Analyze Polish language coverage in tokenization.

        Args:
            text: Text to analyze

        Returns:
            Coverage statistics
        """
        tokens = self.tokenize(text)

        total_tokens = len(tokens)
        polish_tokens = sum(1 for token in tokens if any(char in token for char in 'ąćęłńóśźż'))
        special_tokens = sum(1 for token in tokens if token in self.polish_special_tokens)
        unknown_tokens = sum(1 for token in tokens if token == self.base_tokenizer.unk_token)

        return {
            "total_tokens": total_tokens,
            "polish_token_ratio": polish_tokens / max(total_tokens, 1),
            "special_token_ratio": special_tokens / max(total_tokens, 1),
            "unknown_token_ratio": unknown_tokens / max(total_tokens, 1),
            "avg_token_length": sum(len(token) for token in tokens) / max(total_tokens, 1)
        }

    def save_pretrained(self, save_directory: str):
        """Save tokenizer to directory."""
        os.makedirs(save_directory, exist_ok=True)

        # Save base tokenizer
        self.base_tokenizer.save_pretrained(save_directory)

        # Save Polish-specific settings
        polish_config = {
            "vocab_size": self.vocab_size,
            "add_polish_tokens": self.add_polish_tokens,
            "preserve_diacritics": self.preserve_diacritics,
            "normalize_whitespace": self.normalize_whitespace,
            "polish_special_tokens": self.polish_special_tokens,
            "token_stats": self.token_stats
        }

        with open(os.path.join(save_directory, "polish_config.json"), 'w') as f:
            json.dump(polish_config, f, indent=2, ensure_ascii=False)

        logger.info(f"Polish tokenizer saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, model_path: str) -> 'PolishTokenizer':
        """Load Polish tokenizer from directory."""
        # Load base tokenizer
        base_tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load Polish config if it exists
        polish_config_path = os.path.join(model_path, "polish_config.json")

        if os.path.exists(polish_config_path):
            with open(polish_config_path, 'r') as f:
                polish_config = json.load(f)

            tokenizer = cls(
                base_tokenizer=base_tokenizer,
                vocab_size=polish_config.get("vocab_size", 32000),
                add_polish_tokens=polish_config.get("add_polish_tokens", True),
                preserve_diacritics=polish_config.get("preserve_diacritics", True),
                normalize_whitespace=polish_config.get("normalize_whitespace", True)
            )

            # Restore special tokens and stats
            tokenizer.polish_special_tokens = polish_config.get("polish_special_tokens", [])
            tokenizer.token_stats = polish_config.get("token_stats", {})
        else:
            # Create with default settings
            tokenizer = cls(base_tokenizer=base_tokenizer)

        logger.info(f"Polish tokenizer loaded from {model_path}")
        return tokenizer

def train_polish_tokenizer(
    texts: List[str],
    vocab_size: int = 32000,
    model_type: str = "bpe",
    special_tokens: Optional[List[str]] = None
) -> Tokenizer:
    """
    Train a new tokenizer optimized for Polish from scratch.

    Args:
        texts: List of Polish texts for training
        vocab_size: Target vocabulary size
        model_type: Tokenizer model type (bpe, wordpiece, unigram)
        special_tokens: Special tokens to include

    Returns:
        Trained tokenizer
    """
    logger.info(f"Training Polish {model_type} tokenizer with vocab size {vocab_size}")

    # Default special tokens
    if special_tokens is None:
        special_tokens = [
            "<unk>", "<s>", "</s>", "<pad>", "<mask>",
            "<polish>", "</polish>", "<formal>", "</formal>",
            "<question>", "</question>", "<answer>", "</answer>"
        ]

    # Initialize tokenizer based on model type
    if model_type.lower() == "bpe":
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            min_frequency=2
        )
    elif model_type.lower() == "wordpiece":
        tokenizer = Tokenizer(models.WordPiece(unk_token="<unk>"))
        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            min_frequency=2
        )
    elif model_type.lower() == "unigram":
        tokenizer = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            unk_token="<unk>"
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Configure pre-tokenizer for Polish
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=r'\s+', behavior="removed"),
        pre_tokenizers.Punctuation(behavior="isolated")
    ])

    # Configure normalizer (preserve Polish diacritics)
    tokenizer.normalizer = NFD()

    # Configure post-processor
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> $B:1 </s>:1",
        special_tokens=[("<s>", 1), ("</s>", 2)]
    )

    # Train tokenizer
    logger.info("Training tokenizer on Polish texts...")
    tokenizer.train_from_iterator(texts, trainer=trainer)

    logger.info(f"Tokenizer training completed. Vocab size: {tokenizer.get_vocab_size()}")
    return tokenizer

def setup_polish_tokenizer(
    tokenizer_name: str,
    add_polish_tokens: bool = True,
    vocab_size: Optional[int] = None
) -> PolishTokenizer:
    """
    Setup a Polish-optimized tokenizer from a base model.

    Args:
        tokenizer_name: Name or path of base tokenizer
        add_polish_tokens: Whether to add Polish-specific tokens
        vocab_size: Target vocabulary size

    Returns:
        Configured Polish tokenizer
    """
    logger.info(f"Setting up Polish tokenizer from {tokenizer_name}")

    # Load base tokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Create Polish tokenizer
    polish_tokenizer = PolishTokenizer(
        base_tokenizer=base_tokenizer,
        vocab_size=vocab_size or base_tokenizer.vocab_size,
        add_polish_tokens=add_polish_tokens
    )

    return polish_tokenizer

def analyze_tokenizer_efficiency(
    tokenizer: Union[PolishTokenizer, PreTrainedTokenizer],
    polish_texts: List[str],
    comparison_texts: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze tokenizer efficiency on Polish texts.

    Args:
        tokenizer: Tokenizer to analyze
        polish_texts: Polish texts for analysis
        comparison_texts: Non-Polish texts for comparison

    Returns:
        Analysis results
    """
    logger.info("Analyzing tokenizer efficiency on Polish texts...")

    # Analyze Polish texts
    polish_stats = []
    for text in polish_texts:
        if isinstance(tokenizer, PolishTokenizer):
            tokens = tokenizer.tokenize(text)
        else:
            tokens = tokenizer.tokenize(text)

        stats = {
            "text_length": len(text),
            "num_tokens": len(tokens),
            "compression_ratio": len(text) / max(len(tokens), 1),
            "avg_token_length": sum(len(token) for token in tokens) / max(len(tokens), 1),
            "polish_chars_ratio": sum(1 for char in text if char in 'ąćęłńóśźż') / max(len(text), 1)
        }
        polish_stats.append(stats)

    # Calculate averages for Polish texts
    polish_results = {
        "num_samples": len(polish_stats),
        "avg_compression_ratio": sum(s["compression_ratio"] for s in polish_stats) / len(polish_stats),
        "avg_token_length": sum(s["avg_token_length"] for s in polish_stats) / len(polish_stats),
        "avg_tokens_per_text": sum(s["num_tokens"] for s in polish_stats) / len(polish_stats),
        "avg_polish_chars_ratio": sum(s["polish_chars_ratio"] for s in polish_stats) / len(polish_stats)
    }

    results = {"polish": polish_results}

    # Analyze comparison texts if provided
    if comparison_texts:
        comparison_stats = []
        for text in comparison_texts:
            if isinstance(tokenizer, PolishTokenizer):
                tokens = tokenizer.tokenize(text)
            else:
                tokens = tokenizer.tokenize(text)

            stats = {
                "text_length": len(text),
                "num_tokens": len(tokens),
                "compression_ratio": len(text) / max(len(tokens), 1),
                "avg_token_length": sum(len(token) for token in tokens) / max(len(tokens), 1)
            }
            comparison_stats.append(stats)

        comparison_results = {
            "num_samples": len(comparison_stats),
            "avg_compression_ratio": sum(s["compression_ratio"] for s in comparison_stats) / len(comparison_stats),
            "avg_token_length": sum(s["avg_token_length"] for s in comparison_stats) / len(comparison_stats),
            "avg_tokens_per_text": sum(s["num_tokens"] for s in comparison_stats) / len(comparison_stats)
        }

        results["comparison"] = comparison_results

        # Calculate relative efficiency
        polish_efficiency = polish_results["avg_compression_ratio"]
        comparison_efficiency = comparison_results["avg_compression_ratio"]

        results["relative_efficiency"] = {
            "polish_vs_comparison": polish_efficiency / max(comparison_efficiency, 0.1),
            "polish_advantage_percent": ((polish_efficiency - comparison_efficiency) / max(comparison_efficiency, 0.1)) * 100
        }

    return results

def create_polish_subword_vocab(
    polish_texts: List[str],
    target_size: int = 5000,
    min_frequency: int = 5
) -> Dict[str, int]:
    """
    Create Polish-specific subword vocabulary.

    Args:
        polish_texts: Polish texts for vocabulary extraction
        target_size: Target vocabulary size
        min_frequency: Minimum frequency for inclusion

    Returns:
        Polish subword vocabulary
    """
    logger.info(f"Creating Polish subword vocabulary (target size: {target_size})")

    # Extract subwords with Polish morphological patterns
    subword_counter = Counter()

    polish_patterns = [
        # Common endings
        r'\w*ość\b', r'\w*anie\b', r'\w*enie\b', r'\w*ować\b',
        r'\w*ić\b', r'\w*nąć\b', r'\w*em\b', r'\w*ie\b',
        # Prefixes
        r'\bprze\w*', r'\bpo\w*', r'\bna\w*', r'\bza\w*',
        r'\bwy\w*', r'\bdo\w*', r'\bod\w*', r'\bu\w*',
        # Polish-specific character combinations
        r'\w*ą\w*', r'\w*ę\w*', r'\w*ł\w*', r'\w*ń\w*',
        r'\w*ś\w*', r'\w*ź\w*', r'\w*ż\w*', r'\w*ć\w*'
    ]

    for text in polish_texts:
        words = re.findall(r'\b\w+\b', text.lower())

        for word in words:
            if any(char in word for char in 'ąćęłńóśźż'):  # Polish words only
                # Add whole word
                subword_counter[word] += 1

                # Add subwords based on patterns
                for pattern in polish_patterns:
                    matches = re.findall(pattern, word)
                    for match in matches:
                        if len(match) >= 3:  # Minimum subword length
                            subword_counter[match] += 1

                # Add character n-grams for Polish words
                for n in range(2, min(6, len(word) + 1)):
                    for i in range(len(word) - n + 1):
                        ngram = word[i:i+n]
                        if any(char in ngram for char in 'ąćęłńóśźż'):
                            subword_counter[ngram] += 1

    # Filter by frequency and select top subwords
    filtered_subwords = {
        subword: count for subword, count in subword_counter.items()
        if count >= min_frequency
    }

    # Sort by frequency and take top entries
    sorted_subwords = sorted(filtered_subwords.items(), key=lambda x: x[1], reverse=True)
    final_vocab = {subword: idx for idx, (subword, _) in enumerate(sorted_subwords[:target_size])}

    logger.info(f"Created Polish vocabulary with {len(final_vocab)} subwords")
    return final_vocab

def benchmark_tokenizers(
    tokenizers: Dict[str, Union[PolishTokenizer, PreTrainedTokenizer]],
    test_texts: List[str],
    metrics: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark multiple tokenizers on test texts.

    Args:
        tokenizers: Dictionary of tokenizer name -> tokenizer
        test_texts: Test texts for benchmarking
        metrics: Metrics to compute

    Returns:
        Benchmark results
    """
    if metrics is None:
        metrics = ["compression_ratio", "token_length", "speed"]

    results = {}

    for name, tokenizer in tokenizers.items():
        logger.info(f"Benchmarking tokenizer: {name}")

        tokenizer_results = {
            "compression_ratios": [],
            "token_lengths": [],
            "tokenization_times": []
        }

        import time

        for text in test_texts:
            # Measure tokenization time
            start_time = time.time()

            if isinstance(tokenizer, PolishTokenizer):
                tokens = tokenizer.tokenize(text)
            else:
                tokens = tokenizer.tokenize(text)

            end_time = time.time()

            # Calculate metrics
            compression_ratio = len(text) / max(len(tokens), 1)
            avg_token_length = sum(len(token) for token in tokens) / max(len(tokens), 1)
            tokenization_time = end_time - start_time

            tokenizer_results["compression_ratios"].append(compression_ratio)
            tokenizer_results["token_lengths"].append(avg_token_length)
            tokenizer_results["tokenization_times"].append(tokenization_time)

        # Calculate averages
        results[name] = {
            "avg_compression_ratio": sum(tokenizer_results["compression_ratios"]) / len(tokenizer_results["compression_ratios"]),
            "avg_token_length": sum(tokenizer_results["token_lengths"]) / len(tokenizer_results["token_lengths"]),
            "avg_tokenization_time": sum(tokenizer_results["tokenization_times"]) / len(tokenizer_results["tokenization_times"]),
            "total_time": sum(tokenizer_results["tokenization_times"]),
            "texts_per_second": len(test_texts) / sum(tokenizer_results["tokenization_times"])
        }

    return results