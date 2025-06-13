"""
Dataset classes for WronAI Polish language data.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from ..utils.logging import get_logger
from .polish import load_polish_stopwords, normalize_polish_text
from .preprocessing import PolishTextPreprocessor

logger = get_logger(__name__)

class PolishDataset(Dataset):
    """
    Base dataset class for Polish language data.
    """

    def __init__(
        self,
        data_path: Union[str, List[str]],
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        preprocessing: bool = True,
        cache_dir: Optional[str] = None
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessing = preprocessing
        self.cache_dir = cache_dir

        # Initialize preprocessor
        if preprocessing:
            self.preprocessor = PolishTextPreprocessor()

        # Load data
        self.data = self._load_data()

        logger.info(f"Loaded {len(self.data)} samples from {data_path}")

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from files."""
        data = []

        if isinstance(self.data_path, str):
            paths = [self.data_path]
        else:
            paths = self.data_path

        for path in paths:
            path = Path(path)

            if path.suffix == '.json':
                data.extend(self._load_json(path))
            elif path.suffix == '.jsonl':
                data.extend(self._load_jsonl(path))
            elif path.suffix == '.txt':
                data.extend(self._load_text(path))
            else:
                logger.warning(f"Unsupported file format: {path.suffix}")

        return data

    def _load_json(self, path: Path) -> List[Dict[str, Any]]:
        """Load JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        else:
            return [data]

    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """Load JSONL file."""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def _load_text(self, path: Path) -> List[Dict[str, Any]]:
        """Load plain text file."""
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        return [{"text": p, "source": str(path)} for p in paragraphs]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index."""
        item = self.data[idx]

        # Extract text
        text = self._extract_text(item)

        # Preprocess if enabled
        if self.preprocessing:
            text = self.preprocessor.process(text)

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding.input_ids.squeeze(0),
            "attention_mask": encoding.attention_mask.squeeze(0),
            "labels": encoding.input_ids.squeeze(0).clone()
        }

    def _extract_text(self, item: Dict[str, Any]) -> str:
        """Extract text from data item."""
        if "text" in item:
            return item["text"]
        elif "content" in item:
            return item["content"]
        else:
            return str(item)

class InstructionDataset(PolishDataset):
    """
    Dataset for instruction-following data in Polish.
    """

    def __init__(
        self,
        data_path: Union[str, List[str]],
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        instruction_template: str = None,
        preprocessing: bool = True,
        cache_dir: Optional[str] = None
    ):
        self.instruction_template = instruction_template or self._default_template()

        super().__init__(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            preprocessing=preprocessing,
            cache_dir=cache_dir
        )

    def _default_template(self) -> str:
        """Default instruction template for Polish."""
        return "<polish><question>{instruction}</question><answer>{response}</answer></polish>"

    def _extract_text(self, item: Dict[str, Any]) -> str:
        """Extract text from instruction item."""
        if "instruction" in item and "response" in item:
            return self.instruction_template.format(
                instruction=item["instruction"],
                response=item["response"]
            )
        else:
            return super()._extract_text(item)

class ConversationDataset(PolishDataset):
    """
    Dataset for multi-turn conversations in Polish.
    """

    def __init__(
        self,
        data_path: Union[str, List[str]],
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        max_turns: int = 10,
        preprocessing: bool = True,
        cache_dir: Optional[str] = None
    ):
        self.max_turns = max_turns

        super().__init__(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            preprocessing=preprocessing,
            cache_dir=cache_dir
        )

    def _extract_text(self, item: Dict[str, Any]) -> str:
        """Extract text from conversation item."""
        if "conversation" in item:
            turns = item["conversation"][:self.max_turns]

            formatted_turns = []
            for turn in turns:
                role = turn.get("role", "user")
                content = turn.get("content", "")

                if role == "user":
                    formatted_turns.append(f"<question>{content}</question>")
                else:
                    formatted_turns.append(f"<answer>{content}</answer>")

            return f"<polish><dialogue>{''.join(formatted_turns)}</dialogue></polish>"

        return super()._extract_text(item)

class PolishWebDataset(PolishDataset):
    """
    Dataset for Polish web scraped data with quality filtering.
    """

    def __init__(
        self,
        data_path: Union[str, List[str]],
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        min_length: int = 100,
        quality_threshold: float = 0.7,
        preprocessing: bool = True,
        cache_dir: Optional[str] = None
    ):
        self.min_length = min_length
        self.quality_threshold = quality_threshold
        self.polish_stopwords = load_polish_stopwords()

        super().__init__(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            preprocessing=preprocessing,
            cache_dir=cache_dir
        )

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load and filter web data."""
        raw_data = super()._load_data()

        # Filter by quality
        filtered_data = []
        for item in raw_data:
            text = self._extract_text(item)

            if self._is_quality_text(text):
                filtered_data.append(item)

        logger.info(f"Filtered {len(raw_data)} -> {len(filtered_data)} samples")
        return filtered_data

    def _is_quality_text(self, text: str) -> bool:
        """Check if text meets quality criteria."""
        if len(text) < self.min_length:
            return False

        # Check for Polish content
        polish_chars = set('ąćęłńóśźż')
        if not any(char in text.lower() for char in polish_chars):
            return False

        # Check word diversity
        words = text.lower().split()
        if len(words) < 10:
            return False

        unique_words = set(words)
        diversity_ratio = len(unique_words) / len(words)

        if diversity_ratio < 0.3:  # Too repetitive
            return False

        # Check for meaningful content (not just stopwords)
        content_words = [w for w in unique_words if w not in self.polish_stopwords]
        if len(content_words) / len(unique_words) < 0.3:
            return False

        return True

def create_polish_dataset(
    dataset_type: str,
    data_path: Union[str, List[str]],
    tokenizer: AutoTokenizer,
    **kwargs
) -> PolishDataset:
    """
    Factory function to create Polish datasets.

    Args:
        dataset_type: Type of dataset ('text', 'instruction', 'conversation', 'web')
        data_path: Path(s) to data files
        tokenizer: Tokenizer to use
        **kwargs: Additional arguments for dataset

    Returns:
        Configured dataset instance
    """
    dataset_classes = {
        'text': PolishDataset,
        'instruction': InstructionDataset,
        'conversation': ConversationDataset,
        'web': PolishWebDataset
    }

    if dataset_type not in dataset_classes:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    dataset_class = dataset_classes[dataset_type]
    return dataset_class(data_path=data_path, tokenizer=tokenizer, **kwargs)

def load_huggingface_polish_dataset(
    dataset_name: str,
    split: str = "train",
    tokenizer: AutoTokenizer,
    max_samples: Optional[int] = None,
    **kwargs
) -> HFDataset:
    """
    Load Polish dataset from HuggingFace Hub.

    Args:
        dataset_name: Name of HuggingFace dataset
        split: Dataset split to load
        tokenizer: Tokenizer for processing
        max_samples: Maximum number of samples to load
        **kwargs: Additional arguments

    Returns:
        Processed HuggingFace dataset
    """
    # Load dataset
    dataset = load_dataset(dataset_name, split=split)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Tokenization function
    def tokenize_function(examples):
        # Handle different text fields
        texts = []
        for i in range(len(examples.get('text', examples.get('content', [])))):
            if 'text' in examples:
                text = examples['text'][i]
            elif 'content' in examples:
                text = examples['content'][i]
            elif 'instruction' in examples and 'response' in examples:
                instruction = examples['instruction'][i]
                response = examples['response'][i]
                text = f"<polish><question>{instruction}</question><answer>{response}</answer></polish>"
            else:
                text = str(examples[list(examples.keys())[0]][i])

            texts.append(text)

        return tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=kwargs.get('max_length', 2048)
        )

    # Apply tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset