"""
WronAI Data Module

Data processing and utilities for Polish language datasets.
"""

from .dataset import PolishDataset, InstructionDataset, ConversationDataset
from .tokenizer import PolishTokenizer, setup_polish_tokenizer
from .preprocessing import PolishTextPreprocessor, clean_polish_text
from .collectors import PolishDataCollector, InstructionDataCollector


def prepare_polish_data(
        data_path: str,
        tokenizer_name: str = "mistralai/Mistral-7B-v0.1",
        max_length: int = 2048,
        instruction_format: bool = True
):
    """
    Prepare Polish data for training.

    Args:
        data_path: Path to raw data
        tokenizer_name: Name of tokenizer to use
        max_length: Maximum sequence length
        instruction_format: Whether to format as instructions

    Returns:
        Processed dataset ready for training
    """
    # Setup tokenizer
    tokenizer = setup_polish_tokenizer(tokenizer_name)

    # Choose dataset class
    if instruction_format:
        dataset_class = InstructionDataset
    else:
        dataset_class = PolishDataset

    # Create dataset
    dataset = dataset_class(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length
    )

    return dataset


__all__ = [
    "PolishDataset",
    "InstructionDataset",
    "ConversationDataset",
    "PolishTokenizer",
    "PolishTextPreprocessor",
    "PolishDataCollector",
    "InstructionDataCollector",
    "setup_polish_tokenizer",
    "clean_polish_text",
    "prepare_polish_data"
]