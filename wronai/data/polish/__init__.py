"""
Polish language specific data processing utilities.
"""

from .normalization import (
    normalize_polish_text,
    remove_diacritics,
    standardize_quotes,
    fix_spacing
)

from .morphology import (
    PolishMorphologyAnalyzer,
    get_word_forms,
    analyze_polish_sentence
)


def load_polish_stopwords():
    """Load Polish stopwords from file."""
    import os

    stopwords_path = os.path.join(os.path.dirname(__file__), "stopwords.txt")

    if os.path.exists(stopwords_path):
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            return set(word.strip() for word in f if word.strip())
    else:
        # Fallback to basic stopwords
        return {
            'i', 'a', 'o', 'z', 'w', 'na', 'do', 'nie', 'że', 'się', 'to',
            'co', 'jak', 'ale', 'czy', 'gdy', 'już', 'dla', 'od', 'po',
            'lub', 'oraz', 'przez', 'przy', 'pod', 'nad', 'bez', 'mimo'
        }


POLISH_SPECIAL_TOKENS = [
    "<polish>", "</polish>",
    "<formal>", "</formal>",
    "<informal>", "</informal>",
    "<question>", "</question>",
    "<answer>", "</answer>",
    "<summary>", "</summary>",
    "<title>", "</title>",
    "<dialogue>", "</dialogue>"
]

POLISH_DIACRITICS_MAP = {
    'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l',
    'ń': 'n', 'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z',
    'Ą': 'A', 'Ć': 'C', 'Ę': 'E', 'Ł': 'L',
    'Ń': 'N', 'Ó': 'O', 'Ś': 'S', 'Ź': 'Z', 'Ż': 'Z'
}

__all__ = [
    "normalize_polish_text",
    "remove_diacritics",
    "standardize_quotes",
    "fix_spacing",
    "PolishMorphologyAnalyzer",
    "get_word_forms",
    "analyze_polish_sentence",
    "load_polish_stopwords",
    "POLISH_SPECIAL_TOKENS",
    "POLISH_DIACRITICS_MAP"
]