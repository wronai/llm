"""
Text preprocessing utilities for Polish language data.
"""

import re
import unicodedata
from typing import List, Optional, Set

from ..utils.logging import get_logger
from .polish import normalize_polish_text, POLISH_DIACRITICS_MAP, load_polish_stopwords

logger = get_logger(__name__)

class PolishTextPreprocessor:
    """
    Comprehensive text preprocessor for Polish language.
    """

    def __init__(
        self,
        normalize_unicode: bool = True,
        fix_encoding: bool = True,
        standardize_quotes: bool = True,
        fix_whitespace: bool = True,
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        fix_punctuation: bool = True,
        lowercase: bool = False,
        remove_stopwords: bool = False,
        min_length: int = 10,
        max_length: Optional[int] = None
    ):
        self.normalize_unicode = normalize_unicode
        self.fix_encoding = fix_encoding
        self.standardize_quotes = standardize_quotes
        self.fix_whitespace = fix_whitespace
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.fix_punctuation = fix_punctuation
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.min_length = min_length
        self.max_length = max_length

        # Load Polish stopwords if needed
        if remove_stopwords:
            self.stopwords = load_polish_stopwords()
        else:
            self.stopwords = set()

        # Compile regex patterns
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        # HTML tags
        self.html_pattern = re.compile(r'<[^>]+>')

        # URLs
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )

        # Email addresses
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

        # Multiple whitespace
        self.whitespace_pattern = re.compile(r'\s+')

        # Multiple punctuation
        self.punct_pattern = re.compile(r'([.!?]){2,}')

        # Non-Polish characters (but keep basic punctuation and numbers)
        self.non_polish_pattern = re.compile(
            r'[^a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9\s.,!?;:()\-\'"„"…–—]'
        )

    def process(self, text: str) -> str:
        """
        Process text through all preprocessing steps.

        Args:
            text: Input text to process

        Returns:
            Processed text
        """
        if not text or not isinstance(text, str):
            return ""

        original_length = len(text)

        # Step 1: Fix encoding issues
        if self.fix_encoding:
            text = self._fix_encoding(text)

        # Step 2: Normalize Unicode
        if self.normalize_unicode:
            text = self._normalize_unicode(text)

        # Step 3: Remove HTML
        if self.remove_html:
            text = self._remove_html(text)

        # Step 4: Remove URLs
        if self.remove_urls:
            text = self._remove_urls(text)

        # Step 5: Remove emails
        if self.remove_emails:
            text = self._remove_emails(text)

        # Step 6: Standardize quotes
        if self.standardize_quotes:
            text = self._standardize_quotes(text)

        # Step 7: Fix punctuation
        if self.fix_punctuation:
            text = self._fix_punctuation(text)

        # Step 8: Fix whitespace
        if self.fix_whitespace:
            text = self._fix_whitespace(text)

        # Step 9: Lowercase
        if self.lowercase:
            text = text.lower()

        # Step 10: Remove stopwords
        if self.remove_stopwords:
            text = self._remove_stopwords(text)

        # Step 11: Length filtering
        if len(text) < self.min_length:
            return ""

        if self.max_length and len(text) > self.max_length:
            text = text[:self.max_length]

        logger.debug(f"Preprocessed text: {original_length} -> {len(text)} chars")
        return text.strip()

    def _fix_encoding(self, text: str) -> str:
        """Fix common encoding issues."""
        # Fix common encoding artifacts
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€\u009d': '"',
            'â€"': '—',
            'â€"': '–',
            'â€¦': '…',
            'Ã¡': 'á',
            'Ã©': 'é',
            'Ã­': 'í',
            'Ã³': 'ó',
            'Ãº': 'ú',
            'Ã±': 'ñ',
        }

        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)

        return text

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        # Normalize to NFKC form