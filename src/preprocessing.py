"""
Text preprocessing module for sentiment analysis.

Applies standard NLP cleaning steps: lowercasing, punctuation removal,
URL removal, stopword removal, and lemmatization.
"""

import re
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def download_nltk_resources() -> None:
    """Download required NLTK data packages if not already present."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except Exception:
            # Re-download if missing or corrupted (e.g. BadZipFile)
            nltk.download(name, quiet=True)


class TextPreprocessor:
    """
    Reusable text preprocessor for movie review sentiment analysis.

    Pipeline: lowercase → remove URLs → remove HTML → remove numbers →
              remove punctuation/special chars → tokenize → remove stopwords →
              lemmatize → join tokens.
    """

    def __init__(self, language: str = "english") -> None:
        """
        Initialize the preprocessor with stopwords and lemmatizer.

        Args:
            language: Language for stopword list (default: english).
        """
        download_nltk_resources()
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

        # Pre-compiled regex patterns for efficiency
        self._url_pattern = re.compile(
            r"https?://\S+|www\.\S+", re.IGNORECASE
        )
        self._html_pattern = re.compile(r"<[^>]+>")
        self._number_pattern = re.compile(r"\d+")
        self._special_char_pattern = re.compile(r"[^a-zA-Z\s]")

    def to_lowercase(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()

    def remove_urls(self, text: str) -> str:
        """Remove HTTP/HTTPS and www URLs from text."""
        return self._url_pattern.sub(" ", text)

    def remove_html(self, text: str) -> str:
        """Remove HTML tags (common in IMDb reviews)."""
        return self._html_pattern.sub(" ", text)

    def remove_numbers(self, text: str) -> str:
        """Remove numeric digits from text."""
        return self._number_pattern.sub(" ", text)

    def remove_punctuation_and_special_chars(self, text: str) -> str:
        """Remove punctuation and special characters, keeping only letters."""
        return self._special_char_pattern.sub(" ", text)

    def tokenize(self, text: str) -> List[str]:
        """Split text into word tokens using NLTK."""
        return word_tokenize(text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Filter out common English stopwords."""
        return [t for t in tokens if t not in self.stop_words and len(t) > 1]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Reduce tokens to their base form using WordNet lemmatization."""
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def preprocess(self, text: str) -> str:
        """
        Apply the full preprocessing pipeline to a single text string.

        Args:
            text: Raw review text.

        Returns:
            Cleaned, space-joined string of lemmatized tokens.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        text = self.to_lowercase(text)
        text = self.remove_urls(text)
        text = self.remove_html(text)
        text = self.remove_numbers(text)
        text = self.remove_punctuation_and_special_chars(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)

        return " ".join(tokens)

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a list of text documents.

        Args:
            texts: List of raw review strings.

        Returns:
            List of preprocessed strings.
        """
        return [self.preprocess(t) for t in texts]


def get_sample_preprocessing_demo(preprocessor: TextPreprocessor) -> None:
    """
    Print a before/after example of text preprocessing.

    Args:
        preprocessor: Initialized TextPreprocessor instance.
    """
    sample = (
        "This movie is AMAZING!!! Visit http://example.com for more. "
        "I loved every minute of it. 10/10 would watch again."
    )
    print("\nPreprocessing Demo:")
    print(f"  Original:  {sample}")
    print(f"  Processed: {preprocessor.preprocess(sample)}")
