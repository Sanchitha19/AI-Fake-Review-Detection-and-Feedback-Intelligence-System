"""
preprocess.py — Production-grade text preprocessing and feature engineering.
Handles text cleaning, NLP normalization, and behavioral feature extraction
(review length, exclamation count, caps ratio, avg word length) that are
classic signals for fake review detection.
"""

import re
import string
from typing import Dict, Any

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from loguru import logger

# Download required NLTK assets at import time (silent)
for _pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    nltk.download(_pkg, quiet=True)


class ReviewPreprocessor:
    """Cleans and normalises raw review text for downstream ML consumption."""

    def __init__(self) -> None:
        self.stop_words: set = set(stopwords.words("english"))
        self.lemmatizer = nltk.WordNetLemmatizer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean_text(self, text: str) -> str:
        """
        Full NLP cleaning pipeline:
        lowercase → strip URLs/HTML → remove non-alpha chars → tokenise →
        remove stopwords → lemmatise → rejoin.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        text = text.lower()
        text = re.sub(r"http\S+|www\S+|<[^>]+>", " ", text)   # URLs / HTML
        text = re.sub(r"[^a-zA-Z\s]", " ", text)               # keep letters only
        text = re.sub(r"\s+", " ", text).strip()

        tokens = word_tokenize(text)
        cleaned = [
            self.lemmatizer.lemmatize(tok)
            for tok in tokens
            if tok not in self.stop_words and len(tok) > 1
        ]
        return " ".join(cleaned)

    def extract_behavioral_features(self, text: str) -> Dict[str, float]:
        """
        Extract handcrafted signals that are strong indicators of fake reviews:
        - review_length      : character count
        - word_count         : number of words
        - exclamation_count  : raw count of '!'
        - caps_ratio         : fraction of uppercase letters
        - avg_word_length    : mean characters per word
        - punctuation_density: punctuation chars / total chars
        - digit_ratio        : digit chars / total chars
        """
        if not isinstance(text, str) or not text.strip():
            return {k: 0.0 for k in (
                "review_length", "word_count", "exclamation_count",
                "caps_ratio", "avg_word_length", "punctuation_density",
                "digit_ratio",
            )}

        words = text.split()
        total_chars = len(text) if len(text) > 0 else 1
        upper_chars = sum(1 for c in text if c.isupper())
        punct_chars = sum(1 for c in text if c in string.punctuation)
        digit_chars = sum(1 for c in text if c.isdigit())

        return {
            "review_length": float(len(text)),
            "word_count": float(len(words)),
            "exclamation_count": float(text.count("!")),
            "caps_ratio": round(upper_chars / total_chars, 4),
            "avg_word_length": round(
                sum(len(w) for w in words) / max(len(words), 1), 4
            ),
            "punctuation_density": round(punct_chars / total_chars, 4),
            "digit_ratio": round(digit_chars / total_chars, 4),
        }

    def process_dataframe(
        self, df: pd.DataFrame, text_column: str = "review_text"
    ) -> pd.DataFrame:
        """
        Apply full preprocessing to a DataFrame:
        - Adds 'cleaned_review' column with NLP-cleaned text.
        - Adds all behavioral feature columns.
        """
        logger.info(f"Preprocessing {len(df)} rows from column '{text_column}'")
        df = df.copy()
        df["cleaned_review"] = df[text_column].apply(self.clean_text)

        features = df[text_column].apply(self.extract_behavioral_features)
        feat_df = pd.DataFrame(features.tolist(), index=df.index)
        df = pd.concat([df, feat_df], axis=1)

        logger.info("Preprocessing complete.")
        return df


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    samples = {
        "review_text": [
            "This product is AMAZING!!! I LOVE it so much!!",
            "Worst purchase ever. Broke after 2 days!!!",
            "It's okay, not the best but works fine for the price.",
            "Fake review detection is important for e-commerce.",
        ]
    }
    df = pd.DataFrame(samples)
    pre = ReviewPreprocessor()
    df = pre.process_dataframe(df)
    print(df[["review_text", "cleaned_review", "caps_ratio", "exclamation_count"]])
