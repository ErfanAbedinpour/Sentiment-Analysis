"""
Utility functions for the sentiment analysis project.

Provides path management, data loading, and shared constants used across modules.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd

# Project root is one level above src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "IMDB Dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Label mapping for binary classification
LABEL_MAP = {"positive": 1, "negative": 0}
INVERSE_LABEL_MAP = {1: "positive", 0: "negative"}


def ensure_directories() -> None:
    """Create required output directories if they do not exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    """
    Load the IMDb movie reviews dataset from the dataset folder.

    Returns:
        DataFrame with 'review' and 'sentiment' columns.

    Raises:
        FileNotFoundError: If the CSV file is missing.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Place 'IMDB Dataset.csv' inside the dataset/ directory."
        )

    df = pd.read_csv(DATASET_PATH)
    return df


def explore_dataset(df: pd.DataFrame) -> None:
    """
    Display basic dataset information: shape, samples, and missing values.

    Args:
        df: Loaded dataset DataFrame.
    """
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    print("\nSample rows:")
    print(df.head(3).to_string(index=False))
    print("\nMissing values per column:")
    missing = df.isnull().sum()
    print(missing.to_string())
    print(f"\nTotal missing values: {missing.sum()}")
    print("\nSentiment value counts:")
    print(df["sentiment"].value_counts().to_string())
    print("=" * 60)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Drops rows with missing review or sentiment, and removes duplicate reviews.

    Args:
        df: Raw dataset DataFrame.

    Returns:
        Cleaned DataFrame with no missing critical values.
    """
    initial_rows = len(df)
    df = df.dropna(subset=["review", "sentiment"])
    df = df.drop_duplicates(subset=["review"])
    df = df.reset_index(drop=True)

    removed = initial_rows - len(df)
    if removed > 0:
        print(f"Removed {removed:,} rows (missing values or duplicates).")

    return df


def encode_labels(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Encode sentiment labels to numeric values for model training.

    Args:
        df: DataFrame with a 'sentiment' column.

    Returns:
        Tuple of (text series, encoded label series).
    """
    texts = df["review"].astype(str)
    labels = df["sentiment"].map(LABEL_MAP)
    return texts, labels
