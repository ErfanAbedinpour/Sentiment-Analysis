"""
Visualization module for sentiment analysis EDA and model evaluation.

Generates and saves charts to the reports/ directory.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import REPORTS_DIR, ensure_directories


# Consistent plot styling across all figures
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


class Visualizer:
    """Generate and save visualization charts for the sentiment analysis project."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        """
        Initialize visualizer with output directory.

        Args:
            output_dir: Directory for saved figures (default: reports/).
        """
        self.output_dir = output_dir or REPORTS_DIR
        ensure_directories()

    def _save_figure(self, fig: plt.Figure, filename: str) -> str:
        """
        Save figure to output directory.

        Args:
            fig: Matplotlib figure object.
            filename: Output filename.

        Returns:
            Full path to saved file.
        """
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {path}")
        return str(path)

    def plot_sentiment_distribution(self, df: pd.DataFrame) -> str:
        """
        Plot bar chart of positive vs negative review counts.

        Args:
            df: DataFrame with 'sentiment' column.

        Returns:
            Path to saved figure.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        counts = df["sentiment"].value_counts()
        colors = ["#e74c3c", "#2ecc71"]
        bars = ax.bar(
            counts.index,
            counts.values,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
        )

        ax.set_title("Sentiment Distribution in IMDb Reviews", fontsize=14, fontweight="bold")
        ax.set_xlabel("Sentiment", fontsize=12)
        ax.set_ylabel("Number of Reviews", fontsize=12)

        for bar, count in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts.values) * 0.01,
                f"{count:,}",
                ha="center",
                va="bottom",
                fontsize=11,
            )

        total = counts.sum()
        ax.text(
            0.98, 0.95,
            f"Total: {total:,} reviews",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        fig.tight_layout()
        return self._save_figure(fig, "sentiment_distribution.png")

    def plot_dataset_statistics(self, df: pd.DataFrame) -> str:
        """
        Plot review length distribution and summary statistics.

        Args:
            df: DataFrame with 'review' and 'sentiment' columns.

        Returns:
            Path to saved figure.
        """
        df = df.copy()
        df["review_length"] = df["review"].str.len()
        df["word_count"] = df["review"].str.split().str.len()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for sentiment, color in zip(["positive", "negative"], ["#2ecc71", "#e74c3c"]):
            subset = df[df["sentiment"] == sentiment]["word_count"]
            axes[0].hist(
                subset,
                bins=50,
                alpha=0.6,
                label=sentiment.capitalize(),
                color=color,
                edgecolor="black",
                linewidth=0.3,
            )

        axes[0].set_title("Word Count Distribution by Sentiment", fontweight="bold")
        axes[0].set_xlabel("Word Count")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()
        axes[0].set_xlim(0, 500)

        stats = df.groupby("sentiment").agg(
            avg_length=("review_length", "mean"),
            avg_words=("word_count", "mean"),
            count=("review", "count"),
        )
        x = np.arange(len(stats))
        width = 0.35
        axes[1].bar(x - width / 2, stats["avg_words"], width, label="Avg Words", color="#3498db")
        axes[1].bar(x + width / 2, stats["avg_length"] / 10, width, label="Avg Chars (/10)", color="#9b59b6")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(stats.index)
        axes[1].set_title("Average Review Statistics", fontweight="bold")
        axes[1].legend()

        fig.tight_layout()
        return self._save_figure(fig, "dataset_statistics.png")

    def plot_confusion_matrix(
        self,
        confusion_mat: np.ndarray,
        model_name: str,
        filename: Optional[str] = None,
    ) -> str:
        """
        Plot heatmap of confusion matrix.

        Args:
            confusion_mat: 2x2 confusion matrix array.
            model_name: Name of the evaluated model.
            filename: Optional custom output filename.

        Returns:
            Path to saved figure.
        """
        fig, ax = plt.subplots(figsize=(7, 6))
        labels = ["Negative", "Positive"]

        sns.heatmap(
            confusion_mat,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={"label": "Count"},
            linewidths=0.5,
            linecolor="gray",
        )

        ax.set_title(
            f"Confusion Matrix — {model_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)

        fig.tight_layout()
        out_name = filename or "confusion_matrix.png"
        return self._save_figure(fig, out_name)

    def plot_model_comparison(self, comparison_df: pd.DataFrame) -> str:
        """
        Plot grouped bar chart comparing model metrics.

        Args:
            comparison_df: DataFrame with Model and metric columns.

        Returns:
            Path to saved figure.
        """
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(comparison_df))
        width = 0.2
        colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

        for i, metric in enumerate(metrics):
            offset = (i - 1.5) * width
            ax.bar(
                x + offset,
                comparison_df[metric],
                width,
                label=metric,
                color=colors[i],
                edgecolor="black",
                linewidth=0.3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df["Model"], rotation=15, ha="right")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right")
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

        fig.tight_layout()
        return self._save_figure(fig, "model_comparison.png")

    def plot_top_words(
        self,
        vectorizer: TfidfVectorizer,
        X: np.ndarray,
        y: np.ndarray,
        top_n: int = 20,
    ) -> Dict[str, str]:
        """
        Plot top words associated with positive and negative sentiment.

        Args:
            vectorizer: Fitted TF-IDF vectorizer.
            X: Feature matrix (sparse).
            y: Label array.
            top_n: Number of top words to display.

        Returns:
            Dictionary with paths to positive and negative word charts.
        """
        feature_names = vectorizer.get_feature_names_out()
        paths: Dict[str, str] = {}

        for label, sentiment, color in [
            (1, "positive", "#2ecc71"),
            (0, "negative", "#e74c3c"),
        ]:
            mask = y == label
            class_tfidf = X[mask].mean(axis=0)
            if hasattr(class_tfidf, "A1"):
                class_tfidf = class_tfidf.A1
            else:
                class_tfidf = np.asarray(class_tfidf).flatten()

            top_indices = class_tfidf.argsort()[-top_n:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            top_scores = [class_tfidf[i] for i in top_indices]

            fig, ax = plt.subplots(figsize=(10, 8))
            y_pos = np.arange(len(top_words))
            ax.barh(y_pos, top_scores, color=color, edgecolor="black", linewidth=0.3)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_words)
            ax.invert_yaxis()
            ax.set_xlabel("Average TF-IDF Score")
            ax.set_title(
                f"Top {top_n} {sentiment.capitalize()} Words",
                fontsize=14,
                fontweight="bold",
            )

            fig.tight_layout()
            filename = f"top_{sentiment}_words.png"
            paths[sentiment] = self._save_figure(fig, filename)

        return paths

    def generate_all(
        self,
        df: pd.DataFrame,
        comparison_df: pd.DataFrame,
        best_model_name: str,
        confusion_mat: np.ndarray,
        vectorizer: Optional[TfidfVectorizer] = None,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> List[str]:
        """
        Generate all project visualizations.

        Args:
            df: Raw dataset DataFrame.
            comparison_df: Model comparison metrics table.
            best_model_name: Name of best model for confusion matrix title.
            confusion_mat: Confusion matrix of best model.
            vectorizer: Fitted TF-IDF vectorizer for word charts.
            X: Full feature matrix for word analysis.
            y: Full label array for word analysis.

        Returns:
            List of paths to all saved figures.
        """
        print("\nGenerating visualizations...")
        saved: List[str] = []

        saved.append(self.plot_sentiment_distribution(df))
        saved.append(self.plot_dataset_statistics(df))
        saved.append(self.plot_model_comparison(comparison_df))
        saved.append(
            self.plot_confusion_matrix(confusion_mat, best_model_name)
        )

        if vectorizer is not None and X is not None and y is not None:
            word_paths = self.plot_top_words(vectorizer, X, y)
            saved.extend(word_paths.values())

        print(f"Generated {len(saved)} visualization(s).")
        return saved
