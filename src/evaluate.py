"""
Model evaluation module for sentiment classification.

Computes accuracy, precision, recall, F1, confusion matrix,
and classification reports for all trained models.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


class ModelEvaluator:
    """Evaluate trained sentiment classification models on a test set."""

    def __init__(self) -> None:
        """Initialize evaluator with empty results storage."""
        self.results: Dict[str, Dict[str, Any]] = {}
        self.predictions: Dict[str, np.ndarray] = {}

    def evaluate_model(
        self,
        name: str,
        pipeline: Pipeline,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate a single model and store metrics.

        Args:
            name: Model identifier.
            pipeline: Fitted sklearn pipeline.
            X_test: Test feature matrix.
            y_test: True test labels.

        Returns:
            Dictionary of evaluation metrics.
        """
        y_pred = pipeline.predict(X_test)
        self.predictions[name] = y_pred

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(
                y_test,
                y_pred,
                target_names=["negative", "positive"],
                zero_division=0,
            ),
        }
        self.results[name] = metrics
        return metrics

    def evaluate_all(
        self,
        pipelines: Dict[str, Pipeline],
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all models and return combined results.

        Args:
            pipelines: Mapping of model name to fitted pipeline.
            X_test: Test feature matrix.
            y_test: True test labels.

        Returns:
            Dictionary of metrics per model.
        """
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        for name, pipeline in pipelines.items():
            print(f"\n--- {name} ---")
            metrics = self.evaluate_model(name, pipeline, X_test, y_test)
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1_score']:.4f}")
            print(f"\n  Confusion Matrix:\n{metrics['confusion_matrix']}")
            print(f"\n  Classification Report:\n{metrics['classification_report']}")

        print("=" * 60)
        return self.results

    def get_comparison_table(self) -> pd.DataFrame:
        """
        Build a comparison table of all model metrics.

        Returns:
            DataFrame with models as rows and metrics as columns.
        """
        rows: List[Dict[str, float]] = []
        for name, metrics in self.results.items():
            rows.append(
                {
                    "Model": name,
                    "Accuracy": metrics["accuracy"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1 Score": metrics["f1_score"],
                }
            )

        df = pd.DataFrame(rows)
        df = df.sort_values("F1 Score", ascending=False).reset_index(drop=True)
        return df

    def print_comparison_table(self) -> pd.DataFrame:
        """
        Print formatted comparison table to console.

        Returns:
            The comparison DataFrame.
        """
        table = self.get_comparison_table()
        print("\n" + "=" * 60)
        print("MODEL COMPARISON TABLE")
        print("=" * 60)
        print(table.to_string(index=False, float_format="%.4f"))
        print("=" * 60)
        return table

    def save_comparison_table(self, path: str) -> None:
        """
        Save comparison table to CSV.

        Args:
            path: Output file path.
        """
        table = self.get_comparison_table()
        table.to_csv(path, index=False)
        print(f"Comparison table saved to: {path}")
