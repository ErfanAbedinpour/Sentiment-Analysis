"""
Model training module for sentiment classification.

Trains and compares Multinomial Naive Bayes, Logistic Regression,
and Linear SVM using TF-IDF features.
"""

from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocessing import TextPreprocessor
from src.utils import MODELS_DIR, ensure_directories


# TF-IDF is suitable for sentiment analysis because:
# 1. It captures word importance relative to the corpus (not just frequency).
# 2. It down-weights common words that appear in both positive and negative reviews.
# 3. It produces sparse, high-dimensional features ideal for text classification.
# 4. It works well with linear models (NB, LR, SVM) on bag-of-words representations.


class ModelTrainer:
    """
    Train and compare multiple sentiment classification models.

    Uses TF-IDF vectorization combined with Multinomial Naive Bayes,
    Logistic Regression, and Linear SVM.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        max_features: int = 10000,
        cv_folds: int = 5,
    ) -> None:
        """
        Initialize trainer configuration.

        Args:
            test_size: Fraction of data held out for testing.
            random_state: Seed for reproducibility.
            max_features: Maximum TF-IDF vocabulary size.
            cv_folds: Number of cross-validation folds.
        """
        self.test_size = test_size
        self.random_state = random_state
        self.max_features = max_features
        self.cv_folds = cv_folds
        self.preprocessor = TextPreprocessor()
        self.vectorizer: TfidfVectorizer | None = None
        self.models: Dict[str, Any] = {}
        self.pipelines: Dict[str, Pipeline] = {}
        self.best_model_name: str = ""
        self.best_pipeline: Pipeline | None = None
        self.X_train: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.y_test: np.ndarray | None = None
        self.cv_results: Dict[str, Dict[str, float]] = {}

    def _build_vectorizer(self) -> TfidfVectorizer:
        """Create a TF-IDF vectorizer with standard parameters for text."""
        return TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )

    def prepare_data(
        self, texts: pd.Series, labels: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess text and split into train/test sets.

        Args:
            texts: Raw review texts.
            labels: Encoded sentiment labels (0/1).

        Returns:
            Tuple of (X_train, X_test, y_train, y_test) as TF-IDF matrices.
        """
        print("\nPreprocessing reviews...")
        cleaned = self.preprocessor.preprocess_batch(texts.tolist())

        self.vectorizer = self._build_vectorizer()
        X = self.vectorizer.fit_transform(cleaned)
        y = labels.values

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        print(f"Training set: {X_train.shape[0]:,} samples")
        print(f"Test set:     {X_test.shape[0]:,} samples")
        print(f"Feature dimension: {X_train.shape[1]:,}")

        return X_train, X_test, y_train, y_test

    def _get_model_configs(self) -> Dict[str, Any]:
        """Return model instances with hyperparameter grids for tuning."""
        return {
            "Multinomial Naive Bayes": {
                "model": MultinomialNB(),
                "params": {"clf__alpha": [0.1, 0.5, 1.0]},
            },
            "Logistic Regression": {
                "model": LogisticRegression(
                    max_iter=1000,
                    random_state=self.random_state,
                    solver="lbfgs",
                ),
                "params": {"clf__C": [0.1, 1.0, 10.0]},
            },
            "Linear SVM": {
                "model": LinearSVC(
                    random_state=self.random_state,
                    max_iter=2000,
                ),
                "params": {"clf__C": [0.1, 1.0, 10.0]},
            },
        }

    def train_all(self) -> Dict[str, Pipeline]:
        """
        Train all models with cross-validated hyperparameter search.

        Returns:
            Dictionary mapping model names to fitted pipelines.
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Call prepare_data() before train_all().")

        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)

        configs = self._get_model_configs()

        for name, config in configs.items():
            print(f"\nTraining: {name}")

            pipeline = Pipeline([("clf", config["model"])])

            grid_search = GridSearchCV(
                pipeline,
                config["params"],
                cv=self.cv_folds,
                scoring="f1",
                n_jobs=-1,
            )
            grid_search.fit(self.X_train, self.y_train)

            best_pipeline = grid_search.best_estimator_
            self.pipelines[name] = best_pipeline
            self.models[name] = best_pipeline.named_steps["clf"]

            cv_scores = cross_val_score(
                best_pipeline,
                self.X_train,
                self.y_train,
                cv=self.cv_folds,
                scoring="f1",
            )
            self.cv_results[name] = {
                "cv_f1_mean": float(cv_scores.mean()),
                "cv_f1_std": float(cv_scores.std()),
                "best_params": grid_search.best_params_,
            }

            print(f"  Best params: {grid_search.best_params_}")
            print(
                f"  CV F1: {cv_scores.mean():.4f} "
                f"(+/- {cv_scores.std() * 2:.4f})"
            )

        print("=" * 60)
        return self.pipelines

    def select_best_model(self, evaluation_results: Dict[str, Dict[str, float]]) -> str:
        """
        Select the best model based on test-set F1 score.

        Args:
            evaluation_results: Metrics per model from evaluation module.

        Returns:
            Name of the best-performing model.
        """
        best_name = max(
            evaluation_results,
            key=lambda k: evaluation_results[k]["f1_score"],
        )
        self.best_model_name = best_name
        self.best_pipeline = self.pipelines[best_name]
        print(f"\nBest model (by F1 Score): {best_name}")
        return best_name

    def save_best_model(self) -> str:
        """
        Save the best model pipeline and vectorizer to disk.

        Returns:
            Path to the saved model artifact.
        """
        if self.best_pipeline is None or self.vectorizer is None:
            raise RuntimeError("No best model selected. Train and evaluate first.")

        ensure_directories()

        artifact = {
            "pipeline": self.best_pipeline,
            "vectorizer": self.vectorizer,
            "preprocessor_config": {"language": "english"},
            "model_name": self.best_model_name,
        }

        model_path = MODELS_DIR / "best_sentiment_model.joblib"
        joblib.dump(artifact, model_path)
        print(f"Model saved to: {model_path}")
        return str(model_path)

    def get_feature_names(self) -> List[str]:
        """Return TF-IDF feature (vocabulary) names."""
        if self.vectorizer is None:
            return []
        return list(self.vectorizer.get_feature_names_out())
