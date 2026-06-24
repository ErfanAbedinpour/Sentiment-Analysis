"""
Prediction module for sentiment analysis.

Loads the saved model and predicts sentiment for custom review text.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

from src.preprocessing import TextPreprocessor
from src.utils import INVERSE_LABEL_MAP, MODELS_DIR


class SentimentPredictor:
    """
    Load a trained sentiment model and predict on new text.

    Example:
        predictor = SentimentPredictor()
        result = predictor.predict_sentiment("This movie was fantastic!")
        # {"sentiment": "positive", "confidence": 0.95}
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Initialize predictor and load model artifact.

        Args:
            model_path: Path to joblib model file. Uses default if None.
        """
        self.model_path = Path(model_path or MODELS_DIR / "best_sentiment_model.joblib")
        self.preprocessor = TextPreprocessor()
        self.pipeline = None
        self.vectorizer = None
        self.model_name: str = ""
        self._load_model()

    def _load_model(self) -> None:
        """Load pipeline and vectorizer from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Run training first: python main.py --train"
            )

        artifact: Dict[str, Any] = joblib.load(self.model_path)
        self.pipeline = artifact["pipeline"]
        self.vectorizer = artifact["vectorizer"]
        self.model_name = artifact.get("model_name", "unknown")

    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for a single review text.

        Args:
            text: Raw movie review string.

        Returns:
            Dictionary with 'sentiment' (positive/negative) and
            'confidence' (probability or decision score).
        """
        if self.pipeline is None or self.vectorizer is None:
            raise RuntimeError("Model not loaded.")

        cleaned = self.preprocessor.preprocess(text)
        features = self.vectorizer.transform([cleaned])

        prediction = int(self.pipeline.predict(features)[0])
        sentiment = INVERSE_LABEL_MAP[prediction]

        confidence = self._get_confidence(features, prediction)

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
        }

    def _get_confidence(self, features: Any, prediction: int) -> float:
        """
        Extract prediction confidence from the classifier.

        Uses predict_proba when available; otherwise maps decision_function
        scores to a 0-1 range via sigmoid.

        Args:
            features: TF-IDF feature vector.
            prediction: Predicted class label (0 or 1).

        Returns:
            Confidence score between 0 and 1.
        """
        clf = self.pipeline.named_steps["clf"]

        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(features)[0]
            return float(proba[prediction])

        if hasattr(clf, "decision_function"):
            scores = clf.decision_function(features)[0]
            # Sigmoid mapping for binary SVM decision scores
            confidence = 1.0 / (1.0 + np.exp(-scores))
            return float(confidence)

        return 1.0

    def predict_batch(self, texts: list[str]) -> list[Dict[str, Any]]:
        """
        Predict sentiment for multiple review texts.

        Args:
            texts: List of raw review strings.

        Returns:
            List of prediction dictionaries.
        """
        return [self.predict_sentiment(t) for t in texts]


def predict_sentiment(text: str, model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to predict sentiment for a single text.

    Args:
        text: Raw movie review string.
        model_path: Optional path to saved model.

    Returns:
        Dictionary with sentiment and confidence.
    """
    predictor = SentimentPredictor(model_path=model_path)
    return predictor.predict_sentiment(text)
