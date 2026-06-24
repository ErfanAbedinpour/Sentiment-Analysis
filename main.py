#!/usr/bin/env python3
"""
Sentiment Analysis CLI — IMDb Movie Reviews

Train, evaluate, and predict sentiment (positive/negative) on movie reviews.

Usage:
    python main.py --train              # Full training pipeline
    python main.py --predict "text"     # Predict single review
    python main.py --interactive        # Interactive prediction mode
    python main.py --train --sample 5000  # Train on subset (faster demo)
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import ModelEvaluator
from src.predict import SentimentPredictor, predict_sentiment
from src.preprocessing import TextPreprocessor, get_sample_preprocessing_demo
from src.train import ModelTrainer
from src.utils import (
    REPORTS_DIR,
    encode_labels,
    ensure_directories,
    explore_dataset,
    handle_missing_values,
    load_dataset,
)
from src.visualization import Visualizer


def run_training_pipeline(sample_size: int | None = None) -> None:
    """
    Execute the full training, evaluation, and visualization pipeline.

    Args:
        sample_size: Optional limit on number of samples for faster runs.
    """
    ensure_directories()

    print("\n" + "=" * 60)
    print("IMDb SENTIMENT ANALYSIS — TRAINING PIPELINE")
    print("=" * 60)

    # --- Data Loading ---
    df = load_dataset()
    explore_dataset(df)
    df = handle_missing_values(df)

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"\nUsing sample of {sample_size:,} reviews for training.")

    # --- Preprocessing Demo ---
    preprocessor = TextPreprocessor()
    get_sample_preprocessing_demo(preprocessor)

    # --- EDA Visualizations ---
    visualizer = Visualizer()
    print("\nGenerating EDA visualizations...")
    visualizer.plot_sentiment_distribution(df)
    visualizer.plot_dataset_statistics(df)

    # --- Feature Engineering & Training ---
    texts, labels = encode_labels(df)
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_data(texts, labels)
    trainer.train_all()

    # --- Evaluation ---
    evaluator = ModelEvaluator()
    evaluator.evaluate_all(trainer.pipelines, X_test, y_test)
    comparison_table = evaluator.print_comparison_table()
    evaluator.save_comparison_table(str(REPORTS_DIR / "model_comparison.csv"))

    # --- Best Model Selection & Saving ---
    eval_results = {
        name: {
            "f1_score": m["f1_score"],
            "accuracy": m["accuracy"],
        }
        for name, m in evaluator.results.items()
    }
    best_name = trainer.select_best_model(eval_results)
    trainer.save_best_model()

    # --- Post-training Visualizations ---
    best_cm = evaluator.results[best_name]["confusion_matrix"]
    visualizer.generate_all(
        df=df,
        comparison_df=comparison_table,
        best_model_name=best_name,
        confusion_mat=best_cm,
        vectorizer=trainer.vectorizer,
        X=trainer.vectorizer.transform(
            trainer.preprocessor.preprocess_batch(texts.tolist())
        ),
        y=labels.values,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model: {best_name}")
    print("Model saved: models/best_sentiment_model.joblib")
    print("Reports saved: reports/")
    print("=" * 60)


def run_prediction(text: str) -> None:
    """
    Predict sentiment for a single review and print result.

    Args:
        text: Raw review text.
    """
    result = predict_sentiment(text)
    sentiment_label = result["sentiment"].capitalize()
    print(f"\nInput:  {text}")
    print(f"Output: {sentiment_label}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nFull result: {result}")


def run_interactive_mode() -> None:
    """Allow user to enter custom reviews for sentiment prediction."""
    print("\n" + "=" * 60)
    print("INTERACTIVE SENTIMENT PREDICTION")
    print("Type a movie review and press Enter. Type 'quit' to exit.")
    print("=" * 60)

    try:
        predictor = SentimentPredictor()
        print(f"Loaded model: {predictor.model_name}")
    except FileNotFoundError as exc:
        print(f"\nError: {exc}")
        sys.exit(1)

    while True:
        try:
            text = input("\nEnter review: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if text.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not text:
            print("Please enter a non-empty review.")
            continue

        result = predictor.predict_sentiment(text)
        sentiment = result["sentiment"].capitalize()
        confidence = result["confidence"]
        print(f"  Sentiment:   {sentiment}")
        print(f"  Confidence:  {confidence:.2%}")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="IMDb Movie Review Sentiment Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --train
  python main.py --train --sample 5000
  python main.py --predict "This movie was fantastic!"
  python main.py --interactive
        """,
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run full training, evaluation, and model saving pipeline",
    )
    parser.add_argument(
        "--predict",
        type=str,
        metavar="TEXT",
        help="Predict sentiment for a single review",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter custom reviews interactively",
    )
    parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="Use only N samples for training (faster demo runs)",
    )
    return parser


def main() -> None:
    """Main entry point for the CLI application."""
    parser = build_parser()
    args = parser.parse_args()

    if args.train:
        run_training_pipeline(sample_size=args.sample)
    elif args.predict:
        run_prediction(args.predict)
    elif args.interactive:
        run_interactive_mode()
    else:
        parser.print_help()
        print(
            "\nNo action specified. Use --train, --predict, or --interactive."
        )


if __name__ == "__main__":
    main()
