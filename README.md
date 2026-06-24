# IMDb Movie Review Sentiment Analysis

A university-level **Sentiment Analysis** project that classifies IMDb movie reviews as **Positive** or **Negative** using Python and Scikit-Learn.

## Project Overview

This project implements a complete NLP pipeline for binary sentiment classification:

1. Load and explore the IMDb dataset
2. Preprocess raw text (cleaning, tokenization, lemmatization)
3. Extract TF-IDF features
4. Train and compare three classifiers (Naive Bayes, Logistic Regression, Linear SVM)
5. Evaluate models with standard metrics
6. Select and save the best model
7. Predict sentiment on custom reviews

## Project Structure

```
sentiment-analysis/
├── dataset/
│   └── IMDB Dataset.csv       # IMDb movie reviews (50,000 samples)
├── models/
│   └── best_sentiment_model.joblib  # Saved after training
├── reports/
│   ├── sentiment_distribution.png
│   ├── dataset_statistics.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── top_positive_words.png
│   ├── top_negative_words.png
│   ├── model_comparison.csv
│   └── academic_report.md
├── src/
│   ├── preprocessing.py       # Text cleaning and NLP pipeline
│   ├── train.py               # Model training and comparison
│   ├── evaluate.py            # Metrics and comparison tables
│   ├── predict.py             # Inference on new reviews
│   ├── visualization.py       # Charts and EDA plots
│   └── utils.py               # Data loading and utilities
├── main.py                    # CLI entry point
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone or navigate to the project directory

```bash
cd sentiment-analysis
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

NLTK data (tokenizers, stopwords, WordNet) is downloaded automatically on first run.

## Dataset Description

| Property | Value |
|----------|-------|
| **Source** | IMDb Movie Reviews |
| **File** | `dataset/IMDB Dataset.csv` |
| **Size** | 50,000 reviews |
| **Columns** | `review` (text), `sentiment` (positive/negative) |
| **Balance** | 50% positive, 50% negative |

The dataset contains raw movie review text with HTML tags, punctuation, and varied vocabulary — typical of real-world NLP data.

## Preprocessing Pipeline

Each review goes through the following steps in `src/preprocessing.py`:

| Step | Description |
|------|-------------|
| Lowercasing | Normalize case (`Amazing` → `amazing`) |
| URL removal | Strip `http://` and `www.` links |
| HTML removal | Remove tags like `<br />` |
| Number removal | Remove digits |
| Punctuation removal | Keep only alphabetic characters |
| Tokenization | Split into word tokens (NLTK) |
| Stopword removal | Remove common words (`the`, `is`, `and`) |
| Lemmatization | Reduce words to base form (`loved` → `love`) |

## Feature Engineering — TF-IDF

**TF-IDF (Term Frequency–Inverse Document Frequency)** converts text into numerical vectors.

**Why TF-IDF for sentiment analysis?**

- **Term Frequency** captures how often a word appears in a document.
- **Inverse Document Frequency** penalizes words that appear in many documents.
- Words like *"amazing"* or *"terrible"* get high scores; common words like *"the"* get low scores.
- Produces sparse, high-dimensional features well-suited for linear classifiers.

Configuration used:

- `max_features=10000`
- `ngram_range=(1, 2)` — unigrams and bigrams
- `sublinear_tf=True` — log-scaled term frequency

## Models Compared

| Model | Description |
|-------|-------------|
| **Multinomial Naive Bayes** | Probabilistic classifier; fast and effective for text |
| **Logistic Regression** | Linear classifier with probability outputs |
| **Linear SVM** | Maximum-margin linear separator |

All models use **5-fold cross-validation** with **GridSearchCV** for hyperparameter tuning. The best model is selected by **F1 Score** on the test set.

## How to Train

Run the full pipeline:

```bash
python main.py --train
```

For a faster demo on a subset:

```bash
python main.py --train --sample 5000
```

Training will:

- Load and explore the dataset
- Preprocess all reviews
- Train three models with cross-validation
- Print evaluation metrics and a comparison table
- Save the best model to `models/best_sentiment_model.joblib`
- Generate charts in `reports/`

## How to Predict

### Single review

```bash
python main.py --predict "This movie is amazing and I loved every minute of it."
```

**Example output:**

```
Input:  This movie is amazing and I loved every minute of it.
Output: Positive
Confidence: 95.23%

Full result: {'sentiment': 'positive', 'confidence': 0.9523}
```

### Interactive mode

```bash
python main.py --interactive
```

### Python API

```python
from src.predict import predict_sentiment

result = predict_sentiment("This movie was fantastic!")
print(result)
# {'sentiment': 'positive', 'confidence': 0.95}
```

## Evaluation Metrics

For each model, the following metrics are computed:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | True positives / predicted positives |
| **Recall** | True positives / actual positives |
| **F1 Score** | Harmonic mean of precision and recall |
| **Confusion Matrix** | TP, TN, FP, FN breakdown |
| **Classification Report** | Per-class precision, recall, F1 |

Results are displayed in a comparison table and saved to `reports/model_comparison.csv`.

## Generated Reports

After training, the following files appear in `reports/`:

- `sentiment_distribution.png` — Class balance chart
- `dataset_statistics.png` — Review length statistics
- `model_comparison.png` — Bar chart of model metrics
- `confusion_matrix.png` — Best model confusion matrix
- `top_positive_words.png` — Most indicative positive words
- `top_negative_words.png` — Most indicative negative words
- `model_comparison.csv` — Tabular metrics
- `academic_report.md` — Full university-style report

## Academic Report

A complete academic report is available at `reports/academic_report.md`, covering abstract, methodology, experimental results, and conclusions.

## License

This project is for educational purposes.
