# Setup & Usage Guide

This document explains how to install, run, and use the **IMDb Sentiment Analysis** application.

---

## Prerequisites

Before you begin, make sure you have:

| Requirement | Minimum Version |
|-------------|-----------------|
| Python | 3.10+ |
| pip | Latest recommended |

The dataset must already be present at:

```
dataset/IMDB Dataset.csv
```

No internet download is required for the dataset.

---

## 1. Installation

### Step 1 — Open the project directory

```bash
cd /home/erfan/Desktop/ai/project
```

### Step 2 — Create a virtual environment (recommended)

```bash
python3 -m venv venv
```

### Step 3 — Activate the virtual environment

**Linux / macOS:**

```bash
source venv/bin/activate
```

**Windows (Command Prompt):**

```cmd
venv\Scripts\activate
```

**Windows (PowerShell):**

```powershell
venv\Scripts\Activate.ps1
```

When active, your terminal prompt will show `(venv)`.

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs:

- `pandas`, `numpy` — data handling
- `scikit-learn` — machine learning models
- `nltk` — text preprocessing (tokenization, stopwords, lemmatization)
- `matplotlib`, `seaborn` — charts and visualizations
- `joblib` — model saving and loading

### Step 5 — Verify installation

```bash
python -c "import pandas, sklearn, nltk; print('Installation successful')"
```

NLTK language data (tokenizers, stopwords, WordNet) is downloaded automatically the first time you run the application.

---

## 2. Application Overview

The application is a **CLI (Command Line Interface)** built around `main.py`. It supports three modes:

| Mode | Command | Description |
|------|---------|-------------|
| Train | `--train` | Load data, train models, evaluate, save best model |
| Predict | `--predict "text"` | Classify a single review |
| Interactive | `--interactive` | Enter multiple reviews in a loop |

**Important:** You must run `--train` at least once before using `--predict` or `--interactive`, because those modes load the saved model from `models/best_sentiment_model.joblib`.

---

## 3. Training the Model

### Full training (all 50,000 reviews)

```bash
python main.py --train
```

This runs the complete pipeline:

1. Loads `dataset/IMDB Dataset.csv`
2. Displays dataset shape, samples, and missing values
3. Preprocesses all reviews (cleaning, tokenization, lemmatization)
4. Extracts TF-IDF features
5. Trains three models:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Linear SVM
6. Evaluates each model (accuracy, precision, recall, F1, confusion matrix)
7. Selects the best model by F1 score
8. Saves the model to `models/best_sentiment_model.joblib`
9. Generates charts and reports in `reports/`

Full training may take several minutes depending on your hardware.

### Quick training (subset for testing)

```bash
python main.py --train --sample 5000
```

Uses only 5,000 randomly sampled reviews. Faster for testing that everything works.

### Expected console output (excerpt)

```
============================================================
IMDb SENTIMENT ANALYSIS — TRAINING PIPELINE
============================================================
Shape: 50,000 rows × 2 columns
...
MODEL COMPARISON TABLE
============================================================
                 Model  Accuracy  Precision  Recall  F1 Score
   Logistic Regression    0.8900     0.8900  0.8900    0.8900
...
Best model (by F1 Score): Logistic Regression
Model saved to: models/best_sentiment_model.joblib
============================================================
TRAINING COMPLETE
============================================================
```

---

## 4. Predicting Sentiment

After training, classify any movie review.

### Single review from the command line

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

```bash
python main.py --predict "Terrible acting, boring plot, waste of time."
```

**Example output:**

```
Input:  Terrible acting, boring plot, waste of time.
Output: Negative
Confidence: 92.10%

Full result: {'sentiment': 'negative', 'confidence': 0.9210}
```

### Interactive mode

```bash
python main.py --interactive
```

You can enter reviews one at a time. Type `quit`, `exit`, or `q` to stop.

```
============================================================
INTERACTIVE SENTIMENT PREDICTION
Type a movie review and press Enter. Type 'quit' to exit.
============================================================
Loaded model: Logistic Regression

Enter review: This film was absolutely fantastic!
  Sentiment:   Positive
  Confidence:  94.50%

Enter review: I fell asleep halfway through.
  Sentiment:   Negative
  Confidence:  88.30%

Enter review: quit
Goodbye!
```

---

## 5. Using the Python API

You can also call the prediction function directly from your own Python scripts.

```python
from src.predict import predict_sentiment

# Single prediction
result = predict_sentiment("This movie was fantastic!")
print(result)
# {'sentiment': 'positive', 'confidence': 0.95}
```

### Batch predictions

```python
from src.predict import SentimentPredictor

predictor = SentimentPredictor()
reviews = [
    "Best movie I've ever seen!",
    "Complete waste of money.",
    "It was okay, nothing special.",
]

results = predictor.predict_batch(reviews)
for review, result in zip(reviews, results):
    print(f"{result['sentiment']:8} ({result['confidence']:.0%}) — {review}")
```

### Using the preprocessor directly

```python
from src.preprocessing import TextPreprocessor

preprocessor = TextPreprocessor()
cleaned = preprocessor.preprocess(
    "This movie is AMAZING!!! Visit http://example.com. I loved it."
)
print(cleaned)
# movie amazing visit loved
```

---

## 6. Output Files

After training, these files are created:

### Models (`models/`)

| File | Description |
|------|-------------|
| `best_sentiment_model.joblib` | Saved best model + TF-IDF vectorizer |

### Reports (`reports/`)

| File | Description |
|------|-------------|
| `sentiment_distribution.png` | Positive vs negative class balance |
| `dataset_statistics.png` | Review length and word count stats |
| `model_comparison.png` | Bar chart comparing all models |
| `confusion_matrix.png` | Confusion matrix for the best model |
| `top_positive_words.png` | Most indicative positive words |
| `top_negative_words.png` | Most indicative negative words |
| `model_comparison.csv` | Metrics table (Accuracy, Precision, Recall, F1) |
| `academic_report.md` | Full university-style project report |

---

## 7. Command Reference

```bash
# Show help
python main.py --help

# Train on full dataset
python main.py --train

# Train on 5,000 samples (faster)
python main.py --train --sample 5000

# Predict one review
python main.py --predict "Your review text here"

# Interactive prediction loop
python main.py --interactive
```

---

## 8. Typical Workflow

```
1. Install dependencies     →  pip install -r requirements.txt
2. Train the model          →  python main.py --train
3. Check reports/           →  Review charts and model_comparison.csv
4. Test predictions         →  python main.py --predict "..."
5. Use interactively        →  python main.py --interactive
```

---

## 9. Troubleshooting

### `Model not found at models/best_sentiment_model.joblib`

You have not trained the model yet. Run:

```bash
python main.py --train
```

### `Dataset not found at dataset/IMDB Dataset.csv`

Ensure the CSV file exists:

```bash
ls dataset/
```

The file must be named exactly `IMDB Dataset.csv` (with a space).

### NLTK data errors

If you see errors about missing or corrupted NLTK data, run:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

Then retry training.

### `ModuleNotFoundError: No module named 'pandas'` (or similar)

Dependencies are not installed, or the virtual environment is not activated:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Training is very slow

Use a smaller sample for a quick test:

```bash
python main.py --train --sample 3000
```

---

## 10. Project Structure

```
project/
├── dataset/
│   └── IMDB Dataset.csv      # Input data (required)
├── models/
│   └── best_sentiment_model.joblib  # Created after training
├── reports/                  # Charts and reports (created after training)
├── src/
│   ├── preprocessing.py      # Text cleaning pipeline
│   ├── train.py              # Model training
│   ├── evaluate.py           # Metrics and comparison
│   ├── predict.py            # Inference
│   ├── visualization.py      # Chart generation
│   └── utils.py              # Data loading helpers
├── main.py                   # CLI entry point
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview
└── setup.md                  # This guide
```

---

For a full technical overview, methodology, and academic discussion, see `README.md` and `reports/academic_report.md`.
