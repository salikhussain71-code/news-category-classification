"""Central configuration: paths and constants shared across the project."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("NC_PROJECT_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_TRAIN_PATH = DATA_DIR / "sample" / "sample_train.csv"
SAMPLE_TEST_PATH = DATA_DIR / "sample" / "sample_test.csv"
# Kaggle: amananandrai/ag-news-classification-dataset  (train.csv, test.csv)
DEFAULT_TRAIN_PATH = RAW_DIR / "train.csv"
DEFAULT_TEST_PATH = RAW_DIR / "test.csv"
CLEAN_CACHE_PATH = PROCESSED_DIR / "clean_train.csv"

MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "news_classifier.joblib"
NN_MODEL_PATH = MODELS_DIR / "news_nn.keras"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
README_PATH = PROJECT_ROOT / "README.md"

RANDOM_STATE = 42
VAL_SIZE = 0.15     # share of the training file held out for model selection
TEST_SIZE = 0.20    # only used when no separate test file is supplied

# AG News class index (1-4 in the CSV) -> readable name. Internally labels are 0-3.
CLASS_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
