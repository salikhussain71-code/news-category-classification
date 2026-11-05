# News Category Classification (AG News)

[![CI](https://github.com/YOUR-USERNAME/news-category-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR-USERNAME/news-category-classification/actions)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Keras](https://img.shields.io/badge/Keras-optional-red)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

An end-to-end, reproducible NLP project that classifies news articles into **World, Sports, Business**
and **Sci/Tech**. It covers text preprocessing (tokenisation, stop-word removal, lemmatisation),
**TF-IDF** feature engineering, multiclass classifiers (**Logistic Regression, Linear SVM,
Naive Bayes, Random Forest**), the optional bonus models (**XGBoost, LightGBM, a Keras
feed-forward neural network**), rigorous evaluation and per-category visualisations.

> **Task 2 - Level 1 (NLP Tasks).** Dataset: AG News (Kaggle). Includes all bonus items: word
> frequency bar plots and word clouds per category, a Keras neural network, and XGBoost / LightGBM.

---

## Table of contents
1. [Highlights](#highlights)
2. [Task requirements checklist](#task-requirements-checklist)
3. [Project structure](#project-structure)
4. [Quick start](#quick-start)
5. [Usage](#usage)
6. [Results](#results)
7. [Methodology](#methodology)
8. [Testing & code quality](#testing--code-quality)
9. [Limitations & future work](#limitations--future-work)
10. [License](#license)

## Highlights
* **Clean, modular package** (`src/news_classification/`) - not just a notebook.
* **Leak-free protocol**: train / validation / official-test split. Models are compared and selected on
  the **validation** set; `test.csv` is used only for the final report. TF-IDF is fitted on train only.
* **AG News-aware cleaning**: double-escaped HTML, `#39;` artefacts, optional newswire-prefix removal
  (`--strip-agency`) to measure the "Reuters / AP" shortcut.
* **Up to 7 models** compared on accuracy, macro precision / recall / F1, ROC-AUC and training time.
* **Error analysis**: confusion matrix, per-class F1 and a CSV of the most confident mistakes.
* **Interpretability**: most frequent words and most *discriminative* terms per category.
* Optional **hyper-parameter tuning** (`--tune`), **CLI inference**, a walkthrough **notebook**,
  **unit tests, linting and GitHub Actions CI**.

## Task requirements checklist

| Requirement | Where it is implemented |
|---|---|
| Dataset: AG News (Kaggle) | `data/README.md`, `src/news_classification/data.py` |
| Classify into categories (sports, business, ...) | 4 classes: World, Sports, Business, Sci/Tech |
| Tokenisation, stop-word removal, lemmatisation | `src/news_classification/preprocessing.py` |
| Vectorise with TF-IDF | `src/news_classification/features.py` |
| Multiclass classifier (LR, Random Forest, SVM) | `src/news_classification/models.py` - all three plus Naive Bayes |
| **Bonus:** most frequent words per category (bar plots / word clouds) | `visualize.py` -> `top_words_per_category.png`, `wordclouds_per_category.png` |
| **Bonus:** feed-forward neural network (Keras) | `src/news_classification/nn.py` (`--nn`) |
| **Bonus:** XGBoost / LightGBM | `models.py` (`--xgboost`, `--lightgbm`) |

## Project structure

```
news-category-classification/
├── README.md
├── LICENSE
├── Makefile                      # shortcuts: make train / train-all / test / lint
├── pyproject.toml                # package metadata, pytest & ruff config
├── requirements.txt              # core dependencies
├── requirements-extra.txt        # + XGBoost, LightGBM, TensorFlow/Keras (bonus models)
├── requirements-dev.txt          # + pytest, ruff, jupyter
├── .gitignore
├── .github/workflows/ci.yml      # lint + tests on every push
├── data/
│   ├── README.md                 # how to download the dataset
│   ├── raw/                      # put train.csv and test.csv here (git-ignored)
│   ├── processed/                # cleaned data written by the pipeline (git-ignored)
│   └── sample/                   # synthetic data for smoke tests only
├── docs/methodology.md           # design decisions explained
├── notebooks/
│   └── 01_news_classification.ipynb
├── scripts/                      # generate_sample_data.py, build_notebook.py
├── src/news_classification/
│   ├── config.py                 # paths & constants
│   ├── data.py                   # loading, title+description, label mapping
│   ├── preprocessing.py          # text cleaning
│   ├── features.py               # TF-IDF
│   ├── models.py                 # classifier factory, per-class top terms
│   ├── nn.py                     # Keras feed-forward network (bonus)
│   ├── evaluate.py               # metrics, error analysis
│   ├── visualize.py              # all plots
│   ├── train.py                  # end-to-end training CLI
│   └── predict.py                # inference CLI / API
├── tests/                        # pytest unit & integration tests
├── models/                       # saved models (git-ignored)
└── reports/
    ├── figures/                  # PNG charts generated by training
    └── metrics/                  # CSV / JSON / Markdown metrics, misclassified examples
```

## Quick start

**Requirements:** Python 3.9+ and about 4 GB of free RAM (full 120k-article dataset).

```bash
# 1. Clone
git clone https://github.com/YOUR-USERNAME/news-category-classification.git
cd news-category-classification

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 3. Install
pip install -r requirements.txt
pip install -e .
# Optional bonus models (XGBoost, LightGBM, Keras) - large download:
pip install -r requirements-extra.txt

# 4. Get the data (see data/README.md) -> data/raw/train.csv and data/raw/test.csv
kaggle datasets download -d amananandrai/ag-news-classification-dataset -p data/raw --unzip

# 5. Train, evaluate and generate all figures
python -m news_classification.train
```

NLTK data (`stopwords`, `wordnet`) is downloaded automatically on first run.
No Kaggle account? Download the two CSVs in the browser and place them in `data/raw/`.

## Usage

### Train
```bash
python -m news_classification.train                            # LR, SVM, NB, Random Forest
python -m news_classification.train --models lr svm nb         # fast: skip Random Forest
python -m news_classification.train --xgboost --lightgbm --nn  # + all bonus models
python -m news_classification.train --tune                     # grid-search the best model
python -m news_classification.train --sample 30000             # quick run on a 30k subset
python -m news_classification.train --strip-agency             # drop "Reuters -" / "AP -" prefixes
python -m news_classification.train --help                     # all options
```

> **Runtime guide (full dataset, typical laptop):** preprocessing about 1-2 min; Logistic Regression,
> SVM and Naive Bayes seconds to ~1 min each; **Random Forest several minutes**; XGBoost / LightGBM
> a few minutes; the neural network a few minutes. Use `--models lr svm nb` or `--sample` for quick iterations.

Outputs: `models/news_classifier.joblib`, `reports/metrics/*`, `reports/figures/*.png`, and the
**Results** section of this README (updated automatically).

### Predict
```bash
python -m news_classification.predict "Stocks rally as oil prices fall and earnings beat forecasts"
python -m news_classification.predict --file headlines.txt     # one article per line
python -m news_classification.predict --top 4 "Apple unveils a faster chip for its laptops"
```
Example output (illustrative - scores depend on your trained model):
```
[BUSINESS] Stocks rally as oil prices fall and earnings beat forecasts
            -> Business 9x%, World 3%, Sci/Tech 2%
```

From Python:
```python
from news_classification.predict import NewsClassifier

clf = NewsClassifier()
clf.predict(["Manchester United win the derby 2-1"], top=2)
```

### Notebook
```bash
pip install -r requirements-dev.txt
jupyter notebook notebooks/01_news_classification.ipynb
```

## Results

<!-- RESULTS:START -->
_Run `python -m news_classification.train` on the AG News dataset - this section is then filled in
automatically with your metrics table and figures._
<!-- RESULTS:END -->

**What to expect:** on AG News, TF-IDF with a linear model (Logistic Regression or Linear SVM) typically
reaches roughly 90-92 % test accuracy; Naive Bayes is slightly lower and tree ensembles are usually
a bit lower still. Your exact numbers are written to the table above. The most common mistakes are
between **Business** and **Sci/Tech** - inspect `reports/metrics/misclassified_examples.csv`.

## Methodology

A short summary - see [`docs/methodology.md`](docs/methodology.md) for the reasoning behind every step.

1. **Load** `train.csv` / `test.csv`; combine title + description; shift labels 1-4 to 0-3; drop duplicates.
2. **Clean**: unescape HTML, remove artefacts / URLs, lowercase, regex tokenise, remove stop-words, lemmatise.
3. **Split** the training file 85 / 15 into train / validation (stratified); keep `test.csv` untouched.
4. **Vectorise** with TF-IDF (sub-linear TF, uni+bi-grams, 50k features), fitted on train only.
5. **Train** every model; select the best by validation macro-F1.
6. **Evaluate once** on the official test set; save metrics, error analysis, plots and the model.

## Testing & code quality

```bash
pip install -r requirements-dev.txt
pytest          # 27 tests: preprocessing, data loading, every model, end-to-end CLI
ruff check .    # linting
```
Tests for XGBoost, LightGBM and the Keras network run automatically when those libraries are installed
and are skipped otherwise. GitHub Actions runs lint + tests on Python 3.10, 3.11 and 3.12.

## Limitations & future work
* Bag-of-words cannot capture word meaning or long-range context; categories such as Business and
  Sci/Tech overlap.
* The data is largely 2004-era English newswire; expect lower accuracy on modern or non-news text.
* Ideas: GloVe / fastText embeddings, CNN/LSTM, fine-tuned DistilBERT, SVM probability calibration, a Streamlit demo.

## License
Released under the [MIT License](LICENSE). Dataset licensing is governed by its Kaggle page.
