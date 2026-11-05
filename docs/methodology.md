# Methodology

## 1. Problem
Assign a news article (title + description) to one of four categories: **World, Sports, Business,
Sci/Tech**. This is single-label **multiclass** classification (4 classes, balanced).

## 2. Data protocol (no leakage)
| Split | Source | Used for |
|---|---|---|
| Train (85 % of `train.csv`) | stratified | fitting vectoriser + models |
| Validation (15 % of `train.csv`) | stratified | **model selection** (macro-F1) |
| Test (`test.csv`) | official | final report **only** |

* Title and description are concatenated into one text field.
* Exact duplicates are removed; articles present in both files are removed from training.
* The TF-IDF vectoriser is fitted **only on the training split**.
* Selecting the saved model on the validation set (not the test set) keeps the reported test
  scores an unbiased estimate.

## 3. Preprocessing (`preprocessing.py`)
| Step | Reason |
|---|---|
| Double HTML-unescape + tag removal | AG News contains escaped tags such as `&lt;b&gt;` |
| Remove `#39;`-style entity leftovers, URLs, backslashes | Dataset artefacts / noise |
| Lower-casing | "Market" and "market" are one feature |
| Regex tokenisation (letters only) | Simple, dependency-free tokenisation; drops digits and punctuation |
| Stop-word removal (NLTK English) | Removes uninformative function words. Unlike sentiment analysis, negations are irrelevant for topic classification |
| WordNet lemmatisation (cached) | `stocks/stock`, `rose/rise` collapse into a single feature |
| *(optional)* `--strip-agency` | Removes "Reuters -", "AP -" prefixes |

**Newswire shortcut:** many AG News articles start with an agency tag (e.g. "Reuters", "AP"), which
correlates with the label (Reuters is frequent in Business). The default keeps them, as in the
standard benchmark. Run with `--strip-agency` to measure how much of the accuracy depends on them.

## 4. Feature engineering
**TF-IDF** with sub-linear term frequency, uni+bi-grams, `min_df=2`, `max_df=0.9`, `max_features=50,000`.
Bi-grams capture phrases such as *"wall street"* or *"world cup"*.

## 5. Models (`models.py`)
| Model | Notes |
|---|---|
| Logistic Regression | Multinomial, `C=10`; calibrated probabilities; interpretable weights |
| Linear SVM | Typically the strongest classical text classifier; probabilities are a softmax of margins (relative scores) |
| Complement Naive Bayes | Fast baseline, designed for text |
| Random Forest | 100 trees; included as the non-linear tree baseline (slowest on sparse text) |
| XGBoost *(bonus)* | Gradient boosting, `hist` tree method |
| LightGBM *(bonus)* | Gradient boosting, leaf-wise |
| Keras feed-forward NN *(bonus)* | 50k -> 256 -> 128 -> 4, dropout, Adam, early stopping on validation loss |

The neural network is reported for comparison; the model saved for inference is the best
**scikit-learn-compatible** model (single `joblib` file). The network is saved separately as
`models/news_nn.keras`.

## 6. Evaluation
Accuracy, macro precision / recall / F1, weighted F1, one-vs-rest ROC-AUC; per-class report;
confusion matrix (counts and row-normalised); and an **error-analysis file**
(`reports/metrics/misclassified_examples.csv`, most confident mistakes first).
Macro-F1 is used for model selection because it weights all classes equally.

`--tune` runs `GridSearchCV` (3-fold, macro-F1) for the best model's key hyper-parameters on the
training split. The TF-IDF matrix is fitted once on the whole training split, so the inner CV
scores are very slightly optimistic; validation and test scores are unaffected.

## 7. Interpretation
* Most frequent words per category (bar plots and word clouds). Frequent words are descriptive
  but not necessarily discriminative.
* Highest-weight terms per class from the best linear model / Naive Bayes (`top_model_features.png`).

## 8. Known limitations
* **Business vs Sci/Tech** and **World vs Business** are the usual confusion pairs: many articles
  genuinely sit between categories.
* Bag-of-words ignores word order beyond bi-grams and has no notion of word meaning
  (pre-trained embeddings or transformers would help).
* Categories are coarse and the data is mostly from 2004-era English newswire; performance on
  modern or non-news text will be lower.

## 9. Extensions
Word embeddings (GloVe / fastText), CNN/LSTM, fine-tuning DistilBERT, probability calibration of the SVM,
ensembling, a Streamlit demo.
