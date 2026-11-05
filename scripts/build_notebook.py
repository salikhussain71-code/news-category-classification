"""Builds notebooks/01_news_classification.ipynb (kept as a script so the notebook is reproducible)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s.strip())  # noqa: E731
code = lambda s: nbf.v4.new_code_cell(s.strip())  # noqa: E731

nb.cells = [
md("""
# News Category Classification (AG News)
**Task 2 - NLP Level 1** | Dataset: AG News (Kaggle) | 4 classes: World, Sports, Business, Sci/Tech

The reusable code lives in `src/news_classification/`; run `python -m news_classification.train` for the full
automated pipeline. This notebook walks through it step by step:

1) load -> 2) explore -> 3) preprocess -> 4) split -> 5) TF-IDF -> 6) compare classifiers ->
7) evaluate the best -> 8) words per category (bonus) -> 9) discriminative terms -> 10) optional boosting / neural net -> 11) predict
"""),
code("""
import sys
import tempfile
import time
from pathlib import Path

try:
    import news_classification  # noqa: F401  (works after `pip install -e .`)
except ImportError:
    sys.path.insert(0, str(Path.cwd().parent / "src"))

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, display
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from news_classification import config, visualize
from news_classification.data import load_dataset
from news_classification.evaluate import compute_metrics
from news_classification.features import build_vectorizer
from news_classification.models import MODEL_NAMES, build_classifier, class_probabilities, top_features_per_class
from news_classification.preprocessing import TextPreprocessor

pd.set_option("display.max_colwidth", 140)
FIG_DIR = Path(tempfile.mkdtemp())   # notebook figures are temporary; the CLI saves the final ones
NAMES = [config.CLASS_NAMES[i] for i in sorted(config.CLASS_NAMES)]
"""),
md("## 1. Load the data"),
code("""
if config.DEFAULT_TRAIN_PATH.exists():
    df = load_dataset(config.DEFAULT_TRAIN_PATH)
    test_df = load_dataset(config.DEFAULT_TEST_PATH) if config.DEFAULT_TEST_PATH.exists() else None
else:
    print("AG News CSVs not found in data/raw/ - using the SYNTHETIC sample (results are NOT meaningful).")
    print("See data/README.md to download the real dataset.")
    df, test_df = load_dataset(config.SAMPLE_TRAIN_PATH), load_dataset(config.SAMPLE_TEST_PATH)

print(f"{len(df):,} training articles" + (f", {len(test_df):,} test articles" if test_df is not None else ""))
df.head()
"""),
md("## 2. Explore"),
code("""
display(df["label"].map(config.CLASS_NAMES).value_counts().to_frame("articles"))
display(Image(str(visualize.plot_class_distribution(df["label"], FIG_DIR))))
display(Image(str(visualize.plot_length_distribution(df["text"], df["label"], FIG_DIR))))
"""),
md("## 3. Preprocess\nTokenise (letters only), lowercase, remove HTML artefacts and stop-words, lemmatise."),
code("""
pre = TextPreprocessor()
example = "Reuters - Wall Street stocks rose on Tuesday as oil prices fell &lt;b&gt; #36;5 billion deal"
print("RAW  :", example)
print("CLEAN:", pre(example))
"""),
code("""
df["clean_text"] = pre.transform_series(df["text"])
if test_df is not None:
    test_df["clean_text"] = pre.transform_series(test_df["text"])
df[["text", "clean_text", "label"]].head()
"""),
md("## 4. Split\nTrain / validation (stratified) from the training file; the test set is kept aside for the end."),
code("""
if test_df is None:
    df, test_df = train_test_split(df, test_size=config.TEST_SIZE, stratify=df["label"], random_state=config.RANDOM_STATE)
train_df, val_df = train_test_split(df, test_size=config.VAL_SIZE, stratify=df["label"], random_state=config.RANDOM_STATE)
print(f"train={len(train_df):,}  validation={len(val_df):,}  test={len(test_df):,}")
"""),
md("## 5. TF-IDF features (fitted on the training split only)"),
code("""
vec = build_vectorizer()
X_train = vec.fit_transform(train_df["clean_text"])
X_val, X_test = vec.transform(val_df["clean_text"]), vec.transform(test_df["clean_text"])
y_train, y_val, y_test = train_df["label"], val_df["label"], test_df["label"]
print("TF-IDF matrix:", X_train.shape)
"""),
md("## 6. Compare classifiers\nSelection is done on the **validation** set; test metrics are reported alongside."),
code("""
rows, fitted = [], {}
for kind in ("lr", "svm", "nb", "rf"):
    clf = build_classifier(kind)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    secs = time.perf_counter() - t0
    pv, pt = class_probabilities(clf, X_val), class_probabilities(clf, X_test)
    rows.append({"model": MODEL_NAMES[kind],
                 "val_f1_macro": compute_metrics(y_val, pv.argmax(1), pv)["f1_macro"],
                 **{k: v for k, v in compute_metrics(y_test, pt.argmax(1), pt).items()},
                 "fit_seconds": round(secs, 1)})
    fitted[kind] = clf
comparison = pd.DataFrame(rows).sort_values("val_f1_macro", ascending=False).reset_index(drop=True)
comparison.round(4)
"""),
md("## 7. Evaluate the best model on the test set"),
code("""
best_kind = {v: k for k, v in MODEL_NAMES.items()}[comparison.loc[0, "model"]]
best = fitted[best_kind]
pred = class_probabilities(best, X_test).argmax(axis=1)
print("Best model:", MODEL_NAMES[best_kind])
print(classification_report(y_test, pred, target_names=NAMES, digits=4, zero_division=0))
ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=NAMES, cmap="Blues", xticks_rotation=30)
plt.title(MODEL_NAMES[best_kind]); plt.grid(False); plt.show()
"""),
md("## 8. Most frequent words per category (bonus)"),
code("""
freqs = {k: visualize.word_frequencies(train_df.loc[y_train == k, "clean_text"]) for k in config.CLASS_NAMES}
display(Image(str(visualize.plot_top_words(freqs, FIG_DIR))))
display(Image(str(visualize.plot_wordclouds(freqs, FIG_DIR))))
"""),
md("## 9. Most discriminative terms\nFrequent words are shared across categories; model weights show what truly separates them."),
code("""
feats = top_features_per_class(fitted["lr"], vec.get_feature_names_out())
display(Image(str(visualize.plot_top_features(feats, "Logistic Regression", FIG_DIR))))
"""),
md("## 10. Optional bonus models: XGBoost, LightGBM and a Keras neural network\nThese need `pip install -r requirements-extra.txt` and are skipped if the libraries are missing."),
code("""
extras = []
for kind in ("xgb", "lgbm"):
    try:
        clf = build_classifier(kind)
    except ImportError as err:
        print("Skipping", MODEL_NAMES[kind], "-", err)
        continue
    clf.fit(X_train, y_train)
    pt = class_probabilities(clf, X_test)
    extras.append({"model": MODEL_NAMES[kind], **compute_metrics(y_test, pt.argmax(1), pt)})

try:
    from news_classification import nn
    net, history, secs = nn.train_network(X_train, y_train, X_val, y_val, epochs=6, verbose=False)
    pt = nn.predict_proba(net, X_test)
    extras.append({"model": "Neural Network (Keras)", **compute_metrics(y_test, pt.argmax(1), pt)})
    display(Image(str(visualize.plot_training_history(history, FIG_DIR))))
except ImportError as err:
    print("Skipping neural network -", err)

pd.DataFrame(extras).round(4) if extras else print("No optional libraries installed.")
"""),
md("## 11. Try your own headline"),
code("""
def classify(text, top=2):
    X = vec.transform([pre(text)])
    p = class_probabilities(fitted["lr"], X)[0]
    ranked = sorted(zip(NAMES, p), key=lambda t: -t[1])[:top]
    return " | ".join(f"{n} {s:.0%}" for n, s in ranked) + "   <-   " + text

for headline in ["Stocks rally as oil prices fall and earnings beat forecasts",
                 "Manchester United win the derby with a late goal",
                 "Researchers unveil a faster chip for wireless networks",
                 "Peace talks resume as the government announces a ceasefire"]:
    print(classify(headline))
"""),
md("""
## Conclusions
* TF-IDF + a linear classifier is a strong, fast baseline for topic classification.
* Compare the table in step 6 (and step 10) to see how model families differ in accuracy and training time.
* The usual confusions are Business vs Sci/Tech and World vs Business - see the confusion matrix.
* **Next steps:** word embeddings, CNN/LSTM, or fine-tuning a transformer such as DistilBERT.

> If you ran this on the synthetic sample, re-run it on the real AG News dataset before drawing conclusions.
"""),
]
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, "notebooks/01_news_classification.ipynb")
print("notebook written")
