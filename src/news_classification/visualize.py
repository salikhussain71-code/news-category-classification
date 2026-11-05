"""All plotting code. Every function saves a PNG and returns the output path."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: works on servers / CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from . import config  # noqa: E402

CLASS_ORDER = sorted(config.CLASS_NAMES)
CLASS_COLORS = {0: "#1F4E79", 1: "#2E7D32", 2: "#F9A825", 3: "#8E24AA"}
CLASS_CMAPS = {0: "Blues", 1: "Greens", 2: "YlOrBr", 3: "Purples"}

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_distribution(labels: pd.Series, out_dir: Path) -> Path:
    counts = labels.value_counts().reindex(CLASS_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([config.CLASS_NAMES[i] for i in CLASS_ORDER], counts.values,
                  color=[CLASS_COLORS[i] for i in CLASS_ORDER])
    ax.bar_label(bars, fmt="{:,.0f}", padding=3)
    ax.set_title("Class distribution (training data)")
    ax.set_ylabel("Number of articles")
    ax.grid(axis="x", visible=False)
    return _save(fig, out_dir / "class_distribution.png")


def plot_length_distribution(texts: pd.Series, labels: pd.Series, out_dir: Path) -> Path:
    lengths = texts.str.split().str.len()
    upper = float(np.percentile(lengths, 99))
    bins = np.linspace(0, max(upper, 1), 40)
    fig, ax = plt.subplots(figsize=(7, 4))
    for k in CLASS_ORDER:
        ax.hist(lengths[labels == k].clip(upper=upper), bins=bins, alpha=0.45,
                color=CLASS_COLORS[k], label=config.CLASS_NAMES[k])
    ax.set_title("Article length (words, title + description)")
    ax.set_xlabel("Words")
    ax.set_ylabel("Number of articles")
    ax.legend()
    return _save(fig, out_dir / "text_length_distribution.png")


def word_frequencies(clean_texts: pd.Series) -> Counter:
    counter: Counter = Counter()
    for text in clean_texts:
        counter.update(text.split())
    return counter


def plot_top_words(freqs: dict[int, Counter], out_dir: Path, n: int = 15) -> Path:
    """Bar plots of the most frequent words in each category (2 x 2 grid)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, k in zip(axes.ravel(), CLASS_ORDER):
        top = freqs[k].most_common(n)
        words, counts = zip(*top) if top else ((), ())
        ax.barh(list(words)[::-1], list(counts)[::-1], color=CLASS_COLORS[k])
        ax.set_title(config.CLASS_NAMES[k])
        ax.grid(axis="y", visible=False)
    fig.suptitle("Most frequent words per category", fontweight="bold", fontsize=14)
    return _save(fig, out_dir / "top_words_per_category.png")


def plot_wordclouds(freqs: dict[int, Counter], out_dir: Path) -> Path:
    from wordcloud import WordCloud

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, k in zip(axes.ravel(), CLASS_ORDER):
        cloud = WordCloud(width=800, height=450, background_color="white", colormap=CLASS_CMAPS[k],
                          max_words=120, random_state=config.RANDOM_STATE)
        cloud.generate_from_frequencies(dict(freqs[k].most_common(400)))
        ax.imshow(cloud, interpolation="bilinear")
        ax.set_title(config.CLASS_NAMES[k], fontsize=14)
        ax.axis("off")
    return _save(fig, out_dir / "wordclouds_per_category.png")


def plot_confusion_matrix(y_true, y_pred, title: str, out_dir: Path) -> Path:
    """Counts (left) and row-normalised recall (right)."""
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    norm = cm / cm.sum(axis=1, keepdims=True)
    names = [config.CLASS_NAMES[i] for i in CLASS_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panels = ((axes[0], cm, "d", "Counts"), (axes[1], norm, ".1%", "Row-normalised (recall)"))
    for ax, data, fmt, sub in panels:
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(sub, fontsize=11)
        ax.grid(False)
        thresh = data.max() / 2
        for i in range(len(names)):
            for j in range(len(names)):
                value = format(data[i, j], fmt)
                ax.text(j, i, value, ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title, fontweight="bold")
    return _save(fig, out_dir / "confusion_matrix.png")


def plot_model_comparison(results: pd.DataFrame, out_dir: Path) -> Path:
    """Test accuracy and macro-F1 for every model."""
    df = results.sort_values("accuracy")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9, 0.8 * len(df) + 1.8))
    ax.barh(y - 0.2, df["accuracy"], height=0.38, color="#1F4E79", label="Accuracy")
    ax.barh(y + 0.2, df["f1_macro"], height=0.38, color="#F9A825", label="Macro F1")
    ax.set_yticks(y, df["model_name"])
    lo = max(0.0, float(min(df["accuracy"].min(), df["f1_macro"].min())) - 0.05)
    hi = float(max(df["accuracy"].max(), df["f1_macro"].max()))
    ax.set_xlim(lo, min(1.03, hi + 0.05))
    for i, (a, f) in enumerate(zip(df["accuracy"], df["f1_macro"])):
        ax.text(a + 0.002, i - 0.2, f"{a:.3f}", va="center", fontsize=8)
        ax.text(f + 0.002, i + 0.2, f"{f:.3f}", va="center", fontsize=8)
    ax.set_title("Model comparison (test set)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    ax.grid(axis="y", visible=False)
    return _save(fig, out_dir / "model_comparison.png")


def plot_per_class_f1(report: dict, title: str, out_dir: Path) -> Path:
    names = [config.CLASS_NAMES[i] for i in CLASS_ORDER]
    f1 = [report[n]["f1-score"] for n in names]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(names, f1, color=[CLASS_COLORS[i] for i in CLASS_ORDER])
    ax.bar_label(bars, fmt="{:.3f}", padding=3)
    ax.set_ylim(max(0.0, min(f1) - 0.1), 1.0)
    ax.set_title(f"Per-class F1 - {title}", fontsize=11)
    ax.grid(axis="x", visible=False)
    return _save(fig, out_dir / "per_class_f1.png")


def plot_top_features(features: dict[int, list[tuple[str, float]]], title: str, out_dir: Path,
                      n: int = 10) -> Path:
    """Most indicative (highest-weight) terms for each category."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, k in zip(axes.ravel(), CLASS_ORDER):
        feats = features[k][:n]
        ax.barh([t for t, _ in feats][::-1], [w for _, w in feats][::-1], color=CLASS_COLORS[k])
        ax.set_title(config.CLASS_NAMES[k])
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"Most discriminative terms - {title}", fontweight="bold", fontsize=14)
    return _save(fig, out_dir / "top_model_features.png")


def plot_training_history(history: dict, out_dir: Path) -> Path:
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["loss"], "o-", label="train")
    axes[0].plot(epochs, history["val_loss"], "o-", label="validation")
    axes[0].set_title("Neural network - loss")
    axes[0].set_xlabel("Epoch")
    axes[1].plot(epochs, history["accuracy"], "o-", label="train")
    axes[1].plot(epochs, history["val_accuracy"], "o-", label="validation")
    axes[1].set_title("Neural network - accuracy")
    axes[1].set_xlabel("Epoch")
    for ax in axes:
        ax.legend()
    return _save(fig, out_dir / "nn_training_history.png")
