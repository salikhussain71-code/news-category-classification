"""Dataset loading for AG News (and any CSV with title / description / class columns).

Kaggle AG News files (``train.csv`` / ``test.csv``) have the columns
``Class Index`` (1-4), ``Title`` and ``Description``. The two text fields are concatenated
into one ``text`` column; labels are shifted to 0-3.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

logger = logging.getLogger(__name__)


def load_dataset(
    path: str | Path | None = None,
    label_col: str = "Class Index",
    title_col: str | None = "Title",
    text_col: str = "Description",
    drop_duplicates: bool = True,
    sample: int | None = None,
    seed: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Return a frame with columns ``text`` (title + description) and ``label`` (0-3).

    Pass ``title_col=None`` for datasets that only have a single text column.
    ``sample`` draws a stratified random subset (useful for quick experiments).
    """
    path = Path(path) if path else config.DEFAULT_TRAIN_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download the Kaggle 'AG News Classification Dataset' and place train.csv / test.csv in "
            f"'{config.RAW_DIR}' (see data/README.md), or pass --data / --test-data."
        )

    raw = pd.read_csv(path)
    needed = {label_col, text_col} | ({title_col} if title_col else set())
    missing = needed - set(raw.columns)
    if missing:
        raise KeyError(
            f"Column(s) {sorted(missing)} not found in {path.name}. Available: {list(raw.columns)}. "
            "Use --label-col / --title-col / --text-col."
        )

    title = raw[title_col].fillna("").astype(str) if title_col else ""
    body = raw[text_col].fillna("").astype(str)
    text = (title + ". " + body).str.strip(" .") if title_col else body

    labels = pd.to_numeric(raw[label_col], errors="coerce")
    df = pd.DataFrame({"text": text, "label": labels}).dropna(subset=["label"])
    df = df[df["text"].str.strip().str.len() > 0]
    n_start = len(df)
    if drop_duplicates:
        df = df.drop_duplicates(subset="text")
    df["label"] = df["label"].astype(int)
    if df["label"].min() == 1:  # Kaggle uses 1..4 -> shift to 0..3
        df["label"] = df["label"] - 1
    if not set(df["label"].unique()) <= set(config.CLASS_NAMES):
        raise ValueError(f"Unexpected label values: {sorted(df['label'].unique())} (expected 1-4 or 0-3).")

    if sample and sample < len(df):
        df, _ = train_test_split(df, train_size=sample, stratify=df["label"], random_state=seed)
    df = df.reset_index(drop=True)
    logger.info("Loaded %s: %d rows kept (%d duplicates/empty removed).", path.name, len(df),
                n_start - len(df))
    return df
