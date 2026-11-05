"""Evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config

CLASS_LIST = [config.CLASS_NAMES[i] for i in sorted(config.CLASS_NAMES)]


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Multiclass metrics: accuracy, macro precision/recall/F1, weighted F1, one-vs-rest ROC-AUC."""
    labels = sorted(config.CLASS_NAMES)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc_ovr": float(roc_auc_score(y_true, y_prob, multi_class="ovr", labels=labels)),
    }


def detailed_report(y_true, y_pred) -> dict:
    labels = sorted(config.CLASS_NAMES)
    return {
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=CLASS_LIST, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def misclassified(texts: pd.Series, y_true, y_pred, y_prob, limit: int = 300) -> pd.DataFrame:
    """Error-analysis table: the most confidently wrong predictions first."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    wrong = np.flatnonzero(y_true != y_pred)
    df = pd.DataFrame(
        {
            "text": np.asarray(texts)[wrong],
            "true": [config.CLASS_NAMES[i] for i in y_true[wrong]],
            "predicted": [config.CLASS_NAMES[i] for i in y_pred[wrong]],
            "confidence": np.asarray(y_prob)[wrong].max(axis=1),
        }
    )
    return df.sort_values("confidence", ascending=False).head(limit).reset_index(drop=True)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
