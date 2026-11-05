import json

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from news_classification import config, train
from news_classification.data import load_dataset
from news_classification.features import build_vectorizer
from news_classification.models import (
    CORE_MODELS,
    build_classifier,
    class_probabilities,
    top_features_per_class,
)
from news_classification.predict import NewsClassifier
from news_classification.preprocessing import TextPreprocessor


@pytest.fixture(scope="module")
def matrices(sample_train):
    df = load_dataset(sample_train)
    df["clean"] = TextPreprocessor().transform_series(df["text"])
    X_tr, X_te, y_tr, y_te = train_test_split(df["clean"], df["label"], test_size=0.25,
                                              stratify=df["label"], random_state=0)
    vec = build_vectorizer(max_features=3000)
    return vec.fit_transform(X_tr), vec.transform(X_te), y_tr, y_te, vec


@pytest.mark.parametrize("kind", list(CORE_MODELS) + ["xgb", "lgbm"])
def test_every_model_trains_and_scores(matrices, kind):
    if kind == "xgb":
        pytest.importorskip("xgboost")
    if kind == "lgbm":
        pytest.importorskip("lightgbm")
    X_tr, X_te, y_tr, y_te, _ = matrices
    clf = build_classifier(kind).fit(X_tr, y_tr)
    probs = class_probabilities(clf, X_te)
    assert probs.shape == (X_te.shape[0], 4)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert (probs.argmax(axis=1) == y_te).mean() > 0.9


def test_top_features_per_class(matrices):
    X_tr, _, y_tr, _, vec = matrices
    terms = vec.get_feature_names_out()
    lr_feats = top_features_per_class(build_classifier("lr").fit(X_tr, y_tr), terms, n=5)
    assert set(lr_feats) == {0, 1, 2, 3} and all(len(v) == 5 for v in lr_feats.values())
    assert top_features_per_class(build_classifier("rf").fit(X_tr, y_tr), terms) is None


def test_unknown_model():
    with pytest.raises(ValueError):
        build_classifier("bert")


def test_neural_network(matrices):
    pytest.importorskip("keras")
    from news_classification import nn

    X_tr, X_te, y_tr, y_te, _ = matrices
    model, history, _ = nn.train_network(X_tr, y_tr, X_te, y_te, epochs=2, batch_size=64, verbose=False)
    assert len(history["loss"]) >= 1
    assert nn.predict_proba(model, X_te).shape == (X_te.shape[0], 4)


def _isolate(monkeypatch, tmp_path):
    root = config.PROJECT_ROOT
    for name in ("PROCESSED_DIR", "CLEAN_CACHE_PATH", "MODELS_DIR", "BEST_MODEL_PATH", "NN_MODEL_PATH",
                 "REPORTS_DIR", "FIGURES_DIR", "METRICS_DIR", "README_PATH"):
        monkeypatch.setattr(config, name, tmp_path / getattr(config, name).relative_to(root))
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)


def test_end_to_end_with_official_test_file(tmp_path, monkeypatch, sample_train, sample_test):
    _isolate(monkeypatch, tmp_path)
    readme = tmp_path / "README.md"
    body = f"# t\n{train.RESULTS_START}\nPLACEHOLDER_TEXT\n{train.RESULTS_END}\n"
    readme.write_text(body, encoding="utf-8")

    code = train.main(["--data", str(sample_train), "--test-data", str(sample_test),
                       "--models", "lr", "svm", "nb"])
    assert code == 0
    metrics = json.loads((tmp_path / "reports/metrics/best_model_metrics.json").read_text())
    assert metrics["test_metrics"]["accuracy"] > 0.9
    assert metrics["n_test"] == 120
    assert (tmp_path / "reports/figures/confusion_matrix.png").exists()
    assert (tmp_path / "reports/figures/wordclouds_per_category.png").exists()
    assert (tmp_path / "reports/metrics/misclassified_examples.csv").exists()
    assert "PLACEHOLDER_TEXT" not in readme.read_text()
    assert "Logistic Regression" in readme.read_text()

    clf = NewsClassifier(tmp_path / "models/news_classifier.joblib")
    out = clf.predict(["Oil prices and stocks rally as earnings and profit beat market forecasts.",
                       "The striker scored a late goal to win the championship match."], top=2)
    assert [o["category"] for o in out] == ["Business", "Sports"]
    assert len(out[0]["scores"]) == 2


def test_end_to_end_internal_split_with_tuning(tmp_path, monkeypatch, sample_train):
    _isolate(monkeypatch, tmp_path)
    code = train.main(["--data", str(sample_train), "--models", "nb", "--tune", "--no-plots",
                       "--no-update-readme"])
    assert code == 0
    metrics = json.loads((tmp_path / "reports/metrics/best_model_metrics.json").read_text())
    assert metrics["tuning"]["best_params"]


def test_missing_dataset_returns_error_code(tmp_path):
    assert train.main(["--data", str(tmp_path / "missing.csv")]) == 1
