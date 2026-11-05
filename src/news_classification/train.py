"""End-to-end training pipeline for AG News category classification.

    python -m news_classification.train                              # core models
    python -m news_classification.train --models lr svm nb           # skip slow Random Forest
    python -m news_classification.train --xgboost --lightgbm --nn    # + bonus models
    python -m news_classification.train --tune                       # grid-search the best model

Protocol (no leakage):
  1. ``train.csv`` is split into TRAIN and VALIDATION (stratified). Models are compared and the
     best one is selected on **validation macro-F1**.
  2. The official ``test.csv`` is touched only for the final report. (If no test file exists, a
     stratified test split is carved out of the data instead.)
  3. The TF-IDF vectoriser is fitted on TRAIN only.
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from . import __version__, config, visualize
from .data import load_dataset
from .evaluate import compute_metrics, detailed_report, misclassified, save_json
from .features import build_vectorizer
from .models import (
    CORE_MODELS,
    MODEL_NAMES,
    PARAM_GRIDS,
    build_classifier,
    class_probabilities,
    top_features_per_class,
)
from .preprocessing import PreprocessConfig, TextPreprocessor

logger = logging.getLogger("news_classification")

RESULTS_START = "<!-- RESULTS:START -->"
RESULTS_END = "<!-- RESULTS:END -->"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and evaluate news category classifiers.")
    p.add_argument("--data", type=Path, default=None, help="Training CSV (default: data/raw/train.csv).")
    p.add_argument("--test-data", type=Path, default=None,
                   help="Official test CSV (default: data/raw/test.csv if it exists).")
    p.add_argument("--label-col", default="Class Index")
    p.add_argument("--title-col", default="Title", help="Title column ('' if the CSV has none).")
    p.add_argument("--text-col", default="Description")
    p.add_argument("--models", nargs="+", default=list(CORE_MODELS), choices=list(MODEL_NAMES),
                   help="Classifiers to train (default: lr svm nb rf).")
    p.add_argument("--xgboost", action="store_true", help="Also train XGBoost (bonus).")
    p.add_argument("--lightgbm", action="store_true", help="Also train LightGBM (bonus).")
    p.add_argument("--nn", action="store_true", help="Also train a Keras feed-forward network (bonus).")
    p.add_argument("--nn-epochs", type=int, default=8)
    p.add_argument("--sample", type=int, default=None,
                   help="Use a stratified subset of N training articles (quick experiments).")
    p.add_argument("--max-features", type=int, default=50_000)
    p.add_argument("--tune", action="store_true", help="Grid-search the best model (3-fold CV, train only).")
    p.add_argument("--no-lemmatize", action="store_true")
    p.add_argument("--strip-agency", action="store_true",
                   help="Remove newswire prefixes such as 'Reuters -' before training.")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-update-readme", action="store_true")
    p.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    return p.parse_args(argv)


def _evaluate(clf, X_val, y_val, X_test, y_test) -> tuple[dict, dict, object]:
    """Return (validation metrics, test metrics, test probabilities)."""
    pv = class_probabilities(clf, X_val)
    pt = class_probabilities(clf, X_test)
    val = compute_metrics(y_val, pv.argmax(axis=1), pv)
    test = compute_metrics(y_test, pt.argmax(axis=1), pt)
    return val, test, pt


def results_markdown(results: pd.DataFrame, saved_name: str) -> str:
    cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr", "val_f1_macro",
            "fit_seconds"]
    headers = ["Model", "Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "ROC-AUC (OvR)",
               "Val F1 (macro)", "Train time (s)"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:"] * len(cols)) + "|"]
    for _, r in results.sort_values("accuracy", ascending=False).iterrows():
        name = f"**{r['model_name']}**" if r["model_name"] == saved_name else r["model_name"]
        vals = [f"{r[c]:.4f}" if c != "fit_seconds" else f"{r[c]:.1f}" for c in cols]
        lines.append("| " + " | ".join([name] + vals) + " |")
    return "\n".join(lines)


def update_readme(table_md: str, summary: dict, figures: list[str]) -> bool:
    """Insert the latest results between the README markers. Returns True if updated."""
    path = config.README_PATH
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if RESULTS_START not in text or RESULTS_END not in text:
        return False
    m = summary["test_metrics"]
    block = [
        RESULTS_START,
        f"_Generated automatically on {summary['trained_at']} - {summary['n_train']:,} train / "
        f"{summary['n_val']:,} validation / {summary['n_test']:,} test articles, seed {summary['seed']}._",
        "",
        "Metrics below are measured on the **held-out test set**; the bold model was selected on the "
        "validation set and is the one saved for inference.",
        "",
        table_md,
        "",
        f"**Saved model:** {summary['best_model']} - accuracy **{m['accuracy']:.2%}**, "
        f"macro-F1 **{m['f1_macro']:.2%}**, ROC-AUC **{m['roc_auc_ovr']:.4f}**.",
    ]
    if figures:
        block += ["", "| | |", "|:---:|:---:|"]
        cells = [f"![{f}](reports/figures/{f})" for f in figures]
        for i in range(0, len(cells), 2):
            pair = cells[i:i + 2] + [""] * (2 - len(cells[i:i + 2]))
            block.append("| " + " | ".join(pair) + " |")
    block.append(RESULTS_END)
    pattern = re.compile(re.escape(RESULTS_START) + r".*?" + re.escape(RESULTS_END), re.DOTALL)
    path.write_text(pattern.sub(lambda _: "\n".join(block), text, count=1), encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    title_col = args.title_col or None

    # 1. Load -------------------------------------------------------------------------------
    try:
        df = load_dataset(args.data, args.label_col, title_col, args.text_col, sample=args.sample,
                          seed=args.seed)
        test_path = args.test_data or (config.DEFAULT_TEST_PATH if args.data is None else None)
        df_test = None
        if test_path and Path(test_path).exists():
            df_test = load_dataset(test_path, args.label_col, title_col, args.text_col, seed=args.seed)
        elif args.test_data:
            raise FileNotFoundError(f"Test file not found: {args.test_data}")
    except (FileNotFoundError, KeyError, ValueError) as err:
        logger.error("%s", err.args[0] if err.args else err)
        return 1
    if df_test is not None:  # make sure no article appears in both files
        overlap = df["text"].isin(set(df_test["text"]))
        if overlap.any():
            logger.warning("Removing %d training articles that also occur in the test file.",
                           int(overlap.sum()))
            df = df[~overlap].reset_index(drop=True)
    logger.info("Class balance (train file): %s",
                {config.CLASS_NAMES[k]: v for k, v in sorted(df["label"].value_counts().items())})

    # 2. Clean ------------------------------------------------------------------------------
    pre_cfg = PreprocessConfig(lemmatize=not args.no_lemmatize, strip_agency=args.strip_agency)
    preprocessor = TextPreprocessor(pre_cfg)
    logger.info("Cleaning %d articles (tokenise, lowercase, stop-words, lemmatise) ...", len(df))
    df["clean_text"] = preprocessor.transform_series(df["text"])
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_CACHE_PATH, index=False)

    # 3. Split ------------------------------------------------------------------------------
    if df_test is not None:
        df_test["clean_text"] = preprocessor.transform_series(df_test["text"])
        df_test = df_test[df_test["clean_text"].str.len() > 0].reset_index(drop=True)
        df_fit = df
    else:
        logger.info("No separate test file: holding out %.0f%% of the data as the test set.",
                    config.TEST_SIZE * 100)
        df_fit, df_test = train_test_split(df, test_size=config.TEST_SIZE, stratify=df["label"],
                                           random_state=args.seed)
        df_fit, df_test = df_fit.reset_index(drop=True), df_test.reset_index(drop=True)
    df_train, df_val = train_test_split(df_fit, test_size=config.VAL_SIZE, stratify=df_fit["label"],
                                        random_state=args.seed)
    df_train, df_val = df_train.reset_index(drop=True), df_val.reset_index(drop=True)
    logger.info("Train: %d | Validation: %d | Test: %d", len(df_train), len(df_val), len(df_test))
    y_train, y_val, y_test = df_train["label"], df_val["label"], df_test["label"]

    fig_dir = config.FIGURES_DIR
    figures: list[str] = []
    if not args.no_plots:
        visualize.plot_class_distribution(df_train["label"], fig_dir)
        visualize.plot_length_distribution(df_train["text"], df_train["label"], fig_dir)
        freqs = {k: visualize.word_frequencies(df_train.loc[y_train == k, "clean_text"])
                 for k in config.CLASS_NAMES}
        visualize.plot_top_words(freqs, fig_dir)
        visualize.plot_wordclouds(freqs, fig_dir)

    # 4. Vectorise (fit on TRAIN only) --------------------------------------------------------
    vec = build_vectorizer(max_features=args.max_features)
    X_train = vec.fit_transform(df_train["clean_text"])
    X_val = vec.transform(df_val["clean_text"])
    X_test = vec.transform(df_test["clean_text"])
    terms = vec.get_feature_names_out()
    logger.info("TF-IDF matrix: %s", X_train.shape)

    # 5. Train every model ------------------------------------------------------------------
    kinds = list(dict.fromkeys(args.models + (["xgb"] if args.xgboost else []) +
                               (["lgbm"] if args.lightgbm else [])))
    rows, fitted, test_probs = [], {}, {}
    for kind in kinds:
        try:
            clf = build_classifier(kind, args.seed)
        except ImportError as err:
            logger.warning("Skipping %s: %s", MODEL_NAMES[kind], err)
            continue
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_s = time.perf_counter() - t0
        val_m, test_m, pt = _evaluate(clf, X_val, y_val, X_test, y_test)
        rows.append({"model": kind, "model_name": MODEL_NAMES[kind], **test_m,
                     "val_accuracy": val_m["accuracy"], "val_f1_macro": val_m["f1_macro"],
                     "fit_seconds": round(fit_s, 2)})
        fitted[kind], test_probs[kind] = clf, pt
        logger.info("%-24s val F1=%.4f | test acc=%.4f F1=%.4f (%.1fs)", MODEL_NAMES[kind],
                    val_m["f1_macro"], test_m["accuracy"], test_m["f1_macro"], fit_s)
    if not fitted:
        logger.error("No model could be trained.")
        return 1
    results = pd.DataFrame(rows)

    # 6. Select the best sklearn model on VALIDATION macro-F1 -------------------------------
    best_kind = results.sort_values("val_f1_macro", ascending=False).iloc[0]["model"]
    best_clf, tuning_info = fitted[best_kind], None
    if args.tune:
        if best_kind in PARAM_GRIDS:
            logger.info("Tuning %s ...", MODEL_NAMES[best_kind])
            n_jobs = 1 if best_kind in ("rf", "xgb", "lgbm") else -1
            search = GridSearchCV(build_classifier(best_kind, args.seed), PARAM_GRIDS[best_kind],
                                  scoring="f1_macro", cv=3, n_jobs=n_jobs, refit=True)
            search.fit(X_train, y_train)
            best_clf = search.best_estimator_
            val_m, test_m, pt = _evaluate(best_clf, X_val, y_val, X_test, y_test)
            tuning_info = {"best_params": {k: str(v) for k, v in search.best_params_.items()},
                           "cv_f1_macro": float(search.best_score_)}
            idx = results.index[results["model"] == best_kind][0]
            for k, v in test_m.items():
                results.at[idx, k] = v
            results.at[idx, "val_f1_macro"] = val_m["f1_macro"]
            results.at[idx, "val_accuracy"] = val_m["accuracy"]
            results.at[idx, "model_name"] = MODEL_NAMES[best_kind] + " (tuned)"
            test_probs[best_kind] = pt
            logger.info("Best params: %s", search.best_params_)
        else:  # pragma: no cover
            logger.warning("No tuning grid for %s", best_kind)
    saved_name = results.loc[results["model"] == best_kind, "model_name"].iloc[0]

    # 7. Optional neural network (comparison only) -------------------------------------------
    nn_history = None
    if args.nn:
        try:
            from . import nn

            logger.info("Training Keras feed-forward network ...")
            net, nn_history, nn_secs = nn.train_network(X_train, y_train, X_val, y_val,
                                                        epochs=args.nn_epochs, seed=args.seed)
            pv, pt = nn.predict_proba(net, X_val), nn.predict_proba(net, X_test)
            val_m = compute_metrics(y_val, pv.argmax(axis=1), pv)
            test_m = compute_metrics(y_test, pt.argmax(axis=1), pt)
            results = pd.concat([results, pd.DataFrame([{
                "model": "nn", "model_name": "Neural Network (Keras)", **test_m,
                "val_accuracy": val_m["accuracy"], "val_f1_macro": val_m["f1_macro"],
                "fit_seconds": round(nn_secs, 2)}])], ignore_index=True)
            test_probs["nn"] = pt
            config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            net.save(config.NN_MODEL_PATH)
        except ImportError as err:
            logger.warning("Skipping neural network: %s", err)

    # 8. Final evaluation of the saved model --------------------------------------------------
    pt = test_probs[best_kind]
    y_pred = pt.argmax(axis=1)
    test_metrics = compute_metrics(y_test, y_pred, pt)
    report = detailed_report(y_test, y_pred)
    logger.info("SAVED MODEL: %s | test accuracy=%.4f macro-F1=%.4f", saved_name,
                test_metrics["accuracy"], test_metrics["f1_macro"])

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)
    table_md = results_markdown(results, saved_name)
    (config.METRICS_DIR / "model_comparison.md").write_text(table_md + "\n", encoding="utf-8")
    misclassified(df_test["text"], y_test, y_pred, pt).to_csv(
        config.METRICS_DIR / "misclassified_examples.csv", index=False)

    summary = {
        "version": __version__,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_train": int(len(df_train)), "n_val": int(len(df_val)), "n_test": int(len(df_test)),
        "seed": args.seed, "preprocessing": preprocessor.to_dict(),
        "tfidf_shape": list(X_train.shape),
        "best_model": saved_name, "tuning": tuning_info, "test_metrics": test_metrics, **report,
    }
    save_json(summary, config.METRICS_DIR / "best_model_metrics.json")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": Pipeline([("vec", vec), ("clf", best_clf)]),
         "preprocess_config": preprocessor.to_dict(),
         "metadata": {"model": saved_name, "test_metrics": test_metrics,
                      "trained_at": summary["trained_at"], "classes": config.CLASS_NAMES}},
        config.BEST_MODEL_PATH, compress=3,
    )
    logger.info("Model saved -> %s", config.BEST_MODEL_PATH)

    # 9. Figures --------------------------------------------------------------------------------
    if not args.no_plots:
        visualize.plot_confusion_matrix(y_test, y_pred, f"Test-set confusion matrix - {saved_name}", fig_dir)
        visualize.plot_model_comparison(results, fig_dir)
        visualize.plot_per_class_f1(report["classification_report"], saved_name, fig_dir)
        figures = ["model_comparison.png", "confusion_matrix.png", "top_words_per_category.png",
                   "wordclouds_per_category.png", "per_class_f1.png"]
        feats = top_features_per_class(best_clf, terms)
        if feats:
            visualize.plot_top_features(feats, saved_name, fig_dir)
            figures.append("top_model_features.png")
        if nn_history:
            visualize.plot_training_history(nn_history, fig_dir)
            figures.append("nn_training_history.png")
        logger.info("Figures saved -> %s", fig_dir)

    if not args.no_update_readme and update_readme(table_md, summary, figures):
        logger.info("README.md results section updated.")

    print("\n" + table_md + f"\n\nSaved model: {saved_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
