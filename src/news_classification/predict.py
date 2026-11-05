"""Inference: classify new news headlines / articles with the saved model.

    python -m news_classification.predict "Stocks rally as oil prices drop"
    python -m news_classification.predict --file headlines.txt      # one article per line
    python -m news_classification.predict --top 4 "Apple unveils new chip"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

from . import config
from .models import class_probabilities
from .preprocessing import PreprocessConfig, TextPreprocessor


class NewsClassifier:
    """Loads the bundle written by ``train`` and predicts the category of raw text."""

    def __init__(self, model_path: str | Path = config.BEST_MODEL_PATH) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model at '{model_path}'. Run `python -m news_classification.train` first."
            )
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.metadata = bundle.get("metadata", {})
        self.preprocessor = TextPreprocessor(PreprocessConfig(**bundle["preprocess_config"]))

    def predict(self, texts: list[str], top: int = 3) -> list[dict]:
        clean = self.preprocessor.transform(texts)
        X = self.pipeline.named_steps["vec"].transform(clean)
        probs = class_probabilities(self.pipeline.named_steps["clf"], X)
        results = []
        for text, p in zip(texts, probs):
            order = np.argsort(p)[::-1][:top]
            results.append(
                {
                    "text": text,
                    "category": config.CLASS_NAMES[int(order[0])],
                    "scores": [(config.CLASS_NAMES[int(i)], float(p[i])) for i in order],
                }
            )
        return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict the category of news text.")
    parser.add_argument("text", nargs="*", help="Article text(s) to classify.")
    parser.add_argument("--file", type=Path, help="Text file with one article per line.")
    parser.add_argument("--model", type=Path, default=config.BEST_MODEL_PATH)
    parser.add_argument("--top", type=int, default=3, help="How many categories to show.")
    args = parser.parse_args(argv)

    try:
        clf = NewsClassifier(args.model)
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return 1

    texts = list(args.text)
    if args.file:
        lines = args.file.read_text(encoding="utf-8").splitlines()
        texts += [ln.strip() for ln in lines if ln.strip()]
    if not texts:
        print("Enter an article or headline (empty line to quit):")
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                break
            texts.append(line)
    for result in clf.predict(texts, top=args.top):
        snippet = result["text"] if len(result["text"]) <= 80 else result["text"][:77] + "..."
        ranked = ", ".join(f"{name} {score:.0%}" for name, score in result["scores"])
        print(f"[{result['category'].upper():8s}] {snippet}\n            -> {ranked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
