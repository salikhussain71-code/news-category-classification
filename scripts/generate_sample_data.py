"""Generate SYNTHETIC AG News-style sample files for smoke tests and CI.

Creates ``sample_train.csv`` and ``sample_test.csv`` with the Kaggle columns
``Class Index`` (1-4), ``Title`` and ``Description``. The text is template based and exists only
so the pipeline and tests run without downloading anything. Never report results from it.

    python scripts/generate_sample_data.py
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

VOCAB = {
    1: ("World", ["president", "minister", "government", "talks", "election", "troops", "border", "UN",
                  "peace", "treaty", "protest", "diplomats", "rebels", "ceasefire", "parliament"],
        ["Iraq", "Baghdad", "Israel", "Beijing", "Moscow", "Tehran", "Kabul", "Paris"]),
    2: ("Sports", ["match", "coach", "league", "goal", "championship", "season", "striker", "tournament",
                   "playoff", "olympic", "medal", "stadium", "quarterback", "innings", "referee"],
        ["Yankees", "Chelsea", "Lakers", "Federer", "Madrid", "Boston", "Sydney", "Brazil"]),
    3: ("Business", ["stocks", "profit", "shares", "earnings", "market", "oil", "inflation", "merger",
                     "investors", "revenue", "bank", "economy", "dollar", "retail", "quarter"],
        ["Wall Street", "Nasdaq", "Boeing", "Citigroup", "Reuters", "Fed", "Tokyo", "Walmart"]),
    4: ("Sci/Tech", ["software", "internet", "researchers", "computer", "chip", "microsoft", "online",
                     "scientists", "satellite", "wireless", "security", "linux", "genome", "nasa", "browser"],
        ["Google", "Intel", "Apple", "Mars", "IBM", "Yahoo", "Cisco", "Sony"]),
}
VERBS = ["announces", "reports", "faces", "plans", "wins", "launches", "warns of", "reveals", "expects"]
TEMPL = [
    "{E} {v} new {a} amid {b} concerns",
    "{a} and {b} dominate talks as {E} {v} changes",
    "{E} {v} {a}, analysts say {b} will follow",
    "Experts: {a} {b} to shape {E} this year",
]


def article(rng: random.Random, cls: int) -> tuple[int, str, str]:
    _, words, ents = VOCAB[cls]
    title = rng.choice(TEMPL).format(E=rng.choice(ents), v=rng.choice(VERBS), a=rng.choice(words),
                                     b=rng.choice(words)).capitalize()
    sentences = []
    for _ in range(rng.randint(2, 3)):
        sentences.append(f"{rng.choice(ents)} officials said {rng.choice(words)} and {rng.choice(words)} "
                         f"remain key as {rng.choice(words)} {rng.choice(VERBS)} {rng.choice(words)}.")
    prefix = rng.choice(["", "", "Reuters - ", "AP - ", "(AFP) - "])
    if rng.random() < 0.15:  # a few off-topic words so classes are not perfectly separable
        other = rng.choice([c for c in VOCAB if c != cls])
        sentences.append(f"Also, {rng.choice(VOCAB[other][1])} was mentioned.")
    return cls, title, prefix + " ".join(sentences)


def write(path: Path, n: int, seed: int) -> None:
    rng = random.Random(seed)
    seen, rows = set(), []
    while len(rows) < n:
        cls = len(rows) % 4 + 1
        row = article(rng, cls)
        if row[1:] in seen:
            continue
        seen.add(row[1:])
        rows.append(row)
    rng.shuffle(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Class Index", "Title", "Description"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} synthetic articles -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/sample"))
    parser.add_argument("--n-train", type=int, default=480)
    parser.add_argument("--n-test", type=int, default=120)
    args = parser.parse_args()
    write(args.out_dir / "sample_train.csv", args.n_train, seed=42)
    write(args.out_dir / "sample_test.csv", args.n_test, seed=7)


if __name__ == "__main__":
    main()
