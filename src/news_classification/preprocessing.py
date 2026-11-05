"""Text preprocessing: tokenisation, lower-casing, stop-word removal, lemmatisation.

AG News specifics handled here
------------------------------
* Escaped HTML (``&lt;b&gt;``, ``&amp;``) and numeric-entity leftovers such as ``#39;`` / ``#36;``.
* Backslashes and stray punctuation.
* Optional removal of newswire prefixes ("Reuters - ", "AP - ") via ``strip_agency`` - these are
  strong shortcuts in the benchmark, so the option lets you measure how much they help.

Tokenisation is regex based (letters only), so no extra NLTK tokenizer models are needed.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Iterable

import nltk
import pandas as pd

logger = logging.getLogger(__name__)

_HTML_TAG = re.compile(r"<[^>]+>")
_NUM_ENTITY = re.compile(r"#\d+;")
_URL = re.compile(r"(https?://\S+|www\.\S+)")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_WHITESPACE = re.compile(r"\s+")
_AGENCY = re.compile(
    r"^\s*\(?\s*(reuters|ap|afp|ap online|associated press|upi|bloomberg|cnn)\s*\)?\s*[-:\u2013\u2014]\s*",
    flags=re.IGNORECASE,
)
_AGENCY_TAG = re.compile(
    r"\(\s*(reuters|ap|afp|upi|ap online|bloomberg)\s*\)\s*(-|:)?|\b(reuters|afp|upi)\b\s*[-:]",
    flags=re.IGNORECASE,
)


def ensure_nltk_resources() -> None:
    """Download the NLTK corpora needed (stopwords, wordnet) if they are missing."""
    for resource, path in (("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")):
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info("Downloading NLTK resource '%s' ...", resource)
            nltk.download(resource, quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
        nltk.data.find("corpora/stopwords")
    except LookupError as exc:  # pragma: no cover - only when offline
        raise RuntimeError(
            "NLTK data could not be downloaded. Run: python -m nltk.downloader stopwords wordnet"
        ) from exc


@lru_cache(maxsize=1)
def _lemmatizer():
    ensure_nltk_resources()
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize("test")  # load the corpus now so failures surface early
    return lemmatizer


@lru_cache(maxsize=1)
def default_stopwords() -> frozenset[str]:
    """NLTK English stop words (all of them: negation is irrelevant for topic classification)."""
    ensure_nltk_resources()
    from nltk.corpus import stopwords

    return frozenset(stopwords.words("english"))


@lru_cache(maxsize=300_000)
def _lemmatize_word(word: str) -> str:
    lemmatizer = _lemmatizer()
    for pos in ("n", "v", "a"):  # noun, verb, adjective: "markets"->"market", "wins"->"win"
        lemma = lemmatizer.lemmatize(word, pos=pos)
        if lemma != word:
            return lemma
    return word


@dataclass(frozen=True)
class PreprocessConfig:
    """Switches for the cleaning pipeline (stored inside the saved model for inference)."""

    lowercase: bool = True
    remove_html: bool = True
    remove_urls: bool = True
    remove_stopwords: bool = True
    lemmatize: bool = True
    strip_agency: bool = False
    min_token_length: int = 2


class TextPreprocessor:
    """Callable text cleaner: ``TextPreprocessor()(raw_text) -> clean_text``."""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()
        self._stopwords = default_stopwords() if self.config.remove_stopwords else frozenset()
        if self.config.lemmatize:
            _lemmatizer()

    def tokenize(self, text: str) -> list[str]:
        """Clean one article and return its list of tokens."""
        cfg = self.config
        if not isinstance(text, str):
            return []
        if cfg.remove_html:
            text = html.unescape(html.unescape(text))  # twice: AG News double-escapes tags
            text = _HTML_TAG.sub(" ", text)
            text = _NUM_ENTITY.sub(" ", text)
        if cfg.remove_urls:
            text = _URL.sub(" ", text)
        if cfg.strip_agency:
            text = _AGENCY_TAG.sub(" ", _AGENCY.sub("", text))
        if cfg.lowercase:
            text = text.lower()
        text = text.replace("\\", " ")
        text = _NON_ALPHA.sub(" ", text)
        text = _WHITESPACE.sub(" ", text).strip()

        tokens = [t for t in text.split(" ") if len(t) >= cfg.min_token_length]
        if cfg.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]
        if cfg.lemmatize:
            tokens = [_lemmatize_word(t) for t in tokens]
        return tokens

    def __call__(self, text: str) -> str:
        return " ".join(self.tokenize(text))

    def transform(self, texts: Iterable[str]) -> list[str]:
        return [self(t) for t in texts]

    def transform_series(self, series: pd.Series) -> pd.Series:
        return series.map(self)

    def to_dict(self) -> dict:
        return asdict(self.config)
