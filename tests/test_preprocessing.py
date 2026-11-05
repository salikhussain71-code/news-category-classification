from news_classification.preprocessing import PreprocessConfig, TextPreprocessor

pre = TextPreprocessor()


def test_lowercase_and_stopwords():
    tokens = pre("The Markets Are Rising").split()
    assert "the" not in tokens and "are" not in tokens
    assert all(t == t.lower() for t in tokens)


def test_lemmatization():
    tokens = pre("Stocks rose as markets were rallying").split()
    assert "stock" in tokens and "market" in tokens


def test_double_escaped_html_and_entities_removed():
    out = pre("Dell&lt;b&gt;&amp;lt;/b&amp;gt; earnings #36;5 billion").split()
    assert out == ["dell", "earnings", "billion"]


def test_urls_digits_backslashes_removed():
    out = pre("See http://x.com\\now 2004 results")
    assert "http" not in out and "2004" not in out and "\\" not in out


def test_agency_prefix_is_kept_by_default_and_stripped_on_request():
    assert "reuters" in pre("Reuters - Stocks rose").split()
    stripped = TextPreprocessor(PreprocessConfig(strip_agency=True))("Reuters - Stocks rose")
    assert "reuters" not in stripped.split()


def test_lemmatization_can_be_disabled():
    raw = TextPreprocessor(PreprocessConfig(lemmatize=False))
    assert "markets" in raw("many markets").split()


def test_non_string_is_safe():
    assert pre(None) == ""  # type: ignore[arg-type]
    assert pre("") == ""
