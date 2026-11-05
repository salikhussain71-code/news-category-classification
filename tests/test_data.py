import pandas as pd
import pytest

from news_classification.data import load_dataset


def test_load_sample(sample_train):
    df = load_dataset(sample_train)
    assert set(df.columns) == {"text", "label"}
    assert set(df["label"]) == {0, 1, 2, 3}
    assert df["text"].is_unique


def test_title_and_description_are_combined(tmp_path):
    f = tmp_path / "a.csv"
    pd.DataFrame({"Class Index": [1, 2], "Title": ["Big win", "Stocks fall"],
                  "Description": ["Team wins cup.", "Markets drop."]}).to_csv(f, index=False)
    df = load_dataset(f)
    assert df["text"].iloc[0] == "Big win. Team wins cup"
    assert df["label"].tolist() == [0, 1]  # 1-4 shifted to 0-3


def test_single_text_column(tmp_path):
    f = tmp_path / "b.csv"
    pd.DataFrame({"label": [0, 3], "body": ["alpha text", "beta text"]}).to_csv(f, index=False)
    df = load_dataset(f, label_col="label", title_col=None, text_col="body")
    assert df["label"].tolist() == [0, 3]


def test_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        load_dataset(tmp_path / "none.csv")


def test_wrong_column_message(sample_train):
    with pytest.raises(KeyError, match="--label-col"):
        load_dataset(sample_train, label_col="nope")


def test_invalid_labels_rejected(tmp_path):
    f = tmp_path / "c.csv"
    frame = {"Class Index": [1, 9], "Title": ["a", "b"], "Description": ["c", "d"]}
    pd.DataFrame(frame).to_csv(f, index=False)
    with pytest.raises(ValueError, match="Unexpected label"):
        load_dataset(f)


def test_duplicates_and_empty_removed(tmp_path):
    f = tmp_path / "d.csv"
    pd.DataFrame({"Class Index": [1, 1, 2, 3], "Title": ["x", "x", "", "y"],
                  "Description": ["same", "same", "", "other"]}).to_csv(f, index=False)
    assert len(load_dataset(f)) == 2


def test_stratified_subsample(sample_train):
    df = load_dataset(sample_train, sample=100)
    assert len(df) == 100
    assert df["label"].value_counts().min() >= 20
