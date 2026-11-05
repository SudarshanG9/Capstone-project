
import pandas as pd
import pytest

from rumor_graph.data import load_split, load_splits


def make_sample_dataset():
    """
    Create a small artificial dataset for testing.
    """

    return pd.DataFrame({
        "tweet_id": ["101", "102"],
        "text": [
            "Example tweet one",
            "Example tweet two"
        ],
        "label": ["false", "true"],
        "dataset": ["twitter15", "twitter15"]
    })


def test_valid_dataset(tmp_path):

    df = make_sample_dataset()

    file_path = tmp_path / "train.csv"
    df.to_csv(file_path, index=False)

    loaded_df = load_split(file_path)

    assert len(loaded_df) == 2
    assert "tweet_id" in loaded_df.columns


def test_missing_column(tmp_path):

    df = make_sample_dataset()

    df = df.drop(columns=["label"])

    file_path = tmp_path / "invalid.csv"
    df.to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="Missing columns"):
        load_split(file_path)


def test_invalid_label(tmp_path):

    df = make_sample_dataset()

    df.loc[0, "label"] = "unknown"

    file_path = tmp_path / "invalid_label.csv"
    df.to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="Unknown labels"):
        load_split(file_path)


def test_duplicate_tweet_ids(tmp_path):

    df = make_sample_dataset()

    df.loc[1, "tweet_id"] = "101"

    file_path = tmp_path / "duplicate.csv"
    df.to_csv(file_path, index=False)

    with pytest.raises(ValueError, match="Duplicate tweet IDs"):
        load_split(file_path)


def test_train_test_overlap(tmp_path):

    train = make_sample_dataset()

    test = pd.DataFrame({
        "tweet_id": ["102", "103"],
        "text": ["Test tweet one", "Test tweet two"],
        "label": ["true", "false"],
        "dataset": ["twitter15", "twitter16"]
    })

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    with pytest.raises(ValueError, match="overlapping tweet IDs"):
        load_splits(train_path, test_path)
