
from pathlib import Path

import pandas as pd


# Columns our research pipeline requires
REQUIRED_COLUMNS = {
    "tweet_id",
    "text",
    "label",
    "dataset",
}

# Valid classification labels
VALID_LABELS = {
    "false",
    "true",
    "non-rumor",
    "unverified",
}


def validate_split(df: pd.DataFrame) -> None:
    """
    Validate the structure and contents of a dataset.
    """

    # 1. Check required columns
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    # 2. Check that the dataset is not empty
    if df.empty:
        raise ValueError("Dataset is empty.")

    # 3. Check missing or blank values
    for column in REQUIRED_COLUMNS:
        values = df[column].astype("string")

        invalid = (
            values.isna()
            | values.str.strip().eq("").fillna(False)
        )

        if invalid.any():
            raise ValueError(
                f"Missing or blank values in '{column}'."
            )

    # 4. Check classification labels
    invalid_labels = (
        set(df["label"].unique()) - VALID_LABELS
    )

    if invalid_labels:
        raise ValueError(
            f"Unknown labels: {sorted(invalid_labels)}"
        )

    # 5. Check duplicate tweet IDs
    duplicates = df["tweet_id"].duplicated()

    if duplicates.any():
        raise ValueError(
            "Duplicate tweet IDs found in the split."
        )


def load_split(file_path: str | Path) -> pd.DataFrame:
    """
    Load and validate a single CSV dataset.
    """

    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    # Read columns as strings to preserve tweet IDs
    df = pd.read_csv(
        file_path,
        dtype="string"
    )

    validate_split(df)

    return df


def load_splits(
    train_path: str | Path,
    test_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and testing datasets.
    Verify that their tweet IDs do not overlap.
    """

    train_df = load_split(train_path)
    test_df = load_split(test_path)

    train_ids = set(train_df["tweet_id"])
    test_ids = set(test_df["tweet_id"])

    overlapping_ids = train_ids & test_ids

    if overlapping_ids:
        raise ValueError(
            "Train and test contain overlapping tweet IDs."
        )

    return train_df, test_df


if __name__ == "__main__":

    train_df, test_df = load_splits(
        "data/splits/train_split.csv",
        "data/splits/test_split.csv",
    )

    print("Dataset successfully loaded!")
    print(f"Training examples: {len(train_df)}")
    print(f"Testing examples: {len(test_df)}")

    print("\nTraining class distribution:")
    print(train_df["label"].value_counts())

    # Report exact text repeated across train/test
    repeated_test_text = test_df["text"].isin(
        set(train_df["text"])
    )

    print(
        "\nTest examples with text also in training:",
        int(repeated_test_text.sum()),
    )
