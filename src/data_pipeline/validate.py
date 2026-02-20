import pandas as pd
from src.common.schemas import FEATURE_COLUMNS, TARGET_COLUMN


def validate_columns(df: pd.DataFrame) -> None:
    expected_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    actual_columns = set(df.columns)

    missing = expected_columns - actual_columns
    extra = actual_columns - expected_columns

    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if extra:
        raise ValueError(f"Unexpected columns: {extra}")


def validate_target_missing_values(df: pd.DataFrame) -> None:
    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing vlaues.")


def validate_target_variance(df: pd.DataFrame) -> None:
    if df[TARGET_COLUMN].nunique() <= 1:
        raise ValueError("Target column has no variance.")


def run_validation(df: pd.DataFrame) -> None:
    validate_columns(df)
    validate_target_missing_values(df)
    validate_target_variance(df)
