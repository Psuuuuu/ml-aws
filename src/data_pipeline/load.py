import pandas as pd
from src.common.io import load_yaml, save_splits
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_csv(path: str) -> pd.DataFrame:

    try:
        return pd.read_csv(path).drop(columns=["id"])

    except FileNotFoundError:
        raise FileNotFoundError("File not found")
    except KeyError:
        raise KeyError("'id' column not present")


def split_data(df: pd.DataFrame, test_size: float, val_size: float, random_state: int):
    train_df, temp_df = train_test_split(
        df, test_size=test_size + val_size, random_state=random_state
    )

    val_ratio = val_size / (test_size + val_size)

    val_df, test_df = train_test_split(
        temp_df, test_size=1 - val_ratio, random_state=random_state
    )

    return train_df, val_df, test_df


def run_data_pipeline(path: Path | str, config_path: Path | str):
    config = load_yaml(config_path)

    df = load_csv(Path(config["dataset"]["raw_data_path"]))

    train_df, val_df, test_df = split_data(
        df=df,
        test_size=config["training"]["test_size"],
        val_size=config["training"]["val_size"],
        random_state=config["training"]["random_state"],
    )

    save_splits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        output_dir=Path(config["dataset"]["processed_path"]),
    )
