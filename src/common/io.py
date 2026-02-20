import yaml
from pathlib import Path


def load_yaml(path: Path | str):
    with open(path) as f:
        return yaml.safe_load(f)


def save_splits(train_df, val_df, test_df, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
