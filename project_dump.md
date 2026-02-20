# Project Structure

```text
ml-aws
├── .gitignore
├── .python-version
├── LICENSE
├── README.md
├── cli_commands.md
├── configs
│   ├── aws.yaml
│   ├── inference.yaml
│   └── train.yaml
├── data
│   ├── processed_data
│   └── raw_data
│       └── raw_data.csv
├── envs
│   ├── local.env
│   ├── prod.env
│   └── staging.env
├── infra
│   ├── ecr
│   ├── ecs
│   ├── iam
│   └── s3
├── notebooks
│   └── exploration.ipynb
├── project_to_markdown.py
├── pyproject.toml
├── scripts
│   ├── build_image.sh
│   ├── push_ecr.sh
│   └── run_local.sh
├── src
│   ├── api
│   │   ├── app.py
│   │   └── routes.py
│   ├── artifacts
│   │   ├── load.py
│   │   └── save.py
│   ├── cli
│   │   ├── predict.py
│   │   ├── retrain.py
│   │   └── train.py
│   ├── common
│   │   ├── io.py
│   │   ├── logging.py
│   │   └── schemas.py
│   ├── data_pipeline
│   │   ├── load.py
│   │   └── validate.py
│   ├── features
│   │   ├── build.py
│   │   └── preprocess.py
│   ├── inference
│   │   ├── pipeline.py
│   │   └── predictor.py
│   └── training
│       ├── evaluate.py
│       ├── pipeline.py
│       ├── train.py
│       └── tune.py
├── tests
│   ├── test_api.py
│   ├── test_features.py
│   ├── test_inference.py
│   └── test_training.py
└── uv.lock
```

# File Contents

## configs/aws.yaml

```

```

## configs/inference.yaml

```

```

## configs/train.yaml

```
dataset:
    raw_data_path: data/raw_data/raw_data.csv
    raw_path: data/raw_data/
    processed_path: data/processed_data/
model:
    type: xgboost
    n_estimators: 50
    max_depth: 3
training:
    test_size: 0.2
    val_size: 0.1
    random_state: 42

```

## data/raw_data/raw_data.csv

```
id,age,gender,course,study_hours,class_attendance,internet_access,sleep_hours,sleep_quality,study_method,facility_rating,exam_difficulty,exam_score
0,21,female,b.sc,7.91,98.8,no,4.9,average,online videos,low,easy,78.3
```

## src/api/app.py

```

```

## src/api/routes.py

```

```

## src/artifacts/load.py

```

```

## src/artifacts/save.py

```

```

## src/cli/predict.py

```

```

## src/cli/retrain.py

```

```

## src/cli/train.py

```

```

## src/common/io.py

```
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

```

## src/common/logging.py

```
import logging
import os


def get_logger(name: str):
    level = os.getenv("LOG_LEVEL", "INFO")

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(filename)s | %(funcName)s | %(message)s",
    )

    return logging.getLogger(name)

```

## src/common/schemas.py

```
from pydantic import BaseModel, Field
from typing import Literal, Annotated




TARGET_COLUMN = "exam_score"

# ==========================================================
# 1. Feature Schema (Single Data Point)
# ==========================================================


class StudentFeatures(BaseModel):
    age: Annotated[
        int,
        Field(
            ...,
            ge=0,
            le=100,
            title="Age of the student",
            description="Age of the student <=0 & >=100",
            examples=[22, 30],
            strict=True,
        ),
    ]
    gender: Annotated[
        Literal["female", "other", "male"],
        Field(
            ...,
            title="Gender",
            description="Gender of the student",
            examples=["female", "other", "male"],
            strict=True,
        ),
    ]
    course: Annotated[
        Literal["b.sc", "diploma", "bca", "b.com", "ba", "bba", "b.tech"],
        Field(
            ...,
            title="Course",
            description="Course enrolled by student",
            examples=["b.sc", "diploma", "bca", "b.com", "ba", "bba", "b.tech"],
            strict=True,
        ),
    ]
    study_hours: Annotated[
        float,
        Field(
            ...,
            ge=0,
            le=24,
            title="Study hours",
            description="Student average study hours",
            examples=[4.5, 6, 8],
            strict=True,
        ),
    ]
    class_attendance: Annotated[
        float,
        Field(
            ...,
            ge=0,
            le=100,
            title="Class attendance %",
            description="Class attendance % >=0 & <=100",
            examples=[40, 50, 100],
            strict=True,
        ),
    ]
    internet_access: Annotated[
        Literal["no", "yes"],
        Field(
            ...,
            title="Internet access",
            description="Internet access",
            examples=["no", "yes"],
            strict=True,
        ),
    ]
    sleep_hours: Annotated[
        float,
        Field(
            ...,
            ge=0,
            le=24,
            title="Sleep hours",
            description="Sleep hours",
            examples=[5, 7, 9],
            strict=True,
        ),
    ]
    sleep_quality: Annotated[
        Literal["average", "poor", "good"],
        Field(
            ...,
            title="Sleep quality",
            description="Sleep quality",
            examples=["average", "poor", "good"],
            strict=True,
        ),
    ]
    study_method: Annotated[
        Literal["online videos", "self-study", "coaching", "group study", "mixed"],
        Field(
            ...,
            title="Study method",
            description="Study method",
            examples=[
                "online videos",
                "self-study",
                "coaching",
                "group study",
                "mixed",
            ],
            strict=True,
        ),
    ]
    facility_rating: Annotated[
        Literal["low", "medium", "high"],
        Field(
            ...,
            title="Facility rating",
            description="Facility rating",
            examples=["low", "medium", "high"],
            strict=True,
        ),
    ]
    exam_difficulty: Annotated[
        Literal["easy", "moderate", "hard"],
        Field(
            ...,
            title="Exam difficulty",
            description="Exam difficulty",
            examples=["easy", "moderate", "hard"],
            strict=True,
        ),
    ]

```

## src/data_pipeline/load.py

```
import pandas as pd
from src.common.io import load_yaml, save_splits
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_csv(path: Path | str) -> pd.DataFrame:

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

```

## src/data_pipeline/validate.py

```

```

## src/features/build.py

```

```

## src/features/preprocess.py

```

```

## src/inference/pipeline.py

```

```

## src/inference/predictor.py

```

```

## src/training/evaluate.py

```

```

## src/training/pipeline.py

```

```

## src/training/train.py

```

```

## src/training/tune.py

```

```

## tests/test_api.py

```

```

## tests/test_features.py

```

```

## tests/test_inference.py

```

```

## tests/test_training.py

```

```

