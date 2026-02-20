from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # adding features
    df = df.copy()
    df["attendance_frac"] = df["class_attendance"] / 100
    df["effective_study"] = df["study_hours"] * df["attendance_frac"]
    sleep_map = {"poor": 1, "average": 2, "good": 3}
    df["effective_sleep"] = df["sleep_hours"] * df["sleep_quality"].map(sleep_map)
    df["study_sleep_balance"] = df["study_hours"] - df["sleep_hours"]
    df["mental_overload"] = ((df["study_hours"] > 6) & (df["sleep_hours"] < 6)).astype(
        int
    )
    difficulty_map = {"easy": 1, "moderate": 2, "hard": 3}
    df["study_per_difficulty"] = df["study_hours"] / df["exam_difficulty"].map(
        difficulty_map
    )
    df["low_attendance"] = (df["attendance_frac"] < 0.6).astype(int)

    return df


def build_preprocessor(config: dict) -> Pipeline:
    feature_engineering = FunctionTransformer(func=add_features, validate=False)

    ordinal_cols: list[str] = ["sleep_quality", "facility_rating", "exam_difficulty"]

    ordinal_categories: list[list[str]] = [
        ["poor", "average", "good", "__missing__"],
        ["low", "medium", "high", "__missing__"],
        ["easy", "moderate", "hard", "__missing__"],
    ]

    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    binary_cols = ["internet_access"]

    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(categories=[["no", "yes"]])),
        ]
    )

    target_cols: list[str] = ["gender", "course", "study_method"]

    target_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "target_encoder",
                TargetEncoder(
                    smooth="auto",
                    target_type="continuous",
                    random_state=config["training"]["random_state"],
                ),
            ),
        ]
    )

    numeric_cols: list[str] = [
        "age",
        "study_hours",
        "class_attendance",
        "sleep_hours",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    engineered_cols = [
        "attendance_frac",
        "effective_study",
        "effective_sleep",
        "study_sleep_balance",
        "mental_overload",
        "study_per_difficulty",
        "low_attendance",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal", ordinal_pipeline, ordinal_cols),
            ("binary", binary_pipeline, binary_cols),
            ("targetEncoding", target_pipeline, target_cols),
            ("numeric", numeric_pipeline, numeric_cols),
            ("engineered", "passthrough", engineered_cols),
        ],
        remainder="drop",
    )

    preprocessor.set_output(transform="pandas")

    return Pipeline(
        steps=[
            ("feature_engineering", feature_engineering),
            ("preprocessing", preprocessor),
        ]
    )
