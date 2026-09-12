"""Shared preprocessing + IO helpers used by train, serve, monitor, promote."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MLFLOW_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT = "credflux-credit-risk"
CHAMPION_MODEL_NAME = "credflux-champion"

TARGET_COL = "target"
PROTECTED_COL = "age_band"
# age is dropped as a raw feature: age_band (the protected attribute) is what
# the fairness gate reasons about, and keeping both risks the model learning
# age directly instead of through the band.
DROP_COLS = [TARGET_COL, PROTECTED_COL, "age"]


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv")


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    feature_df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    numeric_cols = feature_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols, categorical_cols = feature_columns(df)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = df[TARGET_COL]
    return X, y
