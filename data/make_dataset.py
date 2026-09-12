"""Phase 0: load German Credit data, engineer a protected attribute, and produce
reproducible train / holdout / future-stream splits.

Run: python data/make_dataset.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
DATA_DIR = Path(__file__).parent
RAW_PATH = DATA_DIR / "raw_credit_g.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
HOLDOUT_PATH = DATA_DIR / "holdout.csv"
FUTURE_STREAM_PATH = DATA_DIR / "future_stream.csv"
META_PATH = DATA_DIR / "dataset_meta.json"

# Fraction of rows held out for the initial test set, and further split off
# into a "future stream" pool used later to simulate live batches.
HOLDOUT_FRAC = 0.2
FUTURE_STREAM_FRAC = 0.3  # fraction of the *holdout* set reserved as future stream


def age_band(age: int) -> str:
    if age < 25:
        return "under_25"
    if age < 40:
        return "25_to_39"
    if age < 60:
        return "40_to_59"
    return "60_plus"


def load_raw() -> pd.DataFrame:
    if RAW_PATH.exists():
        return pd.read_csv(RAW_PATH)
    bunch = fetch_openml(name="credit-g", version=1, as_frame=True)
    df = bunch.frame.copy()
    df.to_csv(RAW_PATH, index=False)
    return df


def build_dataset() -> pd.DataFrame:
    df = load_raw()

    # Target: 1 = bad credit risk (the event we want to predict/flag), 0 = good.
    df["target"] = (df["class"] == "bad").astype(int)
    df = df.drop(columns=["class"])

    # Protected attribute for the fairness gate (Phase 4): age band.
    df["age"] = df["age"].astype(int)
    df["age_band"] = df["age"].apply(age_band)

    return df


def main():
    df = build_dataset()

    # Split 1: train vs. holdout
    train_df, holdout_df = train_test_split(
        df, test_size=HOLDOUT_FRAC, random_state=RANDOM_STATE, stratify=df["target"]
    )

    # Split 2: carve a "future stream" pool out of the holdout set. This pool
    # simulates live traffic batches that we'll feed into the drift monitor
    # and (later) the shadow-mode challenger, so it must not touch training.
    holdout_df, future_stream_df = train_test_split(
        holdout_df,
        test_size=FUTURE_STREAM_FRAC,
        random_state=RANDOM_STATE,
        stratify=holdout_df["target"],
    )

    train_df.to_csv(TRAIN_PATH, index=False)
    holdout_df.to_csv(HOLDOUT_PATH, index=False)
    future_stream_df.to_csv(FUTURE_STREAM_PATH, index=False)

    meta = {
        "source": "OpenML credit-g (Statlog German Credit)",
        "random_state": RANDOM_STATE,
        "protected_attribute": "age_band",
        "target": "target (1 = bad credit risk)",
        "n_total": len(df),
        "n_train": len(train_df),
        "n_holdout": len(holdout_df),
        "n_future_stream": len(future_stream_df),
        "columns": list(df.columns),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    print(f"train={len(train_df)} holdout={len(holdout_df)} future_stream={len(future_stream_df)}")
    print(f"target rate (train)={train_df['target'].mean():.3f}")
    print(f"age_band distribution (train):\n{train_df['age_band'].value_counts()}")


if __name__ == "__main__":
    main()
