"""Phase 3: train a challenger model on the latest available data.

A challenger is trained the same way a champion is (Phase 1), but on
whatever data has accumulated since the champion was last trained — the
retrain trigger is a detected drift event, not a schedule.
"""
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))
from train.common import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, build_preprocessor, split_xy


def train_challenger(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    random_state: int,
    run_name: str = "challenger",
) -> dict:
    """Trains one challenger candidate and logs it as an MLflow run.
    Returns run metadata plus the fitted model (kept in-memory for shadow scoring).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X_train, y_train = split_xy(train_df)
    X_eval, y_eval = split_xy(eval_df)
    preprocessor = build_preprocessor(train_df)

    with mlflow.start_run(run_name=run_name) as run:
        model = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state, C=1.0)),
        ])
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_eval)[:, 1]
        auc = roc_auc_score(y_eval, proba)

        mlflow.log_param("model_type", "logreg_challenger")
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_train_rows", len(train_df))
        mlflow.log_metric("auc", auc)
        mlflow.sklearn.log_model(model, artifact_path="model", serialization_format="pickle")

        return {
            "run_id": run.info.run_id,
            "auc": float(auc),
            "model": model,
            "n_train_rows": len(train_df),
        }
