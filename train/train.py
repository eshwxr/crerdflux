"""Phase 1: train candidate models, track every run in MLflow, and register the
best one as the champion.

Run: python train/train.py
"""
import json
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))
from train.common import (
    CHAMPION_MODEL_NAME,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    ROOT,
    build_preprocessor,
    load_split,
    split_xy,
)

CHAMPION_PATH = ROOT / "promote" / "champion.json"

CANDIDATES = {
    "logreg_baseline": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "xgboost": lambda: XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42,
    ),
    "xgboost_deep": lambda: XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.03,
        eval_metric="logloss",
        random_state=42,
    ),
}


def evaluate(model, X_holdout, y_holdout):
    proba = model.predict_proba(X_holdout)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_holdout, proba),
        "precision": precision_score(y_holdout, preds, zero_division=0),
        "recall": recall_score(y_holdout, preds, zero_division=0),
    }


def train_one(name, build_fn, X_train, y_train, X_holdout, y_holdout, preprocessor):
    with mlflow.start_run(run_name=name) as run:
        model = Pipeline(steps=[("preprocess", preprocessor), ("clf", build_fn())])
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_holdout, y_holdout)

        mlflow.log_param("model_type", name)
        mlflow.log_metrics(metrics)
        # pickle (not skops) so the Pipeline can embed an XGBClassifier step.
        mlflow.sklearn.log_model(model, artifact_path="model", serialization_format="pickle")

        print(f"[{name}] AUC={metrics['auc']:.4f} precision={metrics['precision']:.4f} recall={metrics['recall']:.4f}")
        return run.info.run_id, metrics


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    train_df = load_split("train")
    holdout_df = load_split("holdout")
    X_train, y_train = split_xy(train_df)
    X_holdout, y_holdout = split_xy(holdout_df)
    preprocessor = build_preprocessor(train_df)

    results = []
    for name, build_fn in CANDIDATES.items():
        run_id, metrics = train_one(name, build_fn, X_train, y_train, X_holdout, y_holdout, preprocessor)
        results.append({"name": name, "run_id": run_id, **metrics})

    best = max(results, key=lambda r: r["auc"])
    model_uri = f"runs:/{best['run_id']}/model"
    registered = mlflow.register_model(model_uri, CHAMPION_MODEL_NAME)

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=CHAMPION_MODEL_NAME,
        version=registered.version,
        stage="Production",
        archive_existing_versions=True,
    )

    champion = {
        "model_name": CHAMPION_MODEL_NAME,
        "version": registered.version,
        "run_id": best["run_id"],
        "model_uri": f"models:/{CHAMPION_MODEL_NAME}/{registered.version}",
        "candidate_name": best["name"],
        "auc": best["auc"],
        "precision": best["precision"],
        "recall": best["recall"],
    }
    CHAMPION_PATH.parent.mkdir(exist_ok=True)
    CHAMPION_PATH.write_text(json.dumps(champion, indent=2))

    n_runs = len(mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT]))
    print(f"\nChampion: {best['name']} (v{registered.version}) AUC={best['auc']:.4f}")
    print(f"Total tracked runs in experiment: {n_runs}")


if __name__ == "__main__":
    main()
