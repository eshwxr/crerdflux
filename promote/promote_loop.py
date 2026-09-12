"""Phase 3 + 4: simulate the full champion-challenger loop end to end, many
times over: inject drift -> detect -> retrain (shadow) -> compare -> fairness
gate -> promote / reject / block. Every decision is audit-logged.

Run: python promote/promote_loop.py
"""
import json
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from fairness.fairness import fairness_gate
from monitor.drift import batch_is_drifted, score_batch
from monitor.inject_and_evaluate import feature_cols, inject_drift, sample_batch
from promote.audit import log_decision
from promote.challenger import train_challenger
from train.common import (
    CHAMPION_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    PROTECTED_COL,
    ROOT,
    load_split,
    split_xy,
)

CHAMPION_PATH = ROOT / "promote" / "champion.json"
RESULTS_PATH = ROOT / "promote" / "promote_loop_results.json"

RANDOM_STATE = 11
N_EVENTS = 20
AUC_PROMOTION_MARGIN = 0.01  # challenger must beat champion by >=1% AUC to be promoted (design decision)
SEVERITIES = ["mild", "moderate", "severe"]
NEW_DATA_BATCH_SIZE = 15
DRIFT_CHECK_BATCH_SIZE = 50


def evaluate_auc(model, df: pd.DataFrame) -> float:
    X, y = split_xy(df)
    proba = model.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, proba))


def evaluate_fairness(model, df: pd.DataFrame, threshold: float = 0.5) -> dict:
    X, y = split_xy(df)
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    return fairness_gate(y.values, preds, df[PROTECTED_COL])


def run():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    rng = np.random.default_rng(RANDOM_STATE)

    train_df = load_split("train")
    holdout_df = load_split("holdout")
    future_pool = load_split("future_stream")
    cols = feature_cols(train_df)

    champion = json.loads(CHAMPION_PATH.read_text())
    champion_model = mlflow.sklearn.load_model(champion["model_uri"])
    champion_auc = evaluate_auc(champion_model, holdout_df)
    champion_version = champion["version"]

    current_train_df = train_df.copy()
    outcomes = []

    for i in range(N_EVENTS):
        loop_start = time.perf_counter()
        severity = SEVERITIES[i % len(SEVERITIES)]

        # 1. Unlabeled incoming traffic used purely to check for feature drift.
        drift_check_batch = inject_drift(
            sample_batch(future_pool, rng, size=DRIFT_CHECK_BATCH_SIZE), train_df, severity, rng
        )
        scored = score_batch(train_df, drift_check_batch, cols)
        drifted = batch_is_drifted(scored)
        flagged_features = scored.loc[scored["flagged"], "feature"].tolist()

        entry = {
            "event_id": i,
            "severity": severity,
            "drift_detected": drifted,
            "flagged_features": flagged_features,
        }

        if not drifted:
            entry["decision"] = "no_action"
            entry["loop_time_sec"] = round(time.perf_counter() - loop_start, 4)
            outcomes.append(entry)
            log_decision(entry)
            continue

        # 2. Retrain trigger: new labeled data has trickled in since the
        # champion was last trained -> fold it into the training set.
        new_labeled_data = sample_batch(future_pool, rng, size=NEW_DATA_BATCH_SIZE)
        current_train_df = pd.concat([current_train_df, new_labeled_data], ignore_index=True)

        # 3. Shadow mode: train the challenger, score it against the champion
        # on the same recent labelled data (holdout) -- challenger's scores
        # are logged for comparison only, never served.
        challenger_result = train_challenger(
            current_train_df, holdout_df, random_state=RANDOM_STATE + i, run_name=f"challenger_event_{i}"
        )
        challenger_model = challenger_result["model"]
        challenger_auc = challenger_result["auc"]
        auc_margin = challenger_auc - champion_auc

        entry["champion_auc"] = round(champion_auc, 4)
        entry["challenger_auc"] = round(challenger_auc, 4)
        entry["auc_margin"] = round(auc_margin, 4)
        entry["auc_margin_required"] = AUC_PROMOTION_MARGIN
        entry["challenger_run_id"] = challenger_result["run_id"]

        if auc_margin < AUC_PROMOTION_MARGIN:
            entry["decision"] = "rejected_insufficient_auc_gain"
            entry["loop_time_sec"] = round(time.perf_counter() - loop_start, 4)
            outcomes.append(entry)
            log_decision(entry)
            continue

        # 4. Fairness gate: a challenger that beats the champion on AUC can
        # still be blocked if it's more biased across the protected attribute.
        fairness_result = evaluate_fairness(challenger_model, holdout_df)
        entry["fairness"] = fairness_result

        if not fairness_result["passed"]:
            entry["decision"] = "blocked_for_fairness"
            entry["loop_time_sec"] = round(time.perf_counter() - loop_start, 4)
            outcomes.append(entry)
            log_decision(entry)
            continue

        # 5. Promote: challenger becomes the new champion.
        model_uri = f"runs:/{challenger_result['run_id']}/model"
        registered = mlflow.register_model(model_uri, CHAMPION_MODEL_NAME)
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=CHAMPION_MODEL_NAME, version=registered.version, stage="Production", archive_existing_versions=True
        )

        champion_model = challenger_model
        champion_auc = challenger_auc
        champion_version = registered.version
        CHAMPION_PATH.write_text(json.dumps({
            "model_name": CHAMPION_MODEL_NAME,
            "version": champion_version,
            "run_id": challenger_result["run_id"],
            "model_uri": f"models:/{CHAMPION_MODEL_NAME}/{champion_version}",
            "candidate_name": f"challenger_event_{i}",
            "auc": champion_auc,
        }, indent=2))

        entry["decision"] = "promoted"
        entry["new_champion_version"] = champion_version
        entry["loop_time_sec"] = round(time.perf_counter() - loop_start, 4)
        outcomes.append(entry)
        log_decision(entry)

    outcomes_df = pd.DataFrame(outcomes)
    retrain_triggered = outcomes_df[outcomes_df["decision"] != "no_action"]
    decision_counts = outcomes_df["decision"].value_counts().to_dict()

    summary = {
        "n_events_simulated": N_EVENTS,
        "n_drift_detected_retrain_triggered": len(retrain_triggered),
        "auc_promotion_margin": AUC_PROMOTION_MARGIN,
        "decision_counts": decision_counts,
        "avg_detection_to_decision_loop_time_sec": round(float(retrain_triggered["loop_time_sec"].mean()), 4)
        if len(retrain_triggered) else None,
        "final_champion_version": champion_version,
        "final_champion_auc": round(champion_auc, 4),
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
