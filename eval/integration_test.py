"""Phase 5: one-command, end-to-end integration test.

raw batch in -> drift detected -> retrain (challenger) -> shadow mode ->
fairness gate -> promote-or-block -> audit logged.

This exercises every phase's code path once, linearly, with assertions at
each stage, as opposed to promote/promote_loop.py's many-iteration
stochastic simulation (which produces the headline numbers).

Run: python eval/integration_test.py
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
from promote.audit import log_decision, read_audit_log
from promote.challenger import train_challenger
from train.common import MLFLOW_TRACKING_URI, PROTECTED_COL, ROOT, load_split, split_xy

AUC_PROMOTION_MARGIN = 0.01
RANDOM_STATE = 123


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    rng = np.random.default_rng(RANDOM_STATE)

    train_df = load_split("train")
    holdout_df = load_split("holdout")
    future_pool = load_split("future_stream")
    cols = feature_cols(train_df)
    audit_len_before = len(read_audit_log())

    loop_start = time.perf_counter()

    # 1. Raw batch in, drift-injected to simulate a real distribution shift.
    raw_batch = sample_batch(future_pool, rng, size=50)
    drifted_batch = inject_drift(raw_batch, train_df, severity="severe", rng=rng)
    print("[1/6] raw batch sampled + drift injected")

    # 2. Drift detected.
    scored = score_batch(train_df, drifted_batch, cols)
    drifted = batch_is_drifted(scored)
    assert drifted, "integration test expects a severe-severity batch to be detected"
    print(f"[2/6] drift detected: {drifted} ({int(scored['flagged'].sum())} features flagged)")

    # 3. Retrain triggered -> challenger (shadow mode: scored, not served).
    new_labeled_data = sample_batch(future_pool, rng, size=15)
    retrain_df = pd.concat([train_df, new_labeled_data], ignore_index=True)
    challenger_result = train_challenger(retrain_df, holdout_df, random_state=RANDOM_STATE, run_name="integration_test_challenger")
    print(f"[3/6] challenger retrained + shadow-scored: AUC={challenger_result['auc']:.4f}")

    # 4. Compare champion vs. challenger.
    champion = json.loads((ROOT / "promote" / "champion.json").read_text())
    champion_model = mlflow.sklearn.load_model(champion["model_uri"])
    X_holdout, y_holdout = split_xy(holdout_df)
    champion_auc = float(roc_auc_score(y_holdout, champion_model.predict_proba(X_holdout)[:, 1]))
    auc_margin = challenger_result["auc"] - champion_auc
    print(f"[4/6] compared: champion AUC={champion_auc:.4f} vs challenger AUC={challenger_result['auc']:.4f} (margin={auc_margin:+.4f})")

    # 5. Fairness gate.
    challenger_preds = (challenger_result["model"].predict_proba(X_holdout)[:, 1] >= 0.5).astype(int)
    fairness_result = fairness_gate(y_holdout.values, challenger_preds, holdout_df[PROTECTED_COL])
    print(f"[5/6] fairness gate: disparity={fairness_result['fpr_disparity']:.4f} passed={fairness_result['passed']}")

    # 6. Decide + audit log.
    if auc_margin < AUC_PROMOTION_MARGIN:
        decision = "rejected_insufficient_auc_gain"
    elif not fairness_result["passed"]:
        decision = "blocked_for_fairness"
    else:
        decision = "promoted"

    entry = {
        "kind": "integration_test",
        "champion_auc": round(champion_auc, 4),
        "challenger_auc": round(challenger_result["auc"], 4),
        "auc_margin": round(auc_margin, 4),
        "fairness": fairness_result,
        "decision": decision,
        "loop_time_sec": round(time.perf_counter() - loop_start, 4),
    }
    log_decision(entry)
    audit_len_after = len(read_audit_log())
    assert audit_len_after == audit_len_before + 1, "expected exactly one new audit entry"
    print(f"[6/6] decision={decision}, logged to audit trail (entries {audit_len_before} -> {audit_len_after})")

    print("\nINTEGRATION TEST PASSED: all 6 stages executed and one audit entry was recorded.")


if __name__ == "__main__":
    main()
