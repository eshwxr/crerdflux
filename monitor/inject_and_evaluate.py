"""Phase 2: drift-injection harness.

Generates clean and synthetically-drifted batches from the "future stream"
pool, runs the PSI/KS monitor on each, and records the true-positive
(detection) rate and false-positive rate across many simulated scenarios.

Run: python monitor/inject_and_evaluate.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from monitor.drift import KS_PVALUE_THRESHOLD, PSI_THRESHOLD, batch_is_drifted, score_batch
from train.common import DROP_COLS, ROOT, load_split

RESULTS_PATH = ROOT / "monitor" / "drift_eval_results.json"
RANDOM_STATE = 7
BATCH_SIZE = 50
N_SCENARIOS_PER_SEVERITY = 15  # clean batches per severity level get the same count
SEVERITIES = ["mild", "moderate", "severe"]

NUMERIC_SHIFT_FACTORS = {"mild": 0.5, "moderate": 1.0, "severe": 2.0}
CATEGORY_SWAP_FRAC = {"mild": 0.15, "moderate": 0.35, "severe": 0.6}


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in DROP_COLS]


def sample_batch(pool: pd.DataFrame, rng: np.random.Generator, size: int = BATCH_SIZE) -> pd.DataFrame:
    idx = rng.choice(pool.index, size=size, replace=True)
    return pool.loc[idx].reset_index(drop=True)


def inject_drift(batch: pd.DataFrame, train_df: pd.DataFrame, severity: str, rng: np.random.Generator) -> pd.DataFrame:
    """Shift a random subset of features to simulate real-world drift:
    numeric features get a mean shift + variance rescale, categorical
    features get some fraction of values swapped to a different category.
    """
    drifted = batch.copy()
    cols = feature_cols(batch)
    n_to_drift = max(1, len(cols) // 3)
    drift_cols = rng.choice(cols, size=n_to_drift, replace=False)

    for col in drift_cols:
        if pd.api.types.is_numeric_dtype(train_df[col]):
            std = train_df[col].std() or 1.0
            shift = NUMERIC_SHIFT_FACTORS[severity] * std
            scale = 1.0 + NUMERIC_SHIFT_FACTORS[severity] * 0.4
            drifted[col] = drifted[col] * scale + shift
        else:
            other_categories = [c for c in train_df[col].unique() if c not in drifted[col].unique()]
            pool_categories = other_categories or train_df[col].unique().tolist()
            frac = CATEGORY_SWAP_FRAC[severity]
            n_swap = int(len(drifted) * frac)
            swap_idx = rng.choice(drifted.index, size=n_swap, replace=False)
            drifted.loc[swap_idx, col] = rng.choice(pool_categories, size=n_swap)

    return drifted


def run():
    rng = np.random.default_rng(RANDOM_STATE)
    train_df = load_split("train")
    future_pool = load_split("future_stream")
    cols = feature_cols(train_df)

    scenario_results = []

    # Clean batches: should NOT be flagged. These generate the false-positive rate.
    for i in range(N_SCENARIOS_PER_SEVERITY * len(SEVERITIES)):
        batch = sample_batch(future_pool, rng)
        scored = score_batch(train_df, batch, cols)
        scenario_results.append({
            "scenario_id": f"clean_{i}",
            "type": "clean",
            "severity": None,
            "n_features_flagged": int(scored["flagged"].sum()),
            "flagged": batch_is_drifted(scored),
        })

    # Drifted batches at each severity: SHOULD be flagged. These generate the detection rate.
    for severity in SEVERITIES:
        for i in range(N_SCENARIOS_PER_SEVERITY):
            batch = sample_batch(future_pool, rng)
            drifted_batch = inject_drift(batch, train_df, severity, rng)
            scored = score_batch(train_df, drifted_batch, cols)
            scenario_results.append({
                "scenario_id": f"{severity}_{i}",
                "type": "drifted",
                "severity": severity,
                "n_features_flagged": int(scored["flagged"].sum()),
                "flagged": batch_is_drifted(scored),
            })

    results_df = pd.DataFrame(scenario_results)

    clean = results_df[results_df["type"] == "clean"]
    drifted = results_df[results_df["type"] == "drifted"]

    false_positive_rate = clean["flagged"].mean()
    detection_rate = drifted["flagged"].mean()
    detection_by_severity = drifted.groupby("severity")["flagged"].mean().to_dict()

    summary = {
        "n_features_monitored": len(cols),
        "n_scenarios_total": len(results_df),
        "n_clean_scenarios": len(clean),
        "n_drifted_scenarios": len(drifted),
        "detection_rate_overall": round(float(detection_rate), 4),
        "false_positive_rate": round(float(false_positive_rate), 4),
        "detection_rate_by_severity": {k: round(float(v), 4) for k, v in detection_by_severity.items()},
        "psi_threshold": PSI_THRESHOLD,
        "ks_pvalue_threshold": KS_PVALUE_THRESHOLD,
    }

    RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
