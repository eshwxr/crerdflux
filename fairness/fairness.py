"""Phase 4: fairness gate.

Computes per-subgroup FPR/FNR and blocks promotion if the disparity across
subgroups of the protected attribute exceeds a threshold.
"""
import numpy as np
import pandas as pd

# Design decision (not measured): block promotion if the gap between the
# best- and worst-treated subgroup's FPR exceeds 5 percentage points.
FPR_DISPARITY_THRESHOLD = 0.05


def subgroup_rates(y_true: np.ndarray, y_pred: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred), "group": groups.values})
    rows = []
    for group, sub in df.groupby("group"):
        negatives = sub[sub["y_true"] == 0]
        positives = sub[sub["y_true"] == 1]
        fpr = (negatives["y_pred"] == 1).mean() if len(negatives) else np.nan
        fnr = (positives["y_pred"] == 0).mean() if len(positives) else np.nan
        rows.append({"group": group, "n": len(sub), "fpr": fpr, "fnr": fnr})
    return pd.DataFrame(rows)


def fairness_gate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series,
    threshold: float = FPR_DISPARITY_THRESHOLD,
) -> dict:
    rates = subgroup_rates(y_true, y_pred, groups)
    valid_fpr = rates["fpr"].dropna()

    if len(valid_fpr) < 2:
        disparity = 0.0
    else:
        disparity = float(valid_fpr.max() - valid_fpr.min())

    passed = disparity <= threshold
    worst_group = valid_fpr.idxmax() if len(valid_fpr) else None
    best_group = valid_fpr.idxmin() if len(valid_fpr) else None

    return {
        "passed": passed,
        "fpr_disparity": round(disparity, 4),
        "threshold": threshold,
        "worst_group": rates.loc[worst_group, "group"] if worst_group is not None else None,
        "best_group": rates.loc[best_group, "group"] if best_group is not None else None,
        "subgroup_rates": rates.round(4).to_dict(orient="records"),
    }


def calibrate_group_thresholds(proba: np.ndarray, y_true: np.ndarray, groups: pd.Series, target_fpr: float) -> dict:
    """Bias-mitigation helper: pick a per-group decision threshold so each
    subgroup's FPR lands near `target_fpr`, instead of using one global 0.5
    cutoff for every group (equalized-odds-style post-processing).
    """
    df = pd.DataFrame({"proba": proba, "y_true": np.asarray(y_true), "group": groups.values})
    thresholds = {}
    for group, sub in df.groupby("group"):
        negatives = sub[sub["y_true"] == 0]
        if len(negatives) == 0:
            thresholds[group] = 0.5
            continue
        candidate_thresholds = np.linspace(0.01, 0.99, 99)
        best_t, best_gap = 0.5, float("inf")
        for t in candidate_thresholds:
            fpr = (negatives["proba"] >= t).mean()
            gap = abs(fpr - target_fpr)
            if gap < best_gap:
                best_t, best_gap = t, gap
        thresholds[group] = float(best_t)
    return thresholds


def apply_group_thresholds(proba: np.ndarray, groups: pd.Series, thresholds: dict) -> np.ndarray:
    group_arr = groups.values
    preds = np.zeros(len(proba), dtype=int)
    for i, (p, g) in enumerate(zip(proba, group_arr)):
        preds[i] = int(p >= thresholds.get(g, 0.5))
    return preds
