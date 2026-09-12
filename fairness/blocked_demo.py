"""Phase 4's headline demo: a challenger that is MORE accurate than the
champion but gets BLOCKED from promotion for being more biased.

This isn't contrived noise -- CredFlux's own logreg champion has a real
~20 percentage-point FPR gap between age bands (under_25 gets flagged as
"bad credit" far more often than 40_to_59, at the same 0.5 threshold, even
on the full 800-row training set). This script surfaces that concretely as
a blocked promotion decision.

Run: python fairness/blocked_demo.py
"""
import sys
from pathlib import Path

import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))
from fairness.fairness import fairness_gate
from promote.audit import log_decision
from train.common import MLFLOW_TRACKING_URI, PROTECTED_COL, build_preprocessor, load_split, split_xy

AUC_PROMOTION_MARGIN = 0.01


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    train_df = load_split("train")
    holdout_df = load_split("holdout")
    X_holdout, y_holdout = split_xy(holdout_df)

    # A deliberately weak "current champion" (stand-in for a stale model),
    # same construction as promote/promotion_mechanism_demo.py.
    weak_train = train_df.sample(n=80, random_state=0)
    X_weak, y_weak = split_xy(weak_train)
    weak_model = Pipeline(steps=[
        ("preprocess", build_preprocessor(weak_train)),
        ("clf", LogisticRegression(max_iter=1000, C=0.001)),
    ])
    weak_model.fit(X_weak, y_weak)
    champion_auc = roc_auc_score(y_holdout, weak_model.predict_proba(X_holdout)[:, 1])

    # The "normal" full-data challenger -- more accurate, but carries the
    # champion's same age-band bias since it uses the identical feature set
    # and a plain 0.5 global threshold.
    X_train, y_train = split_xy(train_df)
    challenger_model = Pipeline(steps=[
        ("preprocess", build_preprocessor(train_df)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    challenger_model.fit(X_train, y_train)
    challenger_proba = challenger_model.predict_proba(X_holdout)[:, 1]
    challenger_auc = roc_auc_score(y_holdout, challenger_proba)
    challenger_preds = (challenger_proba >= 0.5).astype(int)

    auc_margin = challenger_auc - champion_auc
    fairness_result = fairness_gate(y_holdout.values, challenger_preds, holdout_df[PROTECTED_COL])
    decision = "rejected_insufficient_auc_gain"
    if auc_margin >= AUC_PROMOTION_MARGIN:
        decision = "promoted" if fairness_result["passed"] else "blocked_for_fairness"

    entry = {
        "kind": "fairness_demo",
        "note": (
            "Deliberate headline case: challenger beats champion on AUC but is blocked "
            "for exceeding the subgroup FPR disparity threshold. See SOP Phase 4."
        ),
        "champion_auc": round(champion_auc, 4),
        "challenger_auc": round(challenger_auc, 4),
        "auc_margin": round(auc_margin, 4),
        "auc_margin_required": AUC_PROMOTION_MARGIN,
        "fairness": fairness_result,
        "decision": decision,
    }
    log_decision(entry)

    print(f"champion AUC={champion_auc:.4f}  challenger AUC={challenger_auc:.4f}  (+{auc_margin:.4f})")
    print(f"fairness disparity={fairness_result['fpr_disparity']:.4f}  threshold={fairness_result['threshold']}")
    print(f"worst-treated group: {fairness_result['worst_group']}  best-treated group: {fairness_result['best_group']}")
    print(f"decision: {decision}")


if __name__ == "__main__":
    main()
