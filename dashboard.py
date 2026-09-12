"""Phase 5: CredFlux dashboard -- current champion, MLflow version history,
drift-monitor results, promotion/rejection log, and fairness gate results.

Run: streamlit run dashboard.py
"""
import json
import sys
from pathlib import Path

import mlflow
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from promote.audit import read_audit_log
from train.common import CHAMPION_MODEL_NAME, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, ROOT

st.set_page_config(page_title="CredFlux", layout="wide")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

st.title("CredFlux — Credit Risk Model Lifecycle")

# --- Current champion -------------------------------------------------
champion_path = ROOT / "promote" / "champion.json"
champion = json.loads(champion_path.read_text()) if champion_path.exists() else None

col1, col2, col3 = st.columns(3)
if champion:
    col1.metric("Champion version", champion["version"])
    col2.metric("Champion AUC (holdout)", f"{champion['auc']:.4f}")
    col3.metric("Candidate", champion.get("candidate_name", "n/a"))
else:
    st.warning("No champion registered yet — run train/train.py")

st.divider()

# --- MLflow version history -------------------------------------------
st.subheader("MLflow run history")
try:
    runs_df = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT])
    display_cols = [c for c in runs_df.columns if c.startswith("metrics.") or c.startswith("params.") or c in ("run_id", "start_time")]
    st.dataframe(runs_df[display_cols].sort_values("start_time", ascending=False), use_container_width=True)
    st.caption(f"{len(runs_df)} tracked runs total in experiment '{MLFLOW_EXPERIMENT}'")
except Exception as e:
    st.info(f"No MLflow runs found yet ({e})")

st.divider()

# --- Drift monitor results ----------------------------------------------
st.subheader("Drift monitor — injection harness results")
drift_results_path = ROOT / "monitor" / "drift_eval_results.json"
if drift_results_path.exists():
    drift_summary = json.loads(drift_results_path.read_text())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Features monitored", drift_summary["n_features_monitored"])
    c2.metric("Detection rate", f"{drift_summary['detection_rate_overall']*100:.1f}%")
    c3.metric("False-positive rate", f"{drift_summary['false_positive_rate']*100:.1f}%")
    c4.metric("Scenarios run", drift_summary["n_scenarios_total"])
    st.bar_chart(pd.Series(drift_summary["detection_rate_by_severity"], name="detection rate by severity"))
else:
    st.info("Run monitor/inject_and_evaluate.py to populate this section")

st.divider()

# --- Promotion / rejection / fairness audit log --------------------------
st.subheader("Promotion decisions & fairness gate audit trail")
audit_entries = read_audit_log()
if audit_entries:
    audit_df = pd.DataFrame(audit_entries)
    decision_counts = audit_df["decision"].value_counts()
    st.bar_chart(decision_counts)

    display_cols = [c for c in [
        "timestamp", "kind", "event_id", "severity", "drift_detected", "champion_auc",
        "challenger_auc", "auc_margin", "decision",
    ] if c in audit_df.columns]
    st.dataframe(audit_df[display_cols].sort_values("timestamp", ascending=False), use_container_width=True)

    st.markdown("**Fairness-gate headline case** (accurate challenger blocked for bias):")
    fairness_cases = audit_df[audit_df.get("kind") == "fairness_demo"]
    for _, row in fairness_cases.iterrows():
        fr = row["fairness"]
        st.json({
            "champion_auc": row["champion_auc"],
            "challenger_auc": row["challenger_auc"],
            "auc_margin": row["auc_margin"],
            "fpr_disparity": fr["fpr_disparity"],
            "worst_group": fr["worst_group"],
            "decision": row["decision"],
        })
else:
    st.info("Run promote/promote_loop.py, promote/promotion_mechanism_demo.py, fairness/blocked_demo.py to populate this section")
