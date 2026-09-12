"""Phase 2: PSI + KS-test drift monitor.

Compares an incoming batch against the training distribution, per feature,
and flags drift when a feature crosses its threshold.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Defensible standard cutoffs (design decisions, not measured results):
# PSI > 0.2 = significant drift, 0.1-0.2 = moderate, <0.1 = stable.
PSI_THRESHOLD = 0.2
KS_PVALUE_THRESHOLD = 0.01  # stricter alpha; guards against multiple-testing false alarms across ~20 features
N_BINS = 5  # coarser bins keep PSI stable at realistic batch sizes (30-50 rows)


def _psi_from_counts(train_counts: pd.Series, batch_counts: pd.Series) -> float:
    # Add-one (Laplace) smoothing instead of clipping raw percentages: it
    # scales with sample size, so a bin with 0/30 batch rows doesn't produce
    # the same blown-up log-ratio as 0/300 would. Clipping to a fixed floor
    # (e.g. 1e-4) ignores batch size entirely and is what made small batches
    # spuriously "drift" on every rare category.
    train_pct = (train_counts + 1) / (train_counts.sum() + len(train_counts))
    batch_pct = (batch_counts + 1) / (batch_counts.sum() + len(batch_counts))
    return float(((batch_pct - train_pct) * np.log(batch_pct / train_pct)).sum())


def _psi_numeric(train_col: pd.Series, batch_col: pd.Series, bins: int = N_BINS) -> float:
    # Bin edges are fixed from the training distribution so both sides are
    # compared against the same reference buckets.
    quantile_edges = np.unique(np.quantile(train_col, np.linspace(0, 1, bins + 1)))
    if len(quantile_edges) < 3:
        return 0.0
    quantile_edges[0], quantile_edges[-1] = -np.inf, np.inf

    train_counts = pd.cut(train_col, quantile_edges).value_counts(sort=False)
    batch_counts = pd.cut(batch_col, quantile_edges).value_counts(sort=False)
    return _psi_from_counts(train_counts, batch_counts)


def _psi_categorical(train_col: pd.Series, batch_col: pd.Series) -> float:
    categories = sorted(set(train_col.unique()) | set(batch_col.unique()))
    train_counts = train_col.value_counts().reindex(categories, fill_value=0)
    batch_counts = batch_col.value_counts().reindex(categories, fill_value=0)
    return _psi_from_counts(train_counts, batch_counts)


def score_feature(train_col: pd.Series, batch_col: pd.Series) -> dict:
    is_numeric = pd.api.types.is_numeric_dtype(train_col)

    if is_numeric:
        psi = _psi_numeric(train_col, batch_col)
        ks_stat, ks_pvalue = ks_2samp(train_col, batch_col)
    else:
        psi = _psi_categorical(train_col.astype(str), batch_col.astype(str))
        ks_stat, ks_pvalue = None, None

    # PSI is the primary signal (robust to small-batch noise once binned by
    # train quantiles); KS is corroborating evidence only, at a stricter
    # alpha, since testing every feature independently at p<0.05 all but
    # guarantees a spurious flag somewhere once you have ~20 features.
    flagged = psi > PSI_THRESHOLD or (ks_pvalue is not None and ks_pvalue < KS_PVALUE_THRESHOLD)
    return {
        "psi": psi,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "flagged": flagged,
    }


def score_batch(train_df: pd.DataFrame, batch_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        result = score_feature(train_df[col], batch_df[col])
        rows.append({"feature": col, **result})
    return pd.DataFrame(rows)


def batch_is_drifted(scored: pd.DataFrame, min_features_flagged: int = 3) -> bool:
    # Require multiple features to agree before calling the whole batch
    # drifted: with ~20 features tested independently, a single flagged
    # feature is expected noise, not a batch-level drift event.
    return int(scored["flagged"].sum()) >= min_features_flagged
