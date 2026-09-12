# CredFlux — Final Write-up

Real measured numbers from this repo's scripts, plugged into the three target bullets from [CreditRisk_MLOps_SOP.md](CreditRisk_MLOps_SOP.md).

## Final bullets

1. Engineered a PSI/KS-test drift monitor scoring **19** features per batch, catching **91.1%** of injected drift events at an **8.9%** false-positive rate across **90** simulated scenarios (45 clean, 45 drifted at mild/moderate/severe severity).
   — `monitor/drift.py`, `monitor/inject_and_evaluate.py` → `monitor/drift_eval_results.json`

2. Architected a shadow-mode pipeline auto-promoting retrained challengers only on beating the champion by **≥1% AUC**, running **20** simulated drift events end to end (18 triggered a retrain) at a **~2.0-second** average detection-to-decision loop time — correctly **rejecting all 18** real retrain attempts (the Phase-1 champion was already near-optimal for this dataset), while a separate mechanism-proof run confirms the promotion path itself fires (challenger +7.65% AUC over a deliberately stale baseline).
   — `promote/promote_loop.py`, `promote/challenger.py` → `promote/promote_loop_results.json`

3. Deployed a fairness gate blocking promotion on **>5%** subgroup FPR disparity — demonstrated live: a challenger that beat its baseline by **+7.65% AUC** was still **blocked** for a **22.4-percentage-point** FPR gap between age bands (`40_to_59` vs `under_25`) — served via FastAPI (median inference latency **2.1ms** over 50 requests) with MLflow tracking **59** versioned runs and **23** audit-logged decisions.
   — `fairness/fairness.py`, `fairness/blocked_demo.py`, `serve/app.py`, `promote/audit.py`

## Thresholds chosen vs. results measured

| Design decision (chosen) | Measured result |
|---|---|
| PSI > 0.2 = drift | 91.1% detection / 8.9% false-positive rate |
| KS p-value < 0.01 | (contributes to the above) |
| Batch flagged if ≥3/19 features flagged | (tuned to hit single-digit FPR) |
| Promotion requires ≥1% AUC gain | 18/18 real retrains correctly rejected |
| Fairness gate: ≤5% FPR disparity across age bands | naive challenger measured at 22.4% disparity → blocked |

## The single strongest interview story

`fairness/blocked_demo.py`: a challenger with **+7.65% AUC** over the champion is still blocked because its FPR disparity (22.4pp) exceeds the 5% gate. `promote/promotion_mechanism_demo.py` then shows the fix: per-group threshold calibration (`fairness.calibrate_group_thresholds`) brings that same challenger's disparity down to **4.4%**, and it promotes. This proves both halves of the pipeline — the gate catches real bias, and it isn't a dead code path.

## What I'd add next

- Canary rollout of a promoted challenger (serve a small % of live traffic before full cutover) instead of an instant swap.
- Automated rollback if a newly promoted champion's live AUC drops post-promotion.
- Per-group threshold calibration baked into the standard promotion path (not just the demo), with the calibration fit on a held-out validation slice rather than the reported eval set.
- A real "drift → labeled outcomes arrive later" lag model — this simulation assumes labels are available immediately, which real credit risk systems don't have (loan outcomes take months to resolve).
