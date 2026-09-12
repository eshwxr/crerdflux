# CredFlux — Credit Risk MLOps Platform

A production-style MLOps platform for a **credit-risk model**, built to demonstrate the complete model lifecycle:

**Train → Track → Serve → Monitor Drift → Retrain → Shadow Test → Fairness Gate → Promote/Reject → Audit**

CredFlux focuses on the parts of an ML system that matter after a model has been trained: detecting data drift, safely evaluating new models against a champion, preventing unfair promotions, and keeping an auditable record of every decision.

---

## 🎯 Project Highlights

- **19-feature PSI/KS drift monitoring**
- **91.1% drift detection rate** across 90 simulated scenarios
- **8.9% false-positive rate**
- **Champion–challenger promotion pipeline** with a ≥1% AUC improvement requirement
- **20 end-to-end simulated drift events**, with 18 triggering retraining
- **FastAPI inference API** with **2.1 ms median latency** over 50 requests
- **MLflow tracking with 59 versioned runs**
- **Fairness gate** blocking models with >5% subgroup FPR disparity
- Demonstrated fairness failure: **+7.65% AUC improvement but 22.4 percentage-point FPR disparity → blocked**
- **23 audit-logged decisions**
- Streamlit dashboard for model, drift, promotion, and fairness monitoring

---

## 🧠 Architecture

```text
                         ┌──────────────────────┐
                         │   Credit Risk Data   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Model Training     │
                         │ Logistic / XGBoost   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       MLflow         │
                         │ Params / Metrics /   │
                         │ Model Versions       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Champion        │
                         └──────────┬───────────┘
                                    │
                 Incoming Batch ───┼──────────────┐
                                    │              │
                                    ▼              ▼
                         ┌────────────────┐  ┌──────────────┐
                         │ Drift Monitor  │  │ FastAPI      │
                         │ PSI + KS Test  │  │ /predict     │
                         └───────┬────────┘  └──────────────┘
                                 │
                           Drift detected
                                 │
                                 ▼
                         ┌──────────────────────┐
                         │      Retraining      │
                         │      Challenger      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Shadow Mode       │
                         │ Champion vs          │
                         │ Challenger           │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   AUC Promotion      │
                         │     Gate ≥ 1%        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Fairness Gate      │
                         │ FPR disparity ≤ 5%   │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                    ┌──────────┐          ┌──────────┐
                    │ PROMOTE  │          │  BLOCK / │
                    │          │          │  REJECT  │
                    └────┬─────┘          └────┬─────┘
                         │                     │
                         └──────────┬──────────┘
                                    ▼
                         ┌──────────────────────┐
                         │    Audit Trail       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Streamlit Dashboard  │
                         └──────────────────────┘
```

---

## 📂 Repository Structure

```text
CredFlux/
│
├── data/                    # Dataset and data preparation
├── train/                   # Model training and MLflow tracking
├── serve/                   # FastAPI inference service
├── monitor/                 # PSI/KS drift monitoring + injection harness
├── promote/                 # Champion-challenger promotion logic
├── fairness/                # Fairness metrics and promotion gate
├── eval/                    # Evaluation utilities
├── scripts/                 # Supporting scripts
│
├── dashboard.py             # Streamlit monitoring dashboard
├── mlflow.db                # Local MLflow tracking database
├── requirements.txt         # Python dependencies
├── CreditRisk_MLOps_SOP.md  # Build / implementation SOP
└── WRITEUP.md               # Detailed measured results
```

---

## 📊 Results

### 1. Drift Monitoring

CredFlux compares incoming batches against the training distribution using **Population Stability Index (PSI)** and the **Kolmogorov–Smirnov (KS) test**.

| Metric | Result |
|---|---:|
| Features monitored | 19 |
| Simulated scenarios | 90 |
| Clean scenarios | 45 |
| Drifted scenarios | 45 |
| Drift detection rate | **91.1%** |
| False-positive rate | **8.9%** |
| PSI drift threshold | **> 0.2** |
| KS threshold | **p < 0.01** |
| Batch drift rule | **≥3/19 features flagged** |

The drift injection harness creates synthetic distribution shifts at different severities and evaluates whether the monitor catches them.

---

### 2. Champion–Challenger Promotion

A retrained model never automatically replaces the production champion.

The challenger first runs in **shadow mode**, where its predictions are logged while the champion continues serving traffic.

Promotion requires:

```text
Challenger AUC ≥ Champion AUC + 1%
```

Measured results:

- 20 simulated drift events
- 18 events triggered retraining
- ~2.0 seconds average detection-to-decision loop
- All 18 real retraining attempts were correctly rejected because the existing champion was already near-optimal
- A separate mechanism test demonstrated successful promotion with a challenger achieving **+7.65% AUC**

This distinction is important: a rejected challenger is not a failure. It demonstrates that the safety mechanism is working.

---

### 3. Fairness Gate

Before promotion, CredFlux evaluates subgroup **False Positive Rate (FPR)** and **False Negative Rate (FNR)**.

The configured promotion rule is:

```text
FPR disparity ≤ 5%
```

The strongest test case:

> A challenger improved AUC by **+7.65%**, but produced a **22.4 percentage-point FPR disparity** between age groups.

Result:

```text
Accuracy improvement:  +7.65% AUC
Fairness disparity:    22.4 percentage points
Allowed disparity:     5%
Decision:              BLOCKED
```

A separate mechanism-proof run applies per-group threshold calibration, reducing the disparity to **4.4%**, allowing the challenger to promote.

---

## 🚀 Running the Dashboard

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

The dashboard exposes:

- Current champion version
- Champion holdout AUC
- MLflow run history
- Drift detection results
- Detection rate and false-positive rate
- Promotion/rejection history
- Fairness-gate results
- Audit trail
- Fairness-blocked demonstration cases

---

## 🔬 Running the Main Experiments

### Drift evaluation

```bash
python monitor/inject_and_evaluate.py
```

This populates:

```text
monitor/drift_eval_results.json
```

### Champion–challenger loop

```bash
python promote/promote_loop.py
```

### Promotion mechanism demonstration

```bash
python promote/promotion_mechanism_demo.py
```

### Fairness-block demonstration

```bash
python fairness/blocked_demo.py
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| ML | scikit-learn, XGBoost |
| Data | Pandas, NumPy |
| Experiment Tracking | MLflow |
| API | FastAPI |
| API Server | Uvicorn |
| Validation / Statistics | SciPy |
| Dashboard | Streamlit |
| Data Drift | PSI + KS Test |
| Model Governance | Champion–Challenger |
| Responsible AI | FPR/FNR subgroup fairness gate |
| Storage | Local MLflow DB + JSON audit logs |

The project intentionally uses **local infrastructure** rather than Kubernetes, cloud services, or Airflow. The goal is to demonstrate depth in the ML lifecycle rather than add infrastructure for the sake of it.

---

## 🔐 Model Governance & Auditability

Every promotion decision records information such as:

- Timestamp
- Event ID
- Drift severity
- Drift detection status
- Champion AUC
- Challenger AUC
- AUC improvement margin
- Fairness metrics
- Final decision

Possible outcomes include:

```text
PROMOTED
REJECTED
BLOCKED_FOR_FAIRNESS
```

This creates a reproducible trail explaining **why a model was or was not promoted**.

---

## 💡 Key Design Decisions

### Why PSI + KS?

PSI provides an interpretable measure of distribution shift, while the KS test provides a statistical comparison between distributions. Using both gives complementary signals for monitoring feature drift.

### Why Champion–Challenger?

Blindly replacing a production model after retraining is risky. Shadow mode lets the challenger prove itself against the existing champion before it takes control.

### Why a Fairness Gate?

A model can improve aggregate performance while becoming significantly worse for a subgroup. CredFlux therefore treats fairness as a **promotion constraint**, not just an after-the-fact report.

### Why keep rejected models?

Rejected challengers provide evidence that the system's guardrails are functioning correctly. They also make the lifecycle easier to audit and explain.

---

## ⚠️ Current Limitations

This is a local MLOps demonstration rather than a production banking deployment.

Current limitations include:

- Simulated future traffic and injected drift
- Labels are assumed to become available immediately in the simulation
- No cloud deployment
- No Kubernetes
- No Airflow
- No live canary rollout
- Fairness calibration is demonstrated separately rather than being fully embedded into the standard promotion path

---

## 🔮 Next Steps

Potential production extensions:

1. **Canary rollout** — send a small percentage of traffic to a newly promoted model before full cutover.
2. **Automated rollback** — revert the champion if live performance deteriorates.
3. **Integrated group threshold calibration** — include calibration directly in the standard promotion workflow.
4. **Delayed-label monitoring** — model the real-world lag between issuing a loan and receiving its eventual outcome.
5. **Production data connectors** — replace simulated streams with real monitored data sources.

---

## 📌 Resume-Ready Summary

**CredFlux — Credit Risk MLOps Platform**

Built a production-style credit-risk MLOps platform with PSI/KS drift monitoring, champion–challenger shadow deployment, MLflow model tracking, FastAPI serving, and fairness-gated promotion; detected **91.1%** of injected drift across **90** scenarios at **8.9%** false positives, and blocked a **+7.65% AUC** challenger due to **22.4pp** subgroup FPR disparity.

---

## 📚 Project Documentation

- `CreditRisk_MLOps_SOP.md` — implementation plan and engineering requirements
- `WRITEUP.md` — measured results and interview-ready findings
- `dashboard.py` — Streamlit monitoring dashboard
- `requirements.txt` — Python dependencies

