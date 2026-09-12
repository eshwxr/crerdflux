# Credit Risk Model Lifecycle Platform — Build SOP
### Production MLOps: Train · Serve · Monitor · Promote Safely · Gate for Fairness

> **Ground rule (same as ThinkScope):** every feature exists to earn a specific number in a specific bullet. If a feature doesn't feed a bullet, don't build it. No Kubernetes, no cloud, no Airflow — local Docker is enough to prove "I can ship this."

---

## Target Bullets (the finish line — build backward from these)

**Objective:** Build a production MLOps platform for a credit-risk model with drift monitoring, safe champion-challenger promotion, and fairness-gated deployment.

1. Engineered a PSI/KS-test drift monitor scoring `[N]` features per batch, catching `[X]`% of injected drift events at `[X]`% false-positive rate across `[N]` simulated scenarios
2. Architected a shadow-mode pipeline auto-promoting retrained challengers only on beating the champion by `[X]`% AUC, cutting model staleness across `[N]` drift events to a `[X]`-min loop
3. Deployed a fairness gate blocking promotion on `>[X]%` subgroup FPR disparity, served via FastAPI/Docker with MLflow tracking `[N]`+ versioned runs and full audit logs

**Every `[X]` comes from a script in the repo. Keep those scripts — they are your interview proof.**

---

## Phase 0 — Setup & Dataset

**Goal:** a clean repo skeleton and a real credit-risk dataset flowing.

- Pick a public credit-risk / fraud dataset that has at least one usable protected attribute (age band, region, etc.) — you need this for the fairness gate later. Good candidates: German Credit, Give Me Some Credit, or a public loan-default set. Confirm it has a demographic column before committing.
- Repo structure: `/data`, `/train`, `/serve`, `/monitor`, `/promote`, `/fairness`, `/eval`, `/scripts`.
- Split the dataset into: initial training set, a holdout test set, and a "future stream" pool you'll feed in batches later to simulate live traffic.
- **Deliverable:** dataset loads, splits are reproducible from a script.

---

## Phase 1 — Train + Track + Serve (earns Bullet 3's serving/MLflow numbers)

**Goal:** a versioned model served behind an API, with every run tracked.

- Train 2-3 model types (Logistic Regression baseline + XGBoost/LightGBM). Compute AUC, precision/recall on the holdout.
- Wire up **MLflow**: log params, metrics, and the model artifact for every run. This is where your "`[N]`+ versioned runs" number comes from — it accumulates automatically as you experiment.
- Register the best run as the current **champion** in MLflow Model Registry (or a simple version table if you keep it lightweight).
- Build a **FastAPI `/predict` endpoint** that loads the champion and returns a risk score + probability.
- Add basic auth (API-key header is enough).
- **Dockerize** the whole thing.
- **Checkpoint numbers:** # of tracked runs, champion AUC, median `/predict` latency (measure across a batch of requests).

---

## Phase 2 — Drift Monitoring (earns Bullet 1)

**Goal:** a monitor that detects when incoming data no longer looks like training data, plus the harness that proves it works.

- Implement **PSI** (Population Stability Index) and/or **KS-test** per feature, comparing a new incoming batch against the training distribution.
- Set a drift threshold per feature (state your chosen cutoff — e.g. PSI > 0.2 = drift; this is a defensible standard value).
- **Build the drift-injection harness — this is what generates your two headline numbers:**
  - Take clean batches from your "future stream" pool.
  - Create drifted copies by synthetically shifting features (shift a mean, rescale variance, change a category's frequency) at varying severities.
  - Run the monitor across a mix of drifted and clean batches.
  - Record: **% of injected-drift scenarios correctly flagged** (true positive rate) and **% of clean batches wrongly flagged** (false positive rate).
- **Checkpoint numbers:** # features monitored, detection %, false-positive %, # of simulated scenarios run. All four go straight into Bullet 1.

---

## Phase 3 — Champion–Challenger Promotion (earns Bullet 2 — the star phase)

**Goal:** retraining that never blindly replaces a model — new models must *earn* promotion.

- When drift is detected (from Phase 2), trigger a **retrain** on the latest available data → this new model is the **challenger**.
- **Shadow mode:** the challenger scores the same incoming traffic as the champion, but its predictions are *logged, not used*. The champion stays in charge.
- **Promotion rule:** compare champion vs. challenger on recent labelled data. Promote the challenger *only if* it beats the champion by your set margin (e.g. ≥1% AUC). State this margin explicitly — it's a design decision, and being able to defend "why 1%" is a strong interview moment.
- On promotion: the challenger becomes the new champion, logged as a new MLflow version.
- **Simulate the full loop end to end** multiple times: inject drift → detect → retrain → shadow → compare → promote (or reject). Time how long detection-to-promotion takes.
- **Checkpoint numbers:** AUC improvement margin (your threshold), # of simulated drift events run through the full loop, detection-to-retrain loop time. These feed Bullet 2.
- **Keep the rejections.** If a challenger *fails* to beat the champion in some runs, that's a feature working correctly — "my pipeline correctly rejected N challengers that didn't improve" is a great thing to say.

---

## Phase 4 — Fairness Gate + Audit Trail (earns Bullet 3's fairness numbers + Responsible-AI story)

**Goal:** no model gets promoted if it's more biased, and every promotion decision is auditable.

- Before any promotion (from Phase 3), run a **fairness check**: compute **FPR and FNR per subgroup** of your protected attribute (e.g. age bands).
- **Gate rule:** block promotion if subgroup FPR (or FNR) disparity exceeds your threshold (e.g. > 5% gap). State the threshold; be ready to defend it.
- This means a challenger can be *more accurate* but still *blocked* for being *more biased* — build a test case that demonstrates exactly this. It's your single strongest interview story.
- **Audit trail:** log every promotion decision with — drift cause (which features shifted), champion-vs-challenger metrics, fairness check result, timestamp, and final decision (promoted / rejected / blocked-for-fairness). Store as structured JSON or a small DB table.
- **Checkpoint numbers:** subgroup disparity threshold, an example of a blocked-for-fairness model (accuracy gain vs. bias gain), # of audit entries logged.

---

## Phase 5 — Integration, Dashboard, Write-up

**Goal:** tie it together and make it demonstrable.

- A simple dashboard (Streamlit is fine — you already know it) showing: current champion, version history, drift scores over time, promotion/rejection log, fairness gate results. This is what you screen-share in an interview.
- End-to-end integration test: raw data batch in → drift detected → retrain → shadow → fairness gate → promote-or-block → audit logged. One command, full loop.
- Write the final bullets with your **real measured numbers** plugged into every `[X]`.
- Prepare the verbal walkthrough: be ready to explain **why champion-challenger over blind replacement**, **why PSI over KS (or both)**, **why your fairness threshold**, and **what you'd add next** (e.g. canary rollout, automated rollback on live performance drop) — naming the next step shows maturity even for things you didn't build.

---

## Hard Rules (same discipline as ThinkScope)

1. **Every number in every bullet traces to a script in the repo.** The drift harness (Phase 2) and the simulated-loop runner (Phase 3) are the two most important — they generate most of your numbers.
2. **Know which numbers are *thresholds you chose* vs. *results you measured*.** "I set the fairness gate at 5% FPR disparity" (design) ≠ "the monitor caught 92% of injected drift" (result). Mixing these up in an interview is where people get caught. Be crisp about which is which.
3. **Champion-challenger is the differentiator — don't cut it to save effort.** If runway gets tight, cut the dashboard polish or the second/third model type, not the shadow-mode promotion. That feature is why this project stands out.
4. **Keep every "correct rejection" and every "blocked-for-fairness" example.** They're not failures — they're proof your guardrails work, and they're stronger interview material than a clean 100% success rate.
5. **No cloud/K8s/Airflow.** Local Docker + FastAPI + MLflow + Streamlit is the whole stack. Every extra infra piece adds a shallow interview topic instead of deepening the ones you can defend.
6. **Depth check per feature:** for anything you build, make sure you can go 2-3 follow-up questions deep. If you can't explain why PSI works or why shadow mode matters, read up before the interview rather than adding more features to compensate.
