"""Phase 1: FastAPI serving layer for the current champion model.

Run: uvicorn serve.app:app --reload
"""
import json
import os
import sys
import time
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from train.common import DROP_COLS, MLFLOW_TRACKING_URI, ROOT

CHAMPION_PATH = ROOT / "promote" / "champion.json"
API_KEY = os.environ.get("CREDFLUX_API_KEY", "credflux-dev-key")

api_key_header = APIKeyHeader(name="X-API-Key")


def require_api_key(key: str = Depends(api_key_header)) -> str:
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return key


class PredictRequest(BaseModel):
    # Raw feature dict matching the training schema, minus target/protected/age.
    features: dict


class PredictResponse(BaseModel):
    risk_score: float
    probability_bad: float
    champion_version: str
    latency_ms: float


app = FastAPI(title="CredFlux Credit Risk API")
_state = {"model": None, "champion": None}


@app.on_event("startup")
def load_champion():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    champion = json.loads(CHAMPION_PATH.read_text())
    model = mlflow.sklearn.load_model(champion["model_uri"])
    _state["model"] = model
    _state["champion"] = champion


@app.get("/health")
def health():
    return {"status": "ok", "champion_version": _state["champion"]["version"] if _state["champion"] else None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, _: str = Depends(require_api_key)):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    X = pd.DataFrame([req.features])
    proba = _state["model"].predict_proba(X)[:, 1][0]
    latency_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        risk_score=round(float(proba) * 100, 2),
        probability_bad=float(proba),
        champion_version=str(_state["champion"]["version"]),
        latency_ms=round(latency_ms, 3),
    )
