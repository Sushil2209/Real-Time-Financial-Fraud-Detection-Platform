"""FastAPI service that scores one transaction with the XGBoost model.

Run from the project root:  uvicorn src.api:app --reload
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import pandas as pd
import requests
from fastapi import FastAPI, Request
from pydantic import create_model

# Reuse the exact column lists/paths from training so API and model match.
from src.model_features import FEATURE_COLUMNS, MODEL_PATH, RAW_COLUMNS, TEXT_COLUMNS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("fraud-api")

FEAST_URL = os.getenv("FEAST_URL", "http://127.0.0.1:6566/get-online-features")
FEAST_FEATURES = [
    "card_transaction_features:txn_count_24h",
    "card_transaction_features:amount_vs_avg",
    "card_transaction_features:device_changed",
]
DEFAULTS = {"txn_count_24h": 0, "amount_vs_avg": 1.0, "device_changed": 0}

# Build the request model: TransactionID and card1 required, the rest optional (NaN).
_fields = {"TransactionID": (int, ...), "card1": (int, ...)}
for _col in RAW_COLUMNS:
    _type = str if _col in TEXT_COLUMNS else float
    _fields[_col] = (Optional[_type], None)
Transaction = create_model("Transaction", **_fields)

state = {}  # holds the model, categories, and HTTP session


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state["model"] = joblib.load(MODEL_PATH)
    # Category lists learned at training time, stored inside the model.
    arrow = state["model"].get_booster().get_categories(export_to_arrow=True).to_arrow()
    state["categories"] = {name: values.to_pylist() for name, values in arrow if values is not None}
    with requests.Session() as session:
        state["session"] = session
        try:
            yield
        finally:
            del state["session"]


app = FastAPI(title="Fraud Scoring API", lifespan=lifespan)


@app.middleware("http")
async def log_latency(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    log.info("%s %s -> %s in %.1f ms", request.method, request.url.path, response.status_code, ms)
    return response


def get_online_features(card1: int) -> tuple[dict, str]:
    """Return (features, source), falling back when Feast is unavailable."""
    try:
        start = time.perf_counter()
        try:
            response = state["session"].post(
                FEAST_URL,
                json={"features": FEAST_FEATURES, "entities": {"card1": [card1]}},
                timeout=1,
            )
        finally:
            log.info("Feast call took %.1f ms", (time.perf_counter() - start) * 1000)
        response.raise_for_status()
        result = response.json()
        names = result["metadata"]["feature_names"]
        results = result["results"]
        if len(names) != len(results) or len(set(names)) != len(names):
            raise ValueError("Feature names and results do not match")
        by_name = dict(zip(names, results))
        values = {name: by_name[name]["values"][0] for name in DEFAULTS}
        if any(v is None or pd.isna(v) for v in values.values()):
            raise ValueError("Feast returned null feature values")
        return values, "feast"
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
        log.warning("Feast lookup failed for card1=%s: %s", card1, exc)
        return dict(DEFAULTS), "default"


def build_row(txn: dict, feats: dict) -> pd.DataFrame:
    """Build the one-row, 41-column frame in the same order/dtypes as training."""
    row = {**feats, **{c: txn.get(c) for c in RAW_COLUMNS}}
    frame = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    for col in FEATURE_COLUMNS:
        if col in TEXT_COLUMNS:
            # Same categories as training; unseen or missing values become NaN.
            frame[col] = pd.Categorical(frame[col], categories=state["categories"][col])
        else:
            frame[col] = pd.to_numeric(frame[col]).astype("float64")  # None -> NaN
    return frame


def risk_level(score: float) -> str:
    if score < 0.3:
        return "low"
    return "medium" if score <= 0.7 else "high"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(txn: Transaction):
    start = time.perf_counter()
    data = txn.model_dump()
    feats, source = get_online_features(data["card1"])
    row = build_row(data, feats)
    scoring_start = time.perf_counter()
    try:
        fraud_score = float(state["model"].predict_proba(row)[0, 1])
    finally:
        log.info("Model scoring took %.1f ms", (time.perf_counter() - scoring_start) * 1000)
    return {
        "TransactionID": data["TransactionID"],
        "fraud_score": fraud_score,
        "risk_level": risk_level(fraud_score),
        "features_source": source,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }
