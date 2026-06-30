"""FastAPI service that serves the trained champion model.

The container only needs the pickled model + ``serving_metadata.json`` produced by the
``model_train`` pipeline. Incoming raw player records are pushed through the *same*
``preprocess_for_inference`` used offline, eliminating train/serve skew.

Run locally:
    uvicorn mlops_player_rating.serving.app:app --reload
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlops_player_rating.utils import preprocess_for_inference

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "data/06_models"))

app = FastAPI(
    title="FIFA Player Rating API",
    description="Predict a player's FIFA overall rating with the trained MLOps champion model.",
    version="0.1.0",
)

_STATE: dict[str, Any] = {"model": None, "meta": None}


def _load_artifacts() -> None:
    """Load the model + metadata once, lazily, so import never fails without artefacts."""
    if _STATE["model"] is not None:
        return
    model_path = MODEL_DIR / "champion_model.pkl"
    meta_path = MODEL_DIR / "serving_metadata.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Model artefacts not found in {MODEL_DIR}. Run `kedro run` first."
        )
    with open(model_path, "rb") as fh:
        _STATE["model"] = pickle.load(fh)
    with open(meta_path, encoding="utf-8") as fh:
        _STATE["meta"] = json.load(fh)
    logger.info(
        "Loaded model '%s' with %d features",
        _STATE["meta"].get("model_name"),
        len(_STATE["meta"]["feature_columns"]),
    )


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., description="List of raw player records.")


class Prediction(BaseModel):
    predicted_rating: float


class PredictResponse(BaseModel):
    model_name: str
    predictions: list[Prediction]


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "fifa-player-rating-api", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        _load_artifacts()
        return {"status": "ok", "model_name": _STATE["meta"].get("model_name")}
    except Exception as exc:  # noqa: BLE001
        return {"status": "model_not_loaded", "detail": str(exc)}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        _load_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not request.records:
        raise HTTPException(status_code=422, detail="`records` must not be empty.")

    meta = _STATE["meta"]
    raw = pd.DataFrame(request.records)
    features = preprocess_for_inference(
        raw,
        meta["attribute_medians"],
        meta["feature_columns"],
    )
    preds = _STATE["model"].predict(features)
    return PredictResponse(
        model_name=meta.get("model_name", "unknown"),
        predictions=[Prediction(predicted_rating=round(float(p), 2)) for p in preds],
    )
