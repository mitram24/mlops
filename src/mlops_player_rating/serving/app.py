"""FastAPI service that serves the trained champion model."""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mlops_player_rating.core.utils import preprocess_for_inference

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "data/06_models"))

app = FastAPI(
    title="FIFA Player Rating API",
    description="Predict a player's FIFA overall rating with the trained MLOps champion model.",
    version="0.1.0",
)

_STATE: dict[str, Any] = {"model": None, "meta": None}


class PlayerRecord(BaseModel):
    """Raw player attributes accepted by the prediction endpoint."""

    model_config = ConfigDict(extra="forbid")

    preferred_foot: str | None = None
    attacking_work_rate: str | None = None
    defensive_work_rate: str | None = None
    date: str | None = None
    birthday: str | None = None
    age: float | None = None
    height: float | None = Field(default=None, gt=0)
    weight: float | None = Field(default=None, gt=0)

    crossing: float | None = Field(default=None, ge=0, le=100)
    finishing: float | None = Field(default=None, ge=0, le=100)
    heading_accuracy: float | None = Field(default=None, ge=0, le=100)
    short_passing: float | None = Field(default=None, ge=0, le=100)
    volleys: float | None = Field(default=None, ge=0, le=100)
    dribbling: float | None = Field(default=None, ge=0, le=100)
    curve: float | None = Field(default=None, ge=0, le=100)
    free_kick_accuracy: float | None = Field(default=None, ge=0, le=100)
    long_passing: float | None = Field(default=None, ge=0, le=100)
    ball_control: float | None = Field(default=None, ge=0, le=100)
    acceleration: float | None = Field(default=None, ge=0, le=100)
    sprint_speed: float | None = Field(default=None, ge=0, le=100)
    agility: float | None = Field(default=None, ge=0, le=100)
    reactions: float | None = Field(default=None, ge=0, le=100)
    balance: float | None = Field(default=None, ge=0, le=100)
    shot_power: float | None = Field(default=None, ge=0, le=100)
    jumping: float | None = Field(default=None, ge=0, le=100)
    stamina: float | None = Field(default=None, ge=0, le=100)
    strength: float | None = Field(default=None, ge=0, le=100)
    long_shots: float | None = Field(default=None, ge=0, le=100)
    aggression: float | None = Field(default=None, ge=0, le=100)
    interceptions: float | None = Field(default=None, ge=0, le=100)
    positioning: float | None = Field(default=None, ge=0, le=100)
    vision: float | None = Field(default=None, ge=0, le=100)
    penalties: float | None = Field(default=None, ge=0, le=100)
    marking: float | None = Field(default=None, ge=0, le=100)
    standing_tackle: float | None = Field(default=None, ge=0, le=100)
    sliding_tackle: float | None = Field(default=None, ge=0, le=100)
    gk_diving: float | None = Field(default=None, ge=0, le=100)
    gk_handling: float | None = Field(default=None, ge=0, le=100)
    gk_kicking: float | None = Field(default=None, ge=0, le=100)
    gk_positioning: float | None = Field(default=None, ge=0, le=100)
    gk_reflexes: float | None = Field(default=None, ge=0, le=100)

    @model_validator(mode="after")
    def _has_signal(self) -> "PlayerRecord":
        if not self.model_dump(exclude_none=True):
            raise ValueError("record must include at least one known player attribute")
        return self

    @field_validator("preferred_foot")
    @classmethod
    def _valid_foot(cls, value: str | None) -> str | None:
        if value is None:
            return value
        value = value.strip().lower()
        if value not in {"left", "right"}:
            raise ValueError("preferred_foot must be 'left' or 'right'")
        return value


class PredictRequest(BaseModel):
    records: list[PlayerRecord] = Field(..., min_length=1, description="Raw player records.")


class Prediction(BaseModel):
    predicted_rating: float


class PredictResponse(BaseModel):
    model_name: str
    predictions: list[Prediction]


def _load_artifacts() -> None:
    """Load the model and metadata once, lazily."""
    if _STATE["model"] is not None:
        return
    model_path = MODEL_DIR / "champion_model.pkl"
    meta_path = MODEL_DIR / "serving_metadata.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Model artefacts not found in {MODEL_DIR}. Run `kedro run` first.")
    with open(model_path, "rb") as fh:
        _STATE["model"] = pickle.load(fh)
    with open(meta_path, encoding="utf-8") as fh:
        _STATE["meta"] = json.load(fh)
    logger.info("Loaded model '%s' with %d features", _STATE["meta"].get("model_name"), len(_STATE["meta"]["feature_columns"]))


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
        meta = _STATE["meta"]
        raw = pd.DataFrame([record.model_dump(exclude_none=True) for record in request.records])
        features = preprocess_for_inference(raw, meta.get("attribute_medians", {}), meta["feature_columns"])
        preds = _STATE["model"].predict(features)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed after validation.") from exc

    return PredictResponse(
        model_name=meta.get("model_name", "unknown"),
        predictions=[Prediction(predicted_rating=round(float(p), 2)) for p in preds],
    )
