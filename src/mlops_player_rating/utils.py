"""Stateless data transformations shared by the Kedro pipelines *and* the serving API.

Keeping these here (instead of duplicating logic inside nodes) guarantees that a player
scored online by the FastAPI service goes through *exactly* the same cleaning and
feature-engineering steps as the data used for training — the classic train/serve skew
trap that MLOps is meant to avoid.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# --- Domain knowledge about the FIFA / European-Soccer dataset ----------------------

# Identifier columns that must never reach the model.
ID_COLUMNS = ["id", "player_fifa_api_id", "player_api_id"]

# Target column (after name normalisation).
TARGET = "overall_rating"

# ``potential`` is the player's *projected* peak rating: it is almost a duplicate of the
# target (corr ~ 0.77) and would not be available as a clean, independent feature in a
# realistic scoring setting — we treat it as leakage and drop it, just as the identifier
# columns are dropped.
LEAKAGE_COLUMNS = ["potential"]

# Free-text / non-feature columns carried only for feature engineering (age) or display.
NON_FEATURE_COLUMNS = ["player_name", "date", "birthday"]

# The 33 raw FIFA skill ratings (all on a 0-100 scale).
SKILL_COLUMNS = [
    "crossing", "finishing", "heading_accuracy", "short_passing", "volleys",
    "dribbling", "curve", "free_kick_accuracy", "long_passing", "ball_control",
    "acceleration", "sprint_speed", "agility", "reactions", "balance",
    "shot_power", "jumping", "stamina", "strength", "long_shots",
    "aggression", "interceptions", "positioning", "vision", "penalties",
    "marking", "standing_tackle", "sliding_tackle",
    "gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes",
]

# FIFA-style attribute groups — aggregated into mean features that summarise a player's
# profile and are robust to a few missing individual ratings.
ATTRIBUTE_GROUPS: dict[str, list[str]] = {
    "attacking": ["crossing", "finishing", "heading_accuracy", "short_passing", "volleys"],
    "skill": ["dribbling", "curve", "free_kick_accuracy", "long_passing", "ball_control"],
    "movement": ["acceleration", "sprint_speed", "agility", "reactions", "balance"],
    "power": ["shot_power", "jumping", "stamina", "strength", "long_shots"],
    "mentality": ["aggression", "interceptions", "positioning", "vision", "penalties"],
    "defending": ["marking", "standing_tackle", "sliding_tackle"],
    "goalkeeping": ["gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes"],
}

# The work-rate columns are notoriously dirty in this dataset (values such as ``"norm"``,
# ``"stoc"``, ``"_0"`` or stray digits). They are genuinely ordinal, so we map the valid
# labels to {0, 1, 2} and fold every corrupt value into the modal ``medium`` (1).
WORK_RATE_COLUMNS = ["attacking_work_rate", "defensive_work_rate"]
_WORK_RATE_ORDINAL = {"low": 0, "medium": 1, "high": 2}


def normalize_name(name: str) -> str:
    """Turn a raw column header such as ``"GK Diving"`` into ``"gk_diving"``."""
    name = str(name).strip().lower()
    name = name.replace("/", "_").replace(" ", "_")
    name = re.sub(r"[^0-9a-z_]", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with every column header normalised to snake_case."""
    df = df.copy()
    df.columns = [normalize_name(c) for c in df.columns]
    return df


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric view of ``col`` (NaN if the column is absent) — used in feature maths."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def clean_work_rate(series: pd.Series) -> pd.Series:
    """Normalise a work-rate column to the valid set {low, medium, high}.

    Every corrupt / out-of-vocabulary value (``"norm"``, ``"stoc"``, ``"_0"``, digits, …)
    is mapped to the modal category ``"medium"`` so the feature stays usable.
    """
    cleaned = series.astype("object").where(series.notna(), "medium")
    cleaned = cleaned.astype(str).str.strip().str.lower()
    return cleaned.where(cleaned.isin(_WORK_RATE_ORDINAL), "medium")


def apply_value_semantics(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve the dataset's known dirty-value issues (work rates, preferred foot)."""
    df = df.copy()
    for col in WORK_RATE_COLUMNS:
        if col in df.columns:
            df[col] = clean_work_rate(df[col])
    if "preferred_foot" in df.columns:
        foot = df["preferred_foot"].astype(str).str.strip().str.lower()
        df["preferred_foot"] = foot.where(foot.isin(["left", "right"]), "right")
    return df


def fit_attribute_imputer(df: pd.DataFrame) -> dict[str, float]:
    """Learn a per-attribute median from the training data.

    Returned so it can be persisted as a feature-store artefact and re-applied at
    inference time — never recomputed from serving data (which avoids train/serve skew).
    """
    medians: dict[str, float] = {}
    for col in SKILL_COLUMNS:
        if col in df.columns:
            value = pd.to_numeric(df[col], errors="coerce").median()
            medians[col] = float(value) if pd.notna(value) else 0.0
    return medians


def apply_attribute_imputer(df: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    """Fill missing skill ratings using the pre-fitted training medians."""
    df = df.copy()
    for col, median in medians.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(median)
    return df


def _compute_age(df: pd.DataFrame) -> pd.Series:
    """Age (in years) at the time of the FIFA snapshot, from ``date`` - ``birthday``."""
    if "date" in df.columns and "birthday" in df.columns:
        date = pd.to_datetime(df["date"], errors="coerce")
        birthday = pd.to_datetime(df["birthday"], errors="coerce")
        return (date - birthday).dt.days / 365.25
    if "age" in df.columns:  # already provided (e.g. a hand-built serving payload)
        return pd.to_numeric(df["age"], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived features used by the model (row-wise, no fitted state)."""
    df = df.copy()

    # Biometrics: age at snapshot + BMI (weight is in lbs, height in cm in this dataset).
    df["age"] = _compute_age(df).clip(lower=14, upper=50)
    weight_kg = _num(df, "weight") * 0.453592
    height_m = _num(df, "height") / 100.0
    df["bmi"] = weight_kg / (height_m**2)

    # Attribute-group means: compact, missing-robust summaries of a player's profile.
    for group, cols in ATTRIBUTE_GROUPS.items():
        present = [c for c in cols if c in df.columns]
        if present:
            df[f"{group}_mean"] = df[present].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    # Goalkeeper flag: GK ratings dominate the outfield ones for keepers.
    gk_mean = df.get("goalkeeping_mean", pd.Series(np.nan, index=df.index))
    outfield = [f"{g}_mean" for g in ("attacking", "skill", "defending") if f"{g}_mean" in df.columns]
    outfield_mean = df[outfield].mean(axis=1) if outfield else pd.Series(np.nan, index=df.index)
    df["is_goalkeeper"] = (gk_mean > outfield_mean).fillna(False).astype(int)

    # Work rates are ordinal -> encode to {0,1,2} and drop the raw string columns.
    for col in WORK_RATE_COLUMNS:
        if col in df.columns:
            df[f"{col}_ord"] = clean_work_rate(df[col]).map(_WORK_RATE_ORDINAL).astype(int)
            df = df.drop(columns=[col])

    return df


def drop_non_features(df: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    """Drop identifier / leakage / non-feature columns so only model features remain."""
    to_drop = ID_COLUMNS + LEAKAGE_COLUMNS + NON_FEATURE_COLUMNS + (extra or [])
    return df.drop(columns=[c for c in to_drop if c in df.columns])


def preprocess_for_inference(
    raw: pd.DataFrame,
    attribute_medians: dict[str, float],
    feature_columns: list[str],
) -> pd.DataFrame:
    """Full raw -> model-ready transformation used by the serving API.

    Mirrors the offline ``data_cleaning`` + ``data_feat_engeneering`` pipelines, then
    aligns the columns to those seen at training time (missing -> NaN, extras dropped).
    """
    df = normalize_columns(raw)
    df = apply_value_semantics(df)
    df = apply_attribute_imputer(df, attribute_medians)
    df = engineer_features(df)
    df = drop_non_features(df, extra=[TARGET])
    # Align to the exact training schema.
    df = df.reindex(columns=feature_columns)
    return df
