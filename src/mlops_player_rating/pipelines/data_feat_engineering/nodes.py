"""Nodes for the ``data_feat_engeneering`` pipeline.

Adds FIFA-specific features such as age, BMI, skill-group means and the goalkeeper flag.
The emitted metadata documents the offline feature table used by training and serving."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from mlops_player_rating.core.modeling import split_feature_types
from mlops_player_rating.core.utils import LEAKAGE_COLUMNS, NON_FEATURE_COLUMNS, TARGET, engineer_features

logger = logging.getLogger(__name__)

# Derived features created by ``engineer_features`` (documented for the feature store).
DERIVED_FEATURES = [
    "age",
    "bmi",
    "attacking_mean",
    "skill_mean",
    "movement_mean",
    "power_mean",
    "mentality_mean",
    "defending_mean",
    "goalkeeping_mean",
    "is_goalkeeper",
    "attacking_work_rate_ord",
    "defensive_work_rate_ord",
]


def build_features(
    cleaned: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Engineer model features and emit feature metadata.

    Returns:
        ``(feature_table, feature_metadata)`` with dtypes, derived features and the
        numeric/categorical feature split used downstream.
    """
    df = engineer_features(cleaned)

    # Remove leakage (``potential``) and the raw columns consumed by feature engineering
    # (``date``/``birthday`` -> age, ``player_name``); the target is preserved.
    to_drop = LEAKAGE_COLUMNS + NON_FEATURE_COLUMNS
    df = df.drop(columns=[c for c in to_drop if c in df.columns])

    drop_cols = [c for c in params.get("drop_columns", []) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.info("Dropped %d configured columns: %s", len(drop_cols), drop_cols)

    feature_only = df.drop(columns=[TARGET]) if TARGET in df.columns else df
    numeric, categorical = split_feature_types(feature_only)

    metadata = {
        "feature_table": "model_features",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(df)),
        "n_features": int(feature_only.shape[1]),
        "target": TARGET,
        "derived_features": [c for c in DERIVED_FEATURES if c in df.columns],
        "numeric_features": numeric,
        "categorical_features": categorical,
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }
    logger.info(
        "Feature table: %d features (%d numeric, %d categorical)",
        metadata["n_features"],
        len(numeric),
        len(categorical),
    )
    return df, metadata
