"""Nodes for the ``data_cleaning`` pipeline.

Cleans work-rate labels, drops rows that cannot be trained on, and fits the median
imputer reused by inference."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from mlops_player_rating.core.utils import (
    ID_COLUMNS,
    SKILL_COLUMNS,
    TARGET,
    apply_value_semantics,
)

logger = logging.getLogger(__name__)


def clean_data(
    ingested: pd.DataFrame, params: dict[str, Any]
) -> pd.DataFrame:
    """Clean work-rate values, drop unusable rows and impute missing skill ratings.

    Returns:
        ``(cleaned, attribute_imputer)``: the cleaned table and the saved per-attribute
        medians used again by inference.
    """
    df = apply_value_semantics(ingested)

    # Rows with no target cannot be trained or evaluated on - drop them.
    if params.get("drop_null_target", True) and TARGET in df.columns:
        before = len(df)
        df = df[pd.to_numeric(df[TARGET], errors="coerce").notna()].reset_index(drop=True)
        logger.info("Dropped %d rows with a missing target", before - len(df))

    # Drop snapshots that are mostly empty (too few skill ratings to be informative).
    present_skills = [c for c in SKILL_COLUMNS if c in df.columns]
    if present_skills:
        max_missing = params.get("max_skill_missing_fraction", 0.5)
        skill_missing = df[present_skills].isna().mean(axis=1)
        before = len(df)
        df = df[skill_missing <= max_missing].reset_index(drop=True)
        logger.info("Dropped %d rows with > %.0f%% missing skills", before - len(df), max_missing * 100)





    n_missing = int(df.isna().sum().sum())
    logger.info("Cleaned table: %d rows x %d cols, %d residual NaNs", *df.shape, n_missing)
    return df
