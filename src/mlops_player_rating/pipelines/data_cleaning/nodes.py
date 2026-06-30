"""Nodes for the ``data_cleaning`` pipeline.

The general shape (a dedicated cleaning stage between ingestion and feature engineering)
follows the course's preprocessing pipelines, but the cleaning rules themselves
(work-rate label repair, median imputation, target-null drop) are specific to this
dataset and were not taught directly in class.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from mlops_player_rating.utils import (
    ID_COLUMNS,
    SKILL_COLUMNS,
    TARGET,
    apply_attribute_imputer,
    apply_value_semantics,
    fit_attribute_imputer,
)

logger = logging.getLogger(__name__)


def clean_data(
    ingested: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resolve dirty values, drop unusable rows and impute missing skill ratings.

    Returns:
        ``(cleaned, attribute_imputer)`` — the cleaned primary table plus the fitted
        per-attribute median artefact (persisted to the feature store so the online
        service can reuse identical imputation values, avoiding train/serve skew).
    """
    df = apply_value_semantics(ingested)

    # Rows with no target cannot be trained or evaluated on — drop them.
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

    # Fit + apply the per-attribute median imputer, and keep it as a feature-store artefact.
    medians = fit_attribute_imputer(df)
    df = apply_attribute_imputer(df, medians)
    attribute_imputer = {"attribute_medians": medians}

    # Identifier columns carry no signal and risk leakage — drop them.
    df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])

    n_missing = int(df.isna().sum().sum())
    logger.info("Cleaned table: %d rows x %d cols, %d residual NaNs", *df.shape, n_missing)
    return df, attribute_imputer
