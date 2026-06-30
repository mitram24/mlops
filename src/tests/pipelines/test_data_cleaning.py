"""Tests for the ``data_cleaning`` node."""

from __future__ import annotations

import numpy as np

from mlops_player_rating.pipelines.data_cleaning.nodes import clean_data
from mlops_player_rating.core.utils import normalize_columns

_PARAMS = {"drop_null_target": True, "max_skill_missing_fraction": 0.5}


def test_clean_data_drops_null_targets_and_ids(raw_df):
    ingested = normalize_columns(raw_df)
    ingested.loc[ingested.index[:5], "overall_rating"] = np.nan
    cleaned, imputer = clean_data(ingested, _PARAMS)
    # Null-target rows are gone.
    assert cleaned["overall_rating"].isna().sum() == 0
    assert len(cleaned) == len(ingested) - 5
    # Identifier columns dropped.
    for col in ("id", "player_api_id", "player_fifa_api_id"):
        assert col not in cleaned.columns


def test_clean_data_returns_attribute_imputer(raw_df):
    ingested = normalize_columns(raw_df)
    _, imputer = clean_data(ingested, _PARAMS)
    assert "attribute_medians" in imputer
    assert imputer["attribute_medians"]["reactions"] > 0
