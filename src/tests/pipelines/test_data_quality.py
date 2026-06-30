"""Tests for the ``data_quality`` validation node."""

from __future__ import annotations

import numpy as np
import pytest

from mlops_player_rating.pipelines.data_quality.nodes import (
    DataQualityError,
    validate_data_quality,
)


def test_valid_data_passes(raw_df, dq_params):
    ingested, report = validate_data_quality(raw_df, dq_params)
    assert report["passed"] is True
    assert report["n_failed_critical"] == 0
    assert "overall_rating" in ingested.columns  # output is normalised


def test_small_fraction_of_null_targets_is_tolerated(raw_df, dq_params):
    # A couple of null targets is a warning, not a hard failure (dropped in cleaning).
    df = raw_df.copy()
    df.loc[df.index[:2], "overall_rating"] = np.nan
    _, report = validate_data_quality(df, dq_params)
    assert report["passed"] is True


def test_negative_target_is_rejected(raw_df, dq_params):
    bad = raw_df.copy()
    bad.loc[0, "overall_rating"] = -5
    with pytest.raises(DataQualityError):
        validate_data_quality(bad, dq_params)


def test_missing_required_column_is_rejected(raw_df, dq_params):
    bad = raw_df.drop(columns=["reactions"])
    with pytest.raises(DataQualityError):
        validate_data_quality(bad, dq_params)


def test_duplicate_ids_are_rejected(raw_df, dq_params):
    bad = raw_df.copy()
    bad.loc[1, "id"] = bad.loc[0, "id"]
    with pytest.raises(DataQualityError):
        validate_data_quality(bad, dq_params)
