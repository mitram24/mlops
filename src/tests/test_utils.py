"""Tests for the shared, stateless transformations in ``utils``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlops_player_rating import utils


def test_normalize_name_handles_spaces_and_slashes():
    assert utils.normalize_name("GK Diving") == "gk_diving"
    assert utils.normalize_name("Short Passing") == "short_passing"
    assert utils.normalize_name("  Overall Rating ") == "overall_rating"


def test_normalize_columns(raw_df):
    out = utils.normalize_columns(raw_df)
    assert "overall_rating" in out.columns
    assert "gk_reflexes" in out.columns


def test_clean_work_rate_maps_corrupt_values_to_medium():
    s = pd.Series(["high", "low", "medium", "norm", "stoc", "_0", "7", None])
    cleaned = utils.clean_work_rate(s)
    assert set(cleaned.unique()) <= {"low", "medium", "high"}
    # Corrupt entries and nulls collapse to the modal "medium".
    assert cleaned.iloc[3] == "medium" and cleaned.iloc[7] == "medium"


def test_apply_value_semantics_fixes_work_rate_and_foot(raw_df):
    df = utils.apply_value_semantics(utils.normalize_columns(raw_df))
    assert set(df["attacking_work_rate"].unique()) <= {"low", "medium", "high"}
    assert set(df["preferred_foot"].unique()) <= {"left", "right"}


def test_attribute_imputer_roundtrip(raw_df):
    df = utils.normalize_columns(raw_df)
    df.loc[df.index[:10], "finishing"] = np.nan
    medians = utils.fit_attribute_imputer(df)
    imputed = utils.apply_attribute_imputer(df, medians)
    assert imputed["finishing"].isna().sum() == 0
    assert medians["finishing"] > 0


def test_engineer_features_creates_derived_columns(raw_df):
    df = utils.engineer_features(utils.apply_value_semantics(utils.normalize_columns(raw_df)))
    for col in ("age", "bmi", "attacking_mean", "goalkeeping_mean", "is_goalkeeper"):
        assert col in df.columns
    assert (df["age"] >= 14).all() and (df["age"] <= 50).all()
    # Work rates become ordinal integers and the raw string columns are dropped.
    assert "attacking_work_rate_ord" in df.columns
    assert "attacking_work_rate" not in df.columns
    assert df["attacking_work_rate_ord"].isin([0, 1, 2]).all()


def test_drop_non_features_removes_ids_and_leakage(raw_df):
    df = utils.engineer_features(utils.apply_value_semantics(utils.normalize_columns(raw_df)))
    out = utils.drop_non_features(df)
    for col in ("id", "player_api_id", "potential", "date", "birthday", "player_name"):
        assert col not in out.columns
    # The target is preserved (it is not an identifier / leakage column).
    assert "overall_rating" in out.columns


def test_preprocess_for_inference_aligns_to_feature_columns(raw_df):
    df = utils.normalize_columns(raw_df)
    medians = utils.fit_attribute_imputer(df)
    feature_columns = ["reactions", "short_passing", "age", "bmi", "preferred_foot", "is_goalkeeper"]
    out = utils.preprocess_for_inference(raw_df, medians, feature_columns)
    assert list(out.columns) == feature_columns
    assert len(out) == len(raw_df)
