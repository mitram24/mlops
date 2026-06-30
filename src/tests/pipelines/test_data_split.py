"""Tests for the ``data_split`` node."""

from __future__ import annotations

import pandas as pd

from mlops_player_rating.pipelines.data_cleaning.nodes import clean_data
from mlops_player_rating.pipelines.data_feat_engeneering.nodes import build_features
from mlops_player_rating.pipelines.data_split.nodes import split_data
from mlops_player_rating.core.utils import TARGET, normalize_columns

_CLEAN = {"drop_null_target": True, "max_skill_missing_fraction": 0.5}


def _feature_table(raw_df):
    cleaned, _ = clean_data(normalize_columns(raw_df), _CLEAN)
    features, _ = build_features(cleaned, {"drop_columns": []})
    return features


def test_split_shapes_and_target_separation(raw_df):
    features = _feature_table(raw_df)
    x_train, x_test, y_train, y_test = split_data(
        features, {"random_state": 42, "test_size": 0.25}
    )
    assert len(x_train) + len(x_test) == len(features)
    assert len(y_train) == len(x_train)
    # Target must not leak into the design matrix.
    assert TARGET not in x_train.columns
    assert TARGET in y_train.columns


def test_split_is_reproducible(raw_df):
    features = _feature_table(raw_df)
    a = split_data(features, {"random_state": 7, "test_size": 0.25})[0]
    b = split_data(features, {"random_state": 7, "test_size": 0.25})[0]
    pd.testing.assert_frame_equal(a, b)
