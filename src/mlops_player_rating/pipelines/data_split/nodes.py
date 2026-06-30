"""Nodes for the ``data_split`` pipeline.

Splits the feature table into train and test partitions with an explicit ratio and seed."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from mlops_player_rating.core.utils import ID_COLUMNS, LEAKAGE_COLUMNS, NON_FEATURE_COLUMNS, TARGET

logger = logging.getLogger(__name__)


def split_data(
    model_features: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by player id when available, then remove non-feature columns."""
    if TARGET not in model_features.columns:
        raise KeyError(f"Target column '{TARGET}' not found in feature table.")

    y = model_features[[TARGET]]
    x = model_features.drop(columns=[TARGET])
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)

    if "player_api_id" in x.columns and x["player_api_id"].nunique() > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(x, y, groups=x["player_api_id"]))
        x_train, x_test = x.iloc[train_idx].copy(), x.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state, shuffle=True
        )

    to_drop = ID_COLUMNS + LEAKAGE_COLUMNS + NON_FEATURE_COLUMNS
    x_train = x_train.drop(columns=[c for c in to_drop if c in x_train.columns])
    x_test = x_test.drop(columns=[c for c in to_drop if c in x_test.columns])

    for frame in (x_train, x_test, y_train, y_test):
        frame.reset_index(drop=True, inplace=True)

    logger.info("Split -> train=%d rows, test=%d rows, %d features", len(x_train), len(x_test), x_train.shape[1])
    return x_train, x_test, y_train, y_test
