"""Nodes for the ``data_split`` pipeline.

Splits the feature table into train and test partitions with an explicit ratio and seed."""

"""Nodes for the ``data_split`` pipeline.

Splits the feature table into train and test partitions with an explicit ratio and seed."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from mlops_player_rating.core.utils import TARGET, ID_COLUMNS, LEAKAGE_COLUMNS, NON_FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def split_data(
    model_features: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the feature table using GroupShuffleSplit to prevent identity leakage."""
    if TARGET not in model_features.columns:
        raise KeyError(f"Target column '{TARGET}' not found in feature table.")

    y = model_features[[TARGET]]
    x = model_features.drop(columns=[TARGET])

    if "player_api_id" in x.columns:
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=params.get("test_size", 0.2),
            random_state=params.get("random_state", 42)
        )
        train_idx, test_idx = next(gss.split(x, y, groups=x["player_api_id"]))
        x_train, x_test = x.iloc[train_idx].copy(), x.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    else:
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=params.get("test_size", 0.2), random_state=params.get("random_state", 42)
        )

    # Now drop the non-features (IDs, dates) since we're done splitting
    to_drop = ID_COLUMNS + LEAKAGE_COLUMNS + NON_FEATURE_COLUMNS
    for c in to_drop:
        if c in x_train.columns: x_train = x_train.drop(columns=[c])
        if c in x_test.columns: x_test = x_test.drop(columns=[c])

    for frame in (x_train, x_test, y_train, y_test):
        frame.reset_index(drop=True, inplace=True)

    logger.info(
        "Split -> train=%d rows, test=%d rows, %d features",
        len(x_train), len(x_test), x_train.shape[1],
    )
    return x_train, x_test, y_train, y_test
