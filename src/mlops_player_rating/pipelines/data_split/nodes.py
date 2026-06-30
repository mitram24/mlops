"""Nodes for the ``data_split`` pipeline.

The course's bank-example pipeline split with a plain ``df.sample(frac=0.8)`` (see
``split_data/nodes.py``). We use ``sklearn.model_selection.train_test_split`` instead, so
the split ratio and seed are explicit parameters rather than hardcoded, and the row order
isn't disturbed before the split.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from mlops_player_rating.utils import TARGET

logger = logging.getLogger(__name__)


def split_data(
    model_features: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the feature table into train/test design matrices and targets.

    The target is kept on its native 0 to 100 rating scale; no transform is applied
    (see ``modeling.build_model``), so every downstream artefact (metrics, SHAP,
    predictions, the served model) speaks the same, human-readable unit.
    """
    if TARGET not in model_features.columns:
        raise KeyError(f"Target column '{TARGET}' not found in feature table.")

    x = model_features.drop(columns=[TARGET])
    y = model_features[[TARGET]]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=params.get("test_size", 0.2),
        random_state=params.get("random_state", 42),
        shuffle=True,
    )
    for frame in (x_train, x_test, y_train, y_test):
        frame.reset_index(drop=True, inplace=True)

    logger.info(
        "Split -> train=%d rows, test=%d rows, %d features",
        len(x_train),
        len(x_test),
        x_train.shape[1],
    )
    return x_train, x_test, y_train, y_test
