"""Tests for the model factory and metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlops_player_rating.modeling import (
    build_model,
    candidate_estimators,
    regression_metrics,
    split_feature_types,
)


def _toy_xy(n: int = 100):
    rng = np.random.default_rng(0)
    x = pd.DataFrame(
        {
            "reactions": rng.integers(40, 95, n).astype(float),
            "short_passing": rng.integers(40, 95, n).astype(float),
            "age": rng.uniform(17, 38, n),
            "preferred_foot": rng.choice(["left", "right"], n),
        }
    )
    y = (30 + 0.4 * x["reactions"] + 0.3 * x["short_passing"]).clip(1, 99)
    return x, y.to_numpy()


def test_split_feature_types():
    x, _ = _toy_xy()
    numeric, categorical = split_feature_types(x)
    assert "reactions" in numeric and "age" in numeric
    assert categorical == ["preferred_foot"]


def test_build_model_fit_predict_returns_valid_ratings():
    x, y = _toy_xy()
    numeric, categorical = split_feature_types(x)
    model = build_model(candidate_estimators()["linear_regression"], numeric, categorical)
    model.fit(x, y)
    preds = model.predict(x)
    assert preds.shape[0] == len(x)
    assert np.all(np.isfinite(preds))
    assert np.all(preds > 0)


def test_regression_metrics_perfect_prediction():
    y = np.array([60.0, 70.0, 80.0])
    m = regression_metrics(y, y)
    assert m["rmse"] == 0.0
    assert m["mae"] == 0.0
    assert m["r2"] == 1.0
    assert m["within_2_pct"] == 100.0


def test_handle_unknown_category_does_not_crash():
    x, y = _toy_xy()
    numeric, categorical = split_feature_types(x)
    model = build_model(candidate_estimators()["ridge"], numeric, categorical)
    model.fit(x, y)
    unseen = pd.DataFrame(
        {"reactions": [80.0], "short_passing": [75.0], "age": [25.0], "preferred_foot": ["both"]}
    )
    preds = model.predict(unseen)  # unseen category must be ignored, not raise
    assert np.isfinite(preds).all()
