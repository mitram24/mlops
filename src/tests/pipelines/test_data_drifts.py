"""Tests for the ``data_drifts`` node."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlops_player_rating.pipelines.data_drifts.nodes import evaluate_drift


def _reference(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "reactions": rng.normal(65, 8, n),
            "movement_mean": rng.normal(68, 7, n),
            "age": rng.normal(25, 4, n),
            "preferred_foot": rng.choice(["left", "right"], n),
        }
    )


def test_no_drift_when_distributions_match():
    ref = _reference()
    report = evaluate_drift(ref, ref, {"sample_size": 200, "simulate_drift": False})
    assert report["dataset_drift"] is False


def test_simulated_drift_is_detected():
    ref = _reference()
    report = evaluate_drift(
        ref,
        ref,
        {"sample_size": 200, "simulate_drift": True, "drift_factor": 1.5},
    )
    assert report["n_drifted"] >= 1
    assert report["dataset_drift"] is True
    # Highest-PSI feature should be one of the inflated columns.
    assert report["features"][0]["feature"] in {"reactions", "movement_mean", "age"}
