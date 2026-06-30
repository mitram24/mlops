"""Shared pytest fixtures: a small, raw-schema synthetic FIFA-like dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """A tiny dataframe using the *raw* CSV column names (incl. dirty work-rate values)."""
    rng = np.random.default_rng(42)
    n = 120

    skills = [
        "crossing", "finishing", "heading_accuracy", "short_passing", "volleys",
        "dribbling", "curve", "free_kick_accuracy", "long_passing", "ball_control",
        "acceleration", "sprint_speed", "agility", "reactions", "balance",
        "shot_power", "jumping", "stamina", "strength", "long_shots",
        "aggression", "interceptions", "positioning", "vision", "penalties",
        "marking", "standing_tackle", "sliding_tackle",
        "gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes",
    ]

    data: dict[str, np.ndarray] = {
        "id": np.arange(1, n + 1),
        "player_fifa_api_id": rng.integers(1000, 9999, n),
        "player_api_id": np.arange(100000, 100000 + n),
        "player_name": [f"Player {i}" for i in range(n)],
        "date": pd.to_datetime("2014-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
        "birthday": pd.to_datetime("1988-01-01") + pd.to_timedelta(rng.integers(0, 3000, n), unit="D"),
        "height": rng.uniform(165, 200, n).round(2),
        "weight": rng.integers(140, 200, n),
        "preferred_foot": rng.choice(["right", "left"], n),
        # Deliberately include the dataset's notorious corrupt work-rate values.
        "attacking_work_rate": rng.choice(["medium", "high", "low", "norm", "stoc", "y"], n),
        "defensive_work_rate": rng.choice(["medium", "high", "low", "_0", "ormal", "7"], n),
    }
    for s in skills:
        data[s] = rng.integers(30, 95, n).astype(float)

    df = pd.DataFrame(data)
    # Target correlated with a couple of strong attributes, on a realistic 0-100 scale.
    df["overall_rating"] = (
        30 + 0.35 * df["reactions"] + 0.25 * df["short_passing"] + 0.20 * df["ball_control"]
    ).clip(1, 99).round()
    df["potential"] = (df["overall_rating"] + rng.integers(0, 6, n)).clip(1, 99)
    return df


@pytest.fixture
def dq_params() -> dict:
    """Relaxed data-quality thresholds suitable for the tiny test dataset."""
    return {
        "min_rows": 50,
        "min_columns": 10,
        "target": "overall_rating",
        "id_column": "id",
        "max_target_missing_fraction": 0.05,
        "required_columns": ["overall_rating", "reactions", "short_passing", "preferred_foot"],
        "ranges": {"overall_rating": [1, 100], "reactions": [1, 100]},
        "max_missing_fraction": 0.99,
    }
