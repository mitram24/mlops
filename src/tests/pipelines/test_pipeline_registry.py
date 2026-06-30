"""Smoke tests that the pipeline registry wires up every component."""

from __future__ import annotations

from mlops_player_rating.pipeline_registry import register_pipelines

EXPECTED = [
    "data_quality",
    "data_cleaning",
    "data_feat_engeneering",
    "data_split",
    "model_selection",
    "model_train",
    "model_predict",
    "data_drifts",
    "data_processing",
    "modelling",
    "inference",
    "__default__",
]


def test_registry_contains_all_components():
    pipelines = register_pipelines()
    for name in EXPECTED:
        assert name in pipelines, f"missing pipeline: {name}"


def test_default_pipeline_runs_every_stage():
    pipelines = register_pipelines()
    # quality + cleaning + fe + split + selection + train + predict + drift
    assert len(pipelines["__default__"].nodes) == 8
