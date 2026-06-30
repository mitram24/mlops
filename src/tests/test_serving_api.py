"""API-boundary tests for the FastAPI serving app."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mlops_player_rating.serving import app as serving_app


class CapturingModel:
    """Small model stub that records the inference frame passed by the API."""

    def __init__(self) -> None:
        self.features = None

    def predict(self, features):
        self.features = features.copy()
        return [71.236]


@pytest.fixture
def client_with_stubbed_model():
    model = CapturingModel()
    previous_state = serving_app._STATE.copy()
    serving_app._STATE.update(
        {
            "model": model,
            "meta": {
                "model_name": "stubbed_model",
                "feature_columns": [
                    "preferred_foot",
                    "crossing",
                    "finishing",
                    "heading_accuracy",
                    "short_passing",
                    "volleys",
                    "age",
                    "height",
                    "weight",
                    "bmi",
                    "attacking_mean",
                    "attacking_work_rate_ord",
                    "defensive_work_rate_ord",
                ],
                "attribute_medians": {
                    "crossing": 60.0,
                    "finishing": 61.0,
                    "heading_accuracy": 62.0,
                    "short_passing": 63.0,
                    "volleys": 64.0,
                },
            },
        }
    )
    try:
        yield TestClient(serving_app.app), model
    finally:
        serving_app._STATE.clear()
        serving_app._STATE.update(previous_state)


def test_predict_accepts_minimal_valid_record_and_imputes_missing_skills(
    client_with_stubbed_model,
):
    client, model = client_with_stubbed_model

    response = client.post(
        "/predict",
        json={
            "records": [
                {
                    "preferred_foot": "right",
                    "age": 26,
                    "height": 182,
                    "weight": 176,
                }
            ]
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "model_name": "stubbed_model",
        "predictions": [{"predicted_rating": 71.24}],
    }
    assert model.features.loc[0, "finishing"] == 61.0
    assert model.features.loc[0, "short_passing"] == 63.0
    assert model.features.loc[0, "attacking_mean"] == 62.0


@pytest.mark.parametrize(
    "payload",
    [
        {"records": []},
        {"records": [{}]},
        {"records": [{"age": 26, "unknown_field": "x"}]},
        {"records": [{"finishing": 101}]},
    ],
)
def test_predict_rejects_invalid_payloads(client_with_stubbed_model, payload):
    client, _ = client_with_stubbed_model

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
