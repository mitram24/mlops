"""Pipeline definition for ``data_drifts``."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_drift


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=evaluate_drift,
                inputs=["X_train", "X_test", "params:drift"],
                outputs="drift_report",
                name="evaluate_drift_node",
            ),
        ]
    )
