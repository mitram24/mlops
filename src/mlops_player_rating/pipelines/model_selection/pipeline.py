"""Pipeline definition for ``model_selection``."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import select_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=select_model,
                inputs=["X_train", "y_train", "params:model_selection", "params:mlflow"],
                outputs=["model_selection_report", "champion_spec"],
                name="select_model_node",
            ),
        ]
    )
