"""Pipeline definition for ``model_train``."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_champion


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_champion,
                inputs=[
                    "X_train",
                    "X_test",
                    "y_train",
                    "y_test",
                    "champion_spec",
                    "attribute_imputer",
                    "params:model_train",
                    "params:mlflow",
                ],
                outputs=[
                    "champion_model",
                    "model_metrics",
                    "feature_importance",
                    "serving_metadata",
                ],
                name="train_champion_node",
            ),
        ]
    )
