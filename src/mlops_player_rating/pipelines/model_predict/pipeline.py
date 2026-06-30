"""Pipeline definition for ``model_predict``."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import predict


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=predict,
                inputs=["champion_model", "player_sample_raw", "serving_metadata"],
                outputs="predictions",
                name="predict_node",
            ),
        ]
    )
