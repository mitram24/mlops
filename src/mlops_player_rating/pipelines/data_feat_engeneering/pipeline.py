"""Pipeline definition for ``data_feat_engeneering``."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_features


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=build_features,
                inputs=["cleaned", "params:feature_engineering"],
                outputs=["model_features", "feature_metadata"],
                name="build_features_node",
            ),
        ]
    )
