"""Pipeline definition for ``data_quality``."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import validate_data_quality


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=validate_data_quality,
                inputs=["player_raw", "params:data_quality"],
                outputs=["ingested", "data_quality_report"],
                name="validate_data_quality_node",
            ),
        ]
    )
