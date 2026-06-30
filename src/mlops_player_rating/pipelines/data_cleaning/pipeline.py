"""Pipeline definition for ``data_cleaning``."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import clean_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_data,
                inputs=["ingested", "params:cleaning"],
                outputs="cleaned",
                name="clean_data_node",
            ),
        ]
    )
