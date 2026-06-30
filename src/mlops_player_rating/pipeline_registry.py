"""Project pipelines registry.

Every component is an independent, named pipeline so it can be executed on its own
(``kedro run --pipeline data_quality``) exactly as the handout requires, *or* chained
into the full sequential run (``kedro run`` -> ``__default__``).
"""

from __future__ import annotations

from kedro.pipeline import Pipeline

from mlops_player_rating.pipelines import (
    data_cleaning,
    data_drifts,
    data_feat_engineering,
    data_quality,
    data_split,
    model_predict,
    model_selection,
    model_train,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    dq = data_quality.create_pipeline()
    clean = data_cleaning.create_pipeline()
    feat = data_feat_engineering.create_pipeline()
    split = data_split.create_pipeline()
    selection = model_selection.create_pipeline()
    train = model_train.create_pipeline()
    predict = model_predict.create_pipeline()
    drift = data_drifts.create_pipeline()

    # Convenience macro-pipelines -------------------------------------------------
    data_processing = dq + clean + feat
    training_pipeline = data_processing + split + selection + train
    inference_pipeline = data_processing + drift + predict

    pipelines: dict[str, Pipeline] = {
        # --- atomic components (run them in isolation) ---
        "data_quality": dq,
        "data_cleaning": clean,
        "data_feat_engineering": feat,
        "data_feat_engeneering": feat,
        "data_split": split,
        "model_selection": selection,
        "model_train": train,
        "modelling": selection + train,
        "model_predict": predict,
        "data_drifts": drift,
        # --- macro stages ---
        "data_processing": data_processing,
        "training": training_pipeline,
        "inference": inference_pipeline,
        # --- full end-to-end run ---
        "__default__": training_pipeline,
    }
    return pipelines
