"""Nodes for the ``model_selection`` pipeline.

Cross-validates candidate regressors on the training split, logs each run to MLflow, and
selects the model with the lowest mean CV RMSE."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

from mlops_player_rating.core.modeling import (
    build_model,
    candidate_estimators,
    split_feature_types,
)
from mlops_player_rating.core.tracking import setup_mlflow
from mlops_player_rating.core.utils import TARGET

logger = logging.getLogger(__name__)


def select_model(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    params: dict[str, Any],
    mlflow_params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cross-validate candidates and return ``(selection_report, champion)``."""
    y = y_train[TARGET].to_numpy()
    numeric, categorical = split_feature_types(x_train)
    random_state = params.get("random_state", 42)
    cv_folds = params.get("cv_folds", 5)
    wanted = params.get("candidates") or list(candidate_estimators(random_state).keys())

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    estimators = candidate_estimators(random_state)

    mlflow = setup_mlflow(mlflow_params)
    results: list[dict[str, Any]] = []

    with mlflow.start_run(run_name="model_selection"):
        mlflow.log_params({"cv_folds": cv_folds, "n_candidates": len(wanted)})
        for name in wanted:
            if name not in estimators:
                logger.warning("Unknown candidate '%s' - skipping", name)
                continue
            model = build_model(estimators[name], numeric, categorical)
            # n_jobs=1: parallel folds reorder floating-point sums across workers, which
            # breaks exact reproducibility even with a fixed random_state.
            scores = cross_val_score(
                model, x_train, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1
            )
            rmse_mean = float(-scores.mean())
            rmse_std = float(scores.std())
            results.append(
                {"model": name, "cv_rmse_mean": rmse_mean, "cv_rmse_std": rmse_std}
            )
            with mlflow.start_run(run_name=f"cv_{name}", nested=True):
                mlflow.log_param("model", name)
                mlflow.log_param("cv_folds", cv_folds)
                mlflow.log_metric("cv_rmse_mean", rmse_mean)
                mlflow.log_metric("cv_rmse_std", rmse_std)
            logger.info(f"CV {name:<22} RMSE={rmse_mean:.3f} (+/- {rmse_std:.3f})")

        if not results:
            raise ValueError("No valid candidate models were evaluated.")

        results.sort(key=lambda r: r["cv_rmse_mean"])
        champion = results[0]
        mlflow.log_param("champion", champion["model"])
        mlflow.log_metric("champion_cv_rmse", champion["cv_rmse_mean"])

    report = {
        "cv_folds": cv_folds,
        "scoring": "neg_root_mean_squared_error",
        "results": results,
        "champion": champion["model"],
    }
    logger.info(f"Champion model: {champion['model']} (CV RMSE={champion['cv_rmse_mean']:.3f})")
    return report, champion
