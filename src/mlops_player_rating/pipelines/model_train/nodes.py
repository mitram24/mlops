"""Nodes for the ``model_train`` pipeline.

Trains the selected model, evaluates it on the test set, writes SHAP or permutation
importance outputs, and logs metrics, plots and the model to MLflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlops_player_rating.core.modeling import (
    build_model,
    candidate_estimators,
    regression_metrics,
    split_feature_types,
)
from mlops_player_rating.core.tracking import setup_mlflow
from mlops_player_rating.core.utils import TARGET

logger = logging.getLogger(__name__)


def _explain(
    model, x_sample: pd.DataFrame, y_sample: np.ndarray, reporting_dir: Path, params: dict[str, Any]
) -> dict[str, Any]:
    """Compute feature-level explanations and save plots; SHAP first, else permutation."""
    max_display = params.get("shap_max_display", 20)
    random_state = params.get("random_state", 42)
    preprocess = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    feature_names = list(preprocess.get_feature_names_out())
    x_trans = preprocess.transform(x_sample)

    summary_png = reporting_dir / "shap_summary.png"
    bar_png = reporting_dir / "shap_bar.png"

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        try:
            explainer = shap.Explainer(estimator, x_trans)
            explanation = explainer(x_trans)
        except Exception:  # noqa: BLE001 - fall back to the model-agnostic explainer
            explainer = shap.Explainer(estimator.predict, x_trans)
            explanation = explainer(x_trans)

        values = np.asarray(explanation.values)
        if values.ndim == 3:
            values = values[..., 0]
        importance = np.abs(values).mean(axis=0)

        shap.summary_plot(
            values, features=x_trans, feature_names=feature_names, show=False, max_display=max_display
        )
        plt.tight_layout()
        plt.savefig(summary_png, dpi=120, bbox_inches="tight")
        plt.close()

        shap.summary_plot(
            values,
            features=x_trans,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            max_display=max_display,
        )
        plt.tight_layout()
        plt.savefig(bar_png, dpi=120, bbox_inches="tight")
        plt.close()
        method = "shap"
        plots = [str(summary_png), str(bar_png)]
        logger.info("SHAP explanations computed on %d samples", len(x_sample))

    except Exception as exc:  # noqa: BLE001
        logger.warning("SHAP unavailable (%s) - using permutation importance instead", exc)
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            model, x_sample, y_sample, n_repeats=5, random_state=random_state, n_jobs=1
        )
        importance = result.importances_mean
        feature_names = list(x_sample.columns)
        method = "permutation_importance"
        plots = []
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            order = np.argsort(importance)[::-1][:max_display][::-1]
            plt.figure(figsize=(8, 6))
            plt.barh([feature_names[i] for i in order], importance[order])
            plt.xlabel("Permutation importance (increase in error)")
            plt.title("Feature importance")
            plt.tight_layout()
            plt.savefig(bar_png, dpi=120, bbox_inches="tight")
            plt.close()
            plots = [str(bar_png)]
        except Exception:  # noqa: BLE001
            pass

    order = np.argsort(importance)[::-1][:max_display]
    top_features = [
        {"feature": feature_names[i], "importance": float(importance[i])} for i in order
    ]
    return {"method": method, "top_features": top_features, "plots": plots}


def train_champion(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    champion_spec: dict[str, Any],
    attribute_imputer: dict[str, Any],
    params: dict[str, Any],
    mlflow_params: dict[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Fit, evaluate, explain, log and register the champion model.

    Returns ``(model, metrics, explainability, serving_metadata)``.
    """
    champion_name = champion_spec["model"]
    random_state = params.get("random_state", 42)
    y_tr = y_train[TARGET].to_numpy()
    y_te = y_test[TARGET].to_numpy()

    numeric, categorical = split_feature_types(x_train)
    estimator = candidate_estimators(random_state)[champion_name]
    model = build_model(estimator, numeric, categorical)

    logger.info("Training champion '%s' on %d rows...", champion_name, len(x_train))
    model.fit(x_train, y_tr)

    preds = model.predict(x_test)
    metrics = regression_metrics(y_te, preds)
    logger.info(
        f"Test metrics | RMSE={metrics['rmse']:.3f} MAE={metrics['mae']:.3f} "
        f"R2={metrics['r2']:.4f} MAPE={metrics['mape_pct']:.2f}% within2={metrics['within_2_pct']:.1f}%"
    )

    reporting_dir = Path(params.get("reporting_dir", "data/08_reporting"))
    reporting_dir.mkdir(parents=True, exist_ok=True)
    sample_n = min(params.get("shap_sample_size", 150), len(x_train))
    x_sample = x_train.sample(n=sample_n, random_state=random_state)
    y_sample = y_train.loc[x_sample.index, TARGET].to_numpy()
    explainability = _explain(model, x_sample, y_sample, reporting_dir, params)

    # --- MLflow: log params, metrics, artefacts and register the model ---------------
    mlflow = setup_mlflow(mlflow_params)
    with mlflow.start_run(run_name=f"train_{champion_name}"):
        mlflow.log_param("model", champion_name)
        mlflow.log_param("target_transform", "none")
        mlflow.log_param("n_features", x_train.shape[1])
        mlflow.log_param("explainability_method", explainability["method"])
        mlflow.log_metrics(metrics)
        for plot in explainability["plots"]:
            if Path(plot).exists():
                mlflow.log_artifact(plot, artifact_path="explainability")
        try:
            import mlflow.sklearn  # ensure the sklearn flavor is loaded (mlflow 3.x)

            name = mlflow_params.get("registered_model_name", "fifa_player_rating_model")
            # Force cloudpickle: MLflow 3.x defaults to skops, which rejects the numpy
            # ufuncs inside our TransformedTargetRegressor as "untrusted types".
            try:
                mlflow.sklearn.log_model(
                    model,
                    name="model",
                    registered_model_name=name,
                    serialization_format="cloudpickle",
                )
            except TypeError:
                # Older MLflow uses `artifact_path` instead of `name`.
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=name,
                    serialization_format="cloudpickle",
                )
            logger.info("Logged + registered model '%s' in MLflow", name)
        except Exception as exc:  # noqa: BLE001 - registry is best-effort in local mode
            logger.warning("MLflow model logging failed (%s); model still saved via Kedro", exc)

    serving_metadata = {
        "model_name": champion_name,
        "target": TARGET,
        "target_transform": "none",
        "feature_columns": list(x_train.columns),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "attribute_medians": attribute_imputer.get("attribute_medians", {}),
        "metrics": metrics,
    }
    return model, metrics, explainability, serving_metadata
