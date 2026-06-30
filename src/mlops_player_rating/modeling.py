"""Model factory: a single source of truth for the preprocessing + estimator stack.

Both ``model_selection`` (cross-validation) and ``model_train`` (final fit) build their
models here, so the candidate compared in selection is byte-for-byte the model that gets
trained and shipped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_feature_types(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Partition columns into numeric vs categorical based on dtype."""
    numeric = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in x.columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Impute + scale numerics and impute + one-hot encode categoricals.

    ``handle_unknown="ignore"`` keeps the model robust to categories never seen during
    training — important for a service that will receive arbitrary real-world input.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def candidate_estimators(random_state: int = 42) -> dict[str, object]:
    """The model zoo compared during model selection.

    ``n_jobs=1`` on the forest: parallel tree building reorders floating-point sums
    across threads, so a fixed ``random_state`` alone does not give bit-identical
    reruns under ``n_jobs=-1``. Single-threaded is slower but exactly reproducible.
    """
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=10.0, random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=500,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=random_state,
        ),
    }


def build_model(
    estimator: object,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Assemble preprocessing + estimator into a single fitted-end-to-end pipeline.

    Unlike a right-skewed monetary target, ``overall_rating`` is bounded and roughly
    symmetric (about 42 to 90, mean about 68 on the committed sample), so no target
    transform is needed: ``.predict`` returns a rating directly and the serving API
    needs no post-processing.
    """
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_features, categorical_features)),
            ("model", estimator),
        ]
    )


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """RMSE / MAE / R² / MAPE on the rating scale (the modelling objective)."""
    y_true = np.asarray(y_true, dtype="float64").ravel()
    y_pred = np.asarray(y_pred, dtype="float64").ravel()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-9, None))) * 100)
    # Share of predictions within ±2 rating points — an intuitive business metric.
    within_2 = float(np.mean(np.abs(y_true - y_pred) <= 2.0) * 100)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape_pct": mape, "within_2_pct": within_2}
