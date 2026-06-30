"""Nodes for the ``model_predict`` pipeline (offline / batch scoring).

Same batch-scoring shape as the course's bank-example ``model_predict/nodes.py``
(``model.predict`` over a raw feature table). We add an ``abs_error`` column when the
target happens to be present in the input, useful for sanity-checking a batch against
ground truth without changing the scoring path itself.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mlops_player_rating.utils import TARGET, normalize_name, preprocess_for_inference

logger = logging.getLogger(__name__)


def predict(
    champion_model,
    raw_inference: pd.DataFrame,
    serving_metadata: dict,
) -> pd.DataFrame:
    """Score a batch of *raw* player snapshots with the trained model.

    Goes through the same ``preprocess_for_inference`` path as the online API, proving
    offline/online parity. If the input carries the ground-truth ``overall_rating`` the
    absolute error is reported too.
    """
    raw = raw_inference.copy()
    id_col = next((c for c in raw.columns if normalize_name(c) == "id"), None)
    ids = raw[id_col].to_numpy() if id_col else np.arange(len(raw))
    name_col = next((c for c in raw.columns if normalize_name(c) == "player_name"), None)

    features = preprocess_for_inference(
        raw,
        serving_metadata["attribute_medians"],
        serving_metadata["feature_columns"],
    )
    preds = champion_model.predict(features)

    out = pd.DataFrame({"id": ids, "predicted_rating": np.round(preds, 2)})
    if name_col is not None:
        out.insert(1, "player_name", raw[name_col].to_numpy())

    target_col = next((c for c in raw.columns if normalize_name(c) == TARGET), None)
    if target_col is not None:
        actual = pd.to_numeric(raw[target_col], errors="coerce").to_numpy()
        out["actual_rating"] = actual
        out["abs_error"] = np.abs(actual - preds).round(2)

    logger.info(f"Scored {len(out)} players (mean prediction={float(np.mean(preds)):.1f})")
    return out
