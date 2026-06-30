"""Nodes for the ``data_drifts`` pipeline.

Compares a current batch against the training reference using PSI and the
Kolmogorov-Smirnov test. Optional synthetic drift can be injected for monitoring checks.

PSI is computed by summing ``(current_pct - reference_pct) * ln(current_pct / reference_pct)``
across reference-quantile buckets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


def _psi_numeric(ref: pd.Series, cur: pd.Series, buckets: int = 10) -> float:
    ref = pd.to_numeric(ref, errors="coerce").dropna()
    cur = pd.to_numeric(cur, errors="coerce").dropna()
    if len(ref) < 2 or len(cur) < 1:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, buckets + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_dist = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_dist = np.histogram(cur, bins=edges)[0] / len(cur)
    eps = 1e-6
    ref_dist = np.clip(ref_dist, eps, None)
    cur_dist = np.clip(cur_dist, eps, None)
    return float(np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)))


def _psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    eps = 1e-6
    ref_dist = ref.astype(str).value_counts(normalize=True)
    cur_dist = cur.astype(str).value_counts(normalize=True)
    categories = set(ref_dist.index) | set(cur_dist.index)
    psi = 0.0
    for cat in categories:
        r = max(float(ref_dist.get(cat, 0.0)), eps)
        a = max(float(cur_dist.get(cat, 0.0)), eps)
        psi += (a - r) * np.log(a / r)
    return float(psi)


def _simulate_drift(df: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Apply a controlled shift to rating-related numeric columns.

    The shifted batch is used only for the drift-monitoring example. It changes
    enough feature distributions to cross the configured feature-level and
    dataset-level thresholds.
    """
    df = df.copy()
    rng = np.random.default_rng(0)
    targets = [
        "age", "attacking_mean", "skill_mean", "movement_mean", "power_mean",
        "mentality_mean", "defending_mean", "reactions", "ball_control",
        "short_passing", "dribbling", "finishing", "crossing", "long_passing",
        "shot_power", "sprint_speed", "vision", "positioning",
    ]
    for col in targets:
        if col in df.columns:
            noise = rng.normal(1.0, 0.05, len(df))
            df[col] = pd.to_numeric(df[col], errors="coerce") * factor * noise
    return df


def evaluate_drift(
    reference: pd.DataFrame, current_source: pd.DataFrame, params: dict[str, Any]
) -> dict[str, Any]:
    """Compute per-feature drift between ``reference`` and a sample of ``current_source``."""
    sample_size = min(params.get("sample_size", 500), len(current_source))
    current = current_source.sample(n=sample_size, random_state=params.get("random_state", 42))

    simulate = params.get("simulate_drift", False)
    if simulate:
        current = _simulate_drift(current, params.get("drift_factor", 1.25))
        logger.info("Injected synthetic drift (factor=%s)", params.get("drift_factor", 1.25))

    psi_threshold = params.get("psi_threshold", 0.2)
    ks_threshold = params.get("ks_pvalue_threshold", 0.05)

    feature_reports: list[dict[str, Any]] = []
    for col in reference.columns:
        is_numeric = pd.api.types.is_numeric_dtype(reference[col])
        if is_numeric:
            psi = _psi_numeric(reference[col], current[col])
            ks_p = float(
                ks_2samp(
                    pd.to_numeric(reference[col], errors="coerce").dropna(),
                    pd.to_numeric(current[col], errors="coerce").dropna(),
                ).pvalue
            )
            drifted = bool(psi > psi_threshold or ks_p < ks_threshold)
        else:
            psi = _psi_categorical(reference[col], current[col])
            ks_p = float("nan")
            drifted = bool(psi > psi_threshold)
        feature_reports.append(
            {
                "feature": col,
                "type": "numeric" if is_numeric else "categorical",
                "psi": round(psi, 4),
                "ks_pvalue": None if np.isnan(ks_p) else round(ks_p, 4),
                "drifted": drifted,
            }
        )

    feature_reports.sort(key=lambda r: r["psi"], reverse=True)
    n_drifted = sum(1 for f in feature_reports if f["drifted"])
    share = n_drifted / len(feature_reports) if feature_reports else 0.0
    dataset_drift = share >= params.get("dataset_drift_share", 0.3)

    report = {
        "n_features": len(feature_reports),
        "sample_size": int(sample_size),
        "simulated_drift": bool(simulate),
        "psi_threshold": psi_threshold,
        "ks_pvalue_threshold": ks_threshold,
        "n_drifted": int(n_drifted),
        "share_drifted": round(share, 4),
        "dataset_drift": bool(dataset_drift),
        "features": feature_reports,
    }

    _maybe_evidently(reference, current, params)

    logger.info(
        "Drift: %d/%d features drifted (%.0f%%) -> dataset_drift=%s",
        n_drifted,
        len(feature_reports),
        share * 100,
        dataset_drift,
    )
    return report


def _maybe_evidently(reference: pd.DataFrame, current: pd.DataFrame, params: dict[str, Any]) -> None:
    """Write an Evidently HTML drift report when Evidently is installed."""
    reporting_dir = Path(params.get("reporting_dir", "data/08_reporting"))
    reporting_dir.mkdir(parents=True, exist_ok=True)
    out = reporting_dir / "evidently_drift_report.html"
    try:
        try:  # Evidently >= 0.4, < 0.6
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report
        except ImportError:  # newer Evidently
            from evidently import Report
            from evidently.presets import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        result = report.run(reference_data=reference, current_data=current)
        saver = getattr(report, "save_html", None) or getattr(result, "save_html", None)
        if saver:
            saver(str(out))
            logger.info("Evidently drift report saved to %s", out)
    except Exception as exc:  # noqa: BLE001
        logger.info("Evidently report skipped (%s)", exc)
