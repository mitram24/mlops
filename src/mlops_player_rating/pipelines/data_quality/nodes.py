"""Nodes for the ``data_quality`` pipeline.

This is the project's first line of defence: a battery of *asserts* over the raw data
(schema, types, ranges, uniqueness, missingness). Critical failures stop the pipeline
before any compute is spent training on bad data; soft failures are recorded as
warnings in a JSON report that lands in ``data/08_reporting``.

Week 1 of the course taught data validation through Great Expectations
(``01_Data_Unit_Tests.ipynb``). We validate the same kinds of things (schema, ranges,
nulls) but with plain Python asserts instead of the GX framework, a lighter dependency
footprint that suits a single-table proof of concept; the brief explicitly allows
"one of the tools from class or your own solution".
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from mlops_player_rating.utils import SKILL_COLUMNS, normalize_columns

logger = logging.getLogger(__name__)


class DataQualityError(AssertionError):
    """Raised when one or more *critical* data-quality checks fail."""


def _check(report: list[dict[str, Any]], name: str, level: str, passed: bool, detail: str) -> None:
    report.append(
        {"name": name, "level": level, "passed": bool(passed), "detail": str(detail)}
    )
    log = logger.info if passed else (logger.error if level == "critical" else logger.warning)
    log("[data-quality] %-32s %s | %s", name, "PASS" if passed else "FAIL", detail)


def validate_data_quality(
    raw: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate the raw dataset and emit the normalised, ingested table + a report.

    Args:
        raw: Raw dataframe loaded straight from the CSV (original column names).
        params: ``data_quality`` parameters (thresholds, expected ranges, ...).

    Returns:
        ``(ingested, report)`` where ``ingested`` is the column-normalised dataframe and
        ``report`` is a JSON-serialisable summary of every check.

    Raises:
        DataQualityError: if any critical check fails.
    """
    df = normalize_columns(raw)
    checks: list[dict[str, Any]] = []

    target = params["target"]
    id_column = params.get("id_column")
    n_rows, n_cols = df.shape

    # --- shape -------------------------------------------------------------------
    _check(checks, "non_empty", "critical", n_rows > 0, f"rows={n_rows}")
    _check(
        checks,
        "min_rows",
        "critical",
        n_rows >= params.get("min_rows", 1),
        f"rows={n_rows} >= {params.get('min_rows', 1)}",
    )
    _check(
        checks,
        "min_columns",
        "critical",
        n_cols >= params.get("min_columns", 1),
        f"cols={n_cols} >= {params.get('min_columns', 1)}",
    )

    # --- required columns --------------------------------------------------------
    required = params.get("required_columns", [target])
    missing_cols = [c for c in required if c not in df.columns]
    _check(
        checks,
        "required_columns_present",
        "critical",
        not missing_cols,
        f"missing={missing_cols}" if missing_cols else "all present",
    )

    # --- target sanity -----------------------------------------------------------
    if target in df.columns:
        tgt = pd.to_numeric(df[target], errors="coerce")
        # The column must be numeric and must have *some* usable values...
        _check(
            checks,
            "target_has_values",
            "critical",
            tgt.notna().any(),
            f"non_null={int(tgt.notna().sum())}",
        )
        _check(
            checks,
            "target_is_numeric",
            "critical",
            tgt.notna().sum() == df[target].notna().sum(),
            f"{target} fully numeric where present",
        )
        # ...but a *small* fraction of null targets is tolerated (dropped in cleaning).
        max_tgt_missing = params.get("max_target_missing_fraction", 0.0)
        tgt_missing = float(tgt.isna().mean())
        _check(
            checks,
            "target_missing_under_threshold",
            "warning" if max_tgt_missing > 0 else "critical",
            tgt_missing <= max_tgt_missing,
            f"missing={tgt_missing:.3%} <= {max_tgt_missing:.0%}",
        )
        _check(
            checks,
            "target_positive",
            "critical",
            bool((tgt.dropna() > 0).all()),
            "all overall_rating > 0",
        )

    # --- id uniqueness -----------------------------------------------------------
    if id_column and id_column in df.columns:
        dupes = int(df[id_column].duplicated().sum())
        _check(checks, "id_unique", "critical", dupes == 0, f"duplicate ids={dupes}")

    # --- duplicate rows ----------------------------------------------------------
    dup_rows = int(df.duplicated().sum())
    _check(checks, "no_duplicate_rows", "warning", dup_rows == 0, f"duplicate rows={dup_rows}")

    # --- numeric ranges ----------------------------------------------------------
    for col, (lo, hi) in params.get("ranges", {}).items():
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            within = bool(((series >= lo) & (series <= hi)).all())
            n_out = int(((series < lo) | (series > hi)).sum())
            _check(checks, f"range_{col}", "warning", within, f"[{lo},{hi}] out_of_range={n_out}")

    # --- skill schema coverage ---------------------------------------------------
    present_skills = [c for c in SKILL_COLUMNS if c in df.columns]
    _check(
        checks,
        "skill_columns_present",
        "critical",
        len(present_skills) >= 25,
        f"{len(present_skills)}/{len(SKILL_COLUMNS)} FIFA skill columns present",
    )

    # --- missingness -------------------------------------------------------------
    max_missing = params.get("max_missing_fraction", 1.0)
    frac = df.isna().mean()
    worst = frac[frac > max_missing]
    _check(
        checks,
        "missing_fraction_under_threshold",
        "warning",
        worst.empty,
        f"cols over {max_missing:.0%}: {list(worst.index)}" if not worst.empty else "ok",
    )

    n_failed_critical = sum(1 for c in checks if not c["passed"] and c["level"] == "critical")
    n_warnings = sum(1 for c in checks if not c["passed"] and c["level"] == "warning")
    report = {
        "dataset": "EuropeanSoccerDatabase",
        "n_rows": int(n_rows),
        "n_columns": int(n_cols),
        "n_checks": len(checks),
        "n_failed_critical": int(n_failed_critical),
        "n_warnings": int(n_warnings),
        "passed": n_failed_critical == 0,
        "checks": checks,
    }

    if n_failed_critical:
        failed = [c["name"] for c in checks if not c["passed"] and c["level"] == "critical"]
        raise DataQualityError(f"Critical data-quality checks failed: {failed}")

    logger.info(
        "[data-quality] PASSED (%d checks, %d warnings) on %d rows x %d cols",
        len(checks),
        n_warnings,
        n_rows,
        n_cols,
    )
    return df, report
