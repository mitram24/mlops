"""Robustness experiments for the report (additive analysis, does not change the pipeline).

Reconstructs the model feature table with the project's own transforms while keeping
``player_api_id``, then compares the headline random split against a player-grouped split,
and runs an error analysis on the random-split test set. Saves figures + a JSON summary.

Run with the system Python from the project root:
    python report/robustness_experiments.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlops_player_rating.modeling import (  # noqa: E402
    build_model,
    candidate_estimators,
    regression_metrics,
    split_feature_types,
)
from mlops_player_rating.utils import (  # noqa: E402
    TARGET,
    apply_attribute_imputer,
    apply_value_semantics,
    drop_non_features,
    engineer_features,
    fit_attribute_imputer,
    normalize_columns,
)

RNG = 42
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def build_table() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Replicate data_quality -> data_cleaning -> data_feat_engeneering, keep player id."""
    raw = pd.read_csv(ROOT / "data" / "01_raw" / "player_attributes.csv")
    df = normalize_columns(raw)
    df = apply_value_semantics(df)
    df = df[df[TARGET].notna()].reset_index(drop=True)          # data_cleaning: drop null target
    medians = fit_attribute_imputer(df)                          # global imputer (pre-split)
    df = apply_attribute_imputer(df, medians)
    df = engineer_features(df)
    groups = df["player_api_id"].copy()                          # keep grouping key
    feats = drop_non_features(df, extra=[])                      # features + target
    y = feats[TARGET].astype(float)
    x = feats.drop(columns=[TARGET])
    return x, y, groups


def fit_eval(x_tr, y_tr, x_te, y_te) -> dict[str, float]:
    numeric, categorical = split_feature_types(x_tr)
    model = build_model(candidate_estimators(RNG)["hist_gradient_boosting"], numeric, categorical)
    model.fit(x_tr, y_tr.to_numpy())
    preds = model.predict(x_te)
    m = regression_metrics(y_te.to_numpy(), preds)
    return m, preds


def main() -> None:
    x, y, groups = build_table()
    n_rows, n_players = len(x), groups.nunique()
    print(f"rows={n_rows} unique_players={n_players} snapshots/player={n_rows/n_players:.2f}")

    # 1) Random split (reproduces the pipeline's evaluation)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=RNG, shuffle=True)
    rnd, preds_rnd = fit_eval(xtr, ytr, xte, yte)
    print("RANDOM  split:", {k: round(v, 4) for k, v in rnd.items()})

    # 2) Player-grouped split (no player in both train and test)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RNG)
    tr_idx, te_idx = next(gss.split(x, y, groups))
    gxtr, gxte = x.iloc[tr_idx], x.iloc[te_idx]
    gytr, gyte = y.iloc[tr_idx], y.iloc[te_idx]
    overlap = len(set(groups.iloc[tr_idx]) & set(groups.iloc[te_idx]))
    grp, _ = fit_eval(gxtr, gytr, gxte, gyte)
    print(f"GROUPED split: {{train_players, test_players overlap={overlap}}}",
          {k: round(v, 4) for k, v in grp.items()})

    # --- Error analysis on the random-split test set --------------------------------
    resid = yte.to_numpy() - preds_rnd
    abs_err = np.abs(resid)
    is_gk = xte["is_goalkeeper"].to_numpy().astype(bool)
    seg = {
        "outfield": float(np.sqrt(np.mean(resid[~is_gk] ** 2))),
        "goalkeeper": float(np.sqrt(np.mean(resid[is_gk] ** 2))),
    }
    # RMSE by rating band
    bands = pd.cut(yte, bins=[0, 60, 65, 70, 75, 80, 100],
                   labels=["<60", "60-65", "65-70", "70-75", "75-80", ">80"])
    band_rmse = {str(b): float(np.sqrt(np.mean(resid[(bands == b).to_numpy()] ** 2)))
                 for b in bands.cat.categories if (bands == b).any()}

    # Figure: residual scatter + error by rating band
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
    ax[0].scatter(yte, preds_rnd, s=6, alpha=0.35, color="#1f77b4", edgecolors="none")
    lo, hi = yte.min(), yte.max()
    ax[0].plot([lo, hi], [lo, hi], color="#d62728", lw=1.0, ls="--")
    ax[0].set_xlabel("actual rating"); ax[0].set_ylabel("predicted rating")
    ax[0].set_title(f"Predicted vs actual (RMSE {rnd['rmse']:.2f})", fontsize=10)
    ax[0].spines[["top", "right"]].set_visible(False)

    ax[1].bar(list(band_rmse.keys()), list(band_rmse.values()), color="#1f77b4")
    ax[1].set_xlabel("actual rating band"); ax[1].set_ylabel("RMSE (pts)")
    ax[1].set_title("Error by rating band", fontsize=10)
    ax[1].spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG / "error_analysis.png", dpi=140, bbox_inches="tight")
    plt.close()

    summary = {
        "n_rows": n_rows, "n_players": int(n_players),
        "snapshots_per_player": round(n_rows / n_players, 3),
        "random_split": rnd, "grouped_split": grp,
        "group_overlap_players": overlap,
        "rmse_by_segment": seg, "rmse_by_band": band_rmse,
        "within_2_overall_pct": float(np.mean(abs_err <= 2) * 100),
    }
    (ROOT / "data" / "08_reporting" / "robustness_report.json").write_text(
        json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
