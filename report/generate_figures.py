"""Generate the full diagnostic figure suite for the report and notebook.

Additive analysis only: reuses the project's own transforms, does not touch the pipeline.
Figures land in report/figures/ and the report-bound ones are copied to data/08_reporting/.
Run from the project root with the project venv:
    .venv/Scripts/python report/generate_figures.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    KFold,
    cross_val_score,
    train_test_split,
)

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
BLUE, RED, GREY = "#1f77b4", "#d62728", "#7f7f7f"
FIG = ROOT / "report" / "figures"
REP = ROOT / "data" / "08_reporting"
FIG.mkdir(parents=True, exist_ok=True)


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


def build():
    raw = pd.read_csv(ROOT / "data" / "01_raw" / "player_attributes.csv")
    df = normalize_columns(raw)
    df = apply_value_semantics(df)
    df = df[df[TARGET].notna()].reset_index(drop=True)
    df = apply_attribute_imputer(df, fit_attribute_imputer(df))
    raw_corr = normalize_columns(raw)
    df = engineer_features(df)
    groups = df["player_api_id"].copy()
    feats = drop_non_features(df, extra=[])
    y = feats[TARGET].astype(float)
    x = feats.drop(columns=[TARGET])
    return x, y, groups


def champion(x_tr):
    num, cat = split_feature_types(x_tr)
    return build_model(candidate_estimators(RNG)["hist_gradient_boosting"], num, cat)


def fig_corr_heatmap(x, y):
    num = x.select_dtypes("number").copy()
    num[TARGET] = y.to_numpy()
    corr = num.corr()
    order = corr[TARGET].abs().sort_values(ascending=False).index[:14]
    c = corr.loc[order, order]
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7)
    for i in range(len(order)):
        for j in range(len(order)):
            ax.text(j, i, f"{c.iloc[i,j]:.2f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(c.iloc[i, j]) > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Pearson r", fontsize=8)
    ax.set_title("Correlation of top features with overall_rating", fontsize=10)
    plt.tight_layout(); plt.savefig(FIG / "correlation_heatmap.png", dpi=140, bbox_inches="tight"); plt.close()


def fig_residuals_calibration(yte, preds):
    resid = yte - preds
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
    ax[0].scatter(preds, resid, s=6, alpha=0.3, color=BLUE, edgecolors="none")
    ax[0].axhline(0, color=RED, lw=1, ls="--")
    ax[0].set_xlabel("predicted rating"); ax[0].set_ylabel("residual (actual - pred)")
    ax[0].set_title("Residuals vs predicted", fontsize=10); clean(ax[0])
    # calibration: bin predictions, plot mean actual vs mean predicted
    bins = np.linspace(preds.min(), preds.max(), 13)
    idx = np.digitize(preds, bins)
    xs, ys = [], []
    for b in range(1, len(bins)):
        m = idx == b
        if m.sum() > 5:
            xs.append(preds[m].mean()); ys.append(yte[m].mean())
    lo, hi = min(xs), max(xs)
    ax[1].plot([lo, hi], [lo, hi], color=GREY, ls="--", lw=1, label="ideal")
    ax[1].plot(xs, ys, "o-", color=BLUE, ms=4, label="model")
    ax[1].set_xlabel("mean predicted (bin)"); ax[1].set_ylabel("mean actual (bin)")
    ax[1].set_title("Calibration", fontsize=10); ax[1].legend(fontsize=8, frameon=False); clean(ax[1])
    plt.tight_layout(); plt.savefig(FIG / "residuals_calibration.png", dpi=140, bbox_inches="tight"); plt.close()


def fig_learning_curve(x, y):
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=RNG, shuffle=True)
    fracs = [0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
    tr_rmse, te_rmse = [], []
    for f in fracs:
        n = max(int(len(xtr) * f), 50)
        xs = xtr.iloc[:n]; ys = ytr.iloc[:n]
        m = champion(xs); m.fit(xs, ys.to_numpy())
        tr_rmse.append(regression_metrics(ys.to_numpy(), m.predict(xs))["rmse"])
        te_rmse.append(regression_metrics(yte.to_numpy(), m.predict(xte))["rmse"])
    sizes = [int(len(xtr) * f) for f in fracs]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(sizes, te_rmse, "o-", color=BLUE, label="held-out RMSE")
    ax.plot(sizes, tr_rmse, "o--", color=RED, label="train RMSE")
    ax.set_xlabel("training rows"); ax.set_ylabel("RMSE (pts)")
    ax.set_title("Learning curve (sample sufficiency)", fontsize=10)
    ax.legend(fontsize=8, frameon=False); clean(ax)
    plt.tight_layout(); plt.savefig(FIG / "learning_curve.png", dpi=140, bbox_inches="tight"); plt.close()
    return list(zip(sizes, [round(v, 3) for v in te_rmse]))


def fig_cv_boxplot(x, y):
    xtr, _, ytr, _ = train_test_split(x, y, test_size=0.2, random_state=RNG, shuffle=True)
    num, cat = split_feature_types(xtr)
    cv = KFold(n_splits=5, shuffle=True, random_state=RNG)
    names = ["hist_gradient_boosting", "random_forest", "ridge", "linear_regression"]
    labels = ["HistGB", "RandomForest", "Ridge", "Linear"]
    scores = []
    for nm in names:
        mdl = build_model(candidate_estimators(RNG)[nm], num, cat)
        s = -cross_val_score(mdl, xtr, ytr.to_numpy(), cv=cv,
                             scoring="neg_root_mean_squared_error")
        scores.append(s)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    bp = ax.boxplot(scores, tick_labels=labels, patch_artist=True, widths=0.6)
    for patch in bp["boxes"]:
        patch.set_facecolor(BLUE); patch.set_alpha(0.6)
    ax.set_ylabel("5-fold CV RMSE (pts)")
    ax.set_title("Model selection: CV RMSE per candidate", fontsize=10)
    ax.tick_params(axis="x", labelsize=8); clean(ax)
    plt.tight_layout(); plt.savefig(FIG / "cv_scores.png", dpi=140, bbox_inches="tight"); plt.close()


def fig_drift_psi():
    clean_r = json.load(open(REP / "drift_report.json"))
    sim_r = json.load(open(REP / "drift_report_simulated.json"))
    cf = {f["feature"]: f["psi"] for f in clean_r["features"]}
    sf = {f["feature"]: f["psi"] for f in sim_r["features"]}
    top = sorted(sf, key=lambda k: sf[k], reverse=True)[:14][::-1]
    ypos = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.barh(ypos + 0.2, [sf[k] for k in top], height=0.4, color=RED, label="simulated")
    ax.barh(ypos - 0.2, [cf.get(k, 0) for k in top], height=0.4, color=BLUE, label="clean")
    ax.axvline(0.1, color=GREY, ls=":", lw=1); ax.axvline(0.2, color="black", ls="--", lw=1)
    ax.text(0.1, len(top) - 0.3, " PSI 0.1", fontsize=7, color=GREY)
    ax.text(0.2, len(top) - 0.3, " PSI 0.2 (drift)", fontsize=7)
    ax.set_yticks(ypos); ax.set_yticklabels(top, fontsize=7.5)
    ax.set_xlabel("PSI"); ax.set_title("Drift by feature: clean vs simulated batch", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right"); clean(ax)
    plt.tight_layout(); plt.savefig(FIG / "drift_psi.png", dpi=140, bbox_inches="tight"); plt.close()


def fig_split_comparison():
    r = json.load(open(REP / "robustness_report.json"))
    metrics = ["rmse", "mae"]
    rnd = [r["random_split"][m] for m in metrics]
    grp = [r["grouped_split"][m] for m in metrics]
    xpos = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.bar(xpos - 0.2, rnd, 0.4, color=BLUE, label="random split")
    ax.bar(xpos + 0.2, grp, 0.4, color=RED, label="player-grouped split")
    for i, (a, b) in enumerate(zip(rnd, grp)):
        ax.text(i - 0.2, a + 0.01, f"{a:.2f}", ha="center", fontsize=7)
        ax.text(i + 0.2, b + 0.01, f"{b:.2f}", ha="center", fontsize=7)
    ax.set_xticks(xpos); ax.set_xticklabels(["RMSE", "MAE"])
    ax.set_ylabel("points"); ax.set_title("Random vs player-grouped split", fontsize=10)
    ax.legend(fontsize=8, frameon=False); clean(ax)
    plt.tight_layout(); plt.savefig(FIG / "split_comparison.png", dpi=140, bbox_inches="tight"); plt.close()


def main():
    x, y, groups = build()
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=RNG, shuffle=True)
    m = champion(xtr); m.fit(xtr, ytr.to_numpy())
    preds = m.predict(xte)

    fig_corr_heatmap(x, y)
    fig_residuals_calibration(yte.to_numpy(), preds)
    lc = fig_learning_curve(x, y)
    fig_cv_boxplot(x, y)
    fig_drift_psi()
    fig_split_comparison()

    # copy report-bound figures into the committed reporting layer too
    for name in ["correlation_heatmap.png", "residuals_calibration.png", "learning_curve.png",
                 "cv_scores.png", "drift_psi.png", "split_comparison.png", "error_analysis.png",
                 "eda_overview.png"]:
        src = FIG / name
        if src.exists():
            shutil.copy(src, REP / name)
    print("generated figures:", sorted(p.name for p in FIG.glob("*.png")))
    print("learning curve (rows, held-out RMSE):", lc)


if __name__ == "__main__":
    main()
