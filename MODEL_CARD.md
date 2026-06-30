# MLOps Project Report: FIFA Player Rating Prediction (European Soccer Database)

> Every number below comes from an actual `kedro run` on the shipped data sample (seed 42)
> and is reproducible. Re-run the pipeline to regenerate all artefacts.

## 1. Problem, data and success metrics

**Goal.** Predict a football player's FIFA `overall_rating` (0 to 100) from in-game skill
attributes. A scouting platform could use this to rank transfer targets automatically, and a game
studio could use it to flag rating errors before a release ships.

**Data.** We use the European Soccer Database (Kaggle, Mathien), a SQLite database covering
2008 to 2016 across 11 European leagues. The relevant table is `Player_Attributes` (about 184k
per-player FIFA snapshots, 33 skill ratings on a 0 to 100 scale plus `preferred_foot` and two
work-rate fields), enriched with `height`, `weight` and `birthday` joined from the `Player`
table. It is a good test bed because it is messy in the ways real data is: mixed dtypes, missing
values that are sometimes structural and sometimes accidental, dirty categorical fields, a
near-duplicate leakage column, and a time axis. The target itself stays clean and continuous.

The committed CSV is a deterministic 6,000-row sample, so the pipeline runs end to end in about
two minutes and stays reproducible. `scripts/build_dataset.py` regenerates the table at any size
(up to the full 184k rows) from the SQLite file.

**Success metrics.** On the committed sample `overall_rating` is bounded and roughly symmetric
(about 42 to 90, mean about 68), so unlike a right-skewed monetary target it needs no log
transform. We model and report the rating directly.

| Metric | Why | Target (PoC) | Achieved |
|--------|-----|--------------|----------|
| RMSE (pts) | Primary; penalises large rating errors | <= 3.0 | **1.45** |
| MAE (pts) | Typical absolute error, easy to communicate | <= 2.5 | **1.05** |
| R2 | Share of rating variance explained | >= 0.85 | **0.954** |
| MAPE (%) | Relative error for stakeholders | <= 5 % | **1.57 %** |
| Within +/- 2 pts (%) | How often the prediction is close | >= 70 % | **86.8 %** |

The champion model is HistGradientBoosting, selected by 5-fold CV RMSE. All success criteria are
met on the held-out test set.

## 2. Project planning (agile sprints)

We organised the work as the Kedro pipelines themselves, with each pipeline mapped to a sprint:

| Sprint | Theme | Deliverable |
|--------|-------|-------------|
| S1 | Data quality and ingestion | `data_quality` pipeline with 19 assertions, JSON report |
| S2 | Cleaning and feature store | `data_cleaning`, `data_feat_engeneering`, feature metadata |
| S3 | Modelling and tracking | `data_split`, `model_selection`, `model_train` with MLflow |
| S4 | Explainability and reporting | SHAP, metrics, feature-importance artefacts |
| S5 | Serving and containers | FastAPI app, Dockerfile, docker-compose |
| S6 | Monitoring and tests | `data_drifts` (PSI/KS/Evidently), pytest suite |

Each sprint produced a pipeline that runs on its own (`kedro run --pipeline <name>`), so a
component can be re-executed in isolation in production. For example, only `data_drifts` runs on
a fresh batch, without re-running the whole DAG.

## 3. Results and conclusions

### 3.1 Data exploration and cleaning decisions

`potential` (a player's projected peak rating) correlates 0.76 with the target, and a scouting platform would not have a clean, independently sourced version of it at scoring time, so it is dropped to prevent leakage.

Work-rate columns contain invalid labels such as "norm", "stoc", "y", and "le". The pipeline treats valid work rates as ordered categories and maps invalid or missing labels to the modal category, "medium".

Rows missing a target (0.5%) are dropped. Missing skill blocks are median-imputed. These medians are computed on the full cleaned table before the split and saved to disk as a feature-store artefact. The serving API uses this artefact directly to score records without recalculating medians. This is acceptable for the proof-of-concept sample, but a production version should fit imputers inside the training fold or feature-store training job.

Engineered features include `age` (snapshot date minus birthday), `bmi`, seven FIFA-style attribute-group means (attacking, skill, movement, power, mentality, defending, goalkeeping), and an `is_goalkeeper` flag.

### 3.2 Modelling
Five-fold cross-validation over four candidates (`model_selection_report.json`):

| Model | CV RMSE (pts) |
|-------|--------------|
| HistGradientBoosting (champion) | **1.59** |
| Random Forest | 1.93 |
| Linear Regression | 3.29 |
| Ridge | 3.30 |

The gap between the boosted trees (CV RMSE 1.59) and the linear baselines (3.29 to 3.30) shows
that a rating depends on attribute interactions, not a weighted sum. Boosting also edges out the
bagged Random Forest (1.59 vs 1.93): a small learning rate over many iterations keeps cutting bias
on a smooth target. On the held-out test set the champion reaches RMSE 1.45, R2 0.954 and MAE 1.05.

An R2 of 0.954 sounds high for a held-out set, but it makes sense here: FIFA computes `overall_rating` from roughly the same attributes we give the model, so the trees are largely recovering a known scoring rule rather than discovering a hidden one. One thing does temper that number. The dataset holds 1.35 snapshots per player on average, so a plain random split can put the same player in both train and test. We check this directly below.

### 3.2.1 Robustness and validation

We re-split the data by player instead of by row (`GroupShuffleSplit` on `player_api_id`, so no player appears on both sides) to see how much the random split flatters the result.

| Split | RMSE | R2 | Within 2 pts (%) |
|-------|------|-----|------------------|
| Random (shipped) | 1.45 | 0.954 | 86.8 |
| Player-grouped | 1.54 | 0.950 | 84.2 |

The cost of the leakage is small: RMSE rises from 1.45 to 1.54 and the within-2 rate falls from 86.8 to 84.2 percent, but every threshold from Section 1 still holds. We also went back through the correlation table looking for any other near-duplicate of the target; nothing else came close, `reactions` remains the strongest retained correlate. Residual and calibration plots in `notebooks/model_evaluation.ipynb` are unremarkable in a good way, residuals sit centred on zero, and binned predictions track the observed rating closely in the dense middle of the range. The learning curve there covers only the 6,000-row sample, so it tells us the sample is enough for this result, not that more data from the full 184k-row table would not help further.

We checked reproducibility by running the full DAG twice in a clean virtual environment: both runs complete green, `pytest` passes, and the metrics in `model_metrics.json` are bit-identical across runs (RMSE 1.4483 both times). Cross-validation and the Random Forest candidate originally used `n_jobs=-1`, which reorders floating-point sums across parallel workers and breaks exact reproducibility even with a fixed seed; we pinned `n_jobs=1` in `modeling.py` and `model_selection/nodes.py` so a fresh `kedro run` reproduces the same numbers, not just the same champion.

### 3.3 Feature importance and explainability (SHAP)
`model_train` computes SHAP values on the final model, saving `shap_summary.png` and `shap_bar.png`.

`ball_control` and `reactions` top the SHAP ranking, followed by `defending_mean`, `heading_accuracy`, and `dribbling`. This is consistent with the raw EDA where `reactions` had a 0.765 correlation with the target. Adding `is_goalkeeper` and `goalkeeping_mean` lets the same model represent goalkeeper and outfield-player patterns. The pipeline falls back to permutation importance if SHAP is unavailable.

### 3.4 Drift monitoring
`data_drifts` compares a current batch against the training reference using PSI and the KS test
per feature. The committed `drift_report.json` reflects a clean run with 1 of 48 features flagged, so `dataset_drift` is false. With 47 numeric KS tests at a 5% level, one flag is within expected false-positive noise. Flipping `drift.simulate_drift: true` injects a shift into a broad set of attributes, and 18 of the 48 features now trip the alarm, well past the threshold that marks `dataset_drift` true, committed as `drift_report_simulated.json`.

## 4. From proof of concept to production

| Area | Advantage of current choice | Risk | Proposed mitigation |
|------|------------------------------|------|---------------------|
| Compute (Pandas) | Simple, reproducible, zero infra | Single machine; the full table is about 184k rows and real feeds are larger | Port nodes to Spark or Dask (Kedro keeps the same node API), about 2 weeks |
| Ingestion (CSV from SQLite) | One script, transparent | Manual, point-in-time snapshot | Scheduled extract into a warehouse table, about 3 days |
| Orchestration (Kedro CLI) | Modular, testable, layered catalog | No scheduling or retries | Deploy to Airflow or Kubeflow via `kedro-airflow`, about 1 week |
| Tracking (MLflow, SQLite) | Free local versioning and registry | Not multi-user, no auth | Hosted MLflow server with S3/DB backend, about 3 days |
| Feature store (parquet `04_feature`) | Lightweight offline store | No online or low-latency serving, no point-in-time joins | Adopt Feast or Hopsworks for an online store, about 2 weeks |
| Serving (FastAPI + Docker) | Standard, language-agnostic REST | Manual scaling | Kubernetes with autoscaling, or MLflow Model Serving, about 1 week |
| Drift (PSI/KS + Evidently) | Interpretable, dependency-light | Batch only, heuristic thresholds | Scheduled monitoring job with alerting, auto-trigger retrain, about 1 week |
| Data quality (custom asserts) | No heavy dependency, fast | Hand-maintained | Migrate to Great Expectations suites in CI, about 3 days |

Dropping `potential` handles offline leakage. In production, a feature store with CI checks is required to enforce this constraint against live data. The drift monitor currently checks only input distributions, offering no visibility into prediction drift or concept drift. Additionally, the system lacks a CI/CD retrain trigger tied to labelled feedback.

Reproducibility comes from a single random seed (`parameters.yml`), pinned dependencies
(`requirements.txt` and `requirements-lock.txt`), single-threaded model fitting, and persisted
Kedro layers, so any teammate reproduces the same metrics and the same champion from `kedro run`,
not just an approximately similar one.

## 5. Packages and versions

Core stack (exact versions frozen in `requirements-lock.txt`):

| Package | Version | Role |
|---------|---------|------|
| kedro | 0.19.x | Pipeline orchestration and data catalog |
| kedro-datasets | 4.x | CSV/Parquet/JSON/Pickle dataset connectors |
| pandas / numpy | 2.x / 2.x | Data manipulation and numerics |
| pyarrow | 15+ | Parquet I/O for the layered catalog |
| scipy | 1.1x | KS test for drift |
| scikit-learn | 1.x | Preprocessing, model zoo, metrics |
| mlflow | 3.x | Experiment tracking and model registry |
| shap | 0.5x | Model explainability |
| evidently | 0.7.x | HTML drift report |
| matplotlib | 3.x | SHAP and importance plots |
| fastapi / uvicorn / pydantic | see lock file | Model serving |
| pytest / pytest-cov | 8.x / 5.x | Testing |

## 6. How to reproduce

```bash
pip install -r requirements.txt && pip install -e .
kedro run                       # full DAG, all artefacts in data/
pytest                          # 25 unit and pipeline tests
uvicorn mlops_player_rating.serving.app:app --reload   # serve at :8000
```
