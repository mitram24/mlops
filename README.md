# mlops_player_rating

This NOVA IMS MLOps project predicts a football player's FIFA overall rating from in-game attributes in the [European Soccer Database](https://www.kaggle.com/datasets/hugomathien/soccer) (184k snapshots). The project uses Kedro pipelines for data checks, cleaning, feature engineering, model selection, training, batch prediction and drift checks. It also includes MLflow tracking, SHAP outputs, and a FastAPI service packaged with Docker.

> Each stage can run as part of the full Kedro DAG or on its own. For example, `data_drifts`
> can run against a new batch without retraining the model.

### See the results without running anything

Pre-generated outputs are committed directly to the repo:
Metrics in `data/08_reporting/model_metrics.json` show RMSE 1.45, R2 0.954, and MAPE 1.57 %, with 86.8 % of predictions within +/- 2 points. `data/08_reporting/model_selection_report.json` covers the 5-fold CV that picked HistGradientBoosting.
SHAP plots (`data/08_reporting/shap_summary.png`, `data/08_reporting/shap_bar.png`) and drift reports (`data/08_reporting/drift_report.json`, `data/08_reporting/evidently_drift_report.html`) are in the reporting layer. The trained artefact is at `data/06_models/champion_model.pkl`, so the FastAPI container can serve it directly. Batch predictions against actuals are available in `data/07_model_output/predictions.csv`. `MODEL_CARD.md` contains the full technical write-up.

## 1. Pipeline architecture

```
data_quality -> data_cleaning -> data_feat_engeneering -> data_split
                                                              |
                                          model_selection -> model_train
                                                              |
                                              model_predict   +   data_drifts
```

| Kedro pipeline           | Data layer in -> out                    | What it does                                                        |
|--------------------------|-----------------------------------------|---------------------------------------------------------------------|
| `data_quality`           | `01_raw` -> `02_intermediate` + report  | Normalises columns, runs 19 asserts (schema, ranges, ids, nulls)    |
| `data_cleaning`          | `02_intermediate` -> `03_primary`       | Cleans dirty work-rate values, drops null-target rows, fits an imputer |
| `data_feat_engeneering`  | `03_primary` -> `04_feature`            | Derived features plus feature-store metadata (the offline store)    |
| `data_split`             | `04_feature` -> `05_model_input`        | Reproducible train/test split                                       |
| `model_selection`        | `05_model_input` -> report + champion   | 5-fold CV over a model zoo, every run logged to MLflow              |
| `model_train`            | `05_model_input` -> `06_models`         | Fits champion, SHAP, metrics, logs and registers in MLflow          |
| `model_predict`          | `01_raw` sample -> `07_model_output`    | Batch scoring through the exact serving transform                   |
| `data_drifts`            | `05_model_input` -> `08_reporting`      | PSI plus KS drift report (optional Evidently HTML)                  |

## 2. The data

The raw analytical table ships in `data/01_raw/player_attributes.csv`, so the project runs out
of the box. It is one row per FIFA attribute snapshot (33 skill ratings on a 0 to 100 scale,
preferred foot, work rates, plus the player's height, weight and birthday joined from the
`Player` table), with the target column `overall_rating`.

The full dataset has about 184k snapshots. The handout allows a runnable data sample, so the
committed CSV is a deterministic 6,000-row sample (seed 42). The whole pipeline runs end to end
in about two minutes and stays reproducible. Regenerate the table at any size from the original
SQLite file:

```bash
# Full dataset:
python scripts/build_dataset.py --sqlite path/to/database.sqlite --max-rows 0
# Or a sample of N rows (default 50000):
python scripts/build_dataset.py --sqlite path/to/database.sqlite --max-rows 6000
```

## 3. Setup

```bash
cd mlops_player_rating

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
pip install -e .            # makes the `mlops_player_rating` package importable
```

## 4. Run the pipelines

```bash
# Full end-to-end run (quality, cleaning, FE, split, selection, train, predict, drift)
kedro run

# Run a single component
kedro run --pipeline data_quality
kedro run --pipeline data_drifts

# Useful macro-stages
kedro run --pipeline data_processing   # quality + cleaning + FE + split
kedro run --pipeline modelling         # selection + train
```

Artefacts land under `data/` along the numbered layers. Metrics, SHAP plots and the drift report
appear in `data/08_reporting/`.

### MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://localhost:5000  (Experiments and Models registry tab)
```

## 5. Serve the model

Local:
```bash
uvicorn mlops_player_rating.serving.app:app --reload
# open http://localhost:8000/docs
```

Container:
```bash
docker compose up --build       # API on http://localhost:8000
```

Example request (raw column names; any omitted fields are imputed with the saved attribute medians):
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records": [{"reactions": 82, "ball_control": 85, "short_passing": 84,
                    "dribbling": 86, "interceptions": 40, "standing_tackle": 38,
                    "preferred_foot": "left", "height": 170, "weight": 159,
                    "date": "2016-01-01", "birthday": "1987-06-24"}]}'
```

The API uses the same cleaning and feature engineering path as training. The batch
`model_predict` pipeline scores raw players and is usually within 1 to 2 rating points
of the actual value (see `data/07_model_output/predictions.csv`).

## 6. Tests

```bash
pytest                 # unit and pipeline tests (25 tests)
pytest --cov=mlops_player_rating
```

## 7. Simulating drift

```bash
# Edit conf/base/parameters.yml -> drift.simulate_drift: true
kedro run --pipeline data_drifts
# data/08_reporting/drift_report.json -> "dataset_drift": true
```

## 8. How each handout component is covered

| Handout requirement | Where it is implemented |
|---------------------|-------------------------|
| Unit data tests plus feature store | `data_quality` asserts; `data_feat_engeneering` builds the feature table and metadata in `04_feature`. |
| MLflow experimentation and versioning | `model_selection` does CV runs; `model_train` logs metrics/artefacts and calls `register_model`. |
| Metrics plus SHAP explainability | `model_train` writes `model_metrics.json`, `feature_importance.json`, `shap_summary.png`, and `shap_bar.png`. |
| Model serving and containers | FastAPI app in `serving/app.py`, packaged with `Dockerfile` and `docker-compose.yml`. |
| Data drift evaluation | `data_drifts` computes PSI/KS, has a drift-injection switch, and builds an optional Evidently HTML report. |
| Tests for functions and pipelines | Provided directly in `src/tests/`. |

See `MODEL_CARD.md` for results, success metrics, planning and the production discussion.