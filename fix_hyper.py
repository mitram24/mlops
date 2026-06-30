import sys

params_path = r"C:\Users\pedro\Documentos\NOVA IMS\2º semestre- eu\MLOps\projeto\_player_rating_extracted\mlops_player_rating\conf\base\parameters.yml"
nodes_path = r"C:\Users\pedro\Documentos\NOVA IMS\2º semestre- eu\MLOps\projeto\_player_rating_extracted\mlops_player_rating\src\mlops_player_rating\pipelines\model_selection\nodes.py"

with open(params_path, "r", encoding="utf-8") as f:
    params = f.read()

# Append hyperparameter grids
params += """
  candidate_grids:
    ridge:
      alpha: [0.1, 1.0, 10.0, 100.0]
    random_forest:
      n_estimators: [50, 100, 200]
      max_depth: [null, 10, 20]
      min_samples_split: [2, 5]
    hist_gradient_boosting:
      learning_rate: [0.01, 0.05, 0.1]
      max_iter: [100, 200, 500]
      max_leaf_nodes: [31, 63]
  
  tune_n_iter: 5
  tune_cv: 3
"""

with open(params_path, "w", encoding="utf-8") as f:
    f.write(params)

with open(nodes_path, "r", encoding="utf-8") as f:
    nodes = f.read()

# We need to change the cross_val_score to RandomizedSearchCV
import re

nodes = nodes.replace("from sklearn.model_selection import GroupKFold, cross_val_score", "from sklearn.model_selection import GroupKFold, RandomizedSearchCV")
nodes = nodes.replace("from sklearn.model_selection import KFold, cross_val_score", "from sklearn.model_selection import GroupKFold, RandomizedSearchCV")

# Redefine candidate_estimators
new_candidates = """def candidate_estimators() -> dict[str, Any]:
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(),
        "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42),
    }"""
nodes = re.sub(r'def candidate_estimators\(\) -> dict\[str, Any\]:.*?    \}', new_candidates, nodes, flags=re.DOTALL)

# Refactor select_model
old_select = """        cv = GroupKFold(n_splits=5)
        # We negate because scikit-learn returns negative MSE for cross_val_score
        scores = cross_val_score(
            pipeline, x_train, y_train, cv=cv, scoring="neg_mean_squared_error", groups=groups
        )
        rmse = float(np.mean(np.sqrt(-scores)))"""

new_select = """        cv = GroupKFold(n_splits=params.get("tune_cv", 3))
        param_grid = params.get("candidate_grids", {}).get(name)
        
        if param_grid:
            # Map grid parameters to the pipeline step 'regressor'
            grid = {f"regressor__{k}": v for k, v in param_grid.items()}
            search = RandomizedSearchCV(
                pipeline, grid, n_iter=params.get("tune_n_iter", 5),
                cv=cv, scoring="neg_mean_squared_error", n_jobs=-1, random_state=42
            )
            search.fit(x_train, y_train, groups=groups)
            best_pipeline = search.best_estimator_
            rmse = float(np.sqrt(-search.best_score_))
            mlflow.log_params({f"best_{k}": v for k, v in search.best_params_.items()})
        else:
            # Baseline evaluation
            search = RandomizedSearchCV(
                pipeline, {}, n_iter=1,
                cv=cv, scoring="neg_mean_squared_error", n_jobs=-1, random_state=42
            )
            search.fit(x_train, y_train, groups=groups)
            best_pipeline = search.best_estimator_
            rmse = float(np.sqrt(-search.best_score_))"""

nodes = nodes.replace(old_select, new_select)

with open(nodes_path, "w", encoding="utf-8") as f:
    f.write(nodes)

print("Hyperparameters tuned via RandomizedSearchCV successfully.")
