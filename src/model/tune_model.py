# Silence the "unknown categories" warning in this script AND in the sklearn worker processes.
# This must be set BEFORE sklearn is imported.
import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import json
import sys

# MLflow prints emoji run links; Windows consoles (cp1252) cannot encode them.
sys.stdout.reconfigure(errors="replace")

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.model_selection import GridSearchCV

from src.logger.logger import logger

with open("params.yaml") as f:
    params = yaml.safe_load(f)
tune_params = params["tune_model"]
mlflow_params = params["mlflow"]

# MLflow + DagsHub setup (same credentials as train_model.py)
load_dotenv(".env", override=True)
dagshub_token = os.environ.get("DAGSHUB_TOKEN")
assert dagshub_token, "DAGSHUB_TOKEN environment variable is not set"
os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_USERNAME"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
mlflow.set_tracking_uri(mlflow_params["TRACKING_URI"])
mlflow.set_experiment(mlflow_params["EXPERIMENT_NAME"])

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Load the training data only
train = pd.read_csv("data/processed/train.csv")
X_train = train.drop(columns=[tune_params["TARGET"]])
y_train = train[tune_params["TARGET"]]

# Which model won in train_model.py. Only this model is tuned.
with open("models/best_model_name.txt") as f:
    best_model_name = f.read()
print("Tuning:", best_model_name)
logger.info(f"Tuning only the winner of train_model: {best_model_name}")

# Load the untrained pipeline of the winner
pipeline = joblib.load("models/untrained_pipeline.joblib")

# Search space of the winner (params.yaml: PARAM_GRIDS)
param_grid = tune_params["PARAM_GRIDS"][best_model_name]

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=tune_params["CV_FOLDS"],
    scoring=tune_params["SCORING"],
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

# The best pipeline, already trained on the full training data
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_cv_f1 = float(grid_search.best_score_)

print("Best Parameters:", best_params)
print("Best CV F1:", best_cv_f1)
logger.info(f"Tuning done: best_cv_f1={best_cv_f1}")

# Save local files FIRST, so they exist even if MLflow fails
joblib.dump(best_model, "models/best_model.joblib")
with open("reports/best_params.json", "w") as f:
    json.dump({"model": best_model_name, "best_params": best_params, "best_cv_f1": best_cv_f1}, f, indent=4, default=str)

# Log to MLflow and save the run id for evaluate_model.py
with mlflow.start_run(run_name="tune_model") as run:
    mlflow.log_param("selected_model", best_model_name)
    mlflow.log_params({key: str(value) for key, value in best_params.items()})
    mlflow.log_metric("best_cv_f1", best_cv_f1)
    mlflow.log_artifact("reports/best_params.json")
    model_info = mlflow.sklearn.log_model(
        best_model,
        name="model",
        skops_trusted_types=["sklearn.tree._tree.Tree", "numpy.dtype"],
    )
    run_id = run.info.run_id

with open("reports/run_info.json", "w") as f:
    json.dump({"run_id": run_id, "model_path": "model", "model_uri": model_info.model_uri}, f, indent=4)

print("Final pipeline saved to: models/best_model.joblib")
print("MLflow run_id:", run_id)
logger.info(f"Tuned model logged to MLflow run {run_id}")
