import json
import os

import joblib
import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger.logger import logger

load_dotenv(".env", override=True)

dagshub_token = os.environ.get("DAGSHUB_TOKEN")
assert dagshub_token, "DAGSHUB_TOKEN environment variable is not set"
os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_USERNAME"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

with open("params.yaml") as f:
    params = yaml.safe_load(f)

features_params = params["features"]
tune_params = params["tune"]
mlflow_params = params["mlflow"]

target = features_params["target"]

mlflow.set_tracking_uri(mlflow_params["TRACKING_URI"])
try:
    mlflow.set_experiment(mlflow_params["EXPERIMENT_NAME"])
except mlflow.exceptions.MlflowException as exc:
    raise SystemExit(
        "DagsHub rejected the MLflow credentials. Check DAGSHUB_USERNAME and "
        "DAGSHUB_TOKEN in .env."
    ) from exc

# train_model.py ne jo best model chuna, wo yahan se aata hai
with open("reports/train_metrics.json") as f:
    best_model_name = json.load(f)["best_model"]

logger.info(f"Best model from train_model: {best_model_name}")

train = pd.read_csv("data/processed/train.csv")
X = train.drop(columns=[target])
y = train[target]

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

all_models = {
    "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "random_forest": RandomForestClassifier(
        class_weight="balanced",
        random_state=tune_params["random_state"],
    ),
}
model = all_models[best_model_name]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])
pipe = Pipeline([("prep", preprocessor), ("clf", model)])

grid = {f"clf__{key}": value for key, value in tune_params[best_model_name].items()}

search = GridSearchCV(
    pipe,
    grid,
    cv=tune_params["cv_folds"],
    scoring=tune_params["scoring"],
    n_jobs=-1,
)
search.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(search.best_estimator_, "models/tuned_model.pkl")

results = {
    "model": best_model_name,
    "best_params": search.best_params_,
    "best_cv_score": float(search.best_score_),
}
os.makedirs("reports", exist_ok=True)
with open("reports/tune_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

mlflow.start_run(run_name="tune_model")
mlflow.log_param("model", best_model_name)
for key, value in search.best_params_.items():
    mlflow.log_param(key, str(value))
mlflow.log_metric("best_cv_score", float(search.best_score_))
mlflow.log_artifact("reports/tune_results.json")
mlflow.end_run()

logger.info(f"Tuned {best_model_name}, best CV {tune_params['scoring']} = {search.best_score_:.4f}")