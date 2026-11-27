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
from sklearn.model_selection import cross_validate
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
train_params = params["train"]
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

train = pd.read_csv("data/processed/train.csv")
X = train.drop(columns=[target])
y = train[target]

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

models = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        n_estimators=train_params["rf_n_estimators"],
        class_weight="balanced",
        random_state=train_params["random_state"],
    ),
}

pipelines = {}
results = {}

for name, model in models.items():
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])
    pipe = Pipeline([("prep", preprocessor), ("clf", model)])

    scores = cross_validate(
        pipe, X, y,
        cv=train_params["cv_folds"],
        scoring=train_params["scoring"],
    )
    results[name] = {
        "cv_mean": float(scores["test_score"].mean()),
        "cv_std": float(scores["test_score"].std()),
    }
    pipelines[name] = pipe
    logger.info(f"{name}: CV {train_params['scoring']} = {results[name]['cv_mean']:.4f}")

best_name = pd.Series({name: results[name]["cv_mean"] for name in results}).idxmax()
best_pipe = pipelines[best_name]
best_pipe.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(best_pipe, "models/base_model.pkl")

os.makedirs("reports", exist_ok=True)
with open("reports/train_metrics.json", "w") as f:
    json.dump({"best_model": best_name, "results": results}, f, indent=2)

mlflow.start_run(run_name="train_model")
mlflow.log_param("cv_folds", train_params["cv_folds"])
mlflow.log_param("selected_model", best_name)
for name in results:
    mlflow.log_metric(f"{name}_cv_f1", results[name]["cv_mean"])
mlflow.log_artifact("reports/train_metrics.json")
mlflow.end_run()

logger.info(f"Best base model: {best_name}")