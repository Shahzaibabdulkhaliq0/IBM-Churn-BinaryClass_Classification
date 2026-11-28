# Silence the "unknown categories" warning (unseen categories in the test data)
import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import json
import sys

# MLflow prints emoji run links; Windows consoles (cp1252) cannot encode them.
sys.stdout.reconfigure(errors="replace")

import joblib
import matplotlib
# "Agg" saves plots as files without opening windows
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.logger.logger import logger

with open("params.yaml") as f:
    params = yaml.safe_load(f)
eval_params = params["evaluate_model"]
mlflow_params = params["mlflow"]

# MLflow + DagsHub setup (same credentials as train_model.py)
load_dotenv(".env", override=True)
dagshub_token = os.environ.get("DAGSHUB_TOKEN")
assert dagshub_token, "DAGSHUB_TOKEN environment variable is not set"
os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_USERNAME"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
mlflow.set_tracking_uri(mlflow_params["TRACKING_URI"])
mlflow.set_experiment(mlflow_params["EXPERIMENT_NAME"])

os.makedirs("reports/figures", exist_ok=True)

# Load the TEST data. It is used only here, never for model selection.
test = pd.read_csv("data/processed/test.csv")
X_test = test.drop(columns=[eval_params["TARGET"]])
y_test = test[eval_params["TARGET"]]

# Load the final tuned pipeline
model = joblib.load("models/best_model.joblib")

# Predict on the untouched test set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "test_accuracy": float(accuracy_score(y_test, y_pred)),
    "test_precision": float(precision_score(y_test, y_pred)),
    "test_recall": float(recall_score(y_test, y_pred)),
    "test_f1": float(f1_score(y_test, y_pred)),
    "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
}

print(metrics)
print(classification_report(y_test, y_pred))
logger.info(f"Test metrics: {metrics}")

# First 10 real customers next to the predictions
comparison = pd.DataFrame({"Actual": y_test.values[:10], "Predicted": y_pred[:10]})
print(comparison)

with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Confusion matrix (counts)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("reports/figures/confusion_matrix.png", bbox_inches="tight")
plt.close()

# Log metrics and the plot to the SAME MLflow run that tuned the model
with open("reports/run_info.json") as f:
    run_info = json.load(f)

with mlflow.start_run(run_id=run_info["run_id"]):
    mlflow.log_metrics(metrics)
    mlflow.log_artifact("reports/figures/confusion_matrix.png")

# Model registry: register as churn_model and move it to Staging
model_name = eval_params["REGISTERED_MODEL_NAME"]
version = mlflow.register_model(run_info["model_uri"], model_name)
MlflowClient().transition_model_version_stage(
    name=model_name,
    version=version.version,
    stage="Staging",
)

print(f"{model_name} v{version.version} registered in Staging")
logger.info(f"{model_name} v{version.version} registered in Staging")
