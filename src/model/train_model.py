# Silence the "unknown categories" warning in this script AND in the sklearn worker processes.
# This must be set BEFORE sklearn is imported.
import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import sys

# MLflow prints emoji run links; Windows consoles (cp1252) cannot encode them.
sys.stdout.reconfigure(errors="replace")

import joblib
import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.logger.logger import logger

with open("params.yaml") as f:
    params = yaml.safe_load(f)
train_params = params["train_model"]
mlflow_params = params["mlflow"]

# MLflow + DagsHub setup: username plus an access token used as the password.
# override=True prevents a stale token exported in the shell from winning over .env
load_dotenv(".env", override=True)
dagshub_token = os.environ.get("DAGSHUB_TOKEN")
assert dagshub_token, "DAGSHUB_TOKEN environment variable is not set"
os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_USERNAME"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
mlflow.set_tracking_uri(mlflow_params["TRACKING_URI"])
mlflow.set_experiment(mlflow_params["EXPERIMENT_NAME"])

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Load ONLY training data. test.csv is reserved for the final evaluation.
train = pd.read_csv("data/processed/train.csv")
X_train = train.drop(columns=[train_params["TARGET"]])
y_train = train[train_params["TARGET"]]

numerical_columns = X_train.select_dtypes(include="number").columns.tolist()
categorical_columns = X_train.select_dtypes(exclude="number").columns.tolist()

# Numeric: fill missing with median, then scale
numerical_pipeline = Pipeline(steps=[
    ("Imputer", SimpleImputer(strategy="median")),
    ("Scaler", StandardScaler()),
])

# Categorical: fill missing with most frequent, then one-hot encode
categorical_pipeline = Pipeline(steps=[
    ("Imputer", SimpleImputer(strategy="most_frequent")),
    ("Encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("Numerical Pipeline", numerical_pipeline, numerical_columns),
    ("Categorical Pipeline", categorical_pipeline, categorical_columns),
])

# class_weight="balanced" gives the rare class (churn) more attention.
# GradientBoosting has no class_weight option, so it is left as it is.
RANDOM_STATE = train_params["RANDOM_STATE"]
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

# Compare every model with the SAME preprocessor, using CV on TRAINING data only
results = []
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", clone(preprocessor)),
        ("classifier", clone(model)),
    ])

    # Accuracy alone can mislead on imbalanced classes, so F1 (churn class) is the main score
    accuracy = cross_val_score(pipeline, X_train, y_train, cv=train_params["CV_FOLDS"], scoring="accuracy", n_jobs=1)
    f1 = cross_val_score(pipeline, X_train, y_train, cv=train_params["CV_FOLDS"], scoring="f1", n_jobs=1)

    results.append({
        "Model": name,
        "CV Accuracy Mean": accuracy.mean(),
        "CV Accuracy Std": accuracy.std(),
        "CV F1 Mean": f1.mean(),
        "CV F1 Std": f1.std(),
    })

# Highest F1 first
comparison_df = pd.DataFrame(results).sort_values(by="CV F1 Mean", ascending=False)
print(comparison_df.to_string(index=False))

# Automatic selection: the first row is the best model
best_model_name = comparison_df["Model"].iloc[0]
print("Best model:", best_model_name)
logger.info(f"Best model by train CV F1: {best_model_name}")

comparison_df.to_csv("reports/model_comparison.csv", index=False)

# Log the comparison to MLflow
with mlflow.start_run(run_name="model_comparison"):
    mlflow.log_param("CV_FOLDS", train_params["CV_FOLDS"])
    mlflow.log_param("selected_model", best_model_name)
    for row in results:
        key = row["Model"].lower().replace(" ", "_")
        mlflow.log_metric(f"{key}_cv_f1", float(row["CV F1 Mean"]))
        mlflow.log_metric(f"{key}_cv_accuracy", float(row["CV Accuracy Mean"]))
    mlflow.log_artifact("reports/model_comparison.csv")

# Save the winner's name, and an UNTRAINED copy of its full pipeline for tune_model.py
with open("models/best_model_name.txt", "w") as f:
    f.write(best_model_name)

untrained_pipeline = Pipeline(steps=[
    ("preprocessor", clone(preprocessor)),
    ("classifier", clone(models[best_model_name])),
])
joblib.dump(untrained_pipeline, "models/untrained_pipeline.joblib")

print("Saved the best model name and untrained pipeline in: models/")
logger.info(f"Saved untrained pipeline for {best_model_name}")
