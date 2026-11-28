# This file runs BEFORE the tests. It builds a small fake model and feature config,
# so the API tests do not need S3, DagsHub or the real trained model.
import os
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
NO_SERVICE_VALUES = ["No", "No internet service", "No phone service"]

numerical_columns = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "services_count"]
categorical_columns = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]

# Make 400 fake customers with the same columns the real model is trained on
rng = np.random.default_rng(42)
rows = 400
yes_no = ["Yes", "No"]
three = ["Yes", "No", "No internet service"]
fake = pd.DataFrame({
    "gender": rng.choice(["Male", "Female"], rows),
    "SeniorCitizen": rng.integers(0, 2, rows),
    "Partner": rng.choice(yes_no, rows),
    "Dependents": rng.choice(yes_no, rows),
    "tenure": rng.integers(0, 73, rows),
    "PhoneService": rng.choice(yes_no, rows),
    "MultipleLines": rng.choice(["Yes", "No", "No phone service"], rows),
    "InternetService": rng.choice(["DSL", "Fiber optic", "No"], rows),
    "OnlineSecurity": rng.choice(three, rows),
    "OnlineBackup": rng.choice(three, rows),
    "DeviceProtection": rng.choice(three, rows),
    "TechSupport": rng.choice(three, rows),
    "StreamingTV": rng.choice(three, rows),
    "StreamingMovies": rng.choice(three, rows),
    "Contract": rng.choice(["Month-to-month", "One year", "Two year"], rows),
    "PaperlessBilling": rng.choice(yes_no, rows),
    "PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], rows),
    "MonthlyCharges": rng.uniform(18, 119, rows),
    "TotalCharges": rng.uniform(0, 8000, rows),
})
fake["services_count"] = (~fake[SERVICE_COLUMNS].isin(NO_SERVICE_VALUES)).sum(axis=1)
fake_labels = rng.integers(0, 2, rows)

# Same preprocessing as train_model.py
preprocessor = ColumnTransformer(transformers=[
    ("Numerical Pipeline", Pipeline(steps=[
        ("Imputer", SimpleImputer(strategy="median")),
        ("Scaler", StandardScaler()),
    ]), numerical_columns),
    ("Categorical Pipeline", Pipeline(steps=[
        ("Imputer", SimpleImputer(strategy="most_frequent")),
        ("Encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
    ]), categorical_columns),
])
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])
pipeline.fit(fake, fake_labels)

# Save the fake model and feature config into a temporary folder
test_dir = Path(tempfile.mkdtemp())
model_file = test_dir / "best_model.joblib"
config_file = test_dir / "feature_config.json"
joblib.dump(pipeline, model_file)
with open(config_file, "w") as file:
    json.dump({"service_columns": SERVICE_COLUMNS, "no_service_values": NO_SERVICE_VALUES}, file)

# api/main.py reads these paths when it is imported, so set them first
os.environ["MODEL_PATH"] = str(model_file)
os.environ["CONFIG_PATH"] = str(config_file)
