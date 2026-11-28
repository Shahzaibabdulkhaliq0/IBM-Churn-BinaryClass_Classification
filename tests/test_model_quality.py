# Model quality gate: the Production model must meet the thresholds in params.yaml
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import pytest
import yaml
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score

# The project folder is one folder above this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load .env so the token is found on your machine too
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Read the settings from params.yaml
with open(PROJECT_ROOT / "params.yaml") as file:
    params = yaml.safe_load(file)
gate = params["quality_gate"]
model_name = params["evaluate_model"]["REGISTERED_MODEL_NAME"]
target = params["evaluate_model"]["TARGET"]


# Skip locally when there is no token. In CI (CI=true is set by GitHub) never skip:
# a skipped quality gate must not look like a passed one.
@pytest.mark.skipif(
    not os.getenv("DAGSHUB_TOKEN") and not os.getenv("CI"),
    reason="DAGSHUB_TOKEN not set (local run)",
)
def test_production_model_meets_thresholds():
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_USERNAME"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]
    mlflow.set_tracking_uri(params["mlflow"]["TRACKING_URI"])

    # Test data made by build_features.py (run dvc pull first in CI)
    test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")
    X_test = test.drop(columns=[target])
    y_test = test[target]

    model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Production accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    assert accuracy >= gate["MIN_TEST_ACCURACY"], f"Accuracy {accuracy:.4f} is below {gate['MIN_TEST_ACCURACY']}"
    assert f1 >= gate["MIN_TEST_F1"], f"F1 {f1:.4f} is below {gate['MIN_TEST_F1']}"
