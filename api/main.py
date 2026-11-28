# FastAPI app: serves the final model from models/best_model.joblib

# This line hides only UserWarning messages (for example "unknown categories") from the logs.
# It must run BEFORE sklearn is imported.
import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

# json: reads the service-column rule saved by build_features.py
import json

# Path builds file paths that work on any computer
from pathlib import Path

# joblib: loads the trained pipeline
import joblib

# pandas: makes the one-row table the model needs
import pandas as pd

# FastAPI: web framework. Field: adds rules to the input (for example, tenure >= 0)
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# prometheus_client: counts predictions and measures how long they take, for Prometheus/Grafana
from prometheus_client import Counter, Histogram, make_asgi_app

from src.logger.logger import logger

# The project folder is one folder above this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Use the env variables if they are set (tests and Docker), otherwise the default files
MODEL_PATH = Path(os.environ.get("MODEL_PATH", PROJECT_ROOT / "models" / "best_model.joblib"))
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", PROJECT_ROOT / "models" / "feature_config.json"))

# Load the model and the feature rule ONCE when the server starts, not on every request
model = joblib.load(MODEL_PATH)
with open(CONFIG_PATH) as file:
    feature_config = json.load(file)

logger.info(f"Model loaded from {MODEL_PATH}, config={feature_config}")

# Metrics: how many predictions per answer (Yes/No), and how long each prediction takes
PREDICTIONS_TOTAL = Counter("predictions_total", "Total predictions", ["churn"])
PREDICT_LATENCY = Histogram("predict_latency_seconds", "Time taken by one /predict request")

app = FastAPI(title="IBM Telco customer churn API")

# Expose the metrics at /metrics (Prometheus scrapes this URL)
app.mount("/metrics", make_asgi_app())


# The input the client must send: the 19 customer columns (no customerID, no Churn)
class Customer(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


# Front end page: returns only api/index.html (not the other files in this folder)
@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).resolve().parent / "index.html")


# Health check: the server is up and tells which model file it loaded.
# Kubernetes and the CI smoke test use this URL.
@app.get("/health")
def health():
    return {"status": "ok", "model_file": MODEL_PATH.name}


# Prediction: build the same services_count feature as training, then predict churn
@app.post("/predict")
def predict(customer: Customer):
    # Measure how long this request takes
    with PREDICT_LATENCY.time():
        row = customer.model_dump()

        # Same rule as build_features.py: count the services the customer really uses
        services = [row[col] for col in feature_config["service_columns"]]
        row["services_count"] = sum(value not in feature_config["no_service_values"] for value in services)

        # The model expects a table, not a dictionary
        customer_df = pd.DataFrame([row])

        probabilities = model.predict_proba(customer_df)[0]
        prediction = int(model.predict(customer_df)[0])

    answer = "Yes" if prediction == 1 else "No"

    # Count this prediction under its answer
    PREDICTIONS_TOTAL.labels(churn=answer).inc()

    logger.info(f"Prediction: churn={answer}")

    return {
        "churn": answer,
        "churn_probability": float(probabilities[list(model.classes_).index(1)]),
    }
