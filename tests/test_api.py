# API tests: run with "pytest" from the project folder
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

# One valid customer, same format the frontend sends
valid_customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 1026.0,
}


# The server is up and reports the model file
def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# The home page is the HTML frontend
def test_home_page_is_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "Churn" in response.text


# A valid customer gets Yes or No and a probability between 0 and 1
def test_predict_returns_valid_answer():
    response = client.post("/predict", json=valid_customer)
    assert response.status_code == 200
    body = response.json()
    assert body["churn"] in {"Yes", "No"}
    assert 0.0 <= body["churn_probability"] <= 1.0


# Negative tenure is rejected by the input rules
def test_negative_tenure_rejected():
    response = client.post("/predict", json=dict(valid_customer, tenure=-1))
    assert response.status_code == 422


# Negative charges are rejected
def test_negative_charges_rejected():
    response = client.post("/predict", json=dict(valid_customer, MonthlyCharges=-5))
    assert response.status_code == 422


# SeniorCitizen must be 0 or 1
def test_senior_citizen_must_be_0_or_1():
    response = client.post("/predict", json=dict(valid_customer, SeniorCitizen=5))
    assert response.status_code == 422


# A missing column is rejected
def test_missing_column_rejected():
    bad_customer = {key: value for key, value in valid_customer.items() if key != "Contract"}
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422


# Metrics endpoint: Prometheus can read it and it contains our custom counter
def test_metrics_endpoint():
    client.post("/predict", json=valid_customer)
    response = client.get("/metrics/")
    assert response.status_code == 200
    assert "predictions_total" in response.text
