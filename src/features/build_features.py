import json
import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.logger.logger import logger

with open("params.yaml") as f:
    params = yaml.safe_load(f)["build_features"]

target = params["TARGET"]
service_cols = params["SERVICE_COLUMNS"]
no_service_values = params["NO_SERVICE_VALUES"]

df = pd.read_csv("data/interim/cleaned_data.csv", low_memory=False)

# Target: Yes -> 1, No -> 0
df[target] = df[target].map({"Yes": 1, "No": 0})

# New feature: how many services the customer uses (phone, internet, security, streaming...)
df["services_count"] = (~df[service_cols].isin(no_service_values)).sum(axis=1)

# Stratified split: churn is the smaller class, both splits keep the same churn share
train_df, test_df = train_test_split(
    df,
    test_size=params["TEST_SIZE"],
    random_state=params["RANDOM_STATE"],
    stratify=df[target],
)

os.makedirs("data/processed", exist_ok=True)
train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

# Drift baseline comes from the TRAIN split only (no leakage from test data)
baseline = {
    "monthly_charges_mean": float(train_df["MonthlyCharges"].mean()),
    "churn_share": float(train_df[target].mean()),
}
os.makedirs("reports", exist_ok=True)
with open("reports/baseline_stats.json", "w") as f:
    json.dump(baseline, f, indent=4)

# The API needs the same feature rule to build services_count for a new customer
os.makedirs("models", exist_ok=True)
with open("models/feature_config.json", "w") as f:
    json.dump({"service_columns": service_cols, "no_service_values": no_service_values}, f, indent=4)

logger.info(f"Train: {train_df.shape}, Test: {test_df.shape}, baseline={baseline}")
