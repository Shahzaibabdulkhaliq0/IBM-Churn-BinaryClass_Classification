import json
import os

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.logger.logger import logger

with open("params.yaml") as f:
    params = yaml.safe_load(f)

data_params = params["data"]
features_params = params["features"]
split_params = params["split"]

target = features_params["target"]
service_cols = features_params["service_cols"]
no_service_values = features_params["no_service_values"]

df = pd.read_csv(data_params["processed_path"])

df[target] = df[target].map({"Yes": 1, "No": 0})

df["services_count"] = (~df[service_cols].isin(no_service_values)).sum(axis=1)

train_df, test_df = train_test_split(
    df,
    test_size=split_params["test_size"],
    random_state=split_params["random_state"],
    stratify=df[target],
)

os.makedirs("data/processed", exist_ok=True)
train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

baseline = train_df.select_dtypes(include="number").mean().to_dict()
os.makedirs("reports", exist_ok=True)
with open("reports/baseline_stats.json", "w") as f:
    json.dump(baseline, f, indent=2)

logger.info(f"Train: {train_df.shape}, Test: {test_df.shape}")