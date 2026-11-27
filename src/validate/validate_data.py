import json
import os

import pandas as pd
import yaml

from src.logger.logger import logger

with open("params.yaml") as f:
    params = yaml.safe_load(f)

data_params = params["data"]
validate_params = params["validate"]

df = pd.read_csv(data_params["processed_path"])

missing_columns = [col for col in validate_params["expected_columns"] if col not in df.columns]
null_count = int(df.isnull().sum().sum())
row_count = len(df)
target_values = sorted(df[validate_params["target"]].unique().tolist())
target_ok = set(target_values) <= set(validate_params["allowed_target_values"])

passed = (
    len(missing_columns) == 0
    and null_count == 0
    and row_count >= validate_params["min_rows"]
    and target_ok
)

report = {
    "passed": bool(passed),
    "missing_columns": missing_columns,
    "null_count": null_count,
    "row_count": row_count,
    "target_values": target_values,
}

os.makedirs("reports", exist_ok=True)
with open("reports/data_validation.json", "w") as f:
    json.dump(report, f, indent=2)

logger.info(f"Data validation report: {report}")

assert passed, f"Data validation failed. See reports/data_validation.json"