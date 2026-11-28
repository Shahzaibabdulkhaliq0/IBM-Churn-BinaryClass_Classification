import json
import os

import pandas as pd
import yaml

from src.logger.logger import logger

with open("params.yaml") as f:
    params = yaml.safe_load(f)["validate_data"]

df = pd.read_csv("data/interim/cleaned_data.csv", low_memory=False)

# Stop early if a required column is missing, the checks below need these columns
missing = [col for col in params["REQUIRED_COLUMNS"] if col not in df.columns]
assert not missing, f"Missing columns: {missing}"

# Each check is True (passed) or False (failed)
checks = {
    "min_rows": bool(len(df) >= params["MIN_ROWS"]),
    "tenure_in_range": bool(df["tenure"].between(params["MIN_TENURE"], params["MAX_TENURE"]).all()),
    "monthly_charges_not_negative": bool(df["MonthlyCharges"].min() >= params["MIN_MONTHLY_CHARGES"]),
    "churn_allowed": bool(df["Churn"].isin(params["ALLOWED_CHURN_VALUES"]).all()),
    "null_share_ok": bool(df.isnull().mean().max() <= params["MAX_NULL_SHARE"]),
}
all_passed = all(checks.values())

# Save the report first, so you can read it even when a check fails
report = {"checks": checks, "all_passed": all_passed}
os.makedirs("reports", exist_ok=True)
with open("reports/data_validation.json", "w") as f:
    json.dump(report, f, indent=4)

print(report)
logger.info(f"Data validation: {report}")

# assert stops the pipeline (dvc repro, retrain.yml) when any check fails
assert all_passed, f"Data validation FAILED: {checks}"
print("Data validation passed")
