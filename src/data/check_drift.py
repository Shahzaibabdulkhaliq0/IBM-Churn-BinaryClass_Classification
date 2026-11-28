import json
import os

import pandas as pd
import yaml

from src.logger.logger import logger

with open("params.yaml") as f:
    params = yaml.safe_load(f)["check_drift"]

# Baseline = numbers saved from the training split (build_features.py)
with open("reports/baseline_stats.json") as f:
    baseline = json.load(f)

# New data = the cleaned data from make_dataset.py
df = pd.read_csv("data/interim/cleaned_data.csv", low_memory=False)

# Check 1: change in average monthly charges, as a fraction of the baseline average
new_charges_mean = float(df["MonthlyCharges"].mean())
charges_change = abs(new_charges_mean - baseline["monthly_charges_mean"]) / baseline["monthly_charges_mean"]

# Check 2: change in the share of churned customers (Yes)
new_churn_share = float((df["Churn"] == "Yes").mean())
churn_share_change = abs(new_churn_share - baseline["churn_share"])

# Drift is a WARNING, not an error. The pipeline keeps running so you can read the report first.
drift_detected = bool(
    charges_change > params["MAX_MONTHLY_CHARGES_MEAN_CHANGE"]
    or churn_share_change > params["MAX_CHURN_SHARE_CHANGE"]
)

report = {
    "new_monthly_charges_mean": new_charges_mean,
    "baseline_monthly_charges_mean": baseline["monthly_charges_mean"],
    "monthly_charges_mean_change": float(charges_change),
    "new_churn_share": new_churn_share,
    "baseline_churn_share": baseline["churn_share"],
    "churn_share_change": float(churn_share_change),
    "drift_detected": drift_detected,
}
os.makedirs("reports", exist_ok=True)
with open("reports/drift_report.json", "w") as f:
    json.dump(report, f, indent=4)

print(report)
logger.info(
    f"Drift check: drift_detected={drift_detected} "
    f"charges_change={charges_change:.3f} churn_share_change={churn_share_change:.3f}"
)
