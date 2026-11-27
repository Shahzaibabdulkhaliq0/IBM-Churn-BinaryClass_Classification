import json
import os

import pandas as pd
import yaml

from src.logger.logger import logger

with open("params.yaml") as f:
    params = yaml.safe_load(f)

drift_params = params["drift"]

with open(drift_params["baseline_path"]) as f:
    baseline = json.load(f)

current = pd.read_csv(drift_params["current_path"])

columns = [col for col in baseline if col in current.columns]

report = {}
for col in columns:
    base_mean = baseline[col]
    cur_mean = float(current[col].mean())
    change = abs(cur_mean - base_mean) / (abs(base_mean) + 1e-9)
    report[col] = {
        "baseline_mean": base_mean,
        "current_mean": cur_mean,
        "relative_change": change,
        "drifted": bool(change > drift_params["threshold"]),
    }

drifted_cols = [col for col in report if report[col]["drifted"]]

os.makedirs("reports", exist_ok=True)
with open(drift_params["report_path"], "w") as f:
    json.dump(report, f, indent=2)

# Sirf warning hai, pipeline yahan nahi rukta
logger.warning(f"Drifted columns: {drifted_cols}")