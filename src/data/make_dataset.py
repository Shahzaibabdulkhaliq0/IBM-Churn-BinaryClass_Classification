import os

import pandas as pd
import yaml

from src.logger.logger import logger

with open("params.yaml") as f:
    params = yaml.safe_load(f)

data_params = params["data"]

raw = pd.read_csv(data_params["raw_path"])

df = raw.drop(columns=data_params["drop_cols"], errors="ignore").copy()

blank_count = (df["TotalCharges"].astype(str).str.strip() == "").sum()
logger.info(f"Blank TotalCharges rows: {blank_count}")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(), errors="coerce"
).fillna(0.0)

os.makedirs(os.path.dirname(data_params["processed_path"]), exist_ok=True)
df.to_csv(data_params["processed_path"], index=False)

logger.info(f"Cleaned data saved: {df.shape}")