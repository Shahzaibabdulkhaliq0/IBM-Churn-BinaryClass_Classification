import os

import boto3
import pandas as pd
import yaml

from src.logger.logger import logger

with open("params.yaml") as f:
    params = yaml.safe_load(f)["make_dataset"]

# Download the raw dataset from the S3 bucket
raw_path = "data/raw/Telco-Customer-Churn.csv"
os.makedirs(os.path.dirname(raw_path), exist_ok=True)

s3 = boto3.client("s3")
s3.download_file(params["S3_BUCKET"], params["S3_KEY"], raw_path)
logger.info(f"Downloaded raw data from S3 to {raw_path}")

raw = pd.read_csv(raw_path, low_memory=False)

# Remove duplicates BEFORE dropping customerID: two different customers can look the same
# without the ID, but the same customerID twice is a real duplicate row.
df = raw.drop_duplicates()

df = df.drop(columns=params["DROP_COLUMNS"], errors="ignore")

# TotalCharges is text in the raw file. Blank values are new customers (tenure 0), so they become 0.
blank_count = int((df["TotalCharges"].astype(str).str.strip() == "").sum())
logger.info(f"Blank TotalCharges rows: {blank_count}")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(), errors="coerce"
).fillna(0.0)

# NOTE: target mapping, new features and the train/test split are NOT done here.
# They live in build_features.py, so cleaning stays separate from feature engineering.

clean_path = "data/interim/cleaned_data.csv"
os.makedirs(os.path.dirname(clean_path), exist_ok=True)
df.to_csv(clean_path, index=False)

logger.info(f"Cleaned data saved, shape={df.shape}")
