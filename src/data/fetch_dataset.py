# One-time step: download the IBM Telco churn dataset from HuggingFace and upload it to S3.
# make_dataset.py then reads it from S3 (same idea as the Airbnb project).
# Run: python -m src.data.fetch_dataset      (needs: pip install -r requirements-fetch.txt, AWS credentials)

import os

import boto3
import yaml
from datasets import load_dataset

from src.logger.logger import logger

with open("params.yaml") as f:
    params = yaml.safe_load(f)["fetch_dataset"]

os.makedirs(os.path.dirname(params["LOCAL_FILE"]), exist_ok=True)

dataset = load_dataset(params["HF_REPO"], split="train")
df = dataset.to_pandas()
df.to_csv(params["LOCAL_FILE"], index=False)
logger.info(f"Downloaded {len(df)} rows from HuggingFace to {params['LOCAL_FILE']}")

s3 = boto3.client("s3")
s3.upload_file(params["LOCAL_FILE"], params["S3_BUCKET"], params["S3_KEY"])
logger.info(f"Uploaded to s3://{params['S3_BUCKET']}/{params['S3_KEY']}")
