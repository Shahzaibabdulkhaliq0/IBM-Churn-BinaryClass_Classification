import os

import yaml
from datasets import load_dataset

from src.logger.logger import logger

with open("params.yaml") as f:
    params = yaml.safe_load(f)

data_params = params["fetch_dataset"]

os.makedirs(os.path.dirname(data_params["raw_path"]), exist_ok=True)

dataset = load_dataset(data_params["hf_repo"], split="train")
df = dataset.to_pandas()
df.to_csv(data_params["raw_path"], index=False)

logger.info(f"Saved {len(df)} rows to {data_params['raw_path']}")