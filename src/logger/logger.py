import logging
import os

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("churn")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("logs/pipeline.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)