# Automatic promotion: picks the best Staging model and promotes it to Production
# ONLY if it passes the quality gate AND is better than the current Production model.
# If it fails, the current Production model stays. Run: python scripts/promote_model.py

import os
from pathlib import Path
import sys

# The project folder is one folder above this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Make `from src...` work no matter where the script is launched from
sys.path.insert(0, str(PROJECT_ROOT))

# MLflow prints emoji run links; Windows consoles (cp1252) cannot encode them.
sys.stdout.reconfigure(errors="replace")

# yaml: reads params.yaml
import yaml

# dotenv: loads DAGSHUB_USERNAME and DAGSHUB_TOKEN from the .env file
from dotenv import load_dotenv

# mlflow: model registry
import mlflow
from mlflow.tracking import MlflowClient

# Decision rules (tested in tests/test_promotion_rules.py)
from promotion_rules import passes_quality_gate, should_promote

from src.logger.logger import logger

# Same DagsHub login as the training scripts (token comes from .env, never from code)
load_dotenv(PROJECT_ROOT / ".env", override=True)
dagshub_token = os.environ.get("DAGSHUB_TOKEN")
assert dagshub_token, "DAGSHUB_TOKEN environment variable is not set"
os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_USERNAME"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Read the settings from params.yaml
with open(PROJECT_ROOT / "params.yaml") as file:
    params = yaml.safe_load(file)
mlflow.set_tracking_uri(params["mlflow"]["TRACKING_URI"])
model_name = params["evaluate_model"]["REGISTERED_MODEL_NAME"]
gate = params["quality_gate"]

client = MlflowClient()

# All versions of churn_model, split by stage
all_versions = client.search_model_versions(f"name='{model_name}'")
staging_versions = [version for version in all_versions if version.current_stage == "Staging"]
production_versions = [version for version in all_versions if version.current_stage == "Production"]

assert staging_versions, f"No {model_name} version is in Staging. Run evaluate_model first."

# Each version's run holds: best_cv_f1 (selection score, from TRAIN data only),
# test_accuracy and test_f1 (quality gate only, never used to pick the winner)
staging_runs = {version.version: client.get_run(version.run_id) for version in staging_versions}

# Select the best Staging candidate by the TRAIN cross-validation score
staging_cv = {
    number: run.data.metrics.get("best_cv_f1", -1)
    for number, run in staging_runs.items()
}
candidate_version = max(staging_cv, key=staging_cv.get)
candidate_metrics = staging_runs[candidate_version].data.metrics

print("Staging candidates (train CV F1):")
for number in staging_cv:
    print(f"  v{number}: {staging_cv[number]:.4f}")

# Quality gate on the candidate (absolute minimums, from params.yaml)
gate_passed = passes_quality_gate(
    candidate_metrics["test_accuracy"],
    candidate_metrics["test_f1"],
    gate["MIN_TEST_ACCURACY"],
    gate["MIN_TEST_F1"],
)

# Tag the candidate so the result is visible on DagsHub
client.set_model_version_tag(
    name=model_name,
    version=candidate_version,
    key="quality_gate",
    value="passed" if gate_passed else "failed",
)

# Current Production model's train CV score, if there is one
if production_versions:
    production_run = client.get_run(production_versions[0].run_id)
    production_cv = production_run.data.metrics.get("best_cv_f1", -1)
    print(f"Current Production: v{production_versions[0].version} (train CV F1 {production_cv:.4f})")
else:
    production_cv = None
    print("No Production version yet.")

print(
    f"Candidate: v{candidate_version} | train CV {staging_cv[candidate_version]:.4f} | "
    f"test accuracy {candidate_metrics['test_accuracy']:.4f} | "
    f"test F1 {candidate_metrics['test_f1']:.4f} | gate {'PASSED' if gate_passed else 'FAILED'}"
)

# Final decision (rules in scripts/promotion_rules.py)
promote = should_promote(staging_cv[candidate_version], production_cv, gate_passed)
logger.info(f"Promotion decision for v{candidate_version}: {'promote' if promote else 'keep Production'}")

if promote:
    client.transition_model_version_stage(
        name=model_name,
        version=candidate_version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"DECISION: {model_name} v{candidate_version} is now in Production.")
else:
    print("DECISION: Production kept. Candidate failed the gate or is not better.")
