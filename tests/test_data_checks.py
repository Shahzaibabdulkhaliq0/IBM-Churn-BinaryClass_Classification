# Tests for the data checks.
# validate_data.py must STOP on bad data. check_drift.py only WARNS.
# Each test copies params and src into a temp folder, makes a small fake dataset, and runs the script.
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
REQUIRED = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]


def make_data(rows=1200):
    # Same columns the scripts expect. These values pass every check.
    data = {col: ["No"] * rows for col in REQUIRED}
    data["SeniorCitizen"] = [0] * rows
    data["tenure"] = [i % 72 for i in range(rows)]
    data["MonthlyCharges"] = [50.0 + (i % 50) for i in range(rows)]
    data["TotalCharges"] = [float(i % 3000) for i in range(rows)]
    data["Churn"] = ["Yes" if i % 4 == 0 else "No" for i in range(rows)]
    return pd.DataFrame(data)


def make_project(tmp_path, frame):
    # Copy params and src into a temp folder that acts as the project root
    shutil.copy(REPO / "params.yaml", tmp_path / "params.yaml")
    shutil.copytree(REPO / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / "data" / "interim").mkdir(parents=True)
    (tmp_path / "reports").mkdir()
    frame.to_csv(tmp_path / "data" / "interim" / "cleaned_data.csv", index=False)
    return tmp_path


def write_baseline(project, frame):
    # Same kind of baseline that build_features.py saves
    baseline = {
        "monthly_charges_mean": float(frame["MonthlyCharges"].mean()),
        "churn_share": float((frame["Churn"] == "Yes").mean()),
    }
    (project / "reports" / "baseline_stats.json").write_text(json.dumps(baseline))


def run_script(project, module):
    return subprocess.run(
        [sys.executable, "-m", module],
        capture_output=True, text=True, cwd=project,
    )


def read_report(project, name):
    return json.loads((project / "reports" / name).read_text())


# ---------- validate_data.py ----------

def test_validation_passes_on_good_data(tmp_path):
    project = make_project(tmp_path, make_data())
    result = run_script(project, "src.data.validate_data")
    assert result.returncode == 0, result.stderr
    assert read_report(project, "data_validation.json")["all_passed"] is True


def test_validation_fails_on_negative_charges(tmp_path):
    frame = make_data()
    frame.loc[0, "MonthlyCharges"] = -5
    project = make_project(tmp_path, frame)
    result = run_script(project, "src.data.validate_data")
    assert result.returncode != 0
    assert read_report(project, "data_validation.json")["checks"]["monthly_charges_not_negative"] is False


def test_validation_fails_on_unknown_churn_value(tmp_path):
    frame = make_data()
    frame.loc[0, "Churn"] = "Maybe"
    project = make_project(tmp_path, frame)
    result = run_script(project, "src.data.validate_data")
    assert result.returncode != 0
    assert read_report(project, "data_validation.json")["checks"]["churn_allowed"] is False


def test_validation_fails_on_missing_column(tmp_path):
    project = make_project(tmp_path, make_data().drop(columns=["tenure"]))
    result = run_script(project, "src.data.validate_data")
    assert result.returncode != 0
    assert "Missing columns" in result.stderr


def test_validation_fails_on_too_few_rows(tmp_path):
    project = make_project(tmp_path, make_data(rows=100))
    result = run_script(project, "src.data.validate_data")
    assert result.returncode != 0
    assert read_report(project, "data_validation.json")["checks"]["min_rows"] is False


# ---------- check_drift.py ----------

def test_drift_not_detected_on_same_data(tmp_path):
    frame = make_data()
    project = make_project(tmp_path, frame)
    write_baseline(project, frame)
    result = run_script(project, "src.data.check_drift")
    assert result.returncode == 0, result.stderr
    assert read_report(project, "drift_report.json")["drift_detected"] is False


def test_drift_detected_but_pipeline_keeps_running(tmp_path):
    base = make_data()
    project = make_project(tmp_path, base)
    write_baseline(project, base)
    # New data with monthly charges doubled: big drift
    shifted = make_data()
    shifted["MonthlyCharges"] = shifted["MonthlyCharges"] * 2
    shifted.to_csv(project / "data" / "interim" / "cleaned_data.csv", index=False)
    result = run_script(project, "src.data.check_drift")
    # Warning only: the script must still exit with 0
    assert result.returncode == 0, result.stderr
    assert read_report(project, "drift_report.json")["drift_detected"] is True
