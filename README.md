# IBM Telco Customer Churn: End-to-End MLOps Project

[![CI Pipeline](https://github.com/Shahzaibabdulkhaliq0/IBM-Churn-BinaryClass_Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/Shahzaibabdulkhaliq0/IBM-Churn-BinaryClass_Classification/actions/workflows/ci.yml)

A binary classification MLOps project that predicts if a telecom customer will **churn** (`Yes` / `No`) from the IBM Telco dataset (`scikit-learn/churn-prediction` on HuggingFace, 7,043 customers, 21 columns).

The project covers data versioning with DVC and S3, a 7-stage DVC pipeline, experiment tracking and a Model Registry with a quality gate (MLflow on DagsHub), a FastAPI REST API with an HTML form, Docker, CI/CD with GitHub Actions (tests, image build, push to AWS ECR), a manual retraining workflow, and ready-to-use Kubernetes (EKS) and Prometheus files.

The structure follows my NYC Airbnb project: [Airbnb-Room-Type-MultiClass_Classification](https://github.com/Shahzaibabdulkhaliq0/Airbnb-Room-Type-MultiClass_Classification).

---

## Architecture

```text
 HuggingFace ──fetch_dataset.py──► AWS S3 (raw CSV, DVC remote)
                                      │ make_dataset.py
                                      ▼
  DVC pipeline: make_dataset → validate_data → build_features →
                train_model → tune_model → evaluate_model → check_drift
                                      │ runs + registry
                                      ▼
  MLflow on DagsHub:  churn_model  Staging ──quality gate──► Production
                                      │ export_production_model.py
                                      ▼
  GitHub Actions: test → build image → smoke test → push to ECR  (→ EKS, optional)
```

## Project Structure

```text
.
├── .github/workflows/
│   ├── ci.yml                   # Tests, quality gate, image build, smoke test, ECR push
│   └── retrain.yml              # Manual retraining workflow
├── api/
│   ├── main.py                  # FastAPI: /, /health, /predict, /metrics
│   └── index.html               # Form frontend
├── src/
│   ├── data/                    # fetch_dataset, make_dataset, validate_data, check_drift
│   ├── features/                # build_features (target, services_count, split, baseline)
│   ├── model/                   # train_model, tune_model, evaluate_model
│   └── logger/                  # project logger
├── scripts/
│   ├── promote_model.py         # Staging -> Production (quality gate + better than Production)
│   ├── promotion_rules.py       # the decision rules
│   └── export_production_model.py
├── tests/                       # API, data checks, logger, promotion rules, model quality
├── dvc.yaml  params.yaml        # pipeline stages and ALL settings
├── Dockerfile  deployment.yaml  servicemonitor.yaml
└── requirements*.txt
```

## ML Pipeline (DVC)

| Stage | What it does | Output |
|---|---|---|
| `make_dataset` | Downloads the CSV from S3, drops duplicates and `customerID`, fixes `TotalCharges` (text, blank for new customers) | `data/interim/cleaned_data.csv` |
| `validate_data` | Checks columns, row count, tenure and charges ranges, allowed `Churn` values, null share. **Stops** the pipeline on error | `reports/data_validation.json` |
| `build_features` | Target `Yes/No` to `1/0`, new feature `services_count`, stratified train/test split, drift baseline from the train split only | `data/processed/`, `models/feature_config.json` |
| `train_model` | Compares Logistic Regression, Decision Tree, Random Forest, Gradient Boosting with cross-validation (F1 of the churn class) | MLflow run, untrained winner pipeline |
| `tune_model` | `GridSearchCV` on the **winner only**, with its own grid from `params.yaml` | `models/best_model.joblib` |
| `evaluate_model` | Test metrics, confusion matrix, registers `churn_model` in **Staging** | `reports/metrics.json` |
| `check_drift` | Monthly charges mean and churn share against the baseline. **Warning only** | `reports/drift_report.json` |

Cleaning (`make_dataset`) and feature engineering (`build_features`) are separate on purpose.

**Class imbalance:** about 26% of customers churn. The split is stratified, models use `class_weight="balanced"` (Gradient Boosting has no such option), and the main metric is **F1 of the churn class**, not accuracy.

**Quality gate** (`params.yaml`): test accuracy >= 0.70 and test F1 >= 0.55. A failed gate never promotes a model.

## Results

Local run on the real dataset (Logistic Regression won the comparison):

| Metric | Value |
|---|---|
| Test accuracy | 0.738 |
| Test F1 (churn) | 0.613 |
| Test recall (churn) | 0.782 |
| Test ROC-AUC | 0.840 |

## How to Run Locally

Prerequisites: Python 3.13 or newer, Git, AWS credentials for the S3 bucket, a DagsHub account.

```bash
git clone https://github.com/Shahzaibabdulkhaliq0/IBM-Churn-BinaryClass_Classification.git
cd IBM-Churn-BinaryClass_Classification
pip install -r requirements-dev.txt -r requirements.txt
```

Create a `.env` file (it is git-ignored):

```text
DAGSHUB_USERNAME=your_dagshub_username
DAGSHUB_TOKEN=your_dagshub_token
```

**One-time: put the dataset in S3** (needs `pip install -r requirements-fetch.txt` and AWS credentials):

```bash
python -m src.data.fetch_dataset
```

**Run the pipeline:**

```bash
dvc repro                              # all 7 stages
python scripts/promote_model.py        # Staging -> Production, only if the gate passes
python scripts/export_production_model.py
dvc push                               # save data and outputs to S3
```

**Tests and API:**

```bash
pytest -q                              # all tests (the quality-gate test needs the DagsHub token)
pytest -q --ignore=tests/test_model_quality.py
uvicorn api.main:app --reload --port 8000     # then open http://localhost:8000
```

**Docker:**

```bash
docker build -t churn-api .
docker run -d -p 8000:8000 --name churn-api churn-api
```

## CI/CD (GitHub Actions)

- `ci.yml`: on every push and pull request to `main` it runs the tests and the model quality gate (the real Production model must pass the thresholds). On a push to `main` it then builds the Docker image, smoke-tests `/health`, and pushes the image to AWS ECR (`churn-ecr-api`). The EKS deploy job is commented out until you connect a cluster.
- `retrain.yml`: manual. Reruns `dvc repro --force`, promotes the new model only if it passes the gate and beats Production, pushes outputs to S3, commits `dvc.lock` and the metrics, then starts `ci.yml`.

**GitHub repository secrets:**

| Secret | Used for |
|---|---|
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | DVC (S3) and ECR |
| `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN` | MLflow on DagsHub |

The AWS region (`eu-north-1`) and the ECR repository name are written in the workflow `env:` block, so no secret is needed for them.

## Configuration

Every setting and threshold lives in `params.yaml`. Each script reads only its own section (same names as the scripts), and the MLflow settings are in the `mlflow` section.

## Known Limitations

- Training runs locally (`dvc repro`) or in `retrain.yml`; CI only uses the Production model.
- The EKS deploy is optional and commented out; Prometheus scrapes `/metrics/` once EKS and `servicemonitor.yaml` are applied.
