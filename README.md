# prodml

[![CI](https://github.com/Mariam123Hamada/MLOPS_Tasks/actions/workflows/ci.yml/badge.svg)](https://github.com/Mariam123Hamada/MLOPS_Tasks/actions/workflows/ci.yml)

Production-grade MLOps project for predicting **NYC Green Taxi trip duration** using a machine-learning regression model exposed through a **FastAPI** service.

The project demonstrates an end-to-end machine-learning lifecycle:

**Data Versioning → Data Preparation → Feature Engineering → Model Training → Evaluation → Quality Gates → Experiment Tracking → Model Lifecycle → Containerization → CI → Continuous Training**

---

## 1. Project Overview

The goal of this project is to build a reproducible and production-oriented ML system for predicting taxi trip duration.

The system uses historical NYC Green Taxi trip data and performs the following steps:

1. Load and validate the raw dataset.
2. Split and prepare the training and validation datasets.
3. Transform raw trip information into model-ready features.
4. Train a regression model.
5. Evaluate the trained model using MAE, RMSE, and R².
6. Apply an automated model-quality gate.
7. Track experiments and model artifacts with MLflow.
8. Version data and pipeline artifacts with DVC.
9. Package the prediction service with Docker.
10. Run automated checks through GitHub Actions.
11. Run Continuous Training when scheduled, manually triggered, or triggered by an external event.
12. Automatically promote an accepted candidate to **Staging**, while keeping **Production promotion human-controlled**.

---

## 2. Problem Definition

### Input

The model uses trip-level information such as:

- Pickup location
- Dropoff location
- Trip distance
- Other available trip metadata used by the feature-engineering pipeline

### Output

The model predicts:

**Trip duration**

The problem is formulated as a **regression problem**, because the target variable is a continuous numerical value.

---

## 3. High-Level Architecture

```text
                         ┌─────────────────────┐
                         │   Raw Taxi Data     │
                         │  .parquet / DVC     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Data Preparation  │
                         │ pipeline/prepare.py │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Feature Engineering │
                         │ pipeline/featurize  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Model Training    │
                         │ pipeline/train.py   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Evaluation      │
                         │ pipeline/evaluate   │
                         └──────────┬──────────┘
                                    │
                           ┌────────┴────────┐
                           ▼                 ▼
                    Quality Gate          MLflow
                           │                 │
                           ▼                 ▼
                       Staging          Model Registry
                           │
                           ▼
                      Docker Image
                           │
                           ▼
                      Deployment/API
                           │
                           ▼
                 Monitoring / Drift Detection
                           │
                           ▼
                   Continuous Training
```

---

## 4. Repository Structure

```text
MLOPS_Tasks/
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── continuous-training.yml
│
├── data/
│   ├── green_tripdata_2025-08.parquet
│   ├── processed/
│   │   ├── train.parquet
│   │   ├── val.parquet
│   │   └── data_quality.json
│   └── features/
│       ├── X_train.pkl
│       ├── X_val.pkl
│       ├── y_train.pkl
│       ├── y_val.pkl
│       ├── vectorizer.pkl
│       └── feature_stats.json
│
├── metrics/
│   ├── eval_metrics.json
│   └── train_metrics.json
│
├── models/
│   └── model.pkl
│
├── pipeline/
│   ├── prepare.py
│   ├── featurize.py
│   ├── train.py
│   └── evaluate.py
│
├── scripts/
│   ├── model_quality_gate.py
│   └── ct_quality_gate.py
│
├── src/
│   └── prodml/
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── train.py
│       └── ...
│
├── dvc.yaml
├── params.yaml
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

# 5. Data Versioning with DVC

DVC (**Data Version Control**) is used because large datasets and ML artifacts should not be managed directly like normal source-code files.

Instead of committing a large Parquet dataset into Git, DVC tracks the data and stores metadata that allows the exact version to be reproduced.

### DVC commands

Pull data and DVC-tracked artifacts:

```bash
dvc pull
```

Re-run the pipeline:

```bash
dvc repro
```

Inspect pipeline status:

```bash
dvc status
```

Inspect the DAG:

```bash
dvc dag
```

---

# 6. DVC Pipeline

The project defines four main stages in `dvc.yaml`.

```text
                 Raw Data
                    │
                    ▼
              ┌───────────┐
              │  prepare  │
              └─────┬─────┘
                    │
                    ▼
              ┌───────────┐
              │ featurize │
              └─────┬─────┘
                    │
                    ▼
              ┌───────────┐
              │   train   │
              └─────┬─────┘
                    │
                    ▼
              ┌───────────┐
              │ evaluate  │
              └─────┬─────┘
                    │
                    ▼
              Evaluation Metrics
```

## Stage 1 — Prepare

Command:

```bash
python pipeline/prepare.py
```

Responsibilities:

- Load raw taxi data.
- Perform data preparation.
- Create training and validation datasets.
- Generate data-quality metrics.

Outputs include:

```text
data/processed/train.parquet
data/processed/val.parquet
data/processed/data_quality.json
```

## Stage 2 — Featurize

Command:

```bash
python pipeline/featurize.py
```

Responsibilities:

- Transform processed data into model-ready features.
- Build the required feature representation.
- Store train/validation feature matrices.
- Save the fitted vectorizer and feature statistics.

Outputs:

```text
data/features/X_train.pkl
data/features/X_val.pkl
data/features/y_train.pkl
data/features/y_val.pkl
data/features/vectorizer.pkl
data/features/feature_stats.json
```

## Stage 3 — Train

Command:

```bash
python pipeline/train.py
```

Responsibilities:

- Load training features.
- Train the configured regression model.
- Save the trained model.
- Record training metrics.

Output:

```text
models/model.pkl
```

Metrics:

```text
metrics/train_metrics.json
```

## Stage 4 — Evaluate

Command:

```bash
python pipeline/evaluate.py
```

Responsibilities:

- Load the trained model.
- Evaluate it against the validation dataset.
- Calculate regression metrics.
- Apply configured evaluation thresholds.
- Write evaluation metrics.

Output:

```text
metrics/eval_metrics.json
```

---

# 7. Model Evaluation

The main evaluation metrics are:

## MAE — Mean Absolute Error

MAE measures the average absolute difference between the actual and predicted values.

```text
MAE = average(|actual - predicted|)
```

For MAE:

**Lower is better.**

---

## RMSE — Root Mean Squared Error

RMSE measures the square root of the average squared prediction error.

```text
RMSE = sqrt(average((actual - predicted)^2))
```

Large errors are penalized more heavily than with MAE.

For RMSE:

**Lower is better.**

---

## R² — R-squared

R² measures the amount of variance explained by the model.

A value closer to 1 generally indicates a better fit.

For R²:

**Higher is better.**

---

## Example Metrics File

The evaluation output is stored in:

```text
metrics/eval_metrics.json
```

Example:

```json
{
  "mae": 4.2994,
  "rmse": 42.9022,
  "r2": 0.78
}
```

The values above are illustrative; actual values depend on the current pipeline run and dataset.

---

# 8. MLflow Experiment Tracking

MLflow is used to track machine-learning experiments and model lifecycle information.

The training process records information such as:

- MAE
- RMSE
- R²
- Training duration
- Model size
- Data hash
- Model parameters
- Experiment metadata

The training runs are associated with the trip-duration experiment.

Conceptually:

```text
Training Run
     │
     ├── Parameters
     ├── Metrics
     ├── Artifacts
     ├── Model
     └── Data Information
```

This makes it possible to compare different runs and identify which model version produced a particular result.

---

# 9. Model Registry and Promotion

The MLflow Model Registry is used to manage model versions.

The intended lifecycle is:

```text
Candidate Model
      │
      ▼
 Quality Gate
      │
      ▼
   Staging
      │
      ▼
 Human Approval
      │
      ▼
 Production
```

The Continuous Training workflow does **not** automatically promote a candidate directly to Production.

This is intentional because Production deployment is a higher-risk operation and should require a human approval step.

---

# 10. Model Quality Gate

The repository contains:

```text
scripts/model_quality_gate.py
```

This script compares:

```text
New Model MAE
        vs
Production Model MAE
```

The CI rule is:

> Fail CI when the new model's MAE regresses by more than 5%.

The calculation is conceptually:

```text
regression =
(new_mae - production_mae) / production_mae
```

Example:

```text
Production MAE = 4.00
New MAE        = 4.50

Regression = (4.50 - 4.00) / 4.00
           = 12.5%
```

Since:

```text
12.5% > 5%
```

the quality gate fails.

Run the gate locally:

```bash
python scripts/model_quality_gate.py 4.00
```

---

# 11. CI vs Continuous Training Quality Gates

There are two different purposes.

## CI Quality Gate

CI is validating a code/model change.

```text
Bad model
   │
   ▼
MAE regression > 5%
   │
   ▼
exit 1
   │
   ▼
❌ CI fails
```

This is useful because a pull request should not be considered healthy when the proposed model violates the quality requirement.

## Continuous Training Gate

Continuous Training is evaluating whether a newly retrained model deserves promotion.

```text
Candidate Model
      │
      ▼
Compare against Production
      │
      ├── Not good enough
      │       ↓
      │    Reject
      │       ↓
      │    exit 0
      │
      └── Good enough
              ↓
          Promote Staging
```

A rejected candidate is a valid business decision, not a broken pipeline.

Therefore the CT gate should normally keep the workflow successful when the model is rejected.

---

# 12. Continuous Training

The Continuous Training workflow is:

```text
.github/workflows/continuous-training.yml
```

Its purpose is to close the loop between:

```text
New Data
   ↓
Retraining
   ↓
Evaluation
   ↓
Model Decision
   ↓
Staging
```

The workflow supports three trigger mechanisms.

---

## 12.1 Scheduled Training

A weekly GitHub Actions schedule can retrain the model automatically.

Conceptually:

```text
Every week
    ↓
GitHub Actions
    ↓
DVC Pull
    ↓
Training
    ↓
Evaluation
    ↓
Gate
```

This is useful when data is expected to change regularly.

---

## 12.2 Manual Training

The workflow also supports:

```yaml
workflow_dispatch:
```

This allows a developer to manually start the workflow from GitHub Actions.

A manual run can include an input such as:

```yaml
data-version:
```

This gives the operator a way to identify the data version or dataset context associated with the run.

Important:

`data-version` is only an input value by itself. It does not automatically change the DVC dataset unless the workflow explicitly connects that input to a DVC/Git version-selection mechanism.

---

## 12.3 External Event / Data Drift

The workflow also supports:

```yaml
repository_dispatch:
  types:
    - data-drift
```

This allows an external system, such as a drift detector, to request a new training run.

Conceptually:

```text
Monitoring
    │
    ▼
Drift detected
    │
    ▼
repository_dispatch
    │
    ▼
Continuous Training
```

This is particularly useful for monitoring and drift-detection modules.

---

# 13. Continuous Training Pipeline

The complete CT pipeline is:

```text
                    Trigger
                       │
          ┌────────────┼────────────┐
          │            │            │
       Schedule     Manual       Data Drift
          │            │            │
          └────────────┼────────────┘
                       ▼
                    DVC Pull
                       │
                       ▼
                  Validate Data
                       │
                       ▼
                    DVC Repro
                       │
                       ▼
                 Candidate Model
                       │
                       ▼
                   Evaluate
                       │
                       ▼
              Compare with Production
                 ┌─────┴─────┐
                 │           │
               Reject      Accept
                 │           │
                 ▼           ▼
              exit 0      Staging
                             │
                             ▼
                       Build Image
                             │
                             ▼
                          Notify
```

---

# 14. Why Staging Is Automatic but Production Is Not

The Continuous Training workflow can automatically promote a model to **Staging** when it passes the quality criteria.

Production remains behind a human approval gate because:

- Production has real users or downstream systems.
- A metric improvement does not guarantee operational safety.
- Human review is useful before changing the live model.
- Deployment risk is higher than experimentation risk.

Therefore:

```text
Automatic:
Candidate → Staging

Manual approval:
Staging → Production
```

This provides automation while keeping a human in control of the final production release.

---

# 15. Continuous Training Decision Table

| Trigger | Meaning | Implemented in CT | Module 5 / Future |
|---|---|---:|---:|
| Schedule | Weekly automatic retraining | ✅ | — |
| Data drift | Retrain after detected drift | ✅ via `repository_dispatch` interface | Drift detector provides the event |
| Performance degradation | Retrain after model performance drops | Possible extension | ✅ Monitoring-driven |
| New labeled data | Retrain after new labels arrive | Possible extension | ✅ Data ingestion-driven |

The main CT workflow is designed so additional triggers can be connected without changing the core train/evaluate/promote stages.

---

# 16. Docker

Docker is used to provide a reproducible runtime environment.

Build and start services with:

```bash
docker compose up -d --build
```

This helps keep the application environment consistent across development and deployment.

The Dockerized system can include:

- FastAPI application
- MLflow service
- Supporting infrastructure defined by the Compose configuration

---

# 17. FastAPI Prediction Service

The trained model is exposed through a FastAPI application.

Typical endpoints include:

```text
GET  /health
POST /predict
```

### Health Check

```http
GET /health
```

The health endpoint is used to verify that the API service is alive and ready.

### Prediction

```http
POST /predict
```

The endpoint receives trip-related information and returns the predicted trip duration.

A typical request flow is:

```text
Client
  │
  ▼
FastAPI
  │
  ▼
Input Validation
  │
  ▼
Feature Transformation
  │
  ▼
Loaded Model
  │
  ▼
Prediction
  │
  ▼
JSON Response
```

---

# 18. Local Development

## Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project:

```bash
pip install -e .
```

---

## Run tests

```bash
pytest
```

---

## Run linting

```bash
ruff check src tests
```

---

## Reproduce the ML pipeline

```bash
dvc pull
dvc repro
```

---

# 19. Useful Git Commands

Check repository state:

```bash
git status
```

Create a feature branch:

```bash
git checkout -b feature/my-change
```

Stage changes:

```bash
git add .
```

Commit:

```bash
git commit -m "Describe the change"
```

Push:

```bash
git push origin feature/my-change
```

---

# 20. GitHub Actions

The project uses GitHub Actions for automation.

The CI workflow is responsible for validating changes before they are merged.

Typical CI responsibilities include:

```text
Checkout
   ↓
Install Dependencies
   ↓
Lint
   ↓
Run Tests
   ↓
Build
   ↓
Model Quality Gate
```

The Continuous Training workflow is responsible for retraining and evaluating new candidates outside the normal pull-request lifecycle.

---

# 21. Security and Permissions

GitHub Actions workflows should follow the principle of least privilege.

For example:

```yaml
permissions:
  contents: read
```

means the workflow's GitHub token receives read-only repository-content permission.

This is different from workflow triggers:

```text
workflow_dispatch
repository_dispatch
schedule
```

Triggers decide:

**"When should the workflow start?"**

Permissions decide:

**"What is the workflow allowed to access or modify?"**

---

# 22. Reproducibility

One of the main goals of this project is reproducibility.

A reproducible ML run should capture:

```text
Code version
    +
Data version
    +
Parameters
    +
Feature pipeline
    +
Model configuration
    +
Evaluation metrics
```

DVC provides data/pipeline versioning, while MLflow provides experiment and model tracking.

Together they help answer:

- Which data produced this model?
- Which parameters were used?
- Which code version trained it?
- What metrics did it achieve?
- Which model version was promoted?

---

# 23. End-to-End MLOps Lifecycle

The complete lifecycle can be summarized as:

```text
                ┌───────────────┐
                │     Data      │
                └───────┬───────┘
                        ▼
                   DVC Versioning
                        │
                        ▼
                   Data Prepare
                        │
                        ▼
                 Feature Engineering
                        │
                        ▼
                     Training
                        │
                        ▼
                    Evaluation
                        │
                        ▼
                  Quality Gate
                        │
                        ▼
                     MLflow
                        │
                        ▼
                    Staging
                        │
                        ▼
                 Docker / Deployment
                        │
                        ▼
                    Monitoring
                        │
                        ▼
                  Drift Detection
                        │
                        ▼
                Continuous Training
                        │
                        └──────────────► New Candidate
```

This creates a feedback loop instead of treating model training as a one-time process.

---

# 24. Key MLOps Principles Demonstrated

### Reproducibility

The same data, parameters, code, and pipeline should produce traceable results.

### Automation

CI and CT reduce manual steps in testing and retraining.

### Versioning

Git manages source code while DVC manages data and ML pipeline artifacts.

### Experiment Tracking

MLflow records experiments, metrics, parameters, and models.

### Quality Gates

A candidate model must meet objective quality requirements before promotion.

### Controlled Deployment

Staging can be automated, while Production requires human approval.

### Monitoring Feedback Loop

Monitoring and drift detection can trigger retraining.

---

# 25. Common Commands Cheat Sheet

```bash
# Run tests
pytest

# Lint
ruff check src tests

# Pull DVC artifacts
dvc pull

# Reproduce DVC pipeline
dvc repro

# View DVC pipeline graph
dvc dag

# Check DVC status
dvc status

# Run quality gate
python scripts/model_quality_gate.py <production_mae>

# Start Docker services
docker compose up -d --build
```

---

# 26. Conclusion

`prodml` is an end-to-end MLOps project that combines machine learning development with software-engineering and deployment practices.

The project moves beyond simply training a model by introducing:

- Data versioning with DVC
- Reproducible ML pipelines
- Experiment tracking with MLflow
- Automated evaluation
- Model quality gates
- FastAPI model serving
- Docker containerization
- GitHub Actions CI
- Continuous Training
- Automated Staging promotion
- Human-controlled Production promotion
- Monitoring and drift-driven retraining architecture

The final objective is a system where a new data/model cycle can move from:

```text
New Data
   ↓
Retraining
   ↓
Evaluation
   ↓
Quality Decision
   ↓
Staging
```

while maintaining reproducibility, traceability, and deployment safety.

---

## Repository

GitHub:
https://github.com/Mariam123Hamada/MLOPS_Tasks
LinkedIn: 
https://www.linkedin.com/in/mariam-abdelsalam-979843335

Author: Mariam Abdelsalam
