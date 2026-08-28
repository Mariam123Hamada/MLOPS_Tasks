# MLOPS_Tasks

## prodml

Production Machine Learning project for predicting NYC Green Taxi trip duration.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

### Windows PowerShell:
```
.\.venv\Scripts\Activate.ps1
```
### Install the project:
```
pip install -e ".[dev]"
```
## Project Workflow
### 1. Install
```
pip install -e ".[dev]"
```
### 2. Lint
```
ruff check src tests && black --check src tests
```
### 3. Test
```
pytest -v --cov=src/prodml --cov-report=term-missing
```

### 4. Train
```
python -m prodml.train
```
This command trains the model and saves it to:
models/model.pkl
### 5. Serve
```
uvicorn prodml.api.main:app --reload --port 8000
```
The API documentation is available at:
-http://127.0.0.1:8000/docs
## Features

The model uses:
PU_DO: Pickup and dropoff location pair
trip_distance: Trip distance

The prediction target is
Trip duration in minutes
Architecture
```
data.py
    ↓
features.py
    ↓
train.py
    ↓
model.pkl
    ↓
predict.py
    ↓
FastAPI

```

# Your exact workflow now

Run these commands in order.

### 1. Install

```powershell
pip install -e ".[dev]"
```
### 2. Run the notebook

Open:
```
notebooks/00-baseline.ipynb
```
Run:
```
Restart Kernel → Restart & Run All
```
Record:
```
Validation RMSE = ?
Validation MAE = ?
```
Update:
```
reports/module-1.md
```
### 3. Train the production package
```
python -m prodml.train
```
It should print:
```
Validation RMSE: X.XXXX
Validation MAE: X.XXXX
```
### 4. Compare the MAE

The check is:
```
abs(baseline_mae - package_mae) <= 0.05
```
The ideal result is actually:
```
Baseline MAE == Package MAE
```
or extremely close.

If the MAE changes significantly, check these four things:
Same Parquet file.
Same duration filtering.
Same feature engineering.
Same random_state=42 and test_size=0.2

### Author: Mariam Hamada Abdelsalam
