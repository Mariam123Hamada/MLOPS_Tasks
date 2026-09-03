# prodml

Production-grade machine learning project for predicting NYC Green Taxi trip duration using a scikit-learn model served through a FastAPI application.

## Overview

This project trains and serves a regression model that predicts trip duration from trip metadata such as pickup/dropoff location pair and trip distance. It includes:

- a data loading and feature engineering pipeline
- model training and serialization
- ONNX export for portability checks
- a FastAPI prediction service
- Docker packaging for deployment
- automated tests for model and API behavior

## Project Structure

```text
.
├── src/
│   └── prodml/
│       ├── api/
│       │   ├── main.py
│       │   └── schemas.py
│       ├── benchmark.py
│       ├── config.py
│       ├── data.py
│       ├── export.py
│       ├── features.py
│       ├── logging_conf.py
│       ├── predict.py
│       └── train.py
├── tests/
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.single
│   └── docker-compose.yml
├── models/
├── data/
├── notebooks/
├── reports/
├── pyproject.toml
├── .dockerignore
├── .gitignore
├── LICENSE
└── README.md
```

## Features

The model predicts trip duration as a continuous value using:

- `PU_DO`: pickup and dropoff location pair
- `trip_distance`: trip distance in miles

## Requirements

- Python 3.10+
- pip
- Docker Desktop (for containerization)
- Windows PowerShell, macOS Terminal, or Linux shell

## Local Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project with development dependencies:

```powershell
pip install -e ".[dev]"
```

## Run Tests

```powershell
pytest -q
```

For coverage output:

```powershell
pytest -q --cov=src/prodml --cov-report=term-missing
```

## Train the Model

```powershell
python -m prodml.train
```

This creates the model artifact at:

```text
models/model.pkl
```

## Run the API Locally

Start the service:

```powershell
uvicorn prodml.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open the API docs in a browser:

```text
http://127.0.0.1:8000/docs
```

## Example Prediction Request

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"PU_DO":"1_2","trip_distance":5.0}'
```

Example response:

```json
{
  "prediction": 15.74,
  "model_version": "0.1.0",
  "correlation_id": "...",
  "latency_ms": 1.24
}
```

## Health Check

```powershell
curl.exe http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"healthy"}
```

## Docker

### Build the multi-stage image

```powershell
docker build -f docker/Dockerfile -t prodml-api:0.1.0 -t prodml-api:latest .
```

### Build the single-stage comparison image

```powershell
docker build -f docker/Dockerfile.single -t prodml-api-single:0.1.0 .
```

### Run the container

```powershell
docker run -d --name prodml-api -p 8000:8000 prodml-api:0.1.0
```

Check the service:

```powershell
docker exec prodml-api whoami
curl.exe http://127.0.0.1:8000/health
```

### Docker Compose

```powershell
docker compose -f docker/docker-compose.yml up --build
```

## Docker Publish

Tag and push to Docker Hub:

```powershell
docker login
docker tag prodml-api:0.1.0 <your-dockerhub-user>/prodml-api:0.1.0
docker tag prodml-api:latest <your-dockerhub-user>/prodml-api:latest
docker push <your-dockerhub-user>/prodml-api:0.1.0
docker push <your-dockerhub-user>/prodml-api:latest
```

Then others can run:

```powershell
docker run -p 8000:8000 <your-dockerhub-user>/prodml-api:0.1.0
```

## Model Export

The project also includes ONNX export support for parity and portability checks:

```powershell
python -m prodml.export
```

## Notes

- The package is structured as a `src` layout.
- The app loads the model from the configured runtime path in production.
- The project is designed to be reproducible in a local virtual environment and in Docker.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Author

### Mariam Hamada Abdelsalam
###  Github https://www.linkedin.com/in/mariam-abdelsalam-979843335
