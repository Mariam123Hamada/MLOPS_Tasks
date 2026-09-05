#!/usr/bin/env python
"""
DVC Pipeline Stage: Train Model

This stage:
1. Loads feature vectors from featurize stage
2. Trains a model using parameters from params.yaml
3. Logs metrics and model to MLflow
4. Saves model to models/model.pkl
5. Tracks DVC data hash for reproducibility
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prodml.config import settings
from prodml.logging_conf import configure_logging
from prodml.train import get_dvc_hash, get_git_commit

configure_logging()
logger = logging.getLogger(__name__)

# MLflow setup
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
mlflow.set_experiment("ride-duration")


def train_from_pipeline():
    """Train model using features from featurize stage."""

    logger.info("Starting train stage...")

    # Load parameters
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    train_params = params.get("train", {})
    model_type = train_params.get("model_type", "linear_regression")
    fit_intercept = train_params.get("fit_intercept", True)
    random_state = train_params.get("random_state", 42)

    # Load feature vectors from featurize stage
    logger.info("Loading feature vectors...")
    X_train = pickle.load(open("data/features/X_train.pkl", "rb"))
    X_val = pickle.load(open("data/features/X_val.pkl", "rb"))
    y_train = pickle.load(open("data/features/y_train.pkl", "rb"))
    y_val = pickle.load(open("data/features/y_val.pkl", "rb"))
    vectorizer = pickle.load(open("data/features/vectorizer.pkl", "rb"))

    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}")

    # Train model
    logger.info(f"Training {model_type} model...")
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params(
            {
                "model_type": model_type,
                "fit_intercept": fit_intercept,
                "random_state": random_state,
            }
        )

        # Train
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(X_train, y_train)

        # Evaluate on train/val
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        train_rmse = np.sqrt(np.mean((y_train_pred - y_train) ** 2))
        val_rmse = np.sqrt(np.mean((y_val_pred - y_val) ** 2))
        train_mae = np.mean(np.abs(y_train_pred - y_train))
        val_mae = np.mean(np.abs(y_val_pred - y_val))

        logger.info(f"Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")
        logger.info(f"Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}")

        # Log metrics
        mlflow.log_metrics(
            {
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "train_mae": train_mae,
                "val_mae": val_mae,
            }
        )

        # Create and save pipeline (vectorizer + model)
        pipeline = Pipeline(
            [
                ("vectorizer", vectorizer),
                ("model", model),
            ]
        )

        # Log pipeline to MLflow
        mlflow.sklearn.log_model(pipeline, "model", input_example=X_train[:1])
        logger.info(f"MLflow run ID: {run.info.run_id}")

        # Log DVC data hash for reproducibility
        dvc_hash = get_dvc_hash(settings.data_path)
        git_commit = get_git_commit()

        tags = {
            "dvc_data_hash": dvc_hash or "unknown",
            "git_commit": git_commit or "unknown",
        }
        mlflow.set_tags(tags)
        logger.info(f"Tracked tags: {tags}")

        # Save model to disk
        Path("models").mkdir(exist_ok=True)
        pickle.dump(pipeline, open("models/model.pkl", "wb"))
        logger.info("Model saved to models/model.pkl")

        # Save train metrics for dvc.yaml
        train_metrics = {
            "train_rmse": float(train_rmse),
            "train_mae": float(train_mae),
            "val_rmse": float(val_rmse),
            "val_mae": float(val_mae),
            "n_features": X_train.shape[1],
            "model_type": model_type,
        }
        print("Train Metrices. ", train_metrics)

        Path("metrics").mkdir(exist_ok=True)
        with open("metrics/train_metrics.json", "w") as f:
            json.dump(train_metrics, f, indent=2)
        logger.info("Train metrics saved to metrics/train_metrics.json")


if __name__ == "__main__":
    train_from_pipeline()
