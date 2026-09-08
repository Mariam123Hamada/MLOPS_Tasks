"""
Evaluate stage: Evaluate model performance and generate metrics.
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def evaluate():
    """Evaluate model on validation set."""

    # Load parameters
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    eval_params = params.get("evaluate", {})
    min_r2 = eval_params.get("min_r2_score", 0.0)
    max_rmse = eval_params.get("max_rmse", float("inf"))

    logger.info(f"Evaluate stage: min_r2={min_r2}, max_rmse={max_rmse}")
    # Load model and validation data
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("data/features/X_val.pkl", "rb") as f:
        X_val = pickle.load(f)

    with open("data/features/y_val.pkl", "rb") as f:
        y_val = pickle.load(f)

    y_pred = model.named_steps["model"].predict(X_val)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    logger.info(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    # Create output directory
    output_dir = Path("metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    eval_metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "validation_samples": len(y_val),
    }

    metrics_path = output_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)

    logger.info(f"Saved evaluation metrics to {metrics_path}")

    # Check thresholds
    if r2 < min_r2:
        logger.warning(f"R² {r2:.4f} is below minimum {min_r2}")
    if rmse > max_rmse:
        logger.warning(f"RMSE {rmse:.4f} is above maximum {max_rmse}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluate()
