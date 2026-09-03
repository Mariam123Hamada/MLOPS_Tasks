import hashlib
import logging
import pickle
import subprocess
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import torch
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from torch import nn

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import create_features, prepare_feature_dicts
from prodml.logging_conf import configure_logging

configure_logging()

logger = logging.getLogger(__name__)


def get_git_commit() -> str:
    """Return the current Git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_git_author() -> str:
    """Return the Git author name."""
    try:
        return subprocess.check_output(
            ["git", "config", "user.name"],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_data_hash(path: Path) -> str:
    """Return SHA256 hash of the training data."""
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def log_common_tags(data_version: str) -> None:
    """Log tags shared by every MLflow run."""
    mlflow.set_tags(
        {
            "git_commit": get_git_commit(),
            "data_version": data_version,
            "author": get_git_author(),
        }
    )


def log_requirements() -> None:
    """Log the exact requirements.txt used by the project."""
    requirements_path = Path("requirements.txt")

    if requirements_path.exists():
        mlflow.log_artifact(
            str(requirements_path),
            artifact_path="environment",
        )
    else:
        logger.warning("requirements.txt was not found")


def save_residual_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
) -> None:
    """Create and save a residual plot."""
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted duration")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_feature_importance_plot(
    feature_names: list[str],
    importance: np.ndarray,
    path: Path,
    title: str,
) -> None:
    """Create and save a feature-importance plot."""
    importance = np.asarray(importance)

    if importance.ndim > 1:
        importance = np.mean(np.abs(importance), axis=0)

    importance = np.abs(importance)

    indices = np.argsort(importance)[-20:]

    plt.figure(figsize=(10, 6))
    plt.barh(
        np.array(feature_names)[indices],
        importance[indices],
    )
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def calculate_model_size(path: Path) -> float:
    """Return model size in MB."""
    return path.stat().st_size / (1024 * 1024)


def log_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_duration: float,
    model_size_mb: float,
) -> dict[str, float]:
    """Calculate and log common model metrics."""

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )

    mae = float(mean_absolute_error(y_true, y_pred))

    r2 = float(r2_score(y_true, y_pred))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "train_duration_seconds": train_duration,
        "model_size_mb": model_size_mb,
    }

    mlflow.log_metrics(metrics)

    return metrics


class TripDurationMLP(nn.Module):
    """Small neural network for trip duration prediction."""

    def __init__(self, input_size: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def prepare_dataset():
    """Load data, create features, and create one shared split."""

    df = load_data()
    df = create_features(df)

    df_train, df_val = split_data(df)

    train_dicts = prepare_feature_dicts(df_train)
    val_dicts = prepare_feature_dicts(df_val)

    dv = DictVectorizer()

    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values.astype(np.float32)
    y_val = df_val["duration"].values.astype(np.float32)

    feature_names = list(dv.get_feature_names_out())

    return (
        X_train,
        X_val,
        y_train,
        y_val,
        feature_names,
        dv,
    )


def train_linear_regression(
    X_train,
    X_val,
    y_train,
    y_val,
    feature_names,
    data_version,
) -> dict[str, float]:
    """Train and track Linear Regression baseline."""

    with mlflow.start_run(run_name="linear-regression"):

        mlflow.set_tag("framework", "scikit-learn")

        log_common_tags(data_version)

        mlflow.log_params(
            {
                "model": "LinearRegression",
                "fit_intercept": True,
                "split_seed": settings.random_state,
                "data_version": data_version,
            }
        )

        model = LinearRegression()

        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_duration = time.perf_counter() - start

        y_pred = model.predict(X_val)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            model_path = tmp_path / "model.pkl"

            with open(model_path, "wb") as file:
                pickle.dump(model, file)

            residual_path = tmp_path / "residual_plot.png"

            save_residual_plot(
                y_val,
                y_pred,
                residual_path,
            )

            importance_path = tmp_path / "feature_importance.png"

            save_feature_importance_plot(
                feature_names,
                model.coef_,
                importance_path,
                "Linear Regression Coefficients",
            )

            mlflow.sklearn.log_model(
                model,
                name="model",
            )

            mlflow.log_artifact(
                str(residual_path),
                artifact_path="plots",
            )

            mlflow.log_artifact(
                str(importance_path),
                artifact_path="plots",
            )

            model_size_mb = calculate_model_size(model_path)

        metrics = log_metrics(
            y_val,
            y_pred,
            train_duration,
            model_size_mb,
        )

        log_requirements()

        return metrics


def train_xgboost(
    X_train,
    X_val,
    y_train,
    y_val,
    feature_names,
    data_version,
    params: dict | None = None,
    run_name: str = "xgboost",
    nested: bool = False,
) -> dict[str, float]:
    """Train and track an XGBoost model."""

    params = params or {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "random_state": settings.random_state,
    }

    with mlflow.start_run(
        run_name=run_name,
        nested=nested,
    ):

        mlflow.set_tag("framework", "xgboost")

        log_common_tags(data_version)

        mlflow.log_params(
            {
                **params,
                "split_seed": settings.random_state,
                "data_version": data_version,
            }
        )

        model = xgb.XGBRegressor(
            **params,
        )

        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_duration = time.perf_counter() - start

        y_pred = model.predict(X_val)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            model_path = tmp_path / "model.json"

            model.save_model(model_path)

            residual_path = tmp_path / "residual_plot.png"

            save_residual_plot(
                y_val,
                y_pred,
                residual_path,
            )

            importance_path = tmp_path / "feature_importance.png"

            save_feature_importance_plot(
                feature_names,
                model.feature_importances_,
                importance_path,
                "XGBoost Feature Importance",
            )

            mlflow.xgboost.log_model(
                model,
                name="model",
            )

            mlflow.log_artifact(
                str(residual_path),
                artifact_path="plots",
            )

            mlflow.log_artifact(
                str(importance_path),
                artifact_path="plots",
            )

            model_size_mb = calculate_model_size(model_path)

        metrics = log_metrics(
            y_val,
            y_pred,
            train_duration,
            model_size_mb,
        )

        log_requirements()

        return metrics


def train_pytorch_mlp(
    X_train,
    X_val,
    y_train,
    y_val,
    feature_names,
    data_version,
) -> dict[str, float]:
    """Train and track a small PyTorch MLP."""

    with mlflow.start_run(run_name="pytorch-mlp"):

        mlflow.set_tag("framework", "pytorch")

        params = {
            "hidden_layer_1": 64,
            "hidden_layer_2": 32,
            "learning_rate": 0.001,
            "epochs": 20,
            "batch_size": 256,
            "optimizer": "Adam",
            "split_seed": settings.random_state,
            "data_version": data_version,
        }

        mlflow.log_params(params)

        torch.manual_seed(settings.random_state)

        X_train_dense = X_train.toarray().astype(np.float32)
        X_val_dense = X_val.toarray().astype(np.float32)

        X_train_tensor = torch.tensor(X_train_dense)
        y_train_tensor = torch.tensor(y_train).reshape(-1, 1)

        X_val_tensor = torch.tensor(X_val_dense)

        model = TripDurationMLP(
            input_size=X_train.shape[1],
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params["learning_rate"],
        )

        criterion = nn.MSELoss()

        start = time.perf_counter()

        model.train()

        for _ in range(params["epochs"]):

            permutation = torch.randperm(X_train_tensor.size(0))

            for i in range(
                0,
                X_train_tensor.size(0),
                params["batch_size"],
            ):
                indices = permutation[i : i + params["batch_size"]]

                batch_x = X_train_tensor[indices]
                batch_y = y_train_tensor[indices]

                optimizer.zero_grad()

                predictions = model(batch_x)

                loss = criterion(
                    predictions,
                    batch_y,
                )

                loss.backward()

                optimizer.step()

        train_duration = time.perf_counter() - start

        model.eval()

        with torch.no_grad():
            y_pred = model(X_val_tensor).numpy().reshape(-1)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            model_path = tmp_path / "model.pt"

            torch.save(
                model.state_dict(),
                model_path,
            )

            residual_path = tmp_path / "residual_plot.png"

            save_residual_plot(
                y_val,
                y_pred,
                residual_path,
            )

            first_layer_weights = model.network[0].weight.detach().numpy()

            importance = np.mean(
                np.abs(first_layer_weights),
                axis=0,
            )

            importance_path = tmp_path / "feature_importance.png"

            save_feature_importance_plot(
                feature_names,
                importance,
                importance_path,
                "PyTorch MLP Feature Importance",
            )
            input_example = X_train[:1].toarray()

            mlflow.pytorch.log_model(
                model,
                "model",
                input_example=input_example,
                serialization_format="pickle",
            )

            mlflow.log_artifact(
                str(residual_path),
                artifact_path="plots",
            )

            mlflow.log_artifact(
                str(importance_path),
                artifact_path="plots",
            )

            model_size_mb = calculate_model_size(model_path)

        metrics = log_metrics(
            y_val,
            y_pred,
            train_duration,
            model_size_mb,
        )

        log_requirements()

        return metrics


def run_xgboost_sweep(
    X_train,
    X_val,
    y_train,
    y_val,
    feature_names,
    data_version,
) -> None:
    """Run at least 10 nested XGBoost trials."""

    mlflow.xgboost.autolog(
        log_models=False,
        silent=True,
    )

    search_space = [
        {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": settings.random_state,
        },
        {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": settings.random_state,
        },
        {
            "n_estimators": 400,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": settings.random_state,
        },
        {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": settings.random_state,
        },
        {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": settings.random_state,
        },
        {
            "n_estimators": 400,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
            "random_state": settings.random_state,
        },
        {
            "n_estimators": 200,
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": settings.random_state,
        },
        {
            "n_estimators": 300,
            "max_depth": 7,
            "learning_rate": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": settings.random_state,
        },
        {
            "n_estimators": 400,
            "max_depth": 7,
            "learning_rate": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
            "random_state": settings.random_state,
        },
        {
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.02,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
            "random_state": settings.random_state,
        },
    ]

    with mlflow.start_run(run_name="xgboost-sweep"):

        mlflow.set_tag("framework", "xgboost")
        mlflow.set_tag("run_type", "sweep")

        log_common_tags(data_version)

        mlflow.log_param(
            "number_of_trials",
            len(search_space),
        )

        for index, params in enumerate(search_space, start=1):

            train_xgboost(
                X_train,
                X_val,
                y_train,
                y_val,
                feature_names,
                data_version,
                params=params,
                run_name=f"xgboost-trial-{index}",
                nested=True,
            )


def train_model() -> dict[str, float]:
    """Train and track all model families and XGBoost sweep."""

    mlflow.set_tracking_uri(
        settings.mlflow_tracking_uri,
    )

    mlflow.set_experiment(
        "prodml-trip-duration",
    )

    data_version = get_data_hash(
        settings.data_path,
    )

    (
        X_train,
        X_val,
        y_train,
        y_val,
        feature_names,
        _,
    ) = prepare_dataset()

    linear_metrics = train_linear_regression(
        X_train,
        X_val,
        y_train,
        y_val,
        feature_names,
        data_version,
    )

    train_xgboost(
        X_train,
        X_val,
        y_train,
        y_val,
        feature_names,
        data_version,
    )

    train_pytorch_mlp(
        X_train,
        X_val,
        y_train,
        y_val,
        feature_names,
        data_version,
    )

    run_xgboost_sweep(
        X_train,
        X_val,
        y_train,
        y_val,
        feature_names,
        data_version,
    )

    return linear_metrics


def main() -> None:
    """CLI entry point."""
    train_model()


if __name__ == "__main__":
    main()
