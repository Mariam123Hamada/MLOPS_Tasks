from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


class Settings(BaseSettings):
    data_path: Path = Field(
        default_factory=lambda: repo_path("data", "green_tripdata_2025-08.parquet")
    )
    onnx_model_path: Path = Field(
        default_factory=lambda: repo_path("models", "model.onnx")
    )
    model_path: Path = Field(default_factory=lambda: repo_path("models", "model.pkl"))
    MODEL_NAME: str = "ride-duration-predictor"
    MODEL_STAGE: str = "Production"
    baseline_model_path: Path = Field(
        default_factory=lambda: repo_path("models", "baseline.pkl")
    )

    model_version: str = "0.1.0"
    model_training_date: str = "2026-08-28"

    test_size: float = 0.2
    random_state: int = 42
    min_duration: float = 1.0
    max_duration: float = 60.0
    api_port: int = 8000

    @staticmethod
    def _default_mlflow_tracking_uri() -> str:
        return repo_path("mlruns").as_uri()

    mlflow_tracking_uri: str = Field(
        default_factory=_default_mlflow_tracking_uri,
        validation_alias=AliasChoices("MLFLOW_TRACKING_URI"),
    )

    @field_validator("mlflow_tracking_uri", mode="before")
    @classmethod
    def normalize_mlflow_tracking_uri(cls, value):
        if value is None or value == "":
            return cls._default_mlflow_tracking_uri()
        return value

    model_config = SettingsConfigDict(
        env_prefix="PRODML_",
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
