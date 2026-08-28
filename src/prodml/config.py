from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    data_path: Path = Path("data/green_tripdata_2025-08.parquet")

    model_path: Path = Path("../models/model.pkl")

    baseline_model_path: Path = Path("../models/baseline.pkl")

    test_size: float = 0.2

    random_state: int = 42

    min_duration: float = 1.0

    max_duration: float = 60.0

    api_port: int = 8000

    model_config = SettingsConfigDict(
        env_prefix="PRODML_", env_file=".env", extra="ignore"
    )


settings = Settings()
