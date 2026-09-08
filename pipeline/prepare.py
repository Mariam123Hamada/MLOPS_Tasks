"""
Prepare stage: Load raw data and split into train/val sets.
"""

import json
import logging
from pathlib import Path

import yaml

from prodml.data import load_data, split_data
from prodml.features import create_features

logger = logging.getLogger(__name__)


def prepare():
    """Load raw data and create train/val split."""

    # Load parameters
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    prepare_params = params["prepare"]
    random_state = prepare_params["random_state"]
    validation_split = prepare_params["validation_split"]

    logger.info(
        f"Prepare stage: random_state={random_state}, validation_split={validation_split}"
    )

    # Load and process data
    df = load_data()
    df = create_features(df)

    # Split
    df_train, df_val = split_data(df)

    # Create output directory
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save splits
    df_train.to_parquet(output_dir / "train.parquet", index=False)
    df_val.to_parquet(output_dir / "val.parquet", index=False)

    logger.info(f"Train set: {len(df_train)} rows, Val set: {len(df_val)} rows")

    # Data quality metrics
    quality_metrics = {
        "train_rows": len(df_train),
        "val_rows": len(df_val),
        "train_columns": len(df_train.columns),
        "missing_values_train": df_train.isnull().sum().to_dict(),
    }

    quality_path = output_dir / "data_quality.json"
    with open(quality_path, "w") as f:
        json.dump(quality_metrics, f, indent=2, default=str)

    logger.info(f"Saved data quality metrics to {quality_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prepare()
