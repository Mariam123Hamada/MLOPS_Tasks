"""
Featurize stage: Transform raw features into vectorized features.
"""

import json
import logging
import pickle
from pathlib import Path

import pandas as pd
import yaml
from sklearn.feature_extraction import DictVectorizer

from prodml.features import prepare_feature_dicts

logger = logging.getLogger(__name__)


def featurize():
    """Transform data into vectorized features."""

    # Load parameters
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    featurize_params = params["featurize"]
    min_duration = featurize_params["min_trip_duration"]
    max_duration = featurize_params["max_trip_duration"]

    logger.info(f"Featurize stage: duration range [{min_duration}, {max_duration}]")

    # Load prepared data
    df_train = pd.read_parquet("data/processed/train.parquet")
    df_val = pd.read_parquet("data/processed/val.parquet")

    # Filter by duration
    df_train = df_train[
        (df_train["duration"] >= min_duration) & (df_train["duration"] <= max_duration)
    ]
    df_val = df_val[
        (df_val["duration"] >= min_duration) & (df_val["duration"] <= max_duration)
    ]

    logger.info(f"After filtering: Train {len(df_train)}, Val {len(df_val)}")

    # Prepare feature dicts
    train_dicts = prepare_feature_dicts(df_train)
    val_dicts = prepare_feature_dicts(df_val)

    # Vectorize
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values

    logger.info(f"X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

    # Save outputs
    output_dir = Path("data/features")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "X_train.pkl", "wb") as f:
        pickle.dump(X_train, f)
    with open(output_dir / "X_val.pkl", "wb") as f:
        pickle.dump(X_val, f)
    with open(output_dir / "y_train.pkl", "wb") as f:
        pickle.dump(y_train, f)
    with open(output_dir / "y_val.pkl", "wb") as f:
        pickle.dump(y_val, f)
    with open(output_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(dv, f)

    # Feature statistics
    feature_names = dv.get_feature_names_out()
    feature_stats = {
        "n_features": len(feature_names),
        "feature_names_sample": feature_names[:5].tolist(),
        "sparsity": float(1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1])),
    }

    stats_path = output_dir / "feature_stats.json"
    with open(stats_path, "w") as f:
        json.dump(feature_stats, f, indent=2)

    logger.info(f"Saved feature stats to {stats_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    featurize()
