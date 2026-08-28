import pandas as pd

CATEGORICAL_FEATURES = ["PU_DO"]

NUMERICAL_FEATURES = ["trip_distance"]

FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features."""

    df = df.copy()

    df["PU_DO"] = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)

    return df


def prepare_feature_dicts(
    df: pd.DataFrame,
) -> list[dict[str, object]]:
    """Convert features into dictionaries for DictVectorizer."""

    return df[FEATURES].to_dict(orient="records")
