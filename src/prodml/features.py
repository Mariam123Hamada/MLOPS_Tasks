import pandas as pd

CATEGORICAL_FEATURES = ["PU_DO"]

NUMERICAL_FEATURES = ["trip_distance"]

FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def create_features(
    df: pd.DataFrame | None = None,
    **kwargs: object,
) -> pd.DataFrame:
    """Create model features from a dataframe or a single sample dict."""

    if df is not None:
        df = df.copy()

        if "PU_DO" not in df.columns:
            if {"PULocationID", "DOLocationID"}.issubset(df.columns):
                df["PU_DO"] = (
                    df["PULocationID"].astype(str)
                    + "_"
                    + df["DOLocationID"].astype(str)
                )
            else:
                df["PU_DO"] = None

        return df

    sample = {
        "trip_distance": kwargs.get("trip_distance"),
        "passenger_count": kwargs.get("passenger_count"),
        "PU_DO": kwargs.get("PU_DO"),
    }

    return pd.DataFrame([sample])


def prepare_feature_dicts(
    df: pd.DataFrame,
) -> list[dict[str, object]]:
    """Convert features into dictionaries for DictVectorizer."""

    columns = [c for c in FEATURES if c in df.columns]
    return df[columns].to_dict(orient="records")
