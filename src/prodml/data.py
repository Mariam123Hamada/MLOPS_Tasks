import pandas as pd

from sklearn.model_selection import train_test_split

from prodml.config import settings


def load_data(path: str | None = None) -> pd.DataFrame:
    """Load NYC taxi trip data and create the duration target."""

    data_path = path or str(settings.data_path)

    df = pd.read_parquet(data_path)

    df["duration"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    df = df[
        (df["duration"] >= settings.min_duration)
        & (df["duration"] <= settings.max_duration)
    ]

    df = df[df["trip_distance"] > 0]

    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and validation sets."""

    df_train, df_val = train_test_split(
        df,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )

    return df_train, df_val
