import pandas as pd

from prodml.features import create_features


def test_create_features() -> None:

    df = pd.DataFrame(
        {
            "PULocationID": [10],
            "DOLocationID": [20],
            "trip_distance": [5.0],
        }
    )

    result = create_features(df)

    assert result.loc[0, "PU_DO"] == "10_20"
