import pytest

from prodml.features import create_features


@pytest.mark.parametrize(
    "trip_distance, passenger_count, pu_do",
    [
        (5.0, 2, "12_34"),
        (0.0, 2, "12_34"),
        (5.0, 2, None),
        (5.0, 2, "999_999"),
    ],
)
def test_feature_engineering_edge_cases(
    trip_distance,
    passenger_count,
    pu_do,
):
    result = create_features(
        trip_distance=trip_distance,
        passenger_count=passenger_count,
        PU_DO=pu_do,
    )

    assert result is not None
