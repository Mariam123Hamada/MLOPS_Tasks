from prodml.predict import DurationPredictor


class FakeVectorizer:

    def transform(self, features):
        return features


class FakeModel:

    def predict(self, features):
        return [10.5 for _ in features]


def test_predict_one() -> None:

    predictor = DurationPredictor(
        model=FakeModel(),
        vectorizer=FakeVectorizer(),
    )

    result = predictor.predict_one(
        {
            "PU_DO": "1_2",
            "trip_distance": 5.0,
        }
    )

    assert result == 10.5


def test_predict_batch() -> None:

    predictor = DurationPredictor(
        model=FakeModel(),
        vectorizer=FakeVectorizer(),
    )

    results = predictor.predict_batch(
        [
            {
                "PU_DO": "1_2",
                "trip_distance": 5.0,
            },
            {
                "PU_DO": "2_3",
                "trip_distance": 10.0,
            },
        ]
    )

    assert results == [10.5, 10.5]
