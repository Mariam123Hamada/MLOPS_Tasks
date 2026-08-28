import pickle

import numpy as np
import pandas as pd

from prodml.benchmark import benchmark_onnx, benchmark_pickle
from prodml.benchmark import main as benchmark_main
from prodml.export import export_to_onnx
from prodml.predict import DurationPredictor
from prodml.train import train_model


class FakeVectorizer:
    def transform(self, features):
        return features


class FakeModel:
    def predict(self, features):
        return [10.5 for _ in features]


class BenchmarkFakeModel:
    def predict(self, X):
        return np.array([1.0, 2.0])


class BenchmarkFakeVectorizer:
    def transform(self, data):
        return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)


class BenchmarkFakeSession:
    def get_inputs(self):
        return [type("Input", (), {"name": "features"})()]

    def run(self, *args, **kwargs):
        return [np.array([1.0, 2.0], dtype=np.float32)]


class FakeBenchmarkModel:
    def predict(self, X):
        return np.array([1.0, 2.0])


class FakeBenchmarkVectorizer:
    def transform(self, data):
        return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)


class FakeBenchmarkSession:
    def get_inputs(self):
        return [type("Input", (), {"name": "features"})()]

    def run(self, *args, **kwargs):
        return [np.array([1.0, 2.0], dtype=np.float32)]


def create_predictor() -> DurationPredictor:
    return DurationPredictor(
        model=FakeModel(),
        vectorizer=FakeVectorizer(),
    )


def test_predict_one() -> None:
    predictor = create_predictor()
    result = predictor.predict_one({"PU_DO": "1_2", "trip_distance": 5.0})
    assert result == 10.5


def test_predict_batch() -> None:
    predictor = create_predictor()
    results = predictor.predict_batch(
        [
            {"PU_DO": "1_2", "trip_distance": 5.0},
            {"PU_DO": "2_3", "trip_distance": 10.0},
        ]
    )
    assert results == [10.5, 10.5]


def test_prediction_returns_float() -> None:
    predictor = create_predictor()
    result = predictor.predict_one({"PU_DO": "1_2", "trip_distance": 5.0})
    assert isinstance(result, float)


def test_prediction_is_in_sane_range() -> None:
    predictor = create_predictor()
    result = predictor.predict_one({"PU_DO": "1_2", "trip_distance": 5.0})
    assert 0 < result < 1000


def test_prediction_is_deterministic() -> None:
    predictor = create_predictor()
    features = {"PU_DO": "1_2", "trip_distance": 5.0}
    assert predictor.predict_one(features) == predictor.predict_one(features)


def test_train_model_creates_artifact(tmp_path, monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "PULocationID": [1, 2, 3, 4],
            "DOLocationID": [2, 3, 4, 5],
            "trip_distance": [1.5, 2.0, 3.5, 5.0],
            "lpep_pickup_datetime": pd.to_datetime(
                [
                    "2025-01-01 08:00:00",
                    "2025-01-01 08:15:00",
                    "2025-01-01 08:30:00",
                    "2025-01-01 08:45:00",
                ]
            ),
            "lpep_dropoff_datetime": pd.to_datetime(
                [
                    "2025-01-01 08:20:00",
                    "2025-01-01 08:35:00",
                    "2025-01-01 08:50:00",
                    "2025-01-01 09:10:00",
                ]
            ),
            "duration": [10.0, 12.0, 14.0, 16.0],
        }
    )

    monkeypatch.setattr("prodml.train.load_data", lambda: df)
    monkeypatch.setattr(
        "prodml.train.split_data", lambda data: (data.iloc[:2], data.iloc[2:])
    )
    monkeypatch.setattr("prodml.config.settings.model_path", tmp_path / "model.pkl")

    metrics = train_model()

    assert isinstance(metrics["rmse"], float)
    assert isinstance(metrics["mae"], float)
    assert (tmp_path / "model.pkl").exists()


def test_export_to_onnx_writes_file(monkeypatch, tmp_path) -> None:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LinearRegression

    vectorizer = DictVectorizer()
    vectorizer.fit(
        [
            {"PU_DO": "1_2", "trip_distance": 5.0},
            {"PU_DO": "2_3", "trip_distance": 10.0},
        ]
    )
    model = LinearRegression()
    X = vectorizer.transform(
        [
            {"PU_DO": "1_2", "trip_distance": 5.0},
            {"PU_DO": "2_3", "trip_distance": 10.0},
        ]
    )
    model.fit(X, [12.0, 20.0])

    monkeypatch.setattr(
        "prodml.export.load_artifact",
        lambda model_path: {"model": model, "vectorizer": vectorizer},
    )

    out_path = tmp_path / "artifact.onnx"
    export_to_onnx("ignored.pkl", out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_benchmark_helpers_run(monkeypatch, tmp_path) -> None:
    model = FakeBenchmarkModel()
    vectorizer = FakeBenchmarkVectorizer()
    fake_df = pd.DataFrame(
        {
            "PULocationID": [1, 2],
            "DOLocationID": [2, 3],
            "trip_distance": [1.0, 2.0],
            "lpep_pickup_datetime": pd.to_datetime(
                ["2025-01-01 08:00:00", "2025-01-01 08:20:00"]
            ),
            "lpep_dropoff_datetime": pd.to_datetime(
                ["2025-01-01 08:20:00", "2025-01-01 08:40:00"]
            ),
            "duration": [20.0, 20.0],
        }
    )

    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer}, f)

    monkeypatch.setattr("prodml.benchmark.load_data", lambda: fake_df)
    monkeypatch.setattr("prodml.benchmark.create_features", lambda df: df)
    monkeypatch.setattr(
        "prodml.benchmark.split_data", lambda df: (df.iloc[:1], df.iloc[1:])
    )
    monkeypatch.setattr(
        "prodml.benchmark.pickle.load",
        lambda *args, **kwargs: {"model": model, "vectorizer": vectorizer},
    )
    monkeypatch.setattr(
        "prodml.benchmark.ort.InferenceSession",
        lambda *args, **kwargs: FakeBenchmarkSession(),
    )

    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    pkl_mean, pkl_p95 = benchmark_pickle(model, X, runs=2)
    onnx_mean, onnx_p95 = benchmark_onnx(FakeBenchmarkSession(), "features", X, runs=2)

    assert isinstance(pkl_mean, float)
    assert isinstance(pkl_p95, float)
    assert isinstance(onnx_mean, float)
    assert isinstance(onnx_p95, float)

    monkeypatch.setattr("prodml.benchmark.settings.model_path", model_path)
    monkeypatch.setattr(
        "prodml.benchmark.settings.onnx_model_path", tmp_path / "model.onnx"
    )
    benchmark_main()
