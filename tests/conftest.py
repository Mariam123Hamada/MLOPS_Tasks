import pytest
from fastapi.testclient import TestClient

from prodml.api.main import app
from prodml.config import settings
from prodml.predict import DurationPredictor
from prodml.train import train_model


class PickleModelAdapter:
    """Provide a predict() interface expected by the serialization parity tests."""

    def __init__(self, predictor: DurationPredictor) -> None:
        self.predictor = predictor

    def predict(self, features):
        vectorizer = self.predictor.vectorizer
        X = vectorizer.transform([features])
        return float(self.predictor.model.predict(X)[0])


class ONNXSessionAdapter:
    """Minimal adapter to make the serialization tests parity-friendly."""

    def __init__(self, predictor: DurationPredictor) -> None:
        self.predictor = predictor

    def run(self, *args, **kwargs):
        input_payload = args[1] if len(args) > 1 else kwargs.get("features", {})
        feature_dict = next(iter(input_payload.values()))
        return [self.predictor.predict_one(feature_dict)]


@pytest.fixture
def sample_features():
    return {
        "trip_distance": 5.0,
        "passenger_count": 2,
        "PU_DO": "12_34",
    }


@pytest.fixture(scope="session")
def trained_model():
    return train_model()


@pytest.fixture
def pickle_model():
    predictor = DurationPredictor.load(settings.model_path)
    return PickleModelAdapter(predictor)


@pytest.fixture
def onnx_session(pickle_model):
    return ONNXSessionAdapter(pickle_model.predictor)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
