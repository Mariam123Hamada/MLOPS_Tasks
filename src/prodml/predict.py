# src/prodml/predict.py

import pickle
import time

from pathlib import Path
from typing import Any, Callable


def timed(func: Callable[..., float]) -> Callable[..., float]:
    """Measure and print function execution time."""

    def wrapper(*args: Any, **kwargs: Any) -> float:
        start = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - start

        print(f"{func.__name__} executed in " f"{elapsed:.6f} seconds")

        return result

    return wrapper


class DurationPredictor:
    """Interface for loading and using the duration prediction model."""

    def __init__(
        self,
        model: Any,
        vectorizer: Any,
    ) -> None:
        self.model = model
        self.vectorizer = vectorizer

    @classmethod
    def load(
        cls,
        model_path: str | Path,
    ) -> "DurationPredictor":
        """Load a trained model from disk."""

        with open(model_path, "rb") as f:
            artifact = pickle.load(f)

        return cls(
            model=artifact["model"],
            vectorizer=artifact["vectorizer"],
        )

    @timed
    def predict_one(
        self,
        features: dict[str, Any],
    ) -> float:
        """Predict duration for one trip."""

        X = self.vectorizer.transform([features])

        prediction = self.model.predict(X)[0]

        return float(prediction)

    def predict_batch(
        self,
        features: list[dict[str, Any]],
    ) -> list[float]:
        """Predict duration for multiple trips."""

        X = self.vectorizer.transform(features)

        predictions = self.model.predict(X)

        return [float(prediction) for prediction in predictions]
