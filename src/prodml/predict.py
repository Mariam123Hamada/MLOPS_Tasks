import logging
import pickle
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def timed(func: Callable[..., float]) -> Callable[..., float]:
    """Measure function execution time."""

    def wrapper(*args: Any, **kwargs: Any) -> float:
        start = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - start

        logger.info(
            "Prediction served: function=%s latency_ms=%.3f",
            func.__name__,
            elapsed * 1000,
        )

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

        try:
            with open(model_path, "rb") as f:
                artifact = pickle.load(f)

        except Exception:
            logger.exception(
                "Model load failure: path=%s",
                model_path,
            )
            raise

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
        logger.debug("Feature vector: %s", features)
        trip_distance = features.get("trip_distance")

        if isinstance(trip_distance, (int, float)) and trip_distance > 100:
            logger.warning(
                "Input outside training range: trip_distance=%.2f",
                trip_distance,
            )
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
