import logging
import pickle
import time

import numpy as np
import onnxruntime as ort

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import create_features, prepare_feature_dicts
from prodml.logging_conf import configure_logging

logger = logging.getLogger(__name__)


def benchmark_pickle(
    model,
    X,
    runs: int = 100,
) -> tuple[float, float]:
    """Benchmark Pickle model inference."""

    latencies = []

    for _ in range(runs):
        start = time.perf_counter()

        model.predict(X)

        elapsed = (time.perf_counter() - start) * 1000

        latencies.append(elapsed)

    return (
        float(np.mean(latencies)),
        float(np.percentile(latencies, 95)),
    )


def benchmark_onnx(
    session,
    input_name: str,
    X: np.ndarray,
    runs: int = 100,
) -> tuple[float, float]:
    """Benchmark ONNX model inference."""

    latencies = []

    for _ in range(runs):
        start = time.perf_counter()

        session.run(
            None,
            {input_name: X},
        )

        elapsed = (time.perf_counter() - start) * 1000

        latencies.append(elapsed)

    return (
        float(np.mean(latencies)),
        float(np.percentile(latencies, 95)),
    )


def main() -> None:
    """Benchmark Pickle and ONNX inference."""

    with open(
        settings.model_path,
        "rb",
    ) as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    vectorizer = artifact["vectorizer"]

    df = load_data()

    df = create_features(df)

    _, df_val = split_data(df)

    df_sample = df_val.head(500)

    feature_dicts = prepare_feature_dicts(df_sample)

    X_sparse = vectorizer.transform(feature_dicts)

    X_dense = X_sparse.astype(np.float32).toarray()

    session = ort.InferenceSession(
        str(settings.onnx_model_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    pkl_mean, pkl_p95 = benchmark_pickle(
        model,
        X_sparse,
    )

    onnx_mean, onnx_p95 = benchmark_onnx(
        session,
        input_name,
        X_dense,
    )
    logger.info(
        "Pickle benchmark: mean_latency_ms=%.4f p95_latency_ms=%.4f",
        pkl_mean,
        pkl_p95,
    )

    logger.info(
        "ONNX benchmark: mean_latency_ms=%.4f p95_latency_ms=%.4f",
        onnx_mean,
        onnx_p95,
    )


if __name__ == "__main__":
    configure_logging()
    main()
