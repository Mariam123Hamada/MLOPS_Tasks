import pickle
from pathlib import Path

import numpy as np
from skl2onnx import to_onnx

from prodml.config import settings


def load_artifact(
    model_path: str | Path,
) -> dict:
    """Load the trusted Pickle model artifact."""

    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    return artifact


def export_to_onnx(
    model_path: str | Path,
    onnx_path: str | Path,
) -> None:
    """Export the fitted scikit-learn model to ONNX."""

    artifact = load_artifact(model_path)

    model = artifact["model"]
    vectorizer = artifact["vectorizer"]

    sample = vectorizer.transform(
        [
            {
                "PU_DO": "1_2",
                "trip_distance": 5.0,
            }
        ]
    )

    # DictVectorizer returns a sparse matrix.
    # skl2onnx needs a dense NumPy array here.
    sample_dense = sample.toarray().astype(np.float32)

    onnx_model = to_onnx(
        model,
        sample_dense,
        target_opset=17,
    )

    onnx_path = Path(onnx_path)

    onnx_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())


def main() -> None:
    """Export the trained model to ONNX."""

    export_to_onnx(
        model_path=settings.model_path,
        onnx_path=settings.onnx_model_path,
    )


if __name__ == "__main__":
    main()
