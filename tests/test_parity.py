import pickle

import numpy as np
import onnxruntime as ort

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import create_features, prepare_feature_dicts


def test_pickle_and_onnx_predictions_are_equal() -> None:
    """Verify prediction parity between Pickle and ONNX."""

    with open(settings.model_path, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    vectorizer = artifact["vectorizer"]

    df = load_data()

    df = create_features(df)

    _, df_val = split_data(df)

    df_sample = df_val.head(500)

    feature_dicts = prepare_feature_dicts(df_sample)

    X = vectorizer.transform(feature_dicts)

    # Pickle model prediction
    pred_pkl = model.predict(X)

    # ONNX model prediction
    session = ort.InferenceSession(
        str(settings.onnx_model_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    pred_onnx = session.run(
        None,
        {input_name: X.astype(np.float32).toarray()},
    )[0]

    pred_onnx = np.asarray(pred_onnx).reshape(-1)

    assert np.allclose(
        pred_pkl,
        pred_onnx,
        atol=1e-4,
    )
