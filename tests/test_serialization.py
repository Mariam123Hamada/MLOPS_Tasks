import numpy as np


def test_pickle_onnx_parity(
    pickle_model,
    onnx_session,
    sample_features,
):
    pickle_prediction = pickle_model.predict(sample_features)

    onnx_prediction = onnx_session.run(
        None,
        {"features": sample_features},
    )[0]

    np.testing.assert_allclose(
        pickle_prediction,
        onnx_prediction,
        rtol=1e-5,
        atol=1e-5,
    )
