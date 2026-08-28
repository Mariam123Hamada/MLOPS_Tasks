## Step 04 — Serialization and Model Portability

### ONNX Export

The trained scikit-learn `LinearRegression` model was exported from the trusted Pickle artifact to ONNX format using `skl2onnx`. The model accepts a dynamic batch dimension, allowing inference on different numbers of rows while keeping the feature dimension fixed.

The existing `DictVectorizer` remains responsible for transforming raw feature dictionaries into the numerical feature matrix required by both the Pickle and ONNX models.

### Prediction Parity

A parity test was performed using 500 validation rows. Predictions from the original Pickle model and the exported ONNX model were compared using:

`np.allclose(pred_pkl, pred_onnx, atol=1e-4)`

The parity test passed, confirming that the ONNX export preserves the model's predictions within the specified numerical tolerance.

### Inference Benchmark

Both models were benchmarked on the same batch of 500 validation rows over multiple inference runs.

| Format                | Mean Latency (ms) | P95 Latency (ms) |
| --------------------- | ----------------: | ---------------: |
| Pickle / scikit-learn |      1.2349       |     1.4357 |
| ONNX / ONNX Runtime   |      4.9787       |     19.4300|

The latency values above were measured on the local development machine and should not be generalized to other hardware or deployment environments.

### Serialization Format Comparison

| Format   | Human-readable | Cross-language | Schema-enforced                 | Safe to load from an untrusted source    |
| -------- | -------------- | -------------- | ------------------------------- | ---------------------------------------- |
| JSON     | Yes            | Yes            | No, unless validated separately | Generally yes                            |
| Protobuf | No             | Yes            | Yes                             | Generally yes when using trusted parsers |
| Pickle   | No             | No             | No                              | No                                       |
| ONNX     | No             | Yes            | Yes                             | Yes as a model representation format     |

### Serving Decision

The service uses Pickle for the current scikit-learn and `DictVectorizer` artifact because it preserves the complete Python inference pipeline, while ONNX is exported as a portable runtime representation for parity testing and performance evaluation.

### Security Warning

Pickle can execute arbitrary code during deserialization. Therefore, a `.pkl` file must never be loaded from an untrusted source. Only Pickle artifacts produced and controlled by the project should be loaded.
