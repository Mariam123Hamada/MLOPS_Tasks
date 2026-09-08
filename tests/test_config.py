from prodml.config import settings


def test_default_mlflow_tracking_uri_uses_local_file_store():
    assert settings.mlflow_tracking_uri.startswith("file://")
    assert "localhost:5000" not in settings.mlflow_tracking_uri
