def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200


def test_predict_happy_path(client):
    payload = {
        "trip_distance": 5.0,
        "passenger_count": 2,
        "PU_DO": "12_34",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200


def test_predict_with_mocked_model(client, monkeypatch):
    def fake_predict(*args, **kwargs):
        return 42.5

    monkeypatch.setattr(
        "prodml.api.main.predict",
        fake_predict,
    )

    payload = {
        "trip_distance": 5.0,
        "passenger_count": 2,
        "PU_DO": "12_34",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["prediction"] == 42.5
