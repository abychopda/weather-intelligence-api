"""
Tests for the Weather Intelligence API.

The prediction tests call the real Open-Meteo API, so an internet connection
is required. Run with:  uv run pytest
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# --- Information endpoints ---------------------------------------------------


def test_root_lists_all_endpoints():
    response = client.get("/")
    assert response.status_code == 200
    assert "/predict/index/comfort_climate" in response.json()["endpoints"]
    assert "/predict/category/weather_hazard" in response.json()["endpoints"]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_model_metadata_describes_both_models():
    response = client.get("/model-metadata")
    assert response.status_code == 200
    targets = [model["target"] for model in response.json()]
    assert targets == ["Climate Comfort Index", "Weather Hazard Category"]


# --- Predictions -------------------------------------------------------------


def test_comfort_climate_returns_next_three_days():
    response = client.get("/predict/index/comfort_climate?date=2025-01-01")
    assert response.status_code == 200

    predictions = response.json()["predictions"]["comfort_climate"]
    assert list(predictions) == ["2025-01-02", "2025-01-03", "2025-01-04"]

    # Every prediction must be a valid CCI score.
    for value in predictions.values():
        assert 0 <= value <= 100


def test_weather_hazard_returns_date_seven_days_ahead():
    response = client.get("/predict/category/weather_hazard?date=2025-01-01")
    assert response.status_code == 200

    predictions = response.json()["predictions"]["weather_hazard"]
    assert list(predictions) == ["2025-01-08"]
    assert predictions["2025-01-08"] in [
        "Low Risk", "Moderate Risk", "High Risk", "Extreme Risk"
    ]


def test_works_for_a_2026_date():
    # 2026 data was never used in training, so this checks the API can
    # predict on genuinely new dates.
    response = client.get("/predict/index/comfort_climate?date=2026-03-15")
    assert response.status_code == 200


# --- Error handling ----------------------------------------------------------


def test_wrong_date_format_returns_400():
    response = client.get("/predict/index/comfort_climate?date=15-03-2026")
    assert response.status_code == 400


def test_impossible_date_returns_400():
    response = client.get("/predict/category/weather_hazard?date=2026-02-30")
    assert response.status_code == 400


def test_future_date_returns_400():
    response = client.get("/predict/index/comfort_climate?date=2099-01-01")
    assert response.status_code == 400


def test_missing_date_returns_422():
    response = client.get("/predict/index/comfort_climate")
    assert response.status_code == 422


def test_unknown_endpoint_returns_404():
    response = client.get("/does-not-exist")
    assert response.status_code == 404
