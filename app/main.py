"""
Weather Intelligence API

A FastAPI service that forecasts two weather-derived indices for Sydney:

- Climate Comfort Index (CCI) for the next 3 days (regression)
- Weather Hazard Category (WHC) exactly 7 days ahead (classification)

For each request the API downloads recent weather for Sydney from the
Open-Meteo API, builds the same features the models were trained on, and
returns the model's prediction.
"""

from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import requests
from fastapi import FastAPI, HTTPException, Query
from mlkit_aby.helpers import features, targets, weather

GITHUB_URL = "https://github.com/abychopda/weather-intelligence-api"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

app = FastAPI(
    title="Weather Intelligence API",
    description="Machine learning forecasts of weather-derived indices for Sydney.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Load the trained models once, when the API starts
# ---------------------------------------------------------------------------

# Each .joblib file holds the trained model plus the information needed to
# use it: the feature recipe, the feature column order and test metrics.
cci_models = {
    horizon: joblib.load(MODELS_DIR / f"cci_ridge_h{horizon}.joblib")
    for horizon in [1, 2, 3]
}
whc_model = joblib.load(MODELS_DIR / "whc_logistic_h7.joblib")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def parse_date(date_text: str) -> date:
    """Convert a YYYY-MM-DD string into a date, or return a 400 error."""
    try:
        requested = datetime.strptime(date_text, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date '{date_text}'. Please use the format YYYY-MM-DD.",
        )

    if requested > date.today():
        raise HTTPException(
            status_code=400,
            detail="The date cannot be in the future. Predictions are made from "
            "weather that has already been observed.",
        )

    return requested


def get_features_for_date(requested: date, model: dict):
    """
    Download recent weather and build the model's features for one date.

    Only weather up to and including the requested date is downloaded, so
    the model never sees information from the days it is predicting.
    """
    # The model needs a number of days of history (e.g. for 30-day averages).
    start = requested - timedelta(days=model["history_days"] - 1)

    try:
        hourly = weather.fetch_hourly_weather(str(start), str(requested))
    except requests.RequestException:
        raise HTTPException(
            status_code=503,
            detail="The Open-Meteo weather service could not be reached. "
            "Please try again shortly.",
        )

    daily_features = features.build_daily_features(hourly, model["recipe"])
    row = daily_features[daily_features["date"] == str(requested)]
    X = row[model["feature_columns"]]

    # Safety net: if Open-Meteo returns gaps for this date, the model cannot
    # make a reliable prediction.
    if X.empty or X.isna().any(axis=None):
        raise HTTPException(
            status_code=400,
            detail=f"Complete weather data is not available for {requested}. "
            "Please choose an earlier date.",
        )

    return X


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def root():
    """Project overview and list of available endpoints."""
    return {
        "project": "Weather Intelligence API",
        "description": (
            "Machine learning services forecasting weather-derived indices "
            "for Sydney, NSW, using historical data from Open-Meteo."
        ),
        "version": "1.0.0",
        "github": GITHUB_URL,
        "endpoints": [
            "/",
            "/health",
            "/predict/index/comfort_climate",
            "/predict/category/weather_hazard",
            "/model-metadata",
        ],
        "prediction_targets": {
            "comfort_climate": "Climate Comfort Index (0-100) for the next 3 days",
            "weather_hazard": "Weather Hazard Category exactly 7 days ahead",
        },
        "usage": "/predict/index/comfort_climate?date=2025-01-01",
        "interactive_docs": "/docs",
    }


@app.get("/health")
def health():
    """Confirm the API is running and the models are loaded."""
    return {
        "status": "healthy",
        "message": "Weather Intelligence API is running",
        "models_loaded": len(cci_models) + 1,
    }


@app.get("/predict/index/comfort_climate")
def predict_comfort_climate(
    date: str = Query(..., description="Date in YYYY-MM-DD format", examples=["2025-01-01"]),
):
    """Predict the Climate Comfort Index for the 3 days after the given date."""
    requested = parse_date(date)

    # All three horizon models share the same features, so build them once.
    X = get_features_for_date(requested, cci_models[1])

    predictions = {}
    for horizon, model in cci_models.items():
        forecast_date = requested + timedelta(days=horizon)
        cci = model["pipeline"].predict(X)[0]
        # Keep the prediction inside the index's valid 0-100 range.
        predictions[str(forecast_date)] = round(min(max(cci, 0), 100), 1)

    return {
        "input_date": str(requested),
        "predictions": {"comfort_climate": predictions},
    }


@app.get("/predict/category/weather_hazard")
def predict_weather_hazard(
    date: str = Query(..., description="Date in YYYY-MM-DD format", examples=["2025-01-01"]),
):
    """Predict the Weather Hazard Category exactly 7 days after the given date."""
    requested = parse_date(date)
    X = get_features_for_date(requested, whc_model)

    category = int(whc_model["pipeline"].predict(X)[0])
    forecast_date = requested + timedelta(days=7)

    return {
        "input_date": str(requested),
        "predictions": {
            "weather_hazard": {str(forecast_date): targets.WHC_LABELS[category]}
        },
    }


@app.get("/model-metadata")
def model_metadata():
    """Describe the deployed models, their features and their performance."""
    cci = cci_models[1]
    metadata = [
        {
            "target": "Climate Comfort Index",
            "prediction_type": "Regression",
            "algorithm": cci["algorithm"],
            "forecast_horizon": "3 Days (one model per day ahead)",
            "features": cci["feature_columns"],
            "feature_importance": cci["feature_importance"],
            "performance": {
                f"{horizon}_day_ahead": model["metrics"]
                for horizon, model in cci_models.items()
            },
        },
        {
            "target": "Weather Hazard Category",
            "prediction_type": "Multiclass Classification",
            "algorithm": whc_model["algorithm"],
            "forecast_horizon": "7 Days",
            "classes": whc_model["class_names"],
            "features": whc_model["feature_columns"],
            "feature_importance": whc_model["feature_importance"],
            "performance": whc_model["metrics"],
            "limitations": (
                "Experimental. High and Extreme Risk days are rare and are not "
                "reliably detected by the current model; see the per-class "
                "recall figures. Not suitable as a safety input."
            ),
        },
    ]
    return metadata
