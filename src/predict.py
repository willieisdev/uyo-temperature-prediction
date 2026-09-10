"""
predict.py
----------
Minimal example: load the persisted best model + scaler and predict
ground-truth temperature from NASA POWER-style inputs.

Usage:
    python src/predict.py
"""

import os

import joblib
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")


def load_artifact(path: str = MODEL_PATH):
    return joblib.load(path)


def predict(artifact, nasa_temperature_c, nasa_relative_humidity,
            nasa_solar_irradiance, nasa_wind_speed_ms, lat, lon):
    row = pd.DataFrame([{
        "nasa_temperature_c": nasa_temperature_c,
        "nasa_relative_humidity": nasa_relative_humidity,
        "nasa_solar_irradiance": nasa_solar_irradiance,
        "nasa_wind_speed_ms": nasa_wind_speed_ms,
        "lat": lat,
        "lon": lon,
    }])[artifact["feature_cols"]]
    scaled = artifact["scaler"].transform(row)
    return float(artifact["model"].predict(scaled)[0])


if __name__ == "__main__":
    artifact = load_artifact()
    example = dict(
        nasa_temperature_c=27.4,
        nasa_relative_humidity=68.0,
        nasa_solar_irradiance=1.2,
        nasa_wind_speed_ms=1.5,
        lat=5.0419,
        lon=7.9762,
    )
    pred = predict(artifact, **example)
    print(f"Model used: {artifact['model_name']}")
    print(f"Input: {example}")
    print(f"Calibrated ground-truth temperature prediction: {pred:.2f} \u00b0C")
