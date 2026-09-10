"""
modeling.py
-----------
Predictive, prescriptive, and performance analysis (Ch. 3.6 of the thesis):
 - 80/20 chronological train/test split
 - StandardScaler feature scaling
 - XGBoost (GridSearchCV-tuned), Random Forest, Linear Regression
 - Stacking ensemble (XGBoost + RF + LinearRegression -> LinearRegression meta-learner)
 - MAE / RMSE evaluation
 - persist trained pipeline as a .pkl for downstream use
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "merged_hourly_dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

FEATURE_COLS = [
    "nasa_temperature_c",
    "nasa_relative_humidity",
    "nasa_solar_irradiance",
    "nasa_wind_speed_ms",
    "lat",
    "lon",
]
TARGET_COL = "gt_temperature_c"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    return df.sort_index()


def chronological_split(df: pd.DataFrame, train_frac: float = 0.8):
    n_train = int(len(df) * train_frac)
    train, test = df.iloc[:n_train], df.iloc[n_train:]
    return train, test


def evaluate(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": round(float(mae), 4),
        "MSE": round(float(mse), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
    }


def train_and_evaluate():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    df = load_dataset()
    train_df, test_df = chronological_split(df)
    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}

    # --- Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    results["LinearRegression"] = evaluate(y_test, lr.predict(X_test_s))

    # --- Random Forest ---
    # n_estimators/max_depth capped to keep the persisted model portfolio-sized
    # (an uncapped 300-tree forest serialized to ~450MB with negligible RMSE gain).
    rf = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    results["RandomForest"] = evaluate(y_test, rf.predict(X_test_s))

    # --- XGBoost with GridSearchCV (small grid around thesis-reported params) ---
    param_grid = {
        "learning_rate": [0.05, 0.1],
        "max_depth": [5, 7],
        "n_estimators": [150, 200],
        "subsample": [0.8],
        "reg_alpha": [4],
        "reg_lambda": [1],
    }
    tscv = TimeSeriesSplit(n_splits=3)
    xgb_search = GridSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1),
        param_grid,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        n_jobs=-1,
    )
    xgb_search.fit(X_train_s, y_train)
    best_xgb = xgb_search.best_estimator_
    results["XGBoost"] = evaluate(y_test, best_xgb.predict(X_test_s))
    results["XGBoost_best_params"] = xgb_search.best_params_

    # --- Stacking ensemble: XGBoost + RandomForest + LinearRegression -> LinearRegression meta-learner ---
    stack = StackingRegressor(
        estimators=[
            ("xgb", XGBRegressor(**xgb_search.best_params_, random_state=42, n_jobs=-1)),
            ("rf", RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)),
            ("lr", LinearRegression()),
        ],
        final_estimator=LinearRegression(),
        n_jobs=-1,
    )
    stack.fit(X_train_s, y_train)
    results["StackingEnsemble"] = evaluate(y_test, stack.predict(X_test_s))

    # Persist the best-performing model (lowest test RMSE) + scaler as a single artifact
    candidates = {"LinearRegression": lr, "RandomForest": rf, "XGBoost": best_xgb, "StackingEnsemble": stack}
    best_name = min(
        (k for k in candidates), key=lambda k: results[k]["RMSE"]
    )
    artifact = {
        "model": candidates[best_name],
        "model_name": best_name,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
    }
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(artifact, model_path, compress=3)

    results["n_train"] = len(train_df)
    results["n_test"] = len(test_df)
    results["best_model"] = best_name

    with open(os.path.join(REPORT_DIR, "model_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    res = train_and_evaluate()
    print(json.dumps(res, indent=2))
