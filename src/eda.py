"""
eda.py
------
Descriptive, diagnostic, and correlation analysis (Ch. 3.5 of the thesis):
 - time series plots comparing NASA vs ground-truth temperature
 - correlation heatmap of candidate features against the target
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")


def plot_time_series(df: pd.DataFrame, station: str, out_name: str = "time_series_comparison.png"):
    os.makedirs(FIG_DIR, exist_ok=True)
    sub = df[df["station"] == station].sort_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sub.index, sub["nasa_temperature_c"], label="NASA POWER T2M", alpha=0.8)
    ax.plot(sub.index, sub["gt_temperature_c"], label="Ground truth (station)", alpha=0.8)
    ax.set_title(f"Hourly Temperature: NASA POWER vs Ground Truth ({station})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Temperature (\u00b0C)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(FIG_DIR, out_name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_correlation_heatmap(df: pd.DataFrame, feature_cols, target_col, out_name="correlation_heatmap.png"):
    os.makedirs(FIG_DIR, exist_ok=True)
    corr = df[feature_cols + [target_col]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap: Features vs Ground-Truth Temperature")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, out_name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, corr


if __name__ == "__main__":
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "data", "processed", "merged_hourly_dataset.csv"),
        index_col=0,
        parse_dates=True,
    )
    feature_cols = [
        "nasa_temperature_c",
        "nasa_relative_humidity",
        "nasa_solar_irradiance",
        "nasa_wind_speed_ms",
        "lat",
        "lon",
    ]
    p1 = plot_time_series(df, "astal_uyo")
    p2, corr = plot_correlation_heatmap(df, feature_cols, "gt_temperature_c")
    print("Saved:", p1)
    print("Saved:", p2)
    print(corr["gt_temperature_c"].sort_values(ascending=False))
