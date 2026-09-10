"""
data_pipeline.py
-----------------
Loads ground-truth IoT weather station data and matching NASA POWER
hourly reanalysis data for four study sites, cleans and resamples them
to hourly resolution, merges NASA (features) with station (target)
data, and applies linear interpolation for missing values.

Study sites (Akwa Ibom / Nigeria region):
    astal_uyo     5.0419N, 7.9762E
    daringpotato  5.01N,   7.95E
    dauntedlemon  5.02N,   7.93E
    dazetrumpet   5.02N,   7.93E
"""

import glob
import os

import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

STATIONS = {
    "astal_uyo": {"lat": 5.0419, "lon": 7.9762},
    "daringpotato": {"lat": 5.01, "lon": 7.95},
    "dauntedlemon": {"lat": 5.02, "lon": 7.93},
    "dazetrumpet": {"lat": 5.02, "lon": 7.93},
}


def _find_file(folder: str, pattern: str) -> str:
    matches = glob.glob(os.path.join(RAW_DIR, folder, pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} in {folder}")
    return matches[0]


def load_nasa_power(folder: str) -> pd.DataFrame:
    """Load a NASA POWER hourly CSV, skipping the metadata header block."""
    path = _find_file(folder, "POWER_Point_Hourly_*.csv")
    with open(path, "r") as f:
        lines = f.readlines()
    header_idx = next(i for i, l in enumerate(lines) if l.startswith("YEAR,MO,DY,HR"))
    df = pd.read_csv(path, skiprows=header_idx)
    df["timestamp"] = pd.to_datetime(
        dict(year=df["YEAR"], month=df["MO"], day=df["DY"], hour=df["HR"])
    )
    df = df.rename(
        columns={
            "ALLSKY_SFC_SW_DWN": "nasa_solar_irradiance",
            "T2M": "nasa_temperature_c",
            "RH2M": "nasa_relative_humidity",
            "WS2M": "nasa_wind_speed_ms",
            "WD2M": "nasa_wind_direction_deg",
        }
    )
    keep = ["timestamp"] + [c for c in df.columns if c.startswith("nasa_")]
    df = df[keep].set_index("timestamp").sort_index()
    df = df.replace(-999, np.nan)
    return df


def load_station(folder: str) -> pd.DataFrame:
    """Load a ground-truth IoT station CSV (blockchain sensor log) and
    resample the irregular readings to an hourly mean."""
    path = _find_file(folder, "actor_*.csv")
    usecols_candidates = [
        "timestamp",
        "act.data.temperature_c",
        "act.data.humidity_percent",
        "act.data.pressure_hpa",
        "act.data.wind_spd_ms",
        "act.data.wind_dir_deg",
        "act.data.rain_1hr_mm",
        "act.data.light_intensity_lux",
        "act.data.uv_index",
    ]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in usecols_candidates if c in header]
    df = pd.read_csv(path, usecols=usecols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.rename(columns={c: c.replace("act.data.", "gt_") for c in df.columns})
    df = df.set_index("timestamp").sort_index()
    # Sensor pushes come in every ~15 minutes and out of exact order; resample to hourly mean.
    hourly = df.resample("h").mean()
    return hourly


def build_station_dataset(folder: str) -> pd.DataFrame:
    """Merge one station's ground truth with NASA POWER data on the hour,
    tag with station name and coordinates."""
    nasa = load_nasa_power(folder)
    station = load_station(folder)
    merged = nasa.join(station, how="inner")
    merged["station"] = folder
    merged["lat"] = STATIONS[folder]["lat"]
    merged["lon"] = STATIONS[folder]["lon"]
    return merged


def clean_and_interpolate(df: pd.DataFrame, max_gap_hours: int = 3) -> pd.DataFrame:
    """Drop fully-empty optional sensor columns. Linearly interpolate ONLY
    short genuine sensor micro-outages (runs of at most `max_gap_hours`
    consecutive missing hours), per station, in time order. Longer gaps are
    real data outages (some spanning days-to-weeks in this dataset) and are
    left as NaN so they get dropped rather than fabricated.

    Also adds a `gt_interpolated` flag marking any row whose ground-truth
    columns were filled rather than measured, so it can be audited or
    excluded downstream.
    """
    df = df.copy()
    # Drop columns that are entirely missing for this station (e.g. stations
    # without pressure/wind/rain/light/UV sensors).
    df = df.dropna(axis=1, how="all")
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    parts = []
    for station_name, group in df.groupby("station"):
        group = group.sort_index()
        was_na = group[numeric_cols].isna()
        # limit=max_gap_hours with no limit_direction/limit_area means: only
        # fill runs of up to max_gap_hours consecutive NaNs, from either
        # side, bounded by real values on both ends -- no boundary
        # extrapolation across an open-ended gap.
        group[numeric_cols] = group[numeric_cols].interpolate(
            method="linear", limit=max_gap_hours, limit_area="inside"
        )
        group["gt_interpolated"] = (was_na & group[numeric_cols].notna()).any(axis=1)
        parts.append(group)
    df = pd.concat(parts, axis=0).sort_index()
    return df


def build_full_dataset() -> pd.DataFrame:
    """Build, clean, and concatenate the dataset for all four stations."""
    frames = []
    for folder in STATIONS:
        merged = build_station_dataset(folder)
        frames.append(merged)
    full = pd.concat(frames, axis=0).sort_index()
    full = clean_and_interpolate(full)
    full = full.dropna(subset=["gt_temperature_c"])  # target must be present
    return full


if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    dataset = build_full_dataset()
    out_path = os.path.join(PROCESSED_DIR, "merged_hourly_dataset.csv")
    dataset.to_csv(out_path)
    print(f"Saved {len(dataset)} rows x {dataset.shape[1]} cols -> {out_path}")
    print(dataset.head())
    print(dataset["station"].value_counts())
