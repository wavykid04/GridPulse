"""
MODULE 1: Data Loading & Preprocessing
=======================================
Handles chunked loading, type downcasting, missing value imputation,
and datetime indexing for the UCI Household Power Consumption dataset.
"""

import pandas as pd
import numpy as np
import os
import psutil
import warnings
warnings.filterwarnings("ignore")


def ram_used_gb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def print_ram(label: str = "") -> None:
    print(f"  [RAM] {label}: {ram_used_gb():.2f} GB used")


RAW_COLUMNS = [
    "Date", "Time",
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

DTYPE_MAP = {
    "Global_active_power":    "float32",
    "Global_reactive_power":  "float32",
    "Voltage":                "float32",
    "Global_intensity":       "float32",
    "Sub_metering_1":         "float32",
    "Sub_metering_2":         "float32",
    "Sub_metering_3":         "float32",
}


def load_dataset(filepath: str, chunksize: int = 200_000) -> pd.DataFrame:
    """
    Stream-load the UCI dataset in chunks.

    - Delimiter: semicolon
    - Missing values: '?' replaced with NaN
    - Types: Date/Time as strings initially, numerics as float32
    - Each chunk: cleaned, interpolated, physically clipped before concat
    """
    print(f"\n{'='*60}")
    print("MODULE 1 — Data Loading & Preprocessing")
    print(f"{'='*60}")
    print(f"  Source : {filepath}")
    print_ram("before load")

    chunks, total_rows = [], 0

    reader = pd.read_csv(
        filepath,
        sep=";",
        header=0,
        names=RAW_COLUMNS,
        na_values=["?", "NA", ""],
        dtype={"Date": str, "Time": str, **{k: str for k in DTYPE_MAP}},
        chunksize=chunksize,
        low_memory=False,
    )

    for i, chunk in enumerate(reader):
        chunk = _clean_chunk(chunk)
        chunks.append(chunk)
        total_rows += len(chunk)
        if (i + 1) % 5 == 0:
            print_ram(f"after chunk {i+1} ({total_rows:,} rows)")

    df = pd.concat(chunks, axis=0)
    del chunks

    size_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"\n  Loaded {len(df):,} rows × {df.shape[1]} columns ({size_mb:.0f} MB)")
    print_ram("after concat")
    return df


def _clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Per-chunk cleaning: parse datetime, cast types, interpolate, clip."""
    # 1. Datetime index
    chunk["Datetime"] = pd.to_datetime(
        chunk["Date"] + " " + chunk["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    chunk = chunk.drop(columns=["Date", "Time"])
    chunk = chunk.dropna(subset=["Datetime"])
    chunk = chunk.set_index("Datetime").sort_index()

    # 2. Cast to float32
    for col in DTYPE_MAP:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype("float32")

    # 3. Interpolate short NaN runs (up to 5 consecutive)
    chunk = chunk.interpolate(method="linear", limit=5, limit_direction="both")

    # 4. Physical bounds clipping
    chunk["Voltage"]               = chunk["Voltage"].clip(0, 300)
    chunk["Global_intensity"]      = chunk["Global_intensity"].clip(0, 100)
    chunk["Global_active_power"]   = chunk["Global_active_power"].clip(0, 20)
    chunk["Global_reactive_power"] = chunk["Global_reactive_power"].clip(0, 20)
    for sm in ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]:
        chunk[sm] = chunk[sm].clip(0, 9000)

    return chunk


def assign_meter_ids(df: pd.DataFrame, n_meters: int = 20, seed: int = 42) -> pd.DataFrame:
    """
    Simulate multiple meters from a single-household dataset.
    Injects synthetic theft patterns into 3 random meters.
    Replace with real meter_id column in production.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["Meter_ID"] = rng.integers(1, n_meters + 1, size=len(df)).astype("int16")

    theft_meters = rng.choice(range(1, n_meters + 1), size=3, replace=False)
    mask  = df["Meter_ID"].isin(theft_meters)
    scale = rng.uniform(0.2, 0.5, size=mask.sum()).astype("float32")
    df.loc[mask, "Global_active_power"] *= scale
    print(f"\n  Synthetic theft injected into Meter IDs: {sorted(theft_meters)}")
    return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-minute readings to hourly means per Meter_ID."""
    print("\n  Resampling to hourly …")
    df_hourly = (
        df.groupby("Meter_ID")
          .resample("1h")
          .mean(numeric_only=True)
          .reset_index()
          .set_index("Datetime")
    )
    print(f"  Hourly shape: {df_hourly.shape}")
    print_ram("after resample")
    return df_hourly


def prepare_data(filepath: str, chunksize: int = 200_000, n_meters: int = 20) -> pd.DataFrame:
    """Full pipeline: raw CSV → clean hourly DataFrame with Meter_ID."""
    df = load_dataset(filepath, chunksize=chunksize)
    df = assign_meter_ids(df, n_meters=n_meters)
    df_hourly = resample_hourly(df)
    del df
    return df_hourly
