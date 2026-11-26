import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def hourly_df():
    """Hourly per-meter frame shaped like data_prep.prepare_data output."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=24 * 14, freq="1h")
    frames = []
    for meter in (1, 2, 3):
        hours = idx.hour.values
        base = 1.0 + 0.8 * np.sin(hours / 24 * 2 * np.pi) + rng.normal(0, 0.2, len(idx))
        power = np.clip(base, 0, None).astype("float32")
        if meter == 2:
            power[200:230] = 0.3  # flat / sudden drop segment
        voltage = (240 + rng.normal(0, 2, len(idx))).astype("float32")
        intensity = (power * 1000 / voltage).astype("float32")
        frames.append(pd.DataFrame({
            "Meter_ID": np.int16(meter),
            "Global_active_power": power,
            "Global_reactive_power": np.float32(0.1),
            "Voltage": voltage,
            "Global_intensity": intensity,
            "Raw_negative_flag": np.float32(0.0),
        }, index=idx))
    df = pd.concat(frames)
    df.index.name = "Datetime"
    return df


def write_uci_file(path, n_minutes=3 * 24 * 60, negative_rows=()):
    """Write a small file in the UCI household_power_consumption format."""
    rng = np.random.default_rng(1)
    ts = pd.date_range("2024-01-01", periods=n_minutes, freq="1min")
    power = np.abs(1.0 + 0.5 * np.sin(np.arange(n_minutes) / 300) + rng.normal(0, 0.1, n_minutes))
    power[list(negative_rows)] = -1.5
    voltage = 240 + rng.normal(0, 2, n_minutes)
    df = pd.DataFrame({
        "Date": ts.strftime("%d/%m/%Y"),
        "Time": ts.strftime("%H:%M:%S"),
        "Global_active_power": power.round(3),
        "Global_reactive_power": 0.1,
        "Voltage": voltage.round(2),
        "Global_intensity": (np.abs(power) * 1000 / voltage).round(2),
        "Sub_metering_1": 0.0,
        "Sub_metering_2": 1.0,
        "Sub_metering_3": 17.0,
    })
    df.to_csv(path, sep=";", index=False)
    return path
