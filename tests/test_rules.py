import numpy as np
import pytest

from feature_engineering import (
    RULE_NAMES, compute_all_rules, rule_irregular_tou, rule_negative_reset,
)
from data_prep import _clean_chunk, load_dataset
from conftest import write_uci_file


def test_every_rule_in_unit_interval(hourly_df):
    features = compute_all_rules(hourly_df)
    for col in RULE_NAMES + ["Rule_Score"]:
        s = features[col]
        assert not s.isna().any(), col
        assert s.min() >= 0.0 and s.max() <= 1.0, (col, s.min(), s.max())


def test_r2_normalised_even_with_large_variance(hourly_df):
    df = hourly_df.copy()
    night = df.index.hour.isin([1, 2, 3, 4, 5])
    df.loc[night, "Global_active_power"] *= 50
    r2 = rule_irregular_tou(df)
    assert r2.min() >= 0 and r2.max() <= 1
    assert r2.max() > 0


def test_r4_fires_on_raw_negative_flag(hourly_df):
    df = hourly_df.copy()
    df.iloc[10, df.columns.get_loc("Raw_negative_flag")] = 1.0
    r4 = rule_negative_reset(df)
    assert r4.max() > 0
    assert r4.min() >= 0 and r4.max() <= 1


def test_negative_flag_computed_before_clipping(tmp_path):
    path = write_uci_file(tmp_path / "data.txt", n_minutes=600, negative_rows=range(100, 110))
    df = load_dataset(str(path))
    assert (df["Global_active_power"] >= 0).all()  # still clipped
    assert df["Raw_negative_flag"].sum() == 10


def test_r4_end_to_end_with_negative_readings(tmp_path):
    from data_prep import prepare_data
    path = write_uci_file(tmp_path / "data.txt", negative_rows=range(500, 560))
    df = prepare_data(str(path), n_meters=3)
    features = compute_all_rules(df)
    assert features["R4_negative_reset"].max() > 0
    for col in RULE_NAMES:
        assert features[col].between(0, 1).all(), col


def test_rule_score_not_collapsed_when_constant(hourly_df):
    df = hourly_df.copy()
    df["Global_active_power"] = np.float32(1.0)
    df["Raw_negative_flag"] = np.float32(1.0)
    features = compute_all_rules(df)
    # every row equally suspicious on R4: score must reflect it, not be min-max'd to 0
    assert (features["Rule_Score"] > 0).all()
