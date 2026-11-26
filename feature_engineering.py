"""
MODULE 2: Feature Engineering — 7 Heuristic Rules
===================================================
Each rule produces a normalised score in [0, 1] where
1.0 = highest suspicion of theft.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

NEGATIVE_FLAG_COL = "Raw_negative_flag"


def _minmax_series(s: pd.Series, clip_upper: float = None) -> pd.Series:
    if clip_upper is not None:
        s = s.clip(upper=clip_upper)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.0, index=s.index, dtype="float32")
    return ((s - mn) / (mx - mn)).fillna(0).clip(0, 1).astype("float32")


def _safe_rolling(series: pd.Series, window: int, func: str = "mean") -> pd.Series:
    r = series.rolling(window=window, min_periods=1)
    return getattr(r, func)()


def rule_sudden_drop(df: pd.DataFrame, window_hours: int = 24) -> pd.Series:
    """R1: Sudden Drop — rolling mean drop vs previous 24 h."""
    gap = (
        df.groupby("Meter_ID")["Global_active_power"]
          .transform(lambda x: _safe_rolling(x.shift(1), window_hours) - x)
    )
    gap = gap.clip(lower=0)
    return _minmax_series(gap, clip_upper=gap.quantile(0.995))


def rule_irregular_tou(df: pd.DataFrame) -> pd.Series:
    """R2: Irregular Time-of-Use - rolling 1-5 AM variance, normalised to [0, 1]."""
    var = df.groupby("Meter_ID")["Global_active_power"].transform(
        lambda x: x.where(x.index.hour.isin([1, 2, 3, 4, 5]))
                   .rolling(window=7 * 24, min_periods=1)
                   .var()
                   .fillna(0)
    )
    var = var.fillna(0)
    return _minmax_series(var, clip_upper=var.quantile(0.995))


def rule_flat_readings(df: pd.DataFrame, window: int = 24, tol: float = 1e-4) -> pd.Series:
    """R3: Flat Readings — consecutive identical readings over rolling window."""
    def _flat_ratio(x: pd.Series) -> pd.Series:
        diff = x.diff().abs()
        is_flat = (diff < tol).astype("float32")
        return is_flat.rolling(window, min_periods=1).mean()

    flat = (
        df.groupby("Meter_ID")["Global_active_power"]
          .transform(_flat_ratio)
    )
    return _minmax_series(flat)


def rule_negative_reset(df: pd.DataFrame) -> pd.Series:
    """R4: Negative / non-physical values — flags for tampered meters.

    Reads the raw-negative flag computed before physical clipping
    (see data_prep._clean_chunk). Falls back to checking the power column
    directly when the flag is absent (e.g. unclipped input).
    """
    if NEGATIVE_FLAG_COL in df.columns:
        flag = (df[NEGATIVE_FLAG_COL].fillna(0) > 0).astype("float32")
    else:
        flag = (df["Global_active_power"] < 0).astype("float32")
    propagated = df.assign(_neg=flag.values).groupby("Meter_ID")["_neg"].transform(
        lambda x: x.rolling(24, min_periods=1).max()
    )
    return propagated.fillna(0).clip(0, 1).astype("float32")


def rule_vi_mismatch(df: pd.DataFrame, window: int = 24) -> pd.Series:
    """R5: V×I vs Active Power mismatch — correlation check."""
    df = df.copy()
    df["vi_product"] = (df["Voltage"] * df["Global_intensity"]).astype("float32")

    out = np.zeros(len(df), dtype="float32")
    meter = df["Meter_ID"].values
    for m in pd.unique(meter):
        pos = np.flatnonzero(meter == m)
        grp = df.iloc[pos]
        corr = (
            grp["vi_product"].reset_index(drop=True)
              .rolling(window, min_periods=3)
              .corr(grp["Global_active_power"].reset_index(drop=True))
        )
        out[pos] = ((1 - corr) / 2).clip(0, 1).fillna(0).to_numpy(dtype="float32")

    return pd.Series(out, index=df.index, dtype="float32")


def rule_peer_comparison(df: pd.DataFrame) -> pd.Series:
    """R6: Peer Comparison — deviation from dataset-wide hourly mean."""
    df = df.copy()
    df["_hour"] = df.index.hour
    hourly_mean = df.groupby("_hour")["Global_active_power"].transform("mean")
    deviation = (df["Global_active_power"] - hourly_mean).abs()
    return _minmax_series(deviation, clip_upper=deviation.quantile(0.995))


def rule_sudden_pattern_change(df: pd.DataFrame, window: int = 168) -> pd.Series:
    """R7: Sudden Pattern Change — rolling Z-score of consumption."""
    def _zscore(x: pd.Series) -> pd.Series:
        mu  = x.rolling(window, min_periods=24).mean()
        sig = x.rolling(window, min_periods=24).std().replace(0, np.nan)
        z   = ((x - mu) / sig).abs()
        return z

    z = (
        df.groupby("Meter_ID")["Global_active_power"]
          .transform(_zscore)
    )
    return _minmax_series(z.fillna(0), clip_upper=z.quantile(0.99))


RULE_NAMES = [
    "R1_sudden_drop",
    "R2_irregular_tou",
    "R3_flat_readings",
    "R4_negative_reset",
    "R5_vi_mismatch",
    "R6_peer_comparison",
    "R7_pattern_change",
]

RULE_WEIGHTS = {
    "R1_sudden_drop":     0.20,
    "R2_irregular_tou":   0.10,
    "R3_flat_readings":   0.15,
    "R4_negative_reset":  0.20,
    "R5_vi_mismatch":     0.15,
    "R6_peer_comparison": 0.10,
    "R7_pattern_change":  0.10,
}

RULE_DESCRIPTIONS = {
    "R1_sudden_drop":     "Sudden consumption drop (possible bypass)",
    "R2_irregular_tou":   "Unusual activity during low-usage hours (1–5 AM)",
    "R3_flat_readings":   "Flat/stuck meter readings",
    "R4_negative_reset":  "Non-physical/negative readings (meter tampering)",
    "R5_vi_mismatch":     "Voltage×Current vs Active Power mismatch",
    "R6_peer_comparison": "Abnormal deviation from peer-group usage",
    "R7_pattern_change":  "Sudden consumption pattern change (Z-score anomaly)",
}


def compute_all_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Run all 7 rules and return feature DataFrame with Rule_Score."""
    print("\n" + "="*60)
    print("MODULE 2 — Feature Engineering (7 Rules)")
    print("="*60)

    features = pd.DataFrame(index=df.index)
    features["Meter_ID"] = df["Meter_ID"]

    steps = [
        ("R1: Sudden Drop",      "R1_sudden_drop",     rule_sudden_drop),
        ("R2: Irregular TOU",    "R2_irregular_tou",   rule_irregular_tou),
        ("R3: Flat Readings",    "R3_flat_readings",   rule_flat_readings),
        ("R4: Negative/Reset",   "R4_negative_reset",  rule_negative_reset),
        ("R5: V/I Mismatch",     "R5_vi_mismatch",     rule_vi_mismatch),
        ("R6: Peer Comparison",  "R6_peer_comparison", rule_peer_comparison),
        ("R7: Pattern Change",   "R7_pattern_change",  rule_sudden_pattern_change),
    ]

    for label, col, fn in steps:
        print(f"  {label} …")
        features[col] = fn(df)

    rule_cols = list(RULE_NAMES)
    weights   = np.array([RULE_WEIGHTS[r] for r in rule_cols], dtype="float32")
    rs = (features[rule_cols].values @ weights).astype("float32")
    # Rule weights sum to 1 and every rule is in [0, 1], so the weighted sum is
    # already in [0, 1]. Use it directly instead of batch min-max scaling, which
    # made the score meaningless across runs and collapsed it to 0 when constant.
    features["Rule_Score"] = np.clip(np.nan_to_num(rs), 0, 1).astype("float32")

    print(f"\n  Feature matrix shape: {features.shape}")
    print(f"  Rule_Score — mean={features['Rule_Score'].mean():.3f}, max={features['Rule_Score'].max():.3f}")
    return features
