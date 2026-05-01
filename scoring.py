"""
MODULE 4: Scoring, Explainability & Output
==========================================
Blends IF and Rule scores:
    Final_Risk_Score = (0.6 × IF_Score) + (0.4 × Rule_Score)

Records with Final_Risk_Score > 0.7 receive a plain-English
Primary_Reason identifying the dominant rule violation.
"""

import pandas as pd
import numpy as np
from feature_engineering import RULE_NAMES, RULE_DESCRIPTIONS
import warnings
warnings.filterwarnings("ignore")


IF_WEIGHT       = 0.6
RULE_WEIGHT     = 0.4
ALERT_THRESHOLD = 0.7


def compute_final_score(
    iso_scores:  pd.Series,
    rule_scores: pd.Series,
    if_weight:   float = IF_WEIGHT,
    rule_weight: float = RULE_WEIGHT,
) -> pd.Series:
    """Blend IF and Rule scores into Final_Risk_Score ∈ [0, 1]."""
    assert abs(if_weight + rule_weight - 1.0) < 1e-6
    return (if_weight * iso_scores + rule_weight * rule_scores).clip(0, 1) \
             .rename("Final_Risk_Score").astype("float32")


def explain_row(row: pd.Series, threshold: float = ALERT_THRESHOLD) -> str:
    """
    Human-readable explanation for one record.

    Returns 'Below threshold' if score < threshold.
    Otherwise names the dominant rule and includes secondary rules
    that also scored > 0.5.
    """
    if row["Final_Risk_Score"] < threshold:
        return "Below threshold"

    rule_scores = {r: row[r] for r in RULE_NAMES if r in row.index}
    if not rule_scores or max(rule_scores.values()) == 0:
        return "Isolation Forest anomaly (no dominant rule)"

    top_rule  = max(rule_scores, key=rule_scores.get)
    top_score = rule_scores[top_rule]
    desc      = RULE_DESCRIPTIONS.get(top_rule, top_rule)
    if_score  = row.get("Isolation_Forest_Score", float("nan"))

    secondary = [
        f"{r}={rule_scores[r]:.2f}"
        for r in RULE_NAMES
        if r != top_rule and rule_scores.get(r, 0) > 0.5
    ]
    sec_str = ("; also: " + ", ".join(secondary)) if secondary else ""

    return f"{desc} [{top_rule}={top_score:.2f}{sec_str}] (IF={if_score:.2f})"


def add_primary_reason(result: pd.DataFrame, threshold: float = ALERT_THRESHOLD) -> pd.DataFrame:
    """Apply explain_row across the full DataFrame."""
    print(f"\n  Generating explanations (threshold={threshold}) …")
    result = result.copy()
    result["Primary_Reason"] = result.apply(explain_row, axis=1, threshold=threshold)
    n_flagged = (result["Primary_Reason"] != "Below threshold").sum()
    print(f"  Flagged: {n_flagged:,} / {len(result):,} ({100*n_flagged/len(result):.1f}%)")
    return result


def build_output(
    features:    pd.DataFrame,
    iso_scores:  pd.Series,
    rule_cols:   list,
    if_weight:   float = IF_WEIGHT,
    rule_weight: float = RULE_WEIGHT,
    threshold:   float = ALERT_THRESHOLD,
) -> pd.DataFrame:
    """
    Assemble the final output DataFrame.

    Columns: Timestamp | Meter_ID | Isolation_Forest_Score | Rule_Score
             | Final_Risk_Score | Primary_Reason | R1…R7 (audit trail)
    """
    print("\n" + "="*60)
    print("MODULE 4 — Scoring & Explainability")
    print("="*60)

    result = pd.DataFrame(index=features.index)
    result.index.name = "Timestamp"
    result["Meter_ID"]               = features["Meter_ID"]
    result["Isolation_Forest_Score"] = iso_scores
    result["Rule_Score"]             = features["Rule_Score"]

    for rc in rule_cols:
        result[rc] = features[rc]

    result["Final_Risk_Score"] = compute_final_score(
        iso_scores, features["Rule_Score"], if_weight, rule_weight
    )
    result = add_primary_reason(result, threshold=threshold)

    core = ["Meter_ID", "Isolation_Forest_Score", "Rule_Score",
            "Final_Risk_Score", "Primary_Reason"]
    detail = [c for c in rule_cols if c in result.columns]
    result = result[core + detail]

    print(f"\n  Output shape: {result.shape}")
    print(f"  Score stats:\n{result['Final_Risk_Score'].describe().round(3)}")
    return result


def top_alerts(result: pd.DataFrame, n: int = 20, per_meter: bool = True) -> pd.DataFrame:
    """Return top-N highest-risk records (optionally deduplicated per meter)."""
    flagged = result[result["Primary_Reason"] != "Below threshold"].copy()
    if per_meter:
        flagged = (
            flagged.sort_values("Final_Risk_Score", ascending=False)
                   .groupby("Meter_ID").first().reset_index()
        )
    return flagged.nlargest(n, "Final_Risk_Score")


def save_output(result: pd.DataFrame, filepath: str = "theft_detection_output.csv") -> None:
    result.to_csv(filepath, index=True, float_format="%.4f")
    print(f"\n  Saved full output → {filepath}  ({len(result):,} rows)")


def save_alerts_only(result: pd.DataFrame,
                     filepath: str = "high_risk_alerts.csv",
                     threshold: float = ALERT_THRESHOLD) -> None:
    alerts = result[result["Final_Risk_Score"] >= threshold]
    alerts.to_csv(filepath, index=True, float_format="%.4f")
    print(f"  Saved alerts only → {filepath}  ({len(alerts):,} records)")


def print_summary(result: pd.DataFrame, threshold: float = ALERT_THRESHOLD) -> None:
    flagged = result[result["Final_Risk_Score"] >= threshold]
    meters  = sorted(flagged["Meter_ID"].unique())
    cols    = ["Meter_ID", "Final_Risk_Score", "Rule_Score",
               "Isolation_Forest_Score", "Primary_Reason"]

    print("\n" + "="*60)
    print("  THEFT DETECTION SUMMARY")
    print("="*60)
    print(f"  Total records    : {len(result):,}")
    print(f"  Flagged records  : {len(flagged):,}")
    print(f"  High-risk meters : {meters}")
    print(f"  Alert threshold  : {threshold}")
    print(f"\n  Top 10 highest-risk records:")
    print(flagged[cols].nlargest(10, "Final_Risk_Score").to_string(index=True))
    print("="*60)
