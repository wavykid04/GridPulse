"""
main.py — Electricity Theft Detection Orchestrator
===================================================
Runs the full pipeline end-to-end:

    Module 1  Data Loading & Preprocessing
    Module 2  Feature Engineering (7 Rules)
    Module 3  PCA + Isolation Forest
    Module 4  Scoring, Explainability & Output

Usage
-----
    python main.py --data household_power_consumption.txt
    python main.py --data data.txt --meters 50 --contamination 0.05

    # With cuML GPU backend (auto-detected if installed):
    python main.py --data data.txt

Resource Monitoring
-------------------
RAM  : Tracked per-module via psutil (see ram_used_gb() in Module 1).
VRAM : Tracked via pynvml (see print_vram() in Module 3).
       Install: pip install nvidia-ml-py psutil
"""

import argparse
import time
import sys
import os

# ── Make local modules importable ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from data_prep          import prepare_data, print_ram
from feature_engineering import compute_all_rules, RULE_NAMES
from ml_pipeline        import run_ml_pipeline, save_artefacts
from scoring            import (
    build_output, save_output, save_alerts_only,
    top_alerts, print_summary
)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smart Meter Electricity Theft Detection"
    )
    p.add_argument(
        "--data", default="household_power_consumption.txt",
        help="Path to UCI dataset (default: household_power_consumption.txt)"
    )
    p.add_argument(
        "--meters", type=int, default=20,
        help="Number of synthetic meters to simulate (default: 20)"
    )
    p.add_argument(
        "--chunksize", type=int, default=200_000,
        help="CSV chunk size for streaming load (default: 200000)"
    )
    p.add_argument(
        "--contamination", type=float, default=0.05,
        help="Expected fraction of anomalies for IsoForest (default: 0.05)"
    )
    p.add_argument(
        "--n-estimators", type=int, default=200,
        help="Isolation Forest trees (default: 200)"
    )
    p.add_argument(
        "--pca-variance", type=float, default=0.95,
        help="PCA variance retention threshold (default: 0.95)"
    )
    p.add_argument(
        "--threshold", type=float, default=0.7,
        help="Alert threshold for Final_Risk_Score (default: 0.7)"
    )
    p.add_argument(
        "--output", default="theft_detection_output.csv",
        help="Full output CSV path (default: theft_detection_output.csv)"
    )
    p.add_argument(
        "--alerts", default="high_risk_alerts.csv",
        help="Alerts-only CSV path (default: high_risk_alerts.csv)"
    )
    p.add_argument(
        "--save-models", action="store_true",
        help="Persist scaler/PCA/IsolationForest to disk"
    )
    return p.parse_args()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    t0 = time.time()

    # ── MODULE 1: Data Prep ───────────────────────────────────────────────
    df_hourly = prepare_data(
        filepath=args.data,
        chunksize=args.chunksize,
        n_meters=args.meters,
    )
    print_ram("after Module 1")

    # ── MODULE 2: Feature Engineering ────────────────────────────────────
    features = compute_all_rules(df_hourly)
    del df_hourly   # free raw data — features is our working set now
    print_ram("after Module 2")

    # ── MODULE 3: ML Pipeline ─────────────────────────────────────────────
    iso_scores, artefacts = run_ml_pipeline(
        features=features,
        rule_cols=RULE_NAMES,
        variance_threshold=args.pca_variance,
        n_estimators=args.n_estimators,
        contamination=args.contamination,
    )
    print_ram("after Module 3")

    if args.save_models:
        save_artefacts(artefacts, output_dir="models/")

    # ── MODULE 4: Scoring & Output ────────────────────────────────────────
    result = build_output(
        features=features,
        iso_scores=iso_scores,
        rule_cols=RULE_NAMES,
        threshold=args.threshold,
    )
    print_ram("after Module 4")

    save_output(result, filepath=args.output)
    save_alerts_only(result, filepath=args.alerts, threshold=args.threshold)

    print_summary(result, threshold=args.threshold)

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print("  Done.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    run(args)
