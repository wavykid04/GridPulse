# Smart Meter Electricity Theft Detection
## Hybrid PCA + Isolation Forest System

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE ARCHITECTURE                        │
│                                                                 │
│  Raw CSV (2M rows, per-minute)                                  │
│       │                                                         │
│       ▼  [data_prep.py]                                         │
│  Chunked load (200k rows/chunk) ──► float32 downcast            │
│  '?' → NaN → linear interpolation ──► physical clipping         │
│  Date + Time → DatetimeIndex ──► assign Meter_ID                │
│  Resample 1-minute → 1-hour (÷60 RAM reduction)                 │
│       │                                                         │
│       ▼  [feature_engineering.py]                               │
│  ┌────────────────────────────────────────────┐                 │
│  │  7 Heuristic Rules (all normalised 0→1)   │                 │
│  │  R1  Sudden Drop      (rolling 24h mean)  │                 │
│  │  R2  Irregular TOU    (1–5 AM variance)   │                 │
│  │  R3  Flat Readings    (consecutive ident) │                 │
│  │  R4  Negative/Reset   (non-physical vals) │                 │
│  │  R5  V×I Mismatch     (correlation check) │                 │
│  │  R6  Peer Comparison  (hourly deviation)  │                 │
│  │  R7  Pattern Change   (Z-score rolling)   │                 │
│  └──────────────┬─────────────────────────────┘                 │
│                 │ weighted sum → Rule_Score [0,1]               │
│                 ▼                                               │
│       [ml_pipeline.py]                                          │
│  StandardScaler ──► PCA (95% variance) ──► IsolationForest      │
│  (sklearn CPU)   or  cuML GPU (auto-detect)                     │
│                 │                                               │
│                 │  IF_Score [0,1]                               │
│                 ▼                                               │
│       [scoring.py]                                              │
│  Final = 0.6×IF_Score + 0.4×Rule_Score                         │
│  If Final > 0.7 → generate Primary_Reason                       │
│                 │                                               │
│                 ▼                                               │
│  Output CSV: Timestamp | Meter_ID | IF_Score | Rule_Score       │
│              | Final_Risk_Score | Primary_Reason | R1…R7        │
└─────────────────────────────────────────────────────────────────┘
```

## Hardware Notes

| Resource | Budget | Typical Usage |
|----------|--------|---------------|
| RAM      | 16 GB  | ~2–3 GB (float32 + hourly resample) |
| VRAM     | 6 GB   | ~100–300 MB (PCA-reduced matrix) |
| CPU      | —      | n_jobs=-1 (all cores, sklearn fallback) |

## Installation

```bash
pip install -r requirements.txt

# GPU (optional):
pip install cuml-cu12 cupy-cuda12x   # for CUDA 12
```

## Usage

```bash
# Basic run
python main.py --data household_power_consumption.txt

# Full options
python main.py \
  --data household_power_consumption.txt \
  --meters 50 \
  --contamination 0.05 \
  --n-estimators 300 \
  --pca-variance 0.95 \
  --threshold 0.7 \
  --output results.csv \
  --alerts flagged.csv \
  --save-models
```

## Module Reference

| Module | File | Responsibility |
|--------|------|----------------|
| 1 | `data_prep.py` | Chunked CSV load, cleaning, hourly resample |
| 2 | `feature_engineering.py` | 7 heuristic rules → Rule_Score |
| 3 | `ml_pipeline.py` | StandardScaler + PCA + Isolation Forest |
| 4 | `scoring.py` | Score blending + explainability + CSV output |
| — | `main.py` | CLI orchestrator |

## Output Columns

| Column | Type | Description |
|--------|------|-------------|
| Timestamp | index | Hourly datetime |
| Meter_ID | int | Meter identifier |
| Isolation_Forest_Score | float32 | ML anomaly score [0,1] |
| Rule_Score | float32 | Weighted heuristic score [0,1] |
| Final_Risk_Score | float32 | 0.6×IF + 0.4×Rule [0,1] |
| Primary_Reason | str | Human explanation (if > threshold) |
| R1_sudden_drop … R7_* | float32 | Individual rule audit trail |

## cuML GPU Setup

Your NVIDIA GPU (6 GB VRAM) can accelerate Isolation Forest ~5–20×.

```bash
# Check CUDA version
nvcc --version

# Install matching RAPIDS
pip install cuml-cu12   # CUDA 12
pip install cuml-cu11   # CUDA 11

# Monitor VRAM during run
nvidia-smi -l 1
```

The system auto-detects cuML at startup and falls back to sklearn silently.

## Tuning Recommendations

- **contamination**: Set to your domain estimate of theft prevalence (0.02–0.10).
- **n_estimators**: 200 is robust; increase to 300–500 for production.
- **RULE_WEIGHTS** in `feature_engineering.py`: Adjust per domain expertise.
  R4 (negative readings) is often the most reliable single indicator.
- **PCA variance**: 0.95 works well; try 0.99 if IF scores seem noisy.
