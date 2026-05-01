"""
MODULE 3: Dimensionality Reduction & Anomaly Detection
======================================================
Pipeline: StandardScaler → PCA (95% variance) → Isolation Forest

GPU Acceleration
----------------
Automatically uses cuML (RAPIDS) if installed and an NVIDIA GPU is present.
Your 6 GB VRAM card will comfortably hold the PCA-reduced feature matrix.

Install RAPIDS cuML:
    pip install cuml-cu12   # CUDA 12.x
    # or: https://docs.rapids.ai/install

If cuML is unavailable, falls back to scikit-learn with n_jobs=-1.

VRAM monitoring requires: pip install nvidia-ml-py psutil
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings("ignore")


# ── VRAM monitoring ──────────────────────────────────────────────────────────

def vram_used_mb() -> float:
    """Return GPU VRAM usage in MB (requires pynvml). Returns 0 if unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 2)
    except Exception:
        return 0.0


def print_vram(label: str = "") -> None:
    mb = vram_used_mb()
    if mb > 0:
        print(f"  [VRAM] {label}: {mb:.0f} MB / 6144 MB")


# ── Backend selection ─────────────────────────────────────────────────────────

def _get_isolation_forest_class():
    """Return (IsoForest class, backend label). Prefers cuML, falls back to sklearn."""
    try:
        from cuml.ensemble import IsolationForest as cuIF
        print("  [GPU] cuML detected — GPU-accelerated Isolation Forest active.")
        return cuIF, "cuML (GPU)"
    except ImportError:
        print("  [CPU] cuML not found — using scikit-learn IsolationForest (n_jobs=-1).")
        return IsolationForest, "scikit-learn (CPU)"


# ── Scaling ──────────────────────────────────────────────────────────────────

def scale_features(X: np.ndarray) -> tuple:
    """Standardise to zero mean / unit variance. Returns (X_scaled, scaler)."""
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype("float32")
    print(f"  Scaler fitted — {X.shape[1]} features standardised.")
    return X_scaled, scaler


# ── PCA ──────────────────────────────────────────────────────────────────────

def apply_pca(X_scaled: np.ndarray, variance_threshold: float = 0.95) -> tuple:
    """
    Reduce dimensionality retaining `variance_threshold` of total variance.

    With 7 rule features this typically yields 4–6 components.
    Returns (X_pca, pca_object).
    """
    pca   = PCA(n_components=variance_threshold, svd_solver="full", random_state=42)
    X_pca = pca.fit_transform(X_scaled).astype("float32")
    n_c   = pca.n_components_
    var   = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {X_scaled.shape[1]} → {n_c} components ({var*100:.1f}% variance retained)")
    return X_pca, pca


# ── Isolation Forest ─────────────────────────────────────────────────────────

def fit_isolation_forest(
    X_pca:        np.ndarray,
    n_estimators: int   = 200,
    contamination: float = 0.05,
    max_samples:  str   = "auto",
    random_state: int   = 42,
    n_jobs:       int   = -1,
) -> tuple:
    """
    Fit Isolation Forest and return anomaly scores in [0, 1].

    sklearn's decision_function() → negative = anomalous.
    We negate and min-max normalise so 1.0 = highest theft risk.

    Returns (iso_scores_array, fitted_model).
    """
    IsoForest, backend = _get_isolation_forest_class()
    print_vram("before IsoForest fit")

    kwargs = dict(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=random_state,
    )
    if "scikit" in backend:
        kwargs["n_jobs"] = n_jobs

    model  = IsoForest(**kwargs)

    # Transfer to GPU memory if cuML
    X_fit = X_pca
    if "cuML" in backend:
        try:
            import cupy as cp
            X_fit = cp.array(X_pca)
        except ImportError:
            pass  # cupy missing; cuML may still handle numpy input

    model.fit(X_fit)
    print_vram("after IsoForest fit")

    raw = np.array(model.decision_function(X_fit), dtype="float32")
    try:
        raw = raw.get()  # cupy → numpy
    except AttributeError:
        pass

    scores = _normalise_iso_scores(raw)
    print(f"  [IF]  min={scores.min():.3f}  mean={scores.mean():.3f}  max={scores.max():.3f}")
    print(f"  [IF]  Backend: {backend}")
    return scores, model


def _normalise_iso_scores(raw: np.ndarray) -> np.ndarray:
    """Negate then min-max scale so 1 = most anomalous."""
    negated = -raw
    mn, mx  = negated.min(), negated.max()
    if mx == mn:
        return np.zeros_like(negated, dtype="float32")
    return ((negated - mn) / (mx - mn)).astype("float32")


# ── Full ML pipeline ─────────────────────────────────────────────────────────

def run_ml_pipeline(
    features:           pd.DataFrame,
    rule_cols:          list,
    variance_threshold: float = 0.95,
    n_estimators:       int   = 200,
    contamination:      float = 0.05,
) -> tuple:
    """
    End-to-end: Scale → PCA → Isolation Forest.

    Returns
    -------
    iso_scores : pd.Series aligned to features.index
    artefacts  : dict {scaler, pca, model} for serialisation
    """
    print("\n" + "="*60)
    print("MODULE 3 — Dimensionality Reduction & Anomaly Detection")
    print("="*60)

    X = features[rule_cols].values.astype("float32")
    print(f"  Input: {X.shape[0]:,} samples × {X.shape[1]} features")

    X_scaled, scaler = scale_features(X)
    X_pca,    pca    = apply_pca(X_scaled, variance_threshold)
    iso_arr,  model  = fit_isolation_forest(
        X_pca, n_estimators=n_estimators, contamination=contamination
    )

    iso_scores = pd.Series(
        iso_arr, index=features.index,
        name="Isolation_Forest_Score", dtype="float32"
    )
    return iso_scores, {"scaler": scaler, "pca": pca, "model": model}


# ── Persistence helpers ───────────────────────────────────────────────────────

def save_artefacts(artefacts: dict, output_dir: str = "models") -> None:
    """Persist scaler, PCA and IsolationForest with joblib."""
    import joblib, os
    os.makedirs(output_dir, exist_ok=True)
    for name, obj in artefacts.items():
        path = os.path.join(output_dir, f"{name}.pkl")
        joblib.dump(obj, path)
        print(f"  Saved artefact → {path}")


def load_artefacts(output_dir: str = "models") -> dict:
    import joblib, os
    return {n: joblib.load(os.path.join(output_dir, f"{n}.pkl"))
            for n in ["scaler", "pca", "model"]}
