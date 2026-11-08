"""
Evaluation and Validation Reporting Module for MPD Uncertainty
Generates Stage 6 Validation Report, Platt calibration metrics, error AUROC,
and validation reporting.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_class_statistics(
    df: pd.DataFrame,
    val_col: str = "mpd",
    group_col: str = "label",
    label_names: Dict[Any, str] | None = None,
) -> pd.DataFrame:
    """Computes count, min, max, mean, median, std, p90, p95, p99 overall and by group."""
    records = []

    def get_stats(sub_arr: np.ndarray, name: str) -> Dict[str, Any]:
        arr = sub_arr[~np.isnan(sub_arr)]
        if len(arr) == 0:
            return {"group": name, "count": 0}
        return {
            "group": name,
            "count": int(len(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "p90": float(np.percentile(arr, 90.0)),
            "p95": float(np.percentile(arr, 95.0)),
            "p99": float(np.percentile(arr, 99.0)),
        }

    # Overall
    records.append(get_stats(df[val_col].values, "Overall"))

    # By group
    for val, grp in df.groupby(group_col):
        gname = label_names.get(val, str(val)) if label_names else str(val)
        records.append(get_stats(grp[val_col].values, gname))

    return pd.DataFrame(records)


def run_integrity_audit(
    df: pd.DataFrame,
    expected_rows: int,
    expected_k: int = 25,
    max_bound: float = 0.7072,
    is_pair: bool = False,
) -> Dict[str, Tuple[bool, str]]:
    """Runs Stage 6 integrity checks (OK/FAIL)."""
    checks = {}

    # 1. Ensemble model count == 25
    q_cols = [c for c in df.columns if c.startswith("q_")]
    c1 = (len(q_cols) == expected_k)
    checks["ensemble_model_count_25"] = (c1, f"Found {len(q_cols)} replica columns (expected {expected_k})")

    # 2. U0/U1/min consistency
    c2 = bool(np.allclose(df["mpd"].values, np.minimum(df["u0"].values, df["u1"].values), atol=1e-12))
    checks["u0_u1_min_consistency"] = (c2, "mpd == min(u0, u1)")

    # 3. Row count matches dataset
    c3 = (len(df) == expected_rows)
    checks["row_count_matches"] = (c3, f"Rows = {len(df)} (expected {expected_rows})")

    # 4. Zero NaN
    c4 = not df[["u0", "u1", "mpd"]].isna().any().any()
    checks["zero_nan"] = (c4, "No NaN found in u0/u1/mpd")

    # 5. Zero Inf
    c5 = not np.isinf(df[["u0", "u1", "mpd"]].values).any()
    checks["zero_inf"] = (c5, "No Inf found in u0/u1/mpd")

    # 6. Binary label validity
    valid_labels = {0, 1}
    actual_labels = set(df["label"].unique())
    c6 = actual_labels.issubset(valid_labels)
    lbl_desc = "0=Diff, 1=Same" if is_pair else "0=Benign, 1=Malware"
    checks["binary_label_validity"] = (c6, f"Labels: {actual_labels} ({lbl_desc})")

    # 7. Bounds within [0, 0.7072]
    min_mpd = float(df["mpd"].min())
    max_mpd = float(df["mpd"].max())
    c7 = (min_mpd >= -1e-12 and max_mpd <= max_bound)
    checks["bounds_within_0_to_07072"] = (c7, f"Observed [{min_mpd:.8f}, {max_mpd:.8f}] <= {max_bound}")

    return checks


def plot_lgbm_validation(df: pd.DataFrame, thresholds: Dict[str, float], plots_dir: Path) -> Tuple[Path, Path]:
    """
    LightGBM plots:
      (a) log10(MPD) histogram with 1e-15 floor, non-stacked overlapping series.
      (b) ECDF with log x-axis, T95/T99 lines, and clean stats box.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    floor = 1e-15
    mpd_clean = np.clip(df["mpd"].values, floor, None)
    log_mpd = np.log10(mpd_clean)

    # (a) Histogram
    p1 = plots_dir / "lgbm_log10_mpd_histogram.png"
    plt.figure(figsize=(8, 5))
    benign_mask = (df["label"] == 0)
    malware_mask = (df["label"] == 1)

    bins = np.linspace(-15, 0, 45)
    plt.hist(log_mpd[benign_mask], bins=bins, alpha=0.6, label="Benign (clean)", color="tab:blue", edgecolor="black")
    plt.hist(log_mpd[malware_mask], bins=bins, alpha=0.6, label="Malware (clean)", color="tab:red", edgecolor="black")

    t95_log = np.log10(max(floor, thresholds["T95"]))
    t99_log = np.log10(max(floor, thresholds["T99"]))
    plt.axvline(t95_log, color="black", linestyle="--", label=f"T95 = {thresholds['T95']:.4e}")
    plt.axvline(t99_log, color="purple", linestyle=":", label=f"T99 = {thresholds['T99']:.4e}")

    plt.xlabel("log10(MPD) [floor = 1e-15]")
    plt.ylabel("Sample Count")
    plt.title("LightGBM Clean Validation: log10(MPD) Distribution by Class\n(Note: y-peak below N due to non-stacked binning)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(p1, dpi=300)
    plt.close()

    # (b) ECDF
    p2 = plots_dir / "lgbm_ecdf.png"
    plt.figure(figsize=(8, 5))
    sorted_mpd = np.sort(mpd_clean)
    ecdf = np.arange(1, len(sorted_mpd) + 1) / len(sorted_mpd)

    plt.step(sorted_mpd, ecdf, where="post", color="darkblue", lw=2, label="Clean Validation ECDF")
    plt.axvline(thresholds["T95"], color="black", linestyle="--", label=f"T95: {thresholds['T95']:.4e}")
    plt.axvline(thresholds["T99"], color="purple", linestyle=":", label=f"T99: {thresholds['T99']:.4e}")
    plt.xscale("log")
    plt.xlabel("MPD (log scale, floor 1e-15)")
    plt.ylabel("Cumulative Probability")
    plt.title("LightGBM Clean Validation: Empirical CDF of MPD")

    stats_text = (
        f"N = {len(df)}\n"
        f"Mean = {df['mpd'].mean():.2e}\n"
        f"Median = {df['mpd'].median():.2e}\n"
        f"T90 = {thresholds.get('T90', 0):.2e}\n"
        f"T95 = {thresholds['T95']:.2e}\n"
        f"T99 = {thresholds['T99']:.2e}"
    )
    plt.text(
        0.05, 0.60, stats_text, transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="gray"),
        fontsize=9,
    )
    plt.legend(loc="lower right")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(p2, dpi=300)
    plt.close()

    return p1, p2


def plot_siamese_plots(
    df: pd.DataFrame,
    thresholds: Dict[str, float],
    plots_dir: Path,
    prefix: str = "siamese_pair",
    is_pair: bool = True,
) -> Tuple[Path, Path, Path]:
    """Plots histogram, ECDF, and calibration-check scatter for Siamese SVM."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    lbl0_name = "Diff-class" if is_pair else "Benign"
    lbl1_name = "Same-class" if is_pair else "Malware"
    margin_col = "canonical_margin"

    # 1. Histogram
    p1 = plots_dir / f"{prefix}_mpd_histogram.png"
    plt.figure(figsize=(8, 5))
    m0 = (df["label"] == 0)
    m1 = (df["label"] == 1)

    bins = np.linspace(0.0, 0.7072, 35)
    plt.hist(df.loc[m0, "mpd"], bins=bins, alpha=0.6, label=lbl0_name, color="tab:blue", edgecolor="black", density=True)
    plt.hist(df.loc[m1, "mpd"], bins=bins, alpha=0.6, label=lbl1_name, color="tab:orange", edgecolor="black", density=True)
    plt.axvline(thresholds["T95"], color="black", linestyle="--", label=f"P95 = {thresholds['T95']:.4f}")
    plt.axvline(thresholds["T99"], color="purple", linestyle=":", label=f"P99 = {thresholds['T99']:.4f}")
    plt.yscale("log")
    plt.xlabel("MPD (linear)")
    plt.ylabel("Density (log scale)")
    plt.title(f"{prefix.replace('_', ' ').title()}: MPD Distribution by Relation")
    plt.legend(loc="upper right")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(p1, dpi=300)
    plt.close()

    # 2. ECDF
    p2 = plots_dir / f"{prefix}_ecdf.png"
    plt.figure(figsize=(8, 5))
    sorted_mpd = np.sort(df["mpd"].values)
    ecdf = np.arange(1, len(sorted_mpd) + 1) / len(sorted_mpd)
    plt.step(sorted_mpd, ecdf, where="post", color="darkgreen", lw=2, label="Clean ECDF")
    plt.axvline(thresholds["T95"], color="black", linestyle="--", label=f"P95: {thresholds['T95']:.4f}")
    plt.axvline(thresholds["T99"], color="purple", linestyle=":", label=f"P99: {thresholds['T99']:.4f}")
    plt.xlabel("MPD")
    plt.ylabel("Cumulative Probability")
    plt.title(f"{prefix.replace('_', ' ').title()}: Empirical CDF of MPD")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(p2, dpi=300)
    plt.close()

    # 3. Margin vs MPD scatter
    p3 = plots_dir / f"{prefix}_margin_vs_mpd_scatter.png"
    plt.figure(figsize=(8, 5))
    plt.scatter(df.loc[m0, margin_col], df.loc[m0, "mpd"], alpha=0.5, s=20, label=lbl0_name, color="tab:blue")
    plt.scatter(df.loc[m1, margin_col], df.loc[m1, "mpd"], alpha=0.5, s=20, label=lbl1_name, color="tab:orange")
    plt.axvline(0.0, color="gray", linestyle="--", label="Decision Boundary (margin=0)")
    plt.axhline(thresholds["T95"], color="black", linestyle="--", alpha=0.7, label=f"T95 = {thresholds['T95']:.4f}")
    plt.xlabel("Canonical LinearSVC Margin")
    plt.ylabel("MPD")
    plt.title(f"{prefix.replace('_', ' ').title()}: Canonical Margin vs MPD")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(p3, dpi=300)
    plt.close()

    return p1, p2, p3


def evaluate_calibration_quality(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Computes Brier score and calibration curve metrics."""
    brier = float(brier_score_loss(y_true, y_prob))
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    return {
        "brier_score": brier,
        "fraction_of_positives": prob_true.tolist(),
        "mean_predicted_value": prob_pred.tolist(),
    }


def evaluate_error_auroc(y_true: np.ndarray, y_pred: np.ndarray, mpd: np.ndarray) -> Dict[str, Any]:
    """AUROC of MPD predicting canonical classification errors."""
    is_error = (y_true != y_pred).astype(int)
    n_errors = int(np.sum(is_error))
    if n_errors == 0 or n_errors == len(is_error):
        return {
            "error_auroc": None,
            "n_errors": n_errors,
            "status": "CANNOT_COMPUTE_NO_OR_ALL_ERRORS",
        }
    auc_val = float(roc_auc_score(is_error, mpd))
    return {
        "error_auroc": auc_val,
        "n_errors": n_errors,
        "status": "OK",
    }
