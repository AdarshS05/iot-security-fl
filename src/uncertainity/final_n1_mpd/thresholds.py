"""
Thresholding and Flagging Module for MPD Uncertainty Defense
Reference: Santhosh, Gressel & Darling (2022), Section 4 & 5.

Stages:
  - Clean Baseline Percentiles: T90, T95, T99 computed strictly on CLEAN VALIDATION data.
  - Flag Rule:
      MPD <= T -> CONFIDENT (flag = 0)
      MPD >  T -> UNCERTAIN / FLAGGED (flag = 1)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, List


def compute_percentile_thresholds(clean_mpd: np.ndarray) -> Dict[str, float]:
    """
    Computes 90th, 95th, and 99th percentiles on clean validation MPD.
    Note: On small validation sets (e.g. N=150), T99 rests on ~1-2 samples.
    """
    clean_arr = np.asarray(clean_mpd, dtype=np.float64)
    if len(clean_arr) == 0:
        raise ValueError("Cannot compute percentiles on empty array")

    t90 = float(np.percentile(clean_arr, 90.0))
    t95 = float(np.percentile(clean_arr, 95.0))
    t99 = float(np.percentile(clean_arr, 99.0))

    return {
        "T90": t90,
        "T95": t95,
        "T99": t99,
        "n_samples": len(clean_arr),
        "clean_min": float(np.min(clean_arr)),
        "clean_max": float(np.max(clean_arr)),
        "clean_mean": float(np.mean(clean_arr)),
        "clean_median": float(np.median(clean_arr)),
        "clean_std": float(np.std(clean_arr, ddof=1)) if len(clean_arr) > 1 else 0.0,
    }


def flag_samples(mpd: np.ndarray, threshold: float) -> np.ndarray:
    """
    Paper's decision rule:
      MPD <= T -> CONFIDENT (0)
      MPD >  T -> UNCERTAIN / FLAGGED (1)
    """
    return (np.asarray(mpd, dtype=np.float64) > threshold).astype(int)
