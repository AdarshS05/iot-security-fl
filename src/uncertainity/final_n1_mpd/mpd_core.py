"""
Minimum Prediction Deviation (MPD) Core Module
Reference: Santhosh, Gressel & Darling (2022), "Using Uncertainty as a Defense Against
Adversarial Attacks for Tabular Datasets" (MPD from Darling 2019).

Algorithm 2 (Exact Formulation):
    U0(x) = sqrt( mean_k( (q_k(x) - 0)^2 ) )
    U1(x) = sqrt( mean_k( (q_k(x) - 1)^2 ) )
    MPD(x) = min(U0(x), U1(x))

Theoretical Upper Bound:
    max MPD = sqrt(0.5) ≈ 0.70710678...
    Achieved when half the replicas output 0 and half output 1.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Tuple


def compute_mpd(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes U0, U1, and MPD across ensemble probability predictions.

    Args:
        q: 2D array of shape (N, K) where N is the number of samples/pairs,
           K is the number of ensemble replicas, and q[i, k] in [0, 1].

    Returns:
        (u0, u1, mpd) each as 1D numpy array of shape (N,).
    """
    q_arr = np.asarray(q, dtype=np.float64)
    if q_arr.ndim != 2:
        raise ValueError(f"Expected 2D array (N, K), got shape {q_arr.shape}")
    if q_arr.shape[1] == 0:
        raise ValueError("Ensemble size K cannot be 0")

    # U0 = sqrt(mean((q - 0)^2)) = sqrt(mean(q^2))
    u0 = np.sqrt(np.mean(q_arr ** 2, axis=1))

    # U1 = sqrt(mean((q - 1)^2))
    u1 = np.sqrt(np.mean((q_arr - 1.0) ** 2, axis=1))

    # MPD = min(U0, U1)
    mpd = np.minimum(u0, u1)

    return u0, u1, mpd


def validate_mpd_integrity(
    u0: np.ndarray,
    u1: np.ndarray,
    mpd: np.ndarray,
    expected_count: int | None = None,
    max_bound: float = 0.7072,
) -> Dict[str, Any]:
    """
    Verifies data integrity and theoretical mathematical bounds for MPD outputs.
    """
    u0 = np.asarray(u0)
    u1 = np.asarray(u1)
    mpd = np.asarray(mpd)

    checks = {}

    # 1. Row count match
    if expected_count is not None:
        checks["row_count_matches"] = bool(len(mpd) == expected_count)
    else:
        checks["row_count_matches"] = True

    # 2. Consistent lengths
    checks["length_consistency"] = bool(len(u0) == len(u1) == len(mpd))

    # 3. No NaN
    checks["zero_nan"] = bool(not (np.isnan(u0).any() or np.isnan(u1).any() or np.isnan(mpd).any()))

    # 4. No Inf
    checks["zero_inf"] = bool(not (np.isinf(u0).any() or np.isinf(u1).any() or np.isinf(mpd).any()))

    # 5. Consistency: mpd == min(u0, u1) within numerical tolerance
    checks["min_consistency"] = bool(np.allclose(mpd, np.minimum(u0, u1), atol=1e-12))

    # 6. Bounds check: within [0.0, max_bound]
    min_val = float(np.min(mpd)) if len(mpd) > 0 else 0.0
    max_val = float(np.max(mpd)) if len(mpd) > 0 else 0.0
    checks["bounds_valid"] = bool(min_val >= -1e-12 and max_val <= max_bound)
    checks["observed_min"] = min_val
    checks["observed_max"] = max_val

    all_passed = all(
        v for k, v in checks.items() if k not in ("observed_min", "observed_max")
    )
    checks["all_passed"] = all_passed
    return checks
