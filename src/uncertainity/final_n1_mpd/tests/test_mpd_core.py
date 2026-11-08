import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""
Unit tests for Minimum Prediction Deviation (MPD) Core Module
Verifies exact mathematical properties from Santhosh et al. (2022) and Darling (2019).
Supports both unittest and pytest.
"""

import math
import unittest
import numpy as np

from mpd_core import compute_mpd, validate_mpd_integrity


class TestMPDCore(unittest.TestCase):

    def test_all_zeros(self):
        # All q_k = 0 -> U0 = 0, U1 = 1, MPD = 0
        q = np.zeros((10, 25))
        u0, u1, mpd = compute_mpd(q)
        self.assertTrue(np.allclose(u0, 0.0))
        self.assertTrue(np.allclose(u1, 1.0))
        self.assertTrue(np.allclose(mpd, 0.0))

    def test_all_ones(self):
        # All q_k = 1 -> U0 = 1, U1 = 0, MPD = 0
        q = np.ones((10, 25))
        u0, u1, mpd = compute_mpd(q)
        self.assertTrue(np.allclose(u0, 1.0))
        self.assertTrue(np.allclose(u1, 0.0))
        self.assertTrue(np.allclose(mpd, 0.0))

    def test_all_half(self):
        # All q_k = 0.5 -> U0 = 0.5, U1 = 0.5, MPD = 0.5
        q = np.full((10, 25), 0.5)
        u0, u1, mpd = compute_mpd(q)
        self.assertTrue(np.allclose(u0, 0.5))
        self.assertTrue(np.allclose(u1, 0.5))
        self.assertTrue(np.allclose(mpd, 0.5))

    def test_half_zeros_half_ones(self):
        # Exactly half 0 and half 1 -> U0 = sqrt(0.5), U1 = sqrt(0.5), MPD = sqrt(0.5) ≈ 0.70710678
        row = np.array([0.0] * 10 + [1.0] * 10)
        q = np.tile(row, (5, 1))
        u0, u1, mpd = compute_mpd(q)
        expected = math.sqrt(0.5)
        self.assertTrue(np.allclose(u0, expected))
        self.assertTrue(np.allclose(u1, expected))
        self.assertTrue(np.allclose(mpd, expected))
        self.assertTrue(np.all(mpd <= 0.7071067811865476 + 1e-12))

    def test_min_consistency_random(self):
        rng = np.random.default_rng(42)
        q = rng.uniform(0.0, 1.0, size=(100, 25))
        u0, u1, mpd = compute_mpd(q)
        self.assertTrue(np.allclose(mpd, np.minimum(u0, u1)))
        self.assertTrue(np.all(mpd >= 0.0))
        self.assertTrue(np.all(mpd <= math.sqrt(0.5) + 1e-12))

    def test_integrity_validator(self):
        rng = np.random.default_rng(42)
        q = rng.uniform(0.0, 1.0, size=(150, 25))
        u0, u1, mpd = compute_mpd(q)
        res = validate_mpd_integrity(u0, u1, mpd, expected_count=150, max_bound=0.7072)
        self.assertTrue(res["all_passed"])
        self.assertTrue(res["zero_nan"])
        self.assertTrue(res["zero_inf"])
        self.assertTrue(res["min_consistency"])
        self.assertTrue(res["bounds_valid"])
        self.assertLessEqual(res["observed_max"], 0.7072)

    def test_integrity_validator_detects_nan(self):
        u0 = np.array([0.1, np.nan])
        u1 = np.array([0.9, 0.5])
        mpd = np.array([0.1, 0.5])
        res = validate_mpd_integrity(u0, u1, mpd)
        self.assertFalse(res["zero_nan"])
        self.assertFalse(res["all_passed"])

    def test_integrity_validator_detects_out_of_bounds(self):
        u0 = np.array([0.8])
        u1 = np.array([0.8])
        mpd = np.array([0.75])  # Exceeds max_bound 0.7072
        res = validate_mpd_integrity(u0, u1, mpd, max_bound=0.7072)
        self.assertFalse(res["bounds_valid"])
        self.assertFalse(res["all_passed"])


if __name__ == "__main__":
    unittest.main()
