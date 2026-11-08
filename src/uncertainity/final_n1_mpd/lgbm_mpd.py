"""
LightGBM Minimum Prediction Deviation (MPD) Pipeline
Reproduces Stages 1-4 for the canonical 12-feature tabular model.

Reference: Santhosh, Gressel & Darling (2022); Darling (2019).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.utils import resample

from mpd_core import compute_mpd, validate_mpd_integrity
from thresholds import compute_percentile_thresholds, flag_samples

logger = logging.getLogger(__name__)


class LGBMMPDPipeline:
    # Exact hyperparameters from models/final_n1/entropy_strings_final/lightgbm_entropy_strings.txt
    CANONICAL_HYPERPARAMS = {
        "boosting": "gbdt",
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_data_in_leaf": 20,
        "min_sum_hessian_in_leaf": 0.001,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "feature_fraction": 0.9,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "num_iterations": 800,
        "verbose": -1,
    }

    ADVERSARIAL_HYPERPARAMS = {
        "boosting": "gbdt",
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_data_in_leaf": 20,
        "min_sum_hessian_in_leaf": 0.001,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "feature_fraction": 1.0,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "num_iterations": 800,
        "verbosity": -1,
    }

    FEATURE_COLUMNS = [
        "global_entropy",
        "max_section_entropy",
        "mean_section_entropy",
        "entropy_variance",
        "entropy_high_ratio",
        "num_strings",
        "avg_string_len",
        "max_string_len",
        "num_ip_strings",
        "num_url_strings",
        "num_suspicious_strings",
        "num_high_entropy_strings",
    ]

    CANONICAL_THRESHOLD = 0.5

    def __init__(
        self,
        base_dir: Path,
        n_replicas: int = 25,
        replicas_dir: Path | None = None,
        canonical_model_path: Path | None = None,
        canonical_threshold: float = 0.5,
    ):
        self.base_dir = Path(base_dir)
        self.n_replicas = n_replicas
        self.CANONICAL_THRESHOLD = canonical_threshold

        if replicas_dir is None:
            self.replicas_dir = (
                self.base_dir / "results/uncertainty_final_n1_adv/lgbm/replicas"
            )
        else:
            self.replicas_dir = Path(replicas_dir)
        self.replicas_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: Load canonical booster as hyperparameter / schema source only
        canonical_model_path = Path(canonical_model_path)

        if not canonical_model_path.exists():
            raise FileNotFoundError(f"Canonical model not found: {canonical_model_path}")
        
        self.canonical_model_path = canonical_model_path
        self.canonical_booster = lgb.Booster(model_file=str(canonical_model_path))
        self.replicas: List[lgb.Booster] = []

    def fit_or_load_replicas(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> List[lgb.Booster]:
        """
        Stage 2: Bootstrap resampling (Algorithm 1) and fresh model training.
        Checkpoint every replica to results/uncertainty_final_n1/lgbm/replicas/model_XX.txt
        and resume from existing checkpoints on rerun.
        """
        self.replicas = []
        n_samples = len(X_train)

        for k in range(self.n_replicas):
            ckpt_path = self.replicas_dir / f"model_{k:02d}.txt"

            if ckpt_path.exists():
                print(f"[k={k:02d}] resumed from checkpoint: {ckpt_path.name}")
                booster = lgb.Booster(model_file=str(ckpt_path))
                self.replicas.append(booster)
                continue

            # Resample train with replacement (Algorithm 1)
            X_resampled, y_resampled = resample(
                X_train,
                y_train,
                replace=True,
                n_samples=n_samples,
                random_state=k,
            )

            # Fresh parameters with replica seed
            params = dict(self.ADVERSARIAL_HYPERPARAMS)
            params["seed"] = k
            params["feature_fraction_seed"] = 30000 + k
            params["bagging_seed"] = 400 + k

            num_rounds = params.pop("num_iterations", 800)
            train_data = lgb.Dataset(
                X_resampled,
                label=y_resampled,
                feature_name=self.FEATURE_COLUMNS,
                free_raw_data=False,
            )

            print(f"[k={k:02d}] training fresh replica (n={n_samples}, seed={k})...")
            booster = lgb.train(
                params,
                train_data,
                num_boost_round=num_rounds,
            )

            booster.save_model(str(ckpt_path))
            print(f"[k={k:02d}] saved checkpoint: {ckpt_path.name}")
            self.replicas.append(booster)

        return self.replicas

    def predict_ensemble_probs(self, X: np.ndarray) -> np.ndarray:
        """
        Stage 3: Ensemble prediction.
        Returns q matrix of shape (N, K) with q_k(x) = P(malware).
        """
        if not self.replicas:
            raise RuntimeError("Replicas not loaded. Call fit_or_load_replicas first.")

        q_cols = []
        for booster in self.replicas:
            prob = booster.predict(X)
            q_cols.append(prob)

        return np.column_stack(q_cols)

    def predict_canonical(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (canonical_probs, canonical_preds).
        """
        probs = self.canonical_booster.predict(X)
        preds = (probs >= self.CANONICAL_THRESHOLD).astype(int)
        return probs, preds

    def evaluate_dataset(
        self,
        X: np.ndarray,
        sample_ids: List[str] | None = None,
        labels: np.ndarray | None = None,
        split_name: str = "validation",
        thresholds: Dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """
        Runs Stages 3 & 4 on a given dataset matrix X, returning a DataFrame
        with raw canonical outputs, per-replica probabilities, and MPD in separate columns.
        """
        canon_probs, canon_preds = self.predict_canonical(X)
        q = self.predict_ensemble_probs(X)
        u0, u1, mpd = compute_mpd(q)

        # Integrity check
        val_res = validate_mpd_integrity(u0, u1, mpd, expected_count=len(X))
        if not val_res["all_passed"]:
            raise ValueError(f"MPD integrity check failed on {split_name}: {val_res}")

        data: Dict[str, Any] = {}
        if sample_ids is not None:
            data["sample_id"] = sample_ids
        if labels is not None:
            data["label"] = labels
        data["split"] = split_name

        data["canonical_prob"] = canon_probs
        data["canonical_pred"] = canon_preds

        # Individual replica probabilities q_00 .. q_24
        for k in range(q.shape[1]):
            data[f"q_{k:02d}"] = q[:, k]

        data["u0"] = u0
        data["u1"] = u1
        data["mpd"] = mpd

        # Threshold flags
        if thresholds is not None:
            if "T95" in thresholds:
                data["flag_t95"] = flag_samples(mpd, thresholds["T95"])
            if "T99" in thresholds:
                data["flag_t99"] = flag_samples(mpd, thresholds["T99"])


        return pd.DataFrame(data)
