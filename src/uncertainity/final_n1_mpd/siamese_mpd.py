"""
Siamese SVM Minimum Prediction Deviation (MPD) Pipeline
Implements pair-level and query-level bootstrapped ensemble MPD.

Reference: Santhosh, Gressel & Darling (2022); Darling (2019).
Platt Calibration: Explicit 1-D LogisticRegression on SVM margins (sklearn 1.7 compatible).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.utils import resample

from mpd_core import compute_mpd, validate_mpd_integrity
from thresholds import compute_percentile_thresholds, flag_samples

logger = logging.getLogger(__name__)


def make_difference_matrix(X: sparse.csr_matrix, indices_a: np.ndarray, indices_b: np.ndarray) -> sparse.csr_matrix:
    """Computes sparse |X[a] - X[b]| difference matrix vectorized in C."""
    if len(indices_a) != len(indices_b):
        raise ValueError("Index arrays must have identical length")

    diff = abs(X[indices_a] - X[indices_b])
    diff.eliminate_zeros()
    return diff.tocsr()


def generate_pair_indices(y: np.ndarray, pairs_per_relation: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact canonical pair generation protocol from train_siamese_svm_fewshot.py.
    """
    if pairs_per_relation % 2 != 0:
        raise ValueError("pairs_per_relation must be even")

    class_0 = np.flatnonzero(y == 0)
    class_1 = np.flatnonzero(y == 1)

    if min(len(class_0), len(class_1)) < 2:
        raise ValueError("At least two samples per class required")

    # Same-class pairs: half benign-benign, half malware-malware (a != b)
    pos_a = []
    pos_b = []
    for _ in range(pairs_per_relation // 2):
        for indices in (class_0, class_1):
            a = rng.choice(indices)
            b = rng.choice(indices)
            while a == b:
                b = rng.choice(indices)
            pos_a.append(a)
            pos_b.append(b)

    # Different-class pairs: random cross-class
    neg_a = rng.choice(class_0, size=pairs_per_relation, replace=True)
    neg_b = rng.choice(class_1, size=pairs_per_relation, replace=True)

    a = np.concatenate([np.asarray(pos_a), neg_a])
    b = np.concatenate([np.asarray(pos_b), neg_b])
    pair_y = np.concatenate([
        np.ones(len(pos_a), dtype=np.int32),
        np.zeros(len(neg_a), dtype=np.int32),
    ])

    order = rng.permutation(len(pair_y))
    return a[order], b[order], pair_y[order]


class SiameseMPDPipeline:
    # Exact canonical parameters
    SVM_PARAMS = {
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 20000,
    }

    # Canonical deterministic K=3 support indices from TRAIN (seed=42)
    CANONICAL_SUPPORT_INDICES = {
        0: np.array([53, 448, 535], dtype=int),
        1: np.array([70, 497, 588], dtype=int),
    }

    def __init__(
        self,
        base_dir: Path,
        n_replicas: int = 25,
        replicas_dir: Path | None = None,
    ):
        self.base_dir = Path(base_dir)
        self.n_replicas = n_replicas

        if replicas_dir is None:
            self.replicas_dir = (
                self.base_dir / "results/uncertainty_final_n1/siamese/replicas"
            )
        else:
            self.replicas_dir = Path(replicas_dir)
        self.replicas_dir.mkdir(parents=True, exist_ok=True)

        bundle_path = self.base_dir / "models/siamese_svm_n1/siamese_svm_fewshot.joblib"
        if not bundle_path.exists():
            raise FileNotFoundError(f"Siamese bundle not found: {bundle_path}")

        bundle = joblib.load(bundle_path)
        self.canonical_pair_classifier = bundle["pair_classifier"]
        self.replicas: List[Dict[str, Any]] = []

    def fit_or_load_replicas(
        self,
        X_train: sparse.csr_matrix,
        y_train: np.ndarray,
        pairs_per_relation: int = 20000,
        n_oob_pairs: int = 2000,
    ) -> List[Dict[str, Any]]:
        """
        Stage 2: Bootstrap 699 train samples with replacement (seed=k).
        Rebuild 40,000 pairs, fit LinearSVC, calibrate Platt sigmoid on OOB samples.
        """
        self.replicas = []
        n_samples = X_train.shape[0]
        canonical_support_set = set(self.CANONICAL_SUPPORT_INDICES[0].tolist() + self.CANONICAL_SUPPORT_INDICES[1].tolist())

        # Supports matrix
        support_vecs = {
            0: X_train[self.CANONICAL_SUPPORT_INDICES[0]],
            1: X_train[self.CANONICAL_SUPPORT_INDICES[1]],
        }

        for k in range(self.n_replicas):
            ckpt_path = self.replicas_dir / f"replica_{k:02d}.joblib"

            if ckpt_path.exists():
                print(f"[k={k:02d}] resumed from checkpoint: {ckpt_path.name}")
                rep_data = joblib.load(ckpt_path)
                self.replicas.append(rep_data)
                continue

            # 1. Resample sample indices with replacement
            rng = np.random.default_rng(k)
            resampled_idx = resample(
                np.arange(n_samples),
                replace=True,
                n_samples=n_samples,
                random_state=k,
            )
            oob_idx = np.setdiff1d(np.arange(n_samples), resampled_idx)

            # Rebuild 20k same / 20k diff pairs from resampled samples
            y_resampled = y_train[resampled_idx]
            sub_a, sub_b, pair_y = generate_pair_indices(y_resampled, pairs_per_relation, rng)
            a_idx = resampled_idx[sub_a]
            b_idx = resampled_idx[sub_b]

            print(f"[k={k:02d}] building {len(pair_y)} train pairs (resample distinct={len(np.unique(resampled_idx))}, oob={len(oob_idx)})...")
            X_pairs = make_difference_matrix(X_train, a_idx, b_idx)

            # Fit fresh LinearSVC
            clf = LinearSVC(
                C=self.SVM_PARAMS["C"],
                class_weight=self.SVM_PARAMS["class_weight"],
                max_iter=self.SVM_PARAMS["max_iter"],
                random_state=k,
            )
            clf.fit(X_pairs, pair_y)

            # 2. Platt calibration on OOB samples (cv='prefit' deprecated in sklearn 1.7)
            # a) Pair-level calibrator
            y_oob = y_train[oob_idx]
            oob_sub_a, oob_sub_b, oob_pair_y = generate_pair_indices(
                y_oob, pairs_per_relation=n_oob_pairs // 2, rng=rng
            )
            oob_a = oob_idx[oob_sub_a]
            oob_b = oob_idx[oob_sub_b]
            X_oob_pairs = make_difference_matrix(X_train, oob_a, oob_b)
            oob_pair_margins = clf.decision_function(X_oob_pairs).reshape(-1, 1)

            lr_pair = LogisticRegression(solver="lbfgs", random_state=k)
            lr_pair.fit(oob_pair_margins, oob_pair_y)

            # b) Query-level calibrator: exclude any sample in the canonical support set
            oob_query_indices = [idx for idx in oob_idx if idx not in canonical_support_set]
            oob_query_margins = []
            oob_query_labels = []

            for q_idx in oob_query_indices:
                q_vec = X_train[q_idx]
                q_rep = sparse.vstack([q_vec] * 3, format="csr")

                diff0 = abs(q_rep - support_vecs[0]).tocsr()
                s0 = float(np.mean(clf.decision_function(diff0)))

                diff1 = abs(q_rep - support_vecs[1]).tocsr()
                s1 = float(np.mean(clf.decision_function(diff1)))

                margin = s1 - s0
                oob_query_margins.append(margin)
                oob_query_labels.append(y_train[q_idx])

            lr_query = LogisticRegression(solver="lbfgs", random_state=k)
            lr_query.fit(
                np.array(oob_query_margins).reshape(-1, 1),
                np.array(oob_query_labels),
            )

            rep_data = {
                "classifier": clf,
                "calibrator_pair": lr_pair,
                "calibrator_query": lr_query,
                "replica_id": k,
                "seed": k,
            }
            joblib.dump(rep_data, ckpt_path)
            print(f"[k={k:02d}] saved checkpoint: {ckpt_path.name}")
            self.replicas.append(rep_data)

        return self.replicas

    def evaluate_pairs(
        self,
        X_mat: sparse.csr_matrix,
        indices_a: np.ndarray,
        indices_b: np.ndarray,
        pair_labels: np.ndarray,
        thresholds: Dict[str, float] | None = None,
        split_name: str = "validation",
    ) -> pd.DataFrame:
        """
        Stage 3 & 4 Pair-Level MPD Evaluation.
        q_k = P(same-class) for each pair.
        """
        X_diff = make_difference_matrix(X_mat, indices_a, indices_b)
        canon_margin = self.canonical_pair_classifier.decision_function(X_diff)

        q_cols = []
        for rep in self.replicas:
            clf = rep["classifier"]
            cal_pair = rep["calibrator_pair"]
            margins = clf.decision_function(X_diff).reshape(-1, 1)
            # Probability of same class (class 1)
            probs = cal_pair.predict_proba(margins)[:, 1]
            q_cols.append(probs)

        q = np.column_stack(q_cols)
        u0, u1, mpd = compute_mpd(q)

        val_res = validate_mpd_integrity(u0, u1, mpd, expected_count=len(indices_a))
        if not val_res["all_passed"]:
            raise ValueError(f"MPD integrity check failed on pairs: {val_res}")

        df = pd.DataFrame({
            "idx_a": indices_a,
            "idx_b": indices_b,
            "label": pair_labels,  # 1 = same, 0 = diff
            "split": split_name,
            "canonical_margin": canon_margin,
            "canonical_pred": (canon_margin > 0.0).astype(int),
        })

        for k in range(q.shape[1]):
            df[f"q_{k:02d}"] = q[:, k]

        df["u0"] = u0
        df["u1"] = u1
        df["mpd"] = mpd

        if thresholds is not None:
            if "T95" in thresholds:
                df["flag_t95"] = flag_samples(mpd, thresholds["T95"])
            if "T99" in thresholds:
                df["flag_t99"] = flag_samples(mpd, thresholds["T99"])


        return df

    def evaluate_queries(
        self,
        X_queries: sparse.csr_matrix,
        X_train: sparse.csr_matrix,
        sample_ids: List[str] | None = None,
        labels: np.ndarray | None = None,
        thresholds: Dict[str, float] | None = None,
        split_name: str = "validation",
    ) -> pd.DataFrame:
        """
        Stage 3 & 4 Query-Level MPD Evaluation.
        Uses fixed K=3 supports per class from TRAIN (seed=42).
        q_k = P(malware) for each query sample.
        Captures classifier-training variance for a fixed support set.
        """
        support_vecs = {
            0: X_train[self.CANONICAL_SUPPORT_INDICES[0]],
            1: X_train[self.CANONICAL_SUPPORT_INDICES[1]],
        }

        # Canonical predictions
        canon_margins = []
        for i in range(X_queries.shape[0]):
            q_vec = X_queries[i]
            q_rep = sparse.vstack([q_vec] * 3, format="csr")
            diff0 = abs(q_rep - support_vecs[0]).tocsr()
            s0 = float(np.mean(self.canonical_pair_classifier.decision_function(diff0)))
            diff1 = abs(q_rep - support_vecs[1]).tocsr()
            s1 = float(np.mean(self.canonical_pair_classifier.decision_function(diff1)))
            canon_margins.append(s1 - s0)

        canon_margins = np.array(canon_margins)
        canon_preds = (canon_margins > 0.0).astype(int)

        # Ensemble replica predictions
        q_cols = []
        for rep in self.replicas:
            clf = rep["classifier"]
            cal_query = rep["calibrator_query"]
            rep_margins = []

            for i in range(X_queries.shape[0]):
                q_vec = X_queries[i]
                q_rep = sparse.vstack([q_vec] * 3, format="csr")
                diff0 = abs(q_rep - support_vecs[0]).tocsr()
                s0 = float(np.mean(clf.decision_function(diff0)))
                diff1 = abs(q_rep - support_vecs[1]).tocsr()
                s1 = float(np.mean(clf.decision_function(diff1)))
                rep_margins.append(s1 - s0)

            rep_margins = np.array(rep_margins).reshape(-1, 1)
            probs = cal_query.predict_proba(rep_margins)[:, 1]
            q_cols.append(probs)

        q = np.column_stack(q_cols)
        u0, u1, mpd = compute_mpd(q)

        val_res = validate_mpd_integrity(u0, u1, mpd, expected_count=X_queries.shape[0])
        if not val_res["all_passed"]:
            raise ValueError(f"MPD integrity check failed on queries: {val_res}")

        df_data: Dict[str, Any] = {}
        if sample_ids is not None:
            df_data["sample_id"] = sample_ids
        if labels is not None:
            df_data["label"] = labels
        df_data["split"] = split_name

        df_data["canonical_margin"] = canon_margins
        df_data["canonical_pred"] = canon_preds

        for k in range(q.shape[1]):
            df_data[f"q_{k:02d}"] = q[:, k]

        df_data["u0"] = u0
        df_data["u1"] = u1
        df_data["mpd"] = mpd

        if thresholds is not None:
            if "T95" in thresholds:
                df_data["flag_t95"] = flag_samples(mpd, thresholds["T95"])
            if "T99" in thresholds:
                df_data["flag_t99"] = flag_samples(mpd, thresholds["T99"])


        return pd.DataFrame(df_data)
