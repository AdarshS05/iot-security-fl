"""Bootstrapped-ensemble MPD scaffolding for the Siamese SVM model.
Week 8, Member 3 -- Point 1b: bootstrap-ensemble MPD (Siamese SVM clean baseline).

The saved package (models/siamese_svm1.pkl) is a dict:
    {"model": LinearSVC, "scaler": StandardScaler}

Pipeline (mirrors bootstrap_mpd.py for LightGBM exactly):
  1. Load package -- extract LinearSVC params for cloning (weights NOT reused).
  2. Build a FIXED evaluation set of capped 3,000 (x1,x2,label) pairs using
     |x1-x2| difference vectors (shape N x 18223), seed=42.
  3. Train 25 bootstrap LinearSVC replicas, each Platt-calibrated with
     CalibratedClassifierCV(cv='prefit') -- exactly 25 LinearSVC fits total.
     Each model is checkpointed to disk immediately after fitting.
  4. Compute MPD per Algorithm 2:  MPD = min(mean((q-0)^2), mean((q-1)^2))
  5. Validate + summary JSON/CSV + 3 plots.

NOTE on multi_class='ovr':
  The saved siamese_svm1.pkl was trained with multi_class='ovr' (confirmed via
  get_params()). In sklearn 1.7.2 this parameter still exists with no
  deprecation warning, so it is passed through to the cloned estimators
  unchanged (consistent with the original training).

NOTE on sqrt:
  Algorithm 2 pseudocode omits sqrt; we match it exactly (no sqrt),
  consistent with bootstrap_mpd.py.
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=UserWarning,  module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[3]


# ==============================================================================
# 1. Package loading
# ==============================================================================

def load_svm_package(pkl_path: Path):
    """Load siamese_svm1.pkl; return base_model, scaler, clone_params.

    clone_params is a copy of get_params() with:
      - random_state removed  (each bootstrap seed gets its own)
      - dual forced to False  (confirmed safe: loss=squared_hinge, penalty=l2)
      - multi_class kept as 'ovr' (present in saved model, no warning in 1.7.2)
    """
    try:
        saved = joblib.load(pkl_path)
    except Exception:
        with open(pkl_path, "rb") as f:
            saved = pickle.load(f)

    if not isinstance(saved, dict):
        raise RuntimeError(f"Expected dict at {pkl_path}, got {type(saved)}")
    for key in ("model", "scaler"):
        if key not in saved:
            raise RuntimeError(f"Package missing key '{key}': {list(saved.keys())}")

    base_model = saved["model"]
    scaler = saved["scaler"]

    clone_params = base_model.get_params()
    clone_params.pop("random_state", None)   # seeded per bootstrap iteration

    # dual=False: compatible with loss='squared_hinge' + penalty='l2' (confirmed).
    loss    = clone_params.get("loss", "squared_hinge")
    penalty = clone_params.get("penalty", "l2")
    if loss == "squared_hinge" and penalty in ("l2", "l1"):
        clone_params["dual"] = False
        print("[baseline] dual=False set (compatible: loss=squared_hinge, penalty=l2)")
    else:
        print(f"[baseline] dual left as-is (loss={loss}, penalty={penalty})")

    # multi_class='ovr' -- present in saved model, exists in sklearn 1.7.2,
    # no deprecation warning.  Keep it so clones match original training config.
    print(f"[baseline] loaded {pkl_path.name}")
    print(f"[baseline] scaler n_features={scaler.mean_.shape[0]}")
    print(f"[baseline] clone_params: {clone_params}")
    return base_model, scaler, clone_params


# ==============================================================================
# 2. Build fixed pair evaluation set (capped at max_pairs, seed=42)
# ==============================================================================

def load_tfidf_data(matrix_path: Path, labels_path: Path, filenames_path: Path):
    X = sparse.load_npz(str(matrix_path))

    with open(labels_path, "rb") as f:
        meta = pickle.load(f)
    if isinstance(meta, dict):
        y = meta["labels"]
    elif isinstance(meta, tuple):
        _, y = meta
    else:
        y = meta
    y = np.asarray(y)

    with open(filenames_path, "rb") as f:
        filenames = pickle.load(f)

    print(f"[data] matrix {X.shape}, benign={int((y==0).sum())}, malware={int((y==1).sum())}")
    return X, y, filenames


def generate_fixed_pairs(X_dense: np.ndarray, y: np.ndarray,
                         max_pairs: int = 3000, seed: int = 42):
    """Return (X_diff, pair_labels, left_idx, right_idx).

    Pair-type labels: 1 = same class, 0 = different class.

    Strategy (mirrors generate_pairs.py):
      Same-class : consecutive index pairs within each class label.
      Diff-class : random cross-class pairs balanced to same-class count.
    For 500+500 balanced data this yields 1,996 pairs -- well under 3,000.
    The cap (max_pairs//2 per type) is still enforced explicitly so the
    function stays safe if pair logic or dataset size ever changes.
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y)

    class_indices: dict = {}
    for idx, label in enumerate(y):
        class_indices.setdefault(int(label), []).append(idx)

    # same-class: consecutive pairs within each class
    same_pairs = []
    for lbl, indices in class_indices.items():
        for k in range(len(indices) - 1):
            same_pairs.append((indices[k], indices[k + 1], 1))

    # diff-class: random cross-class, balanced to same-class count
    class_keys = list(class_indices.keys())
    diff_pairs = []
    while len(diff_pairs) < len(same_pairs):
        lbl1, lbl2 = rng.choice(class_keys, size=2, replace=False)
        i = rng.choice(class_indices[lbl1])
        j = rng.choice(class_indices[lbl2])
        diff_pairs.append((i, j, 0))

    # hard cap -- each type gets at most max_pairs//2 entries
    half_cap = max_pairs // 2
    if len(same_pairs) > half_cap:
        sel = rng.permutation(len(same_pairs))[:half_cap]
        same_pairs = [same_pairs[s] for s in sel]
    if len(diff_pairs) > half_cap:
        sel = rng.permutation(len(diff_pairs))[:half_cap]
        diff_pairs = [diff_pairs[s] for s in sel]

    all_pairs = same_pairs + diff_pairs
    perm = rng.permutation(len(all_pairs))
    all_pairs = [all_pairs[p] for p in perm]

    left_idx    = [p[0] for p in all_pairs]
    right_idx   = [p[1] for p in all_pairs]
    pair_labels = np.array([p[2] for p in all_pairs], dtype=int)

    # |x1 - x2| difference vectors
    X_diff = np.abs(X_dense[left_idx] - X_dense[right_idx])

    print(
        f"[pairs] {len(pair_labels)} fixed eval pairs "
        f"({int((pair_labels==1).sum())} same-class, "
        f"{int((pair_labels==0).sum())} diff-class), "
        f"shape {X_diff.shape}"
    )
    return X_diff, pair_labels, left_idx, right_idx


# ==============================================================================
# 3. Bootstrap ensemble -- Platt calibrated, cv='prefit', with checkpointing
# ==============================================================================

def build_bootstrap_ensemble(
    X_pairs_scaled: np.ndarray,
    pair_labels: np.ndarray,
    clone_params: dict,
    n_bootstrap: int = 25,
    checkpoint_dir: Path = None,
):
    """Train n_bootstrap Platt-calibrated LinearSVC replicas.

    Per-iteration protocol (exactly ONE LinearSVC fit per iteration = 25 total):
      1. Bootstrap-resample pairs with replacement (seed=k).
      2. Split 80/20 into train / calibration subsets.
      3. Clone LinearSVC(**clone_params, random_state=k); fit on 80% split.
      4. Wrap the fitted clone in FrozenEstimator so the calibrator cannot
         refit it, then fit CalibratedClassifierCV on the 20% calib split.
         This is the sklearn 1.7 replacement for the deprecated cv='prefit'.
      5. Save checkpoint immediately (checkpoint_dir/model_{k:02d}.pkl) before
         advancing to k+1.  A crash mid-run loses at most one in-progress model.

    cv=5 is NOT used: it retrains LinearSVC 5x per iteration = 125 total fits.
    """
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    n_pairs = len(pair_labels)
    ensemble = []

    for k in range(n_bootstrap):
        # -- resume from checkpoint if it exists --------------------------------
        if checkpoint_dir is not None:
            ckpt = checkpoint_dir / f"model_{k:02d}.pkl"
            if ckpt.exists():
                with open(ckpt, "rb") as f:
                    calibrated = pickle.load(f)
                ensemble.append(calibrated)
                print(f"  [k={k:02d}] resumed from checkpoint {ckpt.name}")
                continue

        rng = np.random.RandomState(k)

        # bootstrap resample
        boot_idx = rng.choice(n_pairs, size=n_pairs, replace=True)
        X_boot   = X_pairs_scaled[boot_idx]
        y_boot   = pair_labels[boot_idx]

        # 80/20 train / calibration split
        n_train  = int(0.8 * len(boot_idx))
        perm     = rng.permutation(len(boot_idx))
        X_train, y_train = X_boot[perm[:n_train]], y_boot[perm[:n_train]]
        X_calib, y_calib = X_boot[perm[n_train:]], y_boot[perm[n_train:]]

        # clone and fit base LinearSVC (ONE fit per iteration)
        params_k = dict(clone_params)
        params_k["random_state"] = k
        fitted_clf = LinearSVC(**params_k)
        fitted_clf.fit(X_train, y_train)

        # Platt calibration on held-out split only.
        # FrozenEstimator wraps fitted_clf so the calibrator cannot refit it --
        # the sklearn 1.7 replacement for the deprecated cv='prefit' pattern.
        calibrated = CalibratedClassifierCV(
            FrozenEstimator(fitted_clf), method="sigmoid"
        )
        calibrated.fit(X_calib, y_calib)

        # -- checkpoint immediately before moving to k+1 ------------------------
        if checkpoint_dir is not None:
            with open(checkpoint_dir / f"model_{k:02d}.pkl", "wb") as f:
                pickle.dump(calibrated, f)

        ensemble.append(calibrated)

        if (k + 1) % 5 == 0 or k == 0:
            tr_acc = accuracy_score(y_train, fitted_clf.predict(X_train))
            ca_acc = accuracy_score(y_calib, fitted_clf.predict(X_calib))
            print(f"  [k={k:02d}] train_acc={tr_acc:.4f}  calib_acc={ca_acc:.4f}")

    return ensemble


# ==============================================================================
# 4. MPD score calculation (Algorithm 2, no sqrt)
# ==============================================================================

def compute_mpd(ensemble: list, X_scaled: np.ndarray):
    """MPD per Algorithm 2 pseudocode (no sqrt).

    q(x)  = predict_proba(x)[:,1] across 25 models  (shape: n_models x n_pairs)
    U0(x) = mean((q - 0)^2)
    U1(x) = mean((q - 1)^2)
    MPD(x)= min(U0, U1)
    """
    preds  = np.stack([m.predict_proba(X_scaled)[:, 1] for m in ensemble], axis=0)
    u0     = np.mean((preds - 0.0) ** 2, axis=0)
    u1     = np.mean((preds - 1.0) ** 2, axis=0)
    mpd    = np.minimum(u0, u1)
    mean_q = np.mean(preds, axis=0)
    return mpd, u0, u1, mean_q


# ==============================================================================
# 5. Validation and statistics
# ==============================================================================

def _stats_block(arr: np.ndarray) -> dict:
    return {
        "count":  int(len(arr)),
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std":    float(np.std(arr)),
        "p90":    float(np.percentile(arr, 90)),
        "p95":    float(np.percentile(arr, 95)),
        "p99":    float(np.percentile(arr, 99)),
    }


def validate_and_summarise(
    ensemble: list,
    mpd_scores: np.ndarray,
    pair_labels: np.ndarray,
    n_bootstrap: int,
    out_dir: Path,
) -> dict:
    # integrity checks
    assert len(ensemble) == n_bootstrap, (
        f"Ensemble count {len(ensemble)} != expected {n_bootstrap}"
    )
    assert not np.isnan(mpd_scores).any(), "NaN in MPD scores"
    assert not np.isinf(mpd_scores).any(), "Inf in MPD scores"
    assert np.all((mpd_scores >= 0) & (mpd_scores <= 0.5)), (
        f"MPD out of [0,0.5]: min={mpd_scores.min():.6f}, max={mpd_scores.max():.6f}"
    )

    same_mpd = mpd_scores[pair_labels == 1]
    diff_mpd = mpd_scores[pair_labels == 0]

    summary = {
        "formula":      "min(E[(q-0)^2], E[(q-1)^2])",
        "n_models":     int(len(ensemble)),
        "n_pairs":      int(len(mpd_scores)),
        "n_same_class": int((pair_labels == 1).sum()),
        "n_diff_class": int((pair_labels == 0).sum()),
        "overall":      _stats_block(mpd_scores),
        "same_class":   _stats_block(same_mpd),
        "diff_class":   _stats_block(diff_mpd),
    }

    json_path = out_dir / "clean_mpd_svm_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] JSON -> {json_path}")
    return summary


# ==============================================================================
# 6. Plots
# ==============================================================================

def _style_axes(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_log_histogram(mpd_scores, pair_labels, summary, docs_dir: Path):
    """Plot 1: log-scale MPD distribution split by pair type."""
    same_mpd = mpd_scores[pair_labels == 1]
    diff_mpd = mpd_scores[pair_labels == 0]
    p95 = summary["overall"]["p95"]
    p99 = summary["overall"]["p99"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(same_mpd, bins=60, alpha=0.65, density=True,
            label=f"Same-class (n={len(same_mpd)})", color="#4C72B0", edgecolor="none")
    ax.hist(diff_mpd, bins=60, alpha=0.65, density=True,
            label=f"Diff-class (n={len(diff_mpd)})", color="#DD8452", edgecolor="none")
    ax.axvline(p95, color="#C44E52", linestyle="--", linewidth=1.5,
               label=f"P95 = {p95:.4f}")
    ax.axvline(p99, color="#8172B2", linestyle=":",  linewidth=1.5,
               label=f"P99 = {p99:.4f}")
    ax.set_yscale("log")
    _style_axes(ax,
                title="MPD Score Distribution by Pair Type -- Siamese SVM (log scale)",
                xlabel="MPD Score", ylabel="Density (log scale)")
    ax.legend(fontsize=10)
    fig.tight_layout()
    out = docs_dir / "mpd_svm_clean_baseline.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] histogram -> {out}")


def plot_ecdf(mpd_scores, summary, docs_dir: Path):
    """Plot 2: empirical CDF with P95/P99 threshold lines."""
    sorted_mpd = np.sort(mpd_scores)
    cdf = np.arange(1, len(sorted_mpd) + 1) / len(sorted_mpd)
    p95 = summary["overall"]["p95"]
    p99 = summary["overall"]["p99"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sorted_mpd, cdf, linewidth=2, color="#4C72B0", label="ECDF (all pairs)")
    ax.axvline(p95, color="#C44E52", linestyle="--", linewidth=1.5,
               label=f"P95 = {p95:.4f}")
    ax.axvline(p99, color="#8172B2", linestyle=":",  linewidth=1.5,
               label=f"P99 = {p99:.4f}")
    ax.axhline(0.95, color="#C44E52", linestyle="--", alpha=0.25, linewidth=0.8)
    ax.axhline(0.99, color="#8172B2", linestyle=":",  alpha=0.25, linewidth=0.8)
    ax.set_xlim(left=0); ax.set_ylim(0, 1.02)
    _style_axes(ax,
                title="Empirical CDF of MPD Scores -- Siamese SVM",
                xlabel="MPD Score", ylabel="Cumulative Probability")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()
    out = docs_dir / "mpd_svm_ecdf_thresholds.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] ECDF -> {out}")


def plot_calibration_check(raw_margins, mpd_scores, pair_labels, docs_dir: Path):
    """Plot 3 (diagnostic): raw decision margin vs calibrated MPD score.

    High-confidence samples (large |margin|) -> low MPD (ensemble agrees).
    Ambiguous samples (margin near 0) -> higher MPD.
    """
    same_mask = pair_labels == 1
    diff_mask = pair_labels == 0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(raw_margins[same_mask], mpd_scores[same_mask],
               alpha=0.35, s=8, color="#4C72B0", label="Same-class", rasterized=True)
    ax.scatter(raw_margins[diff_mask], mpd_scores[diff_mask],
               alpha=0.35, s=8, color="#DD8452", label="Diff-class", rasterized=True)
    ax.axvline(0, color="#555", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Decision boundary (margin=0)")
    _style_axes(ax,
                title="Baseline Decision Margin vs. Calibrated MPD -- Siamese SVM",
                xlabel="Raw LinearSVC Decision Function (margin)", ylabel="MPD Score")
    ax.legend(fontsize=10, markerscale=3)
    ax.grid(True, alpha=0.2, linestyle="--")
    fig.tight_layout()
    out = docs_dir / "mpd_svm_calibration_check.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot] calibration check -> {out}")


# ==============================================================================
# main
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Siamese SVM MPD clean-baseline (25 models, Platt cv=prefit, checkpointed)."
    )
    ap.add_argument("--matrix_path",    type=str,
                    default=str(REPO_ROOT / "data/tfidf/tfidf_matrix.npz"))
    ap.add_argument("--labels_path",    type=str,
                    default=str(REPO_ROOT / "data/tfidf/labels.pkl"))
    ap.add_argument("--filenames_path", type=str,
                    default=str(REPO_ROOT / "data/tfidf/filenames.pkl"))
    ap.add_argument("--model_pkl",      type=str,
                    default=str(REPO_ROOT / "models/siamese_svm1.pkl"))
    ap.add_argument("--n_bootstrap",    type=int, default=25)
    ap.add_argument("--max_pairs",      type=int, default=3000,
                    help="Hard cap on fixed evaluation pair set (balanced per type).")
    ap.add_argument("--pair_seed",      type=int, default=42,
                    help="RNG seed for pair generation and cap sampling.")
    ap.add_argument("--eval_only",      action="store_true",
                    help="Skip training; load ensemble_svm.pkl and recompute MPD.")
    ap.add_argument("--out_dir",        type=str,
                    default=str(REPO_ROOT / "results/training/siamese_svm_bootstrap"))
    ap.add_argument("--docs_dir",       type=str,
                    default=str(REPO_ROOT / "fl-orchestration/docs"))
    args = ap.parse_args()

    out_dir  = Path(args.out_dir);   out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = Path(args.docs_dir); docs_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "ensemble_svm_partial"

    # -- 1. Load package -------------------------------------------------------
    print("\n=== 1. Loading Siamese SVM package ===")
    base_model, scaler, clone_params = load_svm_package(Path(args.model_pkl))

    # -- 2. Fixed pair dataset -------------------------------------------------
    print("\n=== 2. Building fixed pair evaluation set ===")
    X_sparse, y, filenames = load_tfidf_data(
        Path(args.matrix_path), Path(args.labels_path), Path(args.filenames_path)
    )
    X_dense = X_sparse.toarray()
    X_diff, pair_labels, left_idx, right_idx = generate_fixed_pairs(
        X_dense, y, max_pairs=args.max_pairs, seed=args.pair_seed
    )
    X_diff_scaled = scaler.transform(X_diff)   # same scaler for all 25 models

    # -- 3. Bootstrap ensemble -------------------------------------------------
    ensemble_path = out_dir / "ensemble_svm.pkl"
    if args.eval_only and ensemble_path.exists():
        print(f"\n=== 3. Loading existing ensemble ({ensemble_path}) ===")
        with open(ensemble_path, "rb") as f:
            ensemble = pickle.load(f)
        print(f"[ensemble] loaded {len(ensemble)} calibrated models")
    else:
        print(f"\n=== 3. Training {args.n_bootstrap}-model bootstrap ensemble ===")
        print(f"    cv='prefit'  ->  exactly {args.n_bootstrap} LinearSVC fits total")
        print(f"    Checkpoints  ->  {checkpoint_dir}/model_XX.pkl")
        ensemble = build_bootstrap_ensemble(
            X_diff_scaled, pair_labels, clone_params,
            n_bootstrap=args.n_bootstrap,
            checkpoint_dir=checkpoint_dir,
        )
        # consolidate checkpoints into final ensemble.pkl
        with open(ensemble_path, "wb") as f:
            pickle.dump(ensemble, f)
        print(f"[save] ensemble ({len(ensemble)} models) -> {ensemble_path}")

    # -- 4. MPD scores ---------------------------------------------------------
    print("\n=== 4. Computing MPD scores (Algorithm 2) ===")
    mpd_scores, u0, u1, mean_q = compute_mpd(ensemble, X_diff_scaled)
    print(
        f"[mpd] n={len(mpd_scores)}, "
        f"min={mpd_scores.min():.6f}, max={mpd_scores.max():.6f}, "
        f"mean={mpd_scores.mean():.6f}"
    )

    # -- 5. Validation & statistics --------------------------------------------
    print("\n=== 5. Validation & statistics ===")
    summary = validate_and_summarise(
        ensemble, mpd_scores, pair_labels, args.n_bootstrap, out_dir
    )

    csv_path = out_dir / "clean_mpd_svm_summary.csv"
    pd.DataFrame({
        "pair_id":          np.arange(len(pair_labels)),
        "left_idx":         left_idx,
        "right_idx":        right_idx,
        "same_class_label": pair_labels,
        "mpd_score":        mpd_scores,
        "u0":               u0,
        "u1":               u1,
        "mean_q":           mean_q,
    }).to_csv(csv_path, index=False)
    print(f"[save] CSV  -> {csv_path}")

    # -- 6. Plots --------------------------------------------------------------
    print("\n=== 6. Generating plots ===")
    plot_log_histogram(mpd_scores, pair_labels, summary, docs_dir)
    plot_ecdf(mpd_scores, summary, docs_dir)
    raw_margins = base_model.decision_function(X_diff_scaled)
    plot_calibration_check(raw_margins, mpd_scores, pair_labels, docs_dir)

    # -- Terminal validation report --------------------------------------------
    ov  = summary["overall"]
    sc  = summary["same_class"]
    dc  = summary["diff_class"]
    sep = "=" * 64
    print(f"\n{sep}")
    print("  SIAMESE SVM MPD CLEAN-BASELINE -- VALIDATION REPORT")
    print(sep)
    print(f"  Ensemble count    : {len(ensemble):>3d}  [OK] (expected {args.n_bootstrap})")
    print(f"  NaN in MPD        : {int(np.isnan(mpd_scores).sum()):>3d}  [OK]")
    print(f"  Inf in MPD        : {int(np.isinf(mpd_scores).sum()):>3d}  [OK]")
    in_bounds = bool(np.all((mpd_scores >= 0) & (mpd_scores <= 0.5)))
    print(f"  Bounds [0, 0.5]   :  {in_bounds}  [OK]")
    print(f"  Pairs evaluated   : {ov['count']:>5d}  "
          f"({sc['count']} same-class | {dc['count']} diff-class)")
    print()
    print(f"  Overall  min={ov['min']:.6f}  max={ov['max']:.6f}  "
          f"mean={ov['mean']:.6f}  median={ov['median']:.6f}")
    print(f"  Thresholds   P90={ov['p90']:.6f}  P95={ov['p95']:.6f}  "
          f"P99={ov['p99']:.6f}")
    print()
    print(f"  Same-class   mean={sc['mean']:.6f}  P90={sc['p90']:.6f}  "
          f"P95={sc['p95']:.6f}  P99={sc['p99']:.6f}")
    print(f"  Diff-class   mean={dc['mean']:.6f}  P90={dc['p90']:.6f}  "
          f"P95={dc['p95']:.6f}  P99={dc['p99']:.6f}")
    print()
    print("  Artifacts saved:")
    print(f"    {ensemble_path}")
    print(f"    {out_dir / 'clean_mpd_svm_summary.json'}")
    print(f"    {csv_path}")
    print(f"    {docs_dir / 'mpd_svm_clean_baseline.png'}")
    print(f"    {docs_dir / 'mpd_svm_ecdf_thresholds.png'}")
    print(f"    {docs_dir / 'mpd_svm_calibration_check.png'}")
    print(sep)
    print("  Clean baseline verified.  No adversarial / FIGA / triage code.")
    print(sep + "\n")


if __name__ == "__main__":
    main()
