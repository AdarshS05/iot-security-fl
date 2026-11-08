"""
shap_explainability.py

Adds SHAP-based explainability on top of the existing pipeline:
  - LightGBM classifier  -> shap.TreeExplainer (exact, fast)
  - Siamese SVM          -> shap.Explainer (model-agnostic, sampled) via a
                             pairwise-diff wrapper matching train_siamese_svm.py

Usage:
    python shap_explainability.py --model_path model.pkl --data_csv features.csv

    # also explain the Siamese SVM
    python shap_explainability.py --model_path model.pkl --data_csv features.csv \\
        --explain_svm --svm_pipeline_path models/siamese_svm1.pkl \\
        --tfidf_matrix data/tfidf/tfidf_matrix.npz --tfidf_labels data/tfidf/labels.pkl

Outputs (written to --output_dir):
    lightgbm_beeswarm.png, lightgbm_bar.png, lightgbm_waterfall_sample_<i>.png
    lightgbm_shap_values.csv
    svm_beeswarm.png, svm_waterfall_sample_<i>.png, svm_shap_values.csv  (if --explain_svm)
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: never try to open a display window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import joblib
from scipy import sparse
from pandas.api.types import (
    is_bool_dtype,
    is_object_dtype,
    is_string_dtype,
)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_feature_matrix(csv_path: Path, label_col: str = "label"):
    """Load a feature CSV and split into (X, feature_names), dropping any
    label/id columns and non-numeric columns. Uses pd.api.types checks
    instead of select_dtypes string names, since select_dtypes with string
    dtype names is inconsistent across pandas versions (same fix as
    predict_lightgbm.py)."""
    df = pd.read_csv(csv_path)

    drop_cols = [c for c in df.columns if c.lower() in (label_col, "id", "filename", "sha256")]
    df = df.drop(columns=drop_cols, errors="ignore")

    non_numeric = [
        c for c in df.columns
        if is_object_dtype(df[c]) or is_bool_dtype(df[c]) or is_string_dtype(df[c])
        or isinstance(df[c].dtype, pd.CategoricalDtype)
    ]
    if non_numeric:
        print(f"[warn] dropping non-numeric columns before SHAP: {non_numeric}")
        df = df.drop(columns=non_numeric)

    return df, list(df.columns)


_FEATURE_LIST_KEY_CANDIDATES = [
    "feature_names", "features", "selected_features", "feature_cols",
    "used_features", "columns",
]


def get_expected_feature_names(raw_obj, model, feature_key: str = None):
    """Figure out which features (and in what order) the model was actually
    trained on, so the SHAP input can be aligned to it. Tries, in order:
    an explicit --feature_names_key, a feature list stored alongside the
    model in the pickle, then the model's own metadata (LightGBM stores
    this on the fitted object/booster). Returns None if nothing is found,
    meaning no alignment is possible and the caller must fall back to
    using every column in the CSV as-is."""
    if feature_key is not None:
        if not isinstance(raw_obj, dict) or feature_key not in raw_obj:
            raise KeyError(f"--feature_names_key '{feature_key}' not found in the loaded pickle")
        return list(raw_obj[feature_key])

    if isinstance(raw_obj, dict):
        for key in _FEATURE_LIST_KEY_CANDIDATES:
            if key in raw_obj:
                print(f"[info] using feature list from pickle key '{key}'")
                return list(raw_obj[key])

    # LightGBM sklearn API (LGBMClassifier/Regressor) sets this after fit
    names = getattr(model, "feature_name_", None)
    if names is not None:
        print("[info] using feature list from model.feature_name_")
        return list(names)

    # Raw lgb.Booster
    if hasattr(model, "feature_name") and callable(model.feature_name):
        try:
            names = model.feature_name()
            if names:
                print("[info] using feature list from model.feature_name()")
                return list(names)
        except Exception:
            pass

    return None


def align_features(X: pd.DataFrame, expected_names):
    """Reorder/filter X to exactly the columns (and order) the model expects.
    Order matters, not just presence — LightGBM references features by
    position internally, so a mismatched order silently produces wrong
    attributions rather than an error."""
    missing = [f for f in expected_names if f not in X.columns]
    if missing:
        raise ValueError(
            f"The feature CSV is missing {len(missing)} column(s) the model expects: {missing}. "
            f"Make sure --data_csv was produced by the same extractor version the model was trained on."
        )

    extra = [c for c in X.columns if c not in expected_names]
    if extra:
        print(f"[info] dropping {len(extra)} extra column(s) not used by the model: {extra}")

    return X[expected_names]


def load_pickled(path: Path):
    """Load a model/pipeline file saved with plain pickle, joblib (with or
    without compression), or LightGBM's native text format. Tries each in
    turn instead of assuming one, since 'model.pkl' is saved differently
    depending on which tool wrote it."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except pickle.UnpicklingError:
        pass  # fall through and try the alternatives below

    try:
        return joblib.load(path)
    except Exception:
        pass

    try:
        import lightgbm as lgb
        return lgb.Booster(model_file=str(path))
    except Exception as e:
        raise RuntimeError(
            f"Could not load {path} as pickle, joblib, or a LightGBM native model file. "
            f"Check how the model was originally saved (pickle.dump / joblib.dump / "
            f"Booster.save_model) and re-save it accordingly if needed. Last error: {e}"
        )


# --------------------------------------------------------------------------- #
# LightGBM explainability (TreeExplainer path)
# --------------------------------------------------------------------------- #

def explain_lightgbm(model, X: pd.DataFrame, feature_names, output_dir: Path, sample_index: int):
    print(f"[lightgbm] building TreeExplainer over {X.shape[0]} samples, {X.shape[1]} features")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)  # Explanation object: .values, .base_values, .data

    # Global view: ranked, signed feature effects across all samples
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_dir / "lightgbm_beeswarm.png", dpi=150)
    plt.close()

    # Global view: mean |SHAP value| bar ranking
    plt.figure()
    shap.plots.bar(shap_values, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_dir / "lightgbm_bar.png", dpi=150)
    plt.close()

    # Local view: one sample's full additive breakdown
    idx = min(sample_index, X.shape[0] - 1)
    plt.figure()
    shap.plots.waterfall(shap_values[idx], show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(output_dir / f"lightgbm_waterfall_sample_{idx}.png", dpi=150)
    plt.close()

    # Raw values for downstream use (e.g. comparing against GA mutator targets)
    out = pd.DataFrame(shap_values.values, columns=feature_names)
    out.insert(0, "base_value", shap_values.base_values)
    out.to_csv(output_dir / "lightgbm_shap_values.csv", index=False)

    print(f"[lightgbm] wrote plots + shap values to {output_dir}")
    return shap_values


# --------------------------------------------------------------------------- #
# Siamese SVM explainability (model-agnostic path)
#
# Matches train_siamese_svm.py exactly:
#   - trained on |x_i - x_j| (absolute difference between two TF-IDF vectors)
#   - label 1 = same-class pair, label 0 = different-class pair
#   - saved as joblib.dump({"model": LinearSVC, "scaler": StandardScaler}, ...)
#
# Since the SVM scores *pairs*, not single binaries, one side of every pair is
# fixed to a class "anchor" (the mean TF-IDF vector of one class, computed
# from your training data) so the scoring function becomes single-input and
# SHAP-compatible: score(x) = decision_function(scaler(|x - anchor|)).
# With the benign class as the anchor, a higher score reads as "more
# benign-like" and a lower/negative score reads as "more malware-like".
# --------------------------------------------------------------------------- #

def load_tfidf_dataset(matrix_path: Path, labels_path: Path):
    """Mirrors the loader in train_siamese_svm.py so the SVM explainer sees
    data in exactly the same shape/labeling convention it was trained on."""
    X = sparse.load_npz(matrix_path)
    with open(labels_path, "rb") as f:
        metadata = pickle.load(f)

    if isinstance(metadata, dict):
        labels = metadata["labels"]
    elif isinstance(metadata, tuple):
        _, labels = metadata
    elif isinstance(metadata, list):
        labels = metadata
    else:
        raise ValueError(f"Unrecognized labels.pkl format: {type(metadata)}")

    labels = np.asarray(labels)
    if len(labels) != X.shape[0]:
        raise ValueError(f"TF-IDF matrix has {X.shape[0]} rows but labels has {len(labels)} entries")
    return X, labels


def compute_class_anchor(X_sparse, labels: np.ndarray, anchor_class: int) -> np.ndarray:
    """Mean TF-IDF vector for one class — the fixed 'other half' of the pair
    that turns the pairwise Siamese SVM into a single-input function."""
    mask = labels == anchor_class
    if mask.sum() == 0:
        raise ValueError(f"No samples with label {anchor_class} found to build an anchor from")
    anchor = np.asarray(X_sparse[mask].mean(axis=0)).ravel()
    print(f"[svm] anchor built from {int(mask.sum())} class-{anchor_class} samples")
    return anchor


def build_svm_predict_fn(svm_pipeline: dict, anchor: np.ndarray):
    """Wraps the trained pairwise-diff SVM into a single-input scoring
    function SHAP can call: predict_fn(x) = decision_function(scaler(|x - anchor|))."""
    model = svm_pipeline["model"]
    scaler = svm_pipeline["scaler"]

    def predict_fn(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        diffs = np.abs(X - anchor[None, :])
        scaled = scaler.transform(diffs)
        return model.decision_function(scaled)

    return predict_fn


def select_top_variance_columns(X: np.ndarray, k: int):
    """Picks the k highest-variance TF-IDF columns in the sampled data.
    A full vocabulary (often thousands of columns) is computationally
    infeasible for a sampled explainer — k=0 disables this and uses every
    column, which will be slow."""
    if k <= 0 or k >= X.shape[1]:
        return np.arange(X.shape[1])
    variances = X.var(axis=0)
    return np.argsort(variances)[::-1][:k]


def build_reduced_predict_fn(full_predict_fn, anchor: np.ndarray, active_idx: np.ndarray):
    """Restricts SHAP's view to a manageable subset of TF-IDF columns by
    holding every other column fixed at the anchor's value, so only the
    selected terms actually vary during explanation."""
    def reduced_fn(X_reduced: np.ndarray) -> np.ndarray:
        X_reduced = np.atleast_2d(X_reduced)
        full = np.tile(anchor, (X_reduced.shape[0], 1))
        full[:, active_idx] = X_reduced
        return full_predict_fn(full)
    return reduced_fn


def explain_siamese_svm(predict_fn, X: pd.DataFrame, feature_names, output_dir: Path,
                         sample_index: int, background_size: int, random_state: int,
                         eval_sample_size: int):
    n_features = X.shape[1]
    if n_features > 50:
        print(f"[svm] warning: {n_features} features is expensive for a sampled explainer; "
              f"consider a smaller --tfidf_top_k")

    background = shap.sample(X, min(background_size, X.shape[0]), random_state=random_state)

    print(f"[svm] building model-agnostic explainer with background size {len(background)}")
    explainer = shap.Explainer(predict_fn, background)

    eval_X = X.sample(min(eval_sample_size, X.shape[0]), random_state=random_state)
    shap_values = explainer(eval_X)

    plt.figure()
    shap.plots.beeswarm(shap_values, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_dir / "svm_beeswarm.png", dpi=150)
    plt.close()

    idx = min(sample_index, eval_X.shape[0] - 1)
    plt.figure()
    shap.plots.waterfall(shap_values[idx], show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(output_dir / f"svm_waterfall_sample_{idx}.png", dpi=150)
    plt.close()

    out = pd.DataFrame(shap_values.values, columns=feature_names)
    out.insert(0, "base_value", shap_values.base_values)
    out.to_csv(output_dir / "svm_shap_values.csv", index=False)

    print(f"[svm] wrote plots + shap values to {output_dir}")
    return shap_values


# --------------------------------------------------------------------------- #
# Model extraction (handles pickles that wrap the model in a dict)
# --------------------------------------------------------------------------- #

_MODEL_KEY_CANDIDATES = [
    "model", "booster", "clf", "classifier", "estimator",
    "lgb_model", "lightgbm_model", "best_estimator_", "gbm",
]


def extract_model(obj, model_key: str = None):
    """Unwrap the actual model from a loaded pickle. Some training scripts
    save {'model': ..., 'feature_names': ..., 'scaler': ...} instead of the
    bare model object, so a plain pickle.load()/joblib.load() hands back a
    dict rather than something TreeExplainer can use."""
    if not isinstance(obj, dict):
        return obj  # already a model object, nothing to unwrap

    if model_key is not None:
        if model_key not in obj:
            raise KeyError(
                f"--model_key '{model_key}' not found in the loaded dict. "
                f"Available keys: {list(obj.keys())}"
            )
        return obj[model_key]

    for key in _MODEL_KEY_CANDIDATES:
        if key in obj:
            print(f"[info] model.pkl is a dict — using key '{key}' as the model")
            return obj[key]

    # fall back: look for any value that looks like a fitted model (has .predict)
    predict_like = [k for k, v in obj.items() if hasattr(v, "predict")]
    if len(predict_like) == 1:
        key = predict_like[0]
        print(f"[info] model.pkl is a dict — auto-detected key '{key}' as the model")
        return obj[key]

    raise ValueError(
        f"model.pkl loaded as a dict but the actual model could not be auto-detected. "
        f"Available keys: {list(obj.keys())}. "
        f"Re-run with --model_key <key> to specify which one holds the trained model."
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="SHAP explainability for LightGBM and Siamese SVM")
    parser.add_argument("--model_path", required=True, help="Path to pickled LightGBM model")
    parser.add_argument("--data_csv", required=True, help="Path to feature CSV")
    parser.add_argument("--output_dir", default="shap_outputs", help="Directory for plots/CSVs")
    parser.add_argument("--sample_index", type=int, default=0,
                         help="Row index to use for the local waterfall plot")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--model_key", default=None,
                         help="If model.pkl unpickles to a dict, the key holding the "
                              "actual trained model (auto-detected if not given)")
    parser.add_argument("--feature_names_key", default=None,
                         help="If model.pkl unpickles to a dict, the key holding the list "
                              "of features the model was actually trained on (auto-detected "
                              "if not given, falls back to model metadata)")
    parser.add_argument("--drop_features", default=None,
                         help="Comma-separated list of columns to drop from --data_csv before "
                              "SHAP, for when no feature list can be auto-detected from the model")

    parser.add_argument("--explain_svm", action="store_true",
                         help="Also run the model-agnostic explainer on the Siamese SVM")
    parser.add_argument("--svm_pipeline_path", default=None,
                         help="Path to the joblib-saved {'model','scaler'} dict from train_siamese_svm.py")
    parser.add_argument("--tfidf_matrix", default=None,
                         help="Path to the TF-IDF matrix (.npz) used to train the SVM")
    parser.add_argument("--tfidf_labels", default=None,
                         help="Path to the labels.pkl matching --tfidf_matrix")
    parser.add_argument("--anchor_class", type=int, default=0,
                         help="Class (0=benign, 1=malware) whose mean TF-IDF vector is used "
                              "as the fixed anchor for the pairwise SVM")
    parser.add_argument("--tfidf_top_k", type=int, default=30,
                         help="Explain only the top-k highest-variance TF-IDF columns "
                              "(0 = use every column — slow for large vocabularies)")
    parser.add_argument("--background_size", type=int, default=100,
                         help="Background sample size for the SVM explainer")
    parser.add_argument("--eval_sample_size", type=int, default=200,
                         help="Number of rows to actually explain for the SVM (kept small — expensive)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, feature_names = load_feature_matrix(Path(args.data_csv))
    raw_obj = load_pickled(Path(args.model_path))
    model = extract_model(raw_obj, args.model_key)

    if args.drop_features:
        drop = [c.strip() for c in args.drop_features.split(",")]
        present = [c for c in drop if c in X.columns]
        if present:
            print(f"[info] dropping {len(present)} column(s) per --drop_features: {present}")
            X = X.drop(columns=present)
        feature_names = list(X.columns)
    else:
        expected = get_expected_feature_names(raw_obj, model, args.feature_names_key)
        if expected is not None:
            X = align_features(X, expected)
            feature_names = expected
        else:
            print(
                "[warn] could not determine which features the model was trained on — "
                "using every column in the CSV as-is. If TreeExplainer errors on a shape "
                "mismatch below, re-run with --drop_features 'col_a,col_b' or "
                "--feature_names_key <key>."
            )

    explain_lightgbm(model, X, feature_names, output_dir, args.sample_index)

    if args.explain_svm:
        if not (args.svm_pipeline_path and args.tfidf_matrix and args.tfidf_labels):
            print("[error] --explain_svm requires --svm_pipeline_path, --tfidf_matrix, "
                  "and --tfidf_labels", file=sys.stderr)
            sys.exit(1)

        X_tfidf_sparse, tfidf_labels = load_tfidf_dataset(Path(args.tfidf_matrix), Path(args.tfidf_labels))
        anchor = compute_class_anchor(X_tfidf_sparse, tfidf_labels, args.anchor_class)

        svm_pipeline = load_pickled(Path(args.svm_pipeline_path))
        full_predict_fn = build_svm_predict_fn(svm_pipeline, anchor)

        rng = np.random.default_rng(args.random_state)
        sample_n = min(args.background_size + args.eval_sample_size, X_tfidf_sparse.shape[0])
        sample_idx = rng.choice(X_tfidf_sparse.shape[0], size=sample_n, replace=False)
        X_sample = np.asarray(X_tfidf_sparse[sample_idx].todense())

        active_idx = select_top_variance_columns(X_sample, args.tfidf_top_k)
        print(f"[svm] explaining {len(active_idx)} of {X_sample.shape[1]} TF-IDF columns "
              f"(highest-variance terms; rest held at the anchor value)")

        reduced_predict_fn = build_reduced_predict_fn(full_predict_fn, anchor, active_idx)
        X_reduced = pd.DataFrame(X_sample[:, active_idx],
                                  columns=[f"tfidf_term_{i}" for i in active_idx])

        explain_siamese_svm(
            reduced_predict_fn, X_reduced, list(X_reduced.columns), output_dir,
            args.sample_index, args.background_size, args.random_state, args.eval_sample_size,
        )

    print("[done]")


if __name__ == "__main__":
    main()

