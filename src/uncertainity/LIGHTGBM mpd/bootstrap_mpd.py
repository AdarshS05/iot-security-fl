import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

REPO_ROOT = Path(__file__).resolve().parents[3]

import joblib


def load_baseline_package(pkl_path: Path):
    try:
        with open(pkl_path, "rb") as f:
            saved = pickle.load(f)
    except Exception:
        saved = joblib.load(pkl_path)

    if not isinstance(saved, dict):
        raise RuntimeError(
            f"Expected a dict package at {pkl_path}, got {type(saved)}"
        )

    required = {"model", "feature_names"}
    missing = required - set(saved.keys())
    if missing:
        raise RuntimeError(f"Package at {pkl_path} missing keys: {missing}")

    model = saved["model"]
    feature_names = list(saved["feature_names"])
    label_encoder = saved.get("label_encoder", None)

    params = model.get_params()
    params.pop("random_state", None)

    print(f"[baseline] loaded package from {pkl_path.name}")
    print(f"[baseline] n_features = {len(feature_names)}: {feature_names}")
    print(f"[baseline] params: {params}")

    return model, params, feature_names, label_encoder


def load_dataset(path: Path, feature_names: list, label_col: str, label_encoder=None):
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".pkl":
        df = pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    print(f"[data] {path.name}: shape={df.shape}")

    missing_feats = [f for f in feature_names if f not in df.columns]
    if missing_feats:
        raise ValueError(
            f"CSV is missing model features: {missing_feats}. "
            f"Available columns: {df.columns.tolist()}"
        )

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not in CSV columns: {df.columns.tolist()}")

    print(f"[data] label balance:\n{df[label_col].value_counts()}")

    # select EXACTLY the 11 model features, in the model's original order
    X = df[feature_names].copy()

    # cast any bool columns to int — LightGBM/pandas dtype check is strict
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    # encode labels the same way the saved model was trained (string -> int)
    raw_y = df[label_col].values
    if label_encoder is not None:
        y = label_encoder.transform(raw_y)
        print(f"[data] label classes from saved encoder: {list(label_encoder.classes_)}")
    else:
        # fallback: assume malware=1, benign=0
        y = (pd.Series(raw_y).astype(str).str.lower() == "malware").astype(int).values
        print("[warn] no label_encoder in package — assumed malware=1, benign=0")

    return X, y, df


def fit_one_bootstrap(job_id: int, args: dict):
    """One job = one bootstrap replica. Must be top-level/picklable for multiprocessing."""
    X, y, params, seed = args["X"], args["y"], args["params"], args["seed"]
    n = len(X)
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=n, replace=True)
    X_boot, y_boot = X.iloc[idx], y[idx]

    params_clean = dict(params)
    params_clean.pop("verbosity", None)
    params_clean.pop("random_state", None)
    params_clean.pop("n_jobs", None)

    model = LGBMClassifier(**params_clean, random_state=seed, verbosity=-1, n_jobs=1)
    model.fit(X_boot, y_boot)

    full_acc = accuracy_score(y, model.predict(X))
    return {"model": model, "seed": seed, "full_set_accuracy": full_acc}


def compute_mpd(ensemble: list, X: pd.DataFrame, stat: str = "paper"):
    """
    Computes MPD according to Algorithm 2 of the reference paper.

    q(x) = vector of predict_proba(x)[:,1] outputs across all N ensemble models.
    U0(x) = mean((q(x) - 0)^2)   ← no sqrt; matches Algorithm 2 pseudocode
    U1(x) = mean((q(x) - 1)^2)   ← no sqrt; matches Algorithm 2 pseudocode
    MPD(x) = Uq(x) = min(U0(x), U1(x))

    NOTE — sqrt discrepancy:
      Paper body Eq. 2/3 shows sqrt(mean(...)). Algorithm 2 pseudocode omits
      the sqrt. We implement WITHOUT sqrt to match Algorithm 2 exactly.
      See module docstring for full explanation.
    """
    # preds shape: (N_bootstrap_models, N_samples)
    preds = np.stack([m.predict_proba(X)[:, 1] for m in ensemble], axis=0)
    # U0, U1: mean squared deviation from class-0 and class-1 corners
    u0 = np.mean((preds - 0.0) ** 2, axis=0)
    u1 = np.mean((preds - 1.0) ** 2, axis=0)
    # MPD = minimum deviation = uncertainty score
    mpd = np.minimum(u0, u1)
    mean_q = np.mean(preds, axis=0)
    pred_class = (mean_q >= 0.5).astype(int)

    if stat in ["paper", "mpd"]:
        return mpd, u0, u1, pred_class
    elif stat == "std":
        return preds.std(axis=0), u0, u1, pred_class
    elif stat == "min_abs_dev":
        median = np.median(preds, axis=0)
        return np.min(np.abs(preds - median), axis=0), u0, u1, pred_class
    else:
        raise ValueError(f"unknown stat: {stat}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--label_col", type=str, default="label")
    ap.add_argument("--baseline_pkl", type=str,
                     default=str(REPO_ROOT / "models" / "lightgbm_model.pkl"))
    ap.add_argument("--ensemble_pkl", type=str,
                     default=str(REPO_ROOT / "results" / "training" / "lightgbm_bootstrap" / "ensemble.pkl"))
    ap.add_argument("--n_bootstrap", type=int, default=25)
    ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--stat", type=str, default="paper", choices=["paper", "mpd", "std", "min_abs_dev"])
    ap.add_argument("--eval_only", action="store_true", help="Evaluate existing ensemble.pkl without retraining")
    ap.add_argument("--out_dir", type=str,
                     default=str(REPO_ROOT / "results" / "training" / "lightgbm_bootstrap"))
    ap.add_argument("--out_csv", type=str, default=None,
                     help="Custom CSV filename/path (defaults to clean_mpd_baseline_v2.csv)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_model, params, feature_names, label_encoder = load_baseline_package(
        Path(args.baseline_pkl)
    )
    X, y, raw_df = load_dataset(Path(args.data_path), feature_names, args.label_col, label_encoder)

    ensemble_path = Path(args.ensemble_pkl)
    if args.eval_only or ensemble_path.exists():
        print(f"[ensemble] loading existing ensemble from {ensemble_path}")
        with open(ensemble_path, "rb") as f:
            ensemble = pickle.load(f)
    else:
        ensemble = []
        accs = []
        for jid in range(args.n_bootstrap):
            res = fit_one_bootstrap(jid, {"X": X, "y": y, "params": params, "seed": jid})
            ensemble.append(res["model"])
            accs.append(res["full_set_accuracy"])
        accs = np.array(accs)

        baseline_acc = accuracy_score(y, baseline_model.predict(X))

        print(f"\n[sanity check] baseline accuracy: {baseline_acc:.4f}")
        print(f"[sanity check] bootstrap replica accuracy: mean={accs.mean():.4f}, "
              f"std={accs.std():.4f}, min={accs.min():.4f}, max={accs.max():.4f}")

        with open(ensemble_path, "wb") as f:
            pickle.dump(ensemble, f)
        print(f"[save] ensemble ({len(ensemble)} models) -> {ensemble_path}")

    mpd_scores, u0, u1, pred_class = compute_mpd(ensemble, X, stat=args.stat)

    # Build output dataframe preserving sample_id and sample identifiers if present
    res_dict = {
        "sample_id": np.arange(len(raw_df)),
        "mpd_score": mpd_scores,
        "label": y,
        "u0": u0,
        "u1": u1,
        "predicted_class": pred_class,
    }
    for id_col in ["file_name", "filename", "sha256", "md5"]:
        if id_col in raw_df.columns:
            res_dict[id_col] = raw_df[id_col].values

    baseline_dist = pd.DataFrame(res_dict)

    if args.out_csv:
        dist_path = Path(args.out_csv)
    else:
        dist_path = out_dir / "clean_mpd_baseline_v2.csv"

    baseline_dist.to_csv(dist_path, index=False)
    print(f"\n[save] paper-faithful clean MPD baseline distribution -> {dist_path}")
    print(f"[mpd paper formula] evaluated {len(ensemble)} models on {len(X)} samples")
    print(f"[mpd stats] min={mpd_scores.min():.6f}, max={mpd_scores.max():.6f}, "
          f"mean={mpd_scores.mean():.6f}, median={np.median(mpd_scores):.6f}, "
          f"p95={np.percentile(mpd_scores, 95):.6f}")


if __name__ == "__main__":
    main()
