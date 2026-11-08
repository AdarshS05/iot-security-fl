"""
End-to-End Orchestrator for MPD Uncertainty Pipeline (Stages 1 - 7)
Follows Santhosh, Gressel & Darling (2022) and Darling (2019) verbatim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
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

from mpd_core import compute_mpd, validate_mpd_integrity
from thresholds import compute_percentile_thresholds, flag_samples
from lgbm_mpd import LGBMMPDPipeline
from siamese_mpd import SiameseMPDPipeline, generate_pair_indices
from evaluate import (
    compute_class_statistics,
    run_integrity_audit,
    plot_lgbm_validation,
    plot_siamese_plots,
    evaluate_calibration_quality,
    evaluate_error_auroc,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_all")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="End-to-End MPD Pipeline on Canonical Models")
    parser.add_argument("--base-dir", type=str, default="/data/iot-malware-fl/iot-malware-abhi")
    parser.add_argument("--lgbm-model",type=str,default="models/final_n1_adv/lightgbm_adversarial.txt",)
    parser.add_argument("--lgbm-train-csv",type=str,default="models/final_n1_adv/adversarial_train_rows.csv",)
    parser.add_argument("--lgbm-threshold",type=float,default=0.5,)
    parser.add_argument("--n-replicas", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", "--out-dir", dest="output_dir", type=str, default="results/uncertainty_final_n1")
    parser.add_argument("--replicas-dir", type=str, default=None, help="Directory containing pre-trained replicas")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_p = Path(args.output_dir)
    out_dir = out_p if out_p.is_absolute() else (base_dir / args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if args.replicas_dir:
        rep_p = Path(args.replicas_dir)
        replicas_base = rep_p if rep_p.is_absolute() else (base_dir / args.replicas_dir)
    else:
        replicas_base = out_dir

    print("================================================================================")
    print("STAGE 1: Source dataset and baseline initialization")
    print("================================================================================")
    # 1. Tabular features & splits
    tab_df = pd.read_csv(base_dir / "data/features/tabular_n1/tabular_feature.csv").set_index("sample_id")
    train_split = pd.read_csv(base_dir / "data/splits/final_n1/train.csv")
    val_split = pd.read_csv(base_dir / "data/splits/final_n1/validation.csv")
    test_split = pd.read_csv(base_dir / "data/splits/final_n1/test.csv")

    feature_cols = LGBMMPDPipeline.FEATURE_COLUMNS
    X_train_tab = tab_df.loc[train_split["sample_id"], feature_cols].values.astype(np.float64)
    y_train_tab = train_split["label"].map({"benign": 0, "malware": 1}).values.astype(int)

    X_val_tab = tab_df.loc[val_split["sample_id"], feature_cols].values.astype(np.float64)
    y_val_tab = val_split["label"].map({"benign": 0, "malware": 1}).values.astype(int)
    val_sample_ids = val_split["sample_id"].tolist()

    X_test_tab = tab_df.loc[test_split["sample_id"], feature_cols].values.astype(np.float64)
    y_test_tab = test_split["label"].map({"benign": 0, "malware": 1}).values.astype(int)
    test_sample_ids = test_split["sample_id"].tolist()

    # Separate LightGBM training data: clean + adversarial
    adv_csv = Path(args.lgbm_train_csv)
    if not adv_csv.is_absolute():
        adv_csv = base_dir / adv_csv

    adv_df = pd.read_csv(adv_csv)

    # Validate adversarial training rows
    missing = set(feature_cols + ["label"]) - set(adv_df.columns)
    if missing:
        raise ValueError(f"Missing adversarial columns: {missing}")

    if not (adv_df["label"] == "malware").all():
        raise ValueError("Expected all adversarial rows to be malware")

    if "split" in adv_df.columns:
        if not (adv_df["split"] == "train").all():
            raise ValueError("Adversarial rows must belong to the train split")

    X_adv = adv_df[feature_cols].values.astype(np.float64)
    y_adv = np.ones(len(X_adv), dtype=int)

    X_train_lgbm = np.vstack([X_train_tab, X_adv])
    y_train_lgbm = np.concatenate([y_train_tab, y_adv])

    print(f"LightGBM training data: {len(X_train_tab)} clean + {len(X_adv)} adversarial = {len(X_train_lgbm)} rows")

    if len(X_train_lgbm) != 898:
        raise ValueError(f"Expected 898 combined training rows, got {len(X_train_lgbm)}")


    # 2. TF-IDF matrices
    X_train_tfidf = sparse.load_npz(base_dir / "data/features/tfidf_n1/X_train.npz").tocsr()
    y_train_tfidf = y_train_tab  # exact alignment verified in Phase 0

    X_val_tfidf = sparse.load_npz(base_dir / "data/features/tfidf_n1/X_validation.npz").tocsr()
    y_val_tfidf = y_val_tab

    X_test_tfidf = sparse.load_npz(base_dir / "data/features/tfidf_n1/X_test.npz").tocsr()
    y_test_tfidf = y_test_tab

    print(f"Loaded Train: {X_train_tab.shape[0]} samples, Val: {X_val_tab.shape[0]}, Test: {X_test_tab.shape[0]}")
    print(f"Loaded TF-IDF Train: {X_train_tfidf.shape}, Val: {X_val_tfidf.shape}, Test: {X_test_tfidf.shape}")

    print("\n================================================================================")
    print("STAGE 2: Bootstrap resampling and fresh model training (Algorithm 1)")
    print("================================================================================")
    # A. LightGBM Replicas
    lgbm_model_path = Path(args.lgbm_model)
    if not lgbm_model_path.is_absolute():
        lgbm_model_path = base_dir / lgbm_model_path

    lgbm_pipe = LGBMMPDPipeline(base_dir=base_dir, n_replicas=args.n_replicas, replicas_dir=replicas_base / "lgbm/replicas", canonical_model_path=lgbm_model_path, canonical_threshold=args.lgbm_threshold,)

    lgbm_replicas = lgbm_pipe.fit_or_load_replicas(X_train_lgbm, y_train_lgbm)
    # B. Siamese SVM Replicas
    siamese_pipe = SiameseMPDPipeline(
        base_dir=base_dir,
        n_replicas=args.n_replicas,
        replicas_dir=replicas_base / "siamese/replicas",
    )
    siamese_replicas = siamese_pipe.fit_or_load_replicas(
        X_train=X_train_tfidf,
        y_train=y_train_tfidf,
        pairs_per_relation=20000,
        n_oob_pairs=2000,
    )
    print(f"Siamese SVM Replicas ready: {len(siamese_replicas)}")

    print("\n================================================================================")
    print("STAGE 3 & 4: Ensemble prediction and MPD computation on Clean Validation")
    print("================================================================================")
    # 1. LightGBM Validation Clean
    lgbm_val_df = lgbm_pipe.evaluate_dataset(
        X=X_val_tab,
        sample_ids=val_sample_ids,
        labels=y_val_tab,
        split_name="validation",
    )

    # 2. Siamese Query-level Validation Clean
    siamese_val_query_df = siamese_pipe.evaluate_queries(
        X_queries=X_val_tfidf,
        X_train=X_train_tfidf,
        sample_ids=val_sample_ids,
        labels=y_val_tfidf,
        split_name="validation",
    )

    # 3. Siamese Pair-level Validation Clean: fixed seed=42 balanced 2000 pairs (1000 same, 1000 diff)
    rng_val_pairs = np.random.default_rng(args.seed)
    val_a, val_b, val_pair_y = generate_pair_indices(y_val_tfidf, pairs_per_relation=1000, rng=rng_val_pairs)
    siamese_val_pair_df = siamese_pipe.evaluate_pairs(
        X_mat=X_val_tfidf,
        indices_a=val_a,
        indices_b=val_b,
        pair_labels=val_pair_y,
        split_name="validation",
    )

    print("\n================================================================================")
    print("STAGE 5: Thresholds and per-sample flags (T90, T95, T99 on Clean Validation)")
    print("================================================================================")
    thresholds_lgbm = compute_percentile_thresholds(lgbm_val_df["mpd"].values)
    thresholds_siamese_query = compute_percentile_thresholds(siamese_val_query_df["mpd"].values)
    thresholds_siamese_pair = compute_percentile_thresholds(siamese_val_pair_df["mpd"].values)

    all_thresholds = {
        "lgbm": thresholds_lgbm,
        "siamese_query": thresholds_siamese_query,
        "siamese_pair": thresholds_siamese_pair,
    }
    with open(out_dir / "thresholds.json", "w") as f:
        json.dump(all_thresholds, f, indent=2)

    # Add flags to validation DataFrames
    for thresh_dict, df in [
        (thresholds_lgbm, lgbm_val_df),
        (thresholds_siamese_query, siamese_val_query_df),
        (thresholds_siamese_pair, siamese_val_pair_df),
    ]:
        df["flag_t95"] = flag_samples(df["mpd"].values, thresh_dict["T95"])
        df["flag_t99"] = flag_samples(df["mpd"].values, thresh_dict["T99"])

    lgbm_val_df.to_csv(out_dir / "lgbm_clean_validation_mpd.csv", index=False)
    siamese_val_query_df.to_csv(out_dir / "siamese_query_clean_validation_mpd.csv", index=False)
    siamese_val_pair_df.to_csv(out_dir / "siamese_pair_clean_validation_mpd.csv", index=False)

    print(f"LightGBM Clean Validation Thresholds: T90={thresholds_lgbm['T90']:.6e}, T95={thresholds_lgbm['T95']:.6e}, T99={thresholds_lgbm['T99']:.6e}")
    print(f"Siamese Query Clean Thresholds: T90={thresholds_siamese_query['T90']:.6f}, T95={thresholds_siamese_query['T95']:.6f}, T99={thresholds_siamese_query['T99']:.6f}")
    print(f"Siamese Pair Clean Thresholds: T90={thresholds_siamese_pair['T90']:.6f}, T95={thresholds_siamese_pair['T95']:.6f}, T99={thresholds_siamese_pair['T99']:.6f}")

    print("\n================================================================================")
    print("STAGE 6: Validation Report and Clean Baseline Verification")
    print("================================================================================")
    # Integrity checks
    checks_lgbm = run_integrity_audit(lgbm_val_df, expected_rows=150, expected_k=args.n_replicas, is_pair=False)
    checks_s_query = run_integrity_audit(siamese_val_query_df, expected_rows=150, expected_k=args.n_replicas, is_pair=False)
    checks_s_pair = run_integrity_audit(siamese_val_pair_df, expected_rows=2000, expected_k=args.n_replicas, is_pair=True)

    print("\n--- VALIDATION REPORT: INTEGRITY CHECKS ---")
    for name, cdict in [("LightGBM", checks_lgbm), ("Siamese Query", checks_s_query), ("Siamese Pair", checks_s_pair)]:
        print(f"\n[{name}]")
        for k, (ok, desc) in cdict.items():
            status_str = "OK" if ok else "FAIL"
            print(f"  {k:30s} : [{status_str}] ({desc})")

    # Statistical summaries
    stats_lgbm = compute_class_statistics(lgbm_val_df, val_col="mpd", group_col="label", label_names={0: "Benign", 1: "Malware"})
    stats_s_query = compute_class_statistics(siamese_val_query_df, val_col="mpd", group_col="label", label_names={0: "Benign", 1: "Malware"})
    stats_s_pair = compute_class_statistics(siamese_val_pair_df, val_col="mpd", group_col="label", label_names={0: "Diff-class", 1: "Same-class"})

    stats_lgbm.to_csv(out_dir / "clean_mpd_summary_lgbm.csv", index=False)
    stats_s_query.to_csv(out_dir / "clean_mpd_summary_siamese_query.csv", index=False)
    stats_s_pair.to_csv(out_dir / "clean_mpd_summary_siamese_pair.csv", index=False)

    summary_json = {
        "lgbm": stats_lgbm.to_dict(orient="records"),
        "siamese_query": stats_s_query.to_dict(orient="records"),
        "siamese_pair": stats_s_pair.to_dict(orient="records"),
    }
    with open(out_dir / "clean_mpd_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    print("\n--- STATISTICAL SUMMARY BY CLASS ---")
    print("\n[LightGBM MPD Summary]")
    print(stats_lgbm.to_string(index=False))
    print("\n[Siamese Query MPD Summary]")
    print(stats_s_query.to_string(index=False))
    print("\n[Siamese Pair MPD Summary]")
    print(stats_s_pair.to_string(index=False))

    # Calibration quality (Platt scaling on validation)
    cal_lgbm = evaluate_calibration_quality(lgbm_val_df["label"].values, lgbm_val_df["canonical_prob"].values)
    # Siamese probabilities: query-level mean replica q
    mean_q_s_query = np.mean([siamese_val_query_df[f"q_{k:02d}"].values for k in range(args.n_replicas)], axis=0)
    cal_s_query = evaluate_calibration_quality(siamese_val_query_df["label"].values, mean_q_s_query)
    # Siamese pair probabilities: mean replica q
    mean_q_s_pair = np.mean([siamese_val_pair_df[f"q_{k:02d}"].values for k in range(args.n_replicas)], axis=0)
    cal_s_pair = evaluate_calibration_quality(siamese_val_pair_df["label"].values, mean_q_s_pair)

    cal_quality = {
        "lgbm": cal_lgbm,
        "siamese_query": cal_s_query,
        "siamese_pair": cal_s_pair,
    }
    with open(out_dir / "platt_calibration_quality.json", "w") as f:
        json.dump(cal_quality, f, indent=2)

    print("\n--- PLATT CALIBRATION QUALITY (Brier Score on Validation) ---")
    print(f"  LightGBM Brier Score: {cal_lgbm['brier_score']:.6f}")
    print(f"  Siamese Query Brier Score: {cal_s_query['brier_score']:.6f}")
    print(f"  Siamese Pair Brier Score: {cal_s_pair['brier_score']:.6f}")

    # AUROC of MPD predicting canonical-model errors
    err_lgbm = evaluate_error_auroc(lgbm_val_df["label"].values, lgbm_val_df["canonical_pred"].values, lgbm_val_df["mpd"].values)
    err_s_query = evaluate_error_auroc(siamese_val_query_df["label"].values, siamese_val_query_df["canonical_pred"].values, siamese_val_query_df["mpd"].values)
    err_s_pair = evaluate_error_auroc(siamese_val_pair_df["label"].values, siamese_val_pair_df["canonical_pred"].values, siamese_val_pair_df["mpd"].values)

    err_summary = {
        "lgbm": err_lgbm,
        "siamese_query": err_s_query,
        "siamese_pair": err_s_pair,
    }
    with open(out_dir / "mpd_error_auroc.json", "w") as f:
        json.dump(err_summary, f, indent=2)

    print("\n--- AUROC OF MPD AS PREDICTOR OF CANONICAL-MODEL ERRORS ---")
    print(f"  LightGBM Error AUROC: {err_lgbm['error_auroc']} (errors: {err_lgbm['n_errors']}/150)")
    print(f"  Siamese Query Error AUROC: {err_s_query['error_auroc']} (errors: {err_s_query['n_errors']}/150)")
    print(f"  Siamese Pair Error AUROC: {err_s_pair['error_auroc']} (errors: {err_s_pair['n_errors']}/2000)")

    # Generate Stage 6 plots
    plot_lgbm_validation(lgbm_val_df, thresholds_lgbm, plots_dir)
    plot_siamese_plots(siamese_val_pair_df, thresholds_siamese_pair, plots_dir, prefix="siamese_pair", is_pair=True)
    plot_siamese_plots(siamese_val_query_df, thresholds_siamese_query, plots_dir, prefix="siamese_query", is_pair=False)
    print(f"All Stage 6 plots saved to {plots_dir}.")

    print("\n================================================================================")
    print("FINAL STAGE: Single-Pass Evaluation on TEST Split")
    print("================================================================================")
    # Hard Rule 3: Test split is evaluated exactly ONCE at the end.
    test_lgbm_df = lgbm_pipe.evaluate_dataset(
        X=X_test_tab,
        sample_ids=test_sample_ids,
        labels=y_test_tab,
        split_name="test",
        thresholds=thresholds_lgbm,
    )
    test_lgbm_df.to_csv(out_dir / "lgbm_clean_test_mpd.csv", index=False)

    test_s_query_df = siamese_pipe.evaluate_queries(
        X_queries=X_test_tfidf,
        X_train=X_train_tfidf,
        sample_ids=test_sample_ids,
        labels=y_test_tfidf,
        thresholds=thresholds_siamese_query,
        split_name="test",
    )
    test_s_query_df.to_csv(out_dir / "siamese_query_clean_test_mpd.csv", index=False)

    test_stats_lgbm = compute_class_statistics(test_lgbm_df, val_col="mpd", group_col="label", label_names={0: "Benign", 1: "Malware"})
    test_stats_s_query = compute_class_statistics(test_s_query_df, val_col="mpd", group_col="label", label_names={0: "Benign", 1: "Malware"})

    test_summary = {
        "lgbm_test": test_stats_lgbm.to_dict(orient="records"),
        "siamese_query_test": test_stats_s_query.to_dict(orient="records"),
        "lgbm_test_t95_flagged": int(test_lgbm_df["flag_t95"].sum()),
        "lgbm_test_t99_flagged": int(test_lgbm_df["flag_t99"].sum()),
        "siamese_test_t95_flagged": int(test_s_query_df["flag_t95"].sum()),
        "siamese_test_t99_flagged": int(test_s_query_df["flag_t99"].sum()),
    }
    with open(out_dir / "test_evaluation_summary.json", "w") as f:
        json.dump(test_summary, f, indent=2)

    print("\n[LightGBM TEST Clean MPD Stats]")
    print(test_stats_lgbm.to_string(index=False))
    print(f"LGBM Test Flagged at T95: {test_summary['lgbm_test_t95_flagged']}/150, at T99: {test_summary['lgbm_test_t99_flagged']}/150")

    print("\n[Siamese Query TEST Clean MPD Stats]")
    print(test_stats_s_query.to_string(index=False))
    print(f"Siamese Test Flagged at T95: {test_summary['siamese_test_t95_flagged']}/150, at T99: {test_summary['siamese_test_t99_flagged']}/150")

    # Run Manifest
    manifest = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "command": "python3 src/uncertainity/final_n1_mpd/run_all.py",
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": sparse.__version__ if hasattr(sparse, "__version__") else "1.15.3",
            "scikit_learn": "1.7.2",
            "lightgbm": lgb.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "model_paths": {
            "lightgbm": "models/final_n1/entropy_strings_final/lightgbm_entropy_strings.txt",
            "siamese": "models/siamese_svm_n1/siamese_svm_fewshot.joblib",
        },
        "model_hashes": {
            "lightgbm": file_sha256(base_dir / "models/final_n1/entropy_strings_final/lightgbm_entropy_strings.txt"),
            "siamese": file_sha256(base_dir / "models/siamese_svm_n1/siamese_svm_fewshot.joblib"),
        },
        "seeds": {
            "canonical_supports": 42,
            "data_construction": args.seed,
            "replicas": f"seed = k for k in 0..{args.n_replicas - 1}",
        },
        "n_replicas": args.n_replicas,
        "split_files": {
            "train": "data/splits/final_n1/train.csv",
            "validation": "data/splits/final_n1/validation.csv",
            "test": "data/splits/final_n1/test.csv",
        },
    }
    with open(out_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nRun complete! All outputs saved to {out_dir}")


if __name__ == "__main__":
    main()

