"""
Train and evaluate the final N1 LightGBM model using the fixed shared split.

Run from repository root:
  python3 scripts/train_lightgbm_final.py \
    --features data/features/tabular_n1/tabular_features.csv \
    --output-dir models/final_n1

No random split is created. Threshold selection uses validation only.
The test set is evaluated once after model fitting/threshold selection.
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score,
)


METADATA_COLUMNS = {
    "sample_id", "label", "source", "split",
    "binary_path", "binary_sha256",
    "file_name", "filename", "file_path", "md5", "sha256", "sha1",
    "assembly_path", "assembly_sha256",
    "n0_normalized_path", "n0_normalized_sha256",
    "n1_conservative_path", "n1_conservative_normalized_sha256",
    "n2_normalized_path", "n2_normalized_sha256",
    "top_opcodes",
}
EXPECTED_SPLIT_COUNTS = {"train": 699, "validation": 150, "test": 150}


def choose_threshold(y_true, probabilities):
    """Choose threshold maximizing validation F1; deterministic tie-break near 0.5."""
    candidates = np.unique(np.concatenate([
        np.array([0.0, 0.5, 1.0]),
        np.asarray(probabilities, dtype=float),
    ]))
    scored = []
    for threshold in candidates:
        pred = (probabilities >= threshold).astype(int)
        scored.append((f1_score(y_true, pred, zero_division=0), abs(threshold - 0.5), threshold))
    # Highest F1, then closest to 0.5, then higher threshold.
    scored.sort(key=lambda x: (x[0], -x[1], x[2]), reverse=True)
    return float(scored[0][2]), float(scored[0][0])


def prepare_features(train_df, other_dfs, feature_columns):
    """Fit feature typing on train and apply compatible dtypes to all partitions."""
    train_x = train_df[feature_columns].copy()
    transformed = [train_x]

    for frame in other_dfs:
        transformed.append(frame[feature_columns].copy())

    categorical_cols = []
    for col in feature_columns:
        if (
            pd.api.types.is_object_dtype(train_x[col])
            or pd.api.types.is_string_dtype(train_x[col])
            or isinstance(train_x[col].dtype, pd.CategoricalDtype)
        ):
            categorical_cols.append(col)

    # Establish category levels from training data only. Unknown validation/test
    # categories become NaN, which LightGBM handles as missing values.
    for col in categorical_cols:
        categories = pd.Series(train_x[col].dropna().astype(str).unique()).tolist()
        for frame in transformed:
            values = frame[col].astype("string")
            frame[col] = pd.Categorical(values, categories=categories)

    # Coerce remaining columns to numeric; invalid values become NaN.
    for frame in transformed:
        for col in feature_columns:
            if col not in categorical_cols:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
                frame[col] = frame[col].replace([np.inf, -np.inf], np.nan)

    return transformed, categorical_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-dir", default="models/final_n1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--estimators", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    args = parser.parse_args()

    feature_path = Path(args.features).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)

    df = pd.read_csv(feature_path)
    required = {"sample_id", "label", "source", "split", "binary_sha256"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Feature dataset missing required columns: {sorted(missing)}")

    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["split"] = df["split"].astype(str).str.lower().str.strip()
    df["sample_id"] = df["sample_id"].astype(str)
    df["binary_sha256"] = df["binary_sha256"].astype(str).str.lower().str.strip()

    if df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id values in feature dataset.")
    if df["binary_sha256"].duplicated().any():
        raise ValueError("Duplicate raw binary hashes in feature dataset.")
    if set(df["label"]) != {"benign", "malware"}:
        raise ValueError(f"Expected benign/malware labels, found {sorted(set(df['label']))}")
    if set(df["split"]) != set(EXPECTED_SPLIT_COUNTS):
        raise ValueError(f"Unexpected split values: {sorted(set(df['split']))}")

    counts = df["split"].value_counts().to_dict()
    if counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Expected split counts {EXPECTED_SPLIT_COUNTS}, found {counts}")

    # Cross-partition raw/N1 hash leakage checks.
    for col in ("binary_sha256", "n1_conservative_normalized_sha256"):
        if col in df.columns:
            if (df.groupby(col)["split"].nunique() > 1).any():
                raise ValueError(f"Cross-split leakage detected in {col}")

    # Verify the expected class balance in each partition.
    expected_class_counts = {
        "train": {"benign": 349, "malware": 350},
        "validation": {"benign": 75, "malware": 75},
        "test": {"benign": 75, "malware": 75},
    }
    for split, expected in expected_class_counts.items():
        observed = df[df["split"] == split]["label"].value_counts().to_dict()
        if observed != expected:
            raise ValueError(f"{split} class counts differ: expected {expected}, found {observed}")

    # Explicit and stable class mapping.
    df["target"] = df["label"].map({"benign": 0, "malware": 1}).astype(int)

    excluded = METADATA_COLUMNS | {"target"}
    feature_columns = [c for c in df.columns if c not in excluded]
    if not feature_columns:
        raise ValueError("No candidate feature columns remain after excluding metadata.")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()

    # Remove all-null and training-constant columns only. Do not apply the old
    # blanket shortcut-feature list or drop entropy/file-size/entry-point by default.
    retained = []
    dropped_all_null = []
    dropped_constant = []
    for col in feature_columns:
        series = train_df[col]
        if series.notna().sum() == 0:
            dropped_all_null.append(col)
            continue
        if series.nunique(dropna=True) <= 1:
            dropped_constant.append(col)
            continue
        retained.append(col)
    feature_columns = retained
    if not feature_columns:
        raise ValueError("All candidate features were constant or empty in training data.")

    matrices, categorical_cols = prepare_features(
        train_df, [val_df, test_df], feature_columns
    )
    X_train, X_val, X_test = matrices
    y_train = train_df["target"].to_numpy()
    y_val = val_df["target"].to_numpy()
    y_test = test_df["target"].to_numpy()

    # A single-feature AUC audit is diagnostic only. It does not automatically
    # remove a feature: inspect suspicious results for source/collection shortcuts.
    single_feature_auc = {}
    for col in feature_columns:
        if col in categorical_cols:
            continue
        values = pd.to_numeric(X_train[col], errors="coerce")
        if values.notna().nunique() < 2:
            continue
        try:
            auc = roc_auc_score(y_train[values.notna()], values[values.notna()])
            single_feature_auc[col] = float(max(auc, 1.0 - auc))
        except Exception:
            pass

    params = {
        "objective": "binary",
        "n_estimators": args.estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "class_weight": None,
        "random_state": args.seed,
        "n_jobs": -1,
        "verbosity": -1,
        "colsample_bytree": 0.9,
        "subsample": 0.9,
        "subsample_freq": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }
    model = lgb.LGBMClassifier(**params)

    callbacks = [
        lgb.early_stopping(args.early_stopping_rounds, verbose=True),
        lgb.log_evaluation(period=50),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            categorical_feature=categorical_cols if categorical_cols else "auto",
            callbacks=callbacks,
        )

    val_prob = model.predict_proba(X_val)[:, 1]
    threshold, val_best_f1 = choose_threshold(y_val, val_prob)

    # Final held-out test evaluation. No threshold tuning on test.
    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)

    metrics = {
        "threshold_selected_on_validation": threshold,
        "validation_best_f1": val_best_f1,
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, test_prob)),
        "test_pr_auc": float(average_precision_score(y_test, test_prob)),
        "test_confusion_matrix_labels_order": ["benign_0", "malware_1"],
        "test_confusion_matrix": confusion_matrix(y_test, test_pred, labels=[0, 1]).tolist(),
        "test_classification_report": classification_report(
            y_test, test_pred, labels=[0, 1],
            target_names=["benign", "malware"], output_dict=True, zero_division=0
        ),
        "best_iteration": int(model.best_iteration_ or args.estimators),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "lightgbm_model.joblib"
    metadata_path = output_dir / "lightgbm_metadata.json"
    importance_path = output_dir / "feature_importance.csv"
    audit_path = output_dir / "single_feature_auc_audit.csv"

    joblib.dump(model, model_path)

    importance = pd.DataFrame({
        "feature": feature_columns,
        "gain": model.booster_.feature_importance(importance_type="gain"),
        "split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    importance.to_csv(importance_path, index=False)

    pd.DataFrame(
        [{"feature": k, "single_feature_auc_train": v}
         for k, v in sorted(single_feature_auc.items(), key=lambda x: x[1], reverse=True)]
    ).to_csv(audit_path, index=False)

    metadata = {
        "experiment": "final_n1_tabular_lightgbm",
        "label_mapping": {"benign": 0, "malware": 1},
        "split_counts": counts,
        "class_counts": expected_class_counts,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_cols,
        "dropped_all_null_training_features": dropped_all_null,
        "dropped_constant_training_features": dropped_constant,
        "model_parameters": params,
        "metrics": metrics,
        "input_feature_csv": str(feature_path),
        "notes": [
            "Uses the pre-existing final_n1 split; no new random split was created.",
            "Threshold was selected using validation F1 only.",
            "Test metrics were computed after threshold selection.",
            "Identifier, label, source, split, and hash/path fields are excluded from model inputs.",
            "Feature columns with high single-feature AUC are flagged for review, not automatically removed.",
            "Categorical levels are learned from training data and stored in the fitted LightGBM model."
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nFINAL N1 LIGHTGBM RESULTS")
    print(json.dumps(metrics, indent=2))
    print("\nTop 20 features by gain:")
    print(importance.head(20).to_string(index=False))
    print("\nSingle-feature AUC audit (top 15):")
    if not single_feature_auc:
        print("No eligible numeric features.")
    else:
        print(audit_path.read_text()[:0] or
              pd.read_csv(audit_path).head(15).to_string(index=False))
    print(f"\nSaved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved feature importance: {importance_path}")
    print(f"Saved single-feature audit: {audit_path}")


if __name__ == "__main__":
    main()
