
import json
import os
import argparse
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)


# -----------------------------------------------------------------------------
# Model output directory
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LIGHTGBM_MODEL_PATH = MODEL_DIR / "lightgbm_model.pkl"
LIGHTGBM_METADATA_PATH = MODEL_DIR / "lightgbm_metadata.json"

# -----------------------------------------------------------------------------
# Columns that identify a sample rather than describe it.
# -----------------------------------------------------------------------------
IDENTIFIER_COLUMNS = [
    "filename", "file_name", "file_path", "md5", "sha256", "sha1", "top_opcodes"
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

LIGHTGBM_MODEL_PATH = MODEL_DIR / "lightgbm_model.pkl"
LIGHTGBM_METADATA_PATH = MODEL_DIR / "lightgbm_metadata.json"

# -----------------------------------------------------------------------------
# Constant features that have only 1 unique value across the entire dataset.
# -----------------------------------------------------------------------------
CONSTANT_COLUMNS = [
    "elf_type", "os_abi", "endianness", "load_segments", "has_debug_info", "has_packer_signature"
]

# -----------------------------------------------------------------------------
# Compilation and dataset extraction shortcuts (data leaks).
# These features correlate almost perfectly with the label due to structural
# collection differences between how the benign and malware samples were
# sourced, not because of any genuinely malicious content.
# -----------------------------------------------------------------------------
SHORTCUT_COLUMNS = [
    # Static vs. Dynamic linking proxies
    # (malware is 98.5% statically linked, benign is 97.6% dynamically linked)
    "is_statically_linked",
    "program_headers",
    "has_interp",
    "has_dynamic",
    "num_libraries",
    "num_imported_functions",
    "num_dynamic_symbols",
    "import_export_ratio",
    "has_relro",
    "num_uncommon_libraries",
    "num_suspicious_imports",  # 0 for static binaries since there is no dynamic imports table

    # Section & Symbol stripping proxies
    # (100% of benign files have section headers/symbol table stripped during extraction)
    "num_sections",
    "section_headers_present",
    "num_nonstandard_sections",
    "num_executable_sections",
    "text_size_ratio",
    "max_section_entropy",
    "mean_section_entropy",
    "num_static_symbols",
    "is_stripped",

    # ELF header permission flags (collection artifacts, not malicious behavior):
    # 100% of benign samples have rwx_segments == 0 and has_gnu_stack == 1,
    # while malware varies. These collapse to near-constants on the benign
    # side, letting the model shortcut on "how the sample was built" rather
    # than any content-based signal.
    "rwx_segments",
    "has_gnu_stack",

    # Same static/dynamic-linking artifact family as num_imported_functions /
    # num_dynamic_symbols above: 84.6% of malware has exactly 0 exported
    # functions vs 98.6% of benign nonzero. Reflects whether a dynamic export
    # table exists, not malicious content.
    "num_exported_functions",
]




# -----------------------------------------------------------------------------
# Automated leakage sanity check (single-feature AUC / KS / dominant-value
# ratio). Runs before training. Doesn't auto-drop anything -- just warns
# loudly if a *remaining* feature looks suspicious, so a human decides
# whether it's a new artifact (like the three above) or genuine signal
# (like entropy_variance, which scores AUC=0.955 but has smooth overlapping
# distributions rather than a near-constant split, and is kept).
# -----------------------------------------------------------------------------
LEAK_AUC_THRESHOLD = 0.95
LEAK_DOMINANT_RATIO_THRESHOLD = 0.98


def check_remaining_features_for_leakage(X, y):
    from sklearn.metrics import roc_auc_score as _auc

    warnings = []
    for col in X.columns:
        series = X[col]
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            series = series.astype("category").cat.codes

        try:
            auc = _auc(y, series)
            auc = max(auc, 1 - auc)
        except ValueError:
            continue

        dom0 = series[y == 0].value_counts(normalize=True).iloc[0] if (y == 0).any() else np.nan
        dom1 = series[y == 1].value_counts(normalize=True).iloc[0] if (y == 1).any() else np.nan

        if auc >= LEAK_AUC_THRESHOLD or (dom0 >= LEAK_DOMINANT_RATIO_THRESHOLD or dom1 >= LEAK_DOMINANT_RATIO_THRESHOLD):
            warnings.append((col, round(auc, 4), round(dom0, 4), round(dom1, 4)))

    if warnings:
        print("\n" + "!" * 60)
        print("LEAKAGE SANITY CHECK: suspicious remaining feature(s) found.")
        print("These were NOT auto-dropped -- inspect their per-class")
        print("distributions before trusting the model's use of them.")
        print("!" * 60)
        for col, auc, dom0, dom1 in warnings:
            print(f"  - {col:<25} single-feature AUC={auc}  dominant_ratio(class0)={dom0}  dominant_ratio(class1)={dom1}")
        print()
    else:
        print("\nLeakage sanity check: no remaining feature exceeds the "
              f"AUC={LEAK_AUC_THRESHOLD} / dominant-ratio={LEAK_DOMINANT_RATIO_THRESHOLD} thresholds.\n")

    return warnings


def train_robust_model(
    csv_path="data/processed/benign_dataset.csv",
    test_size=0.2,
    val_size=0.1,
    model_dir=None,
):
    # 1. Load the dataset
    if model_dir is None:
        model_dir = MODEL_DIR
    else:
        model_dir = Path(model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Dataset shape (before cleanup): {df.shape}")

    # 2. De-duplicate based on file hash explicitly to prevent data leakage across splits.
    if "sha256" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["sha256"])
        print(f"Removed {before - len(df)} duplicate rows using 'sha256' hash.")
    elif "md5" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["md5"])
        print(f"Removed {before - len(df)} duplicate rows using 'md5' hash.")
    else:
        before = len(df)
        df = df.drop_duplicates()
        print(f"Removed {before - len(df)} fully duplicate rows (no hash columns found).")

    if "label" not in df.columns:
        raise ValueError("Target column 'label' not found in dataset.")

    # 3. Drop unwanted / shortcut / identifier columns
    dropped_cols = IDENTIFIER_COLUMNS + CONSTANT_COLUMNS + SHORTCUT_COLUMNS
    cols_to_drop = [c for c in dropped_cols if c in df.columns]

    print(f"\nDropping {len(cols_to_drop)} unwanted / shortcut columns:")
    for c in cols_to_drop:
        print(f"  - {c}")

    X = df.drop(columns=cols_to_drop + ["label"])
    suspect_features = ["entropy_variance","entry_point","file_size_bytes"]
    for feature in suspect_features:
    	X = X.drop(columns=[feature])
    y = df["label"]

    print(f"\nRemaining robust content-based features ({len(X.columns)}):")
    print(list(X.columns))

    # Encode label (benign -> 0, malware -> 1)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    classes = label_encoder.classes_
    print(f"\nEncoded classes: {dict(zip(classes, label_encoder.transform(classes)))}")

    # 4. Handle categorical / boolean features if any remain
    categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    for col in categorical_cols:
        X[col] = X[col].astype("category")

    # 4b. Automated leakage sanity check on the surviving feature set.
    check_remaining_features_for_leakage(X, y_encoded)

    # 5. Train / validation / test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y_encoded, test_size=test_size, stratify=y_encoded
    )
    val_fraction_of_train = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_fraction_of_train,
        stratify=y_train_full,
    )
    print(f"\nSplit sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 6. Handle class imbalance on the de-duplicated training split
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    print(f"Train class balance -> benign (0): {neg}, malware (1): {pos}, scale_pos_weight={scale_pos_weight:.3f}")

    # 7. Train LightGBM Classifier
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=scale_pos_weight,
        verbosity=-1,
    )

    # NOTE: newer lightgbm sklearn API deprecated the eval_set kwarg name in
    # favor of eval_X/eval_y; pass positionally-compatible kwargs so this
    # works across lightgbm versions without a DeprecationWarning.
    fit_kwargs = dict(
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )
    try:
        model.fit(X_train, y_train, eval_X=X_val, eval_y=y_val, **fit_kwargs)
    except TypeError:
        # Older lightgbm versions don't support eval_X/eval_y.
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], **fit_kwargs)

    print(f"Best training iteration: {model.best_iteration_}")

    # 8. Sweep and tune threshold on the VALIDATION split to maximize F1-score
    val_proba = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)
    f1_scores = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(precisions),
        where=(precisions + recalls) != 0,
    )
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]
    print(f"\nTuned decision threshold (from validation split): {best_threshold:.4f}")

    # 9. Evaluate model on held-out test split using the tuned threshold
    print("\n" + "=" * 50)
    print("Evaluating model performance on held-out test split...")
    print("=" * 50)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= best_threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in classes]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Feature Importance
    importance = model.booster_.feature_importance(importance_type="gain")
    feature_names = model.feature_name_
    feature_imp = pd.DataFrame({"feature": feature_names, "importance": importance})
    feature_imp = feature_imp.sort_values(by="importance", ascending=False)

    print("\nFeature Importance (by Gain):")
    for _, row in feature_imp.iterrows():
        print(f"  {row['feature']:<25} | {row['importance']:>10.2f} |")

    # -------------------------------------------------------------------------
    # Save trained model and all information required for inference
    # -------------------------------------------------------------------------

    model_path = model_dir / "lightgbm_model.pkl"
    metadata_path = model_dir / "lightgbm_metadata.json"

    model_artifact = {
        "model": model,
        "label_encoder": label_encoder,
        "best_threshold": float(best_threshold),
        "feature_names": list(X.columns),
        "n_features": int(len(X.columns)),
        "test_size": test_size,
        "val_size": val_size,
    }

    joblib.dump(
        model_artifact,
        model_path,
        compress=3,
    )

    metadata = {
        "model_type": "LightGBM",
        "model_path": str(model_path),
        "n_features": int(len(X.columns)),
        "feature_names": list(X.columns),
        "classes": classes.tolist(),
        "best_threshold": float(best_threshold),
        "best_iteration": int(model.best_iteration_),
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 50)
    print("MODEL SAVED")
    print("=" * 50)
    print(f"Model    : {model_path}")
    print(f"Metadata : {metadata_path}")

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "best_threshold": best_threshold,
        "best_iteration": model.best_iteration_,
    }

    return model, label_encoder, metrics

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the robust ELF malware/benign classifier."
    )

    parser.add_argument(
        "--csv_path",
        default="data/processed/benign_dataset.csv",
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--val_size",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--model_dir",
        default=None,
        help="Directory where the trained model will be saved.",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    train_robust_model(
        csv_path=args.csv_path,
        test_size=args.test_size,
        val_size=args.val_size,
        model_dir=args.model_dir,
    )
