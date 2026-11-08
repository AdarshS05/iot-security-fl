
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data/features/tabular_n1/tabular_feature.csv"
META = ROOT / "models/final_n1/lightgbm_metadata.json"
OUT = ROOT / "models/final_n1/ablation"

OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load data and original experiment metadata
# --------------------------------------------------

df = pd.read_csv(DATA)

with open(META) as f:
    metadata = json.load(f)

feature_columns = metadata["feature_columns"]
categorical_columns = metadata["categorical_columns"]
dropped_constants = metadata["dropped_constant_training_features"]

df["label"] = df["label"].str.lower()
df["split"] = df["split"].str.lower()

df["target"] = df["label"].map({
    "benign": 0,
    "malware": 1
})

assert df["target"].notna().all()
assert df["sample_id"].is_unique

expected = {"train": 699, "validation": 150, "test": 150}
actual = df["split"].value_counts().to_dict()

assert actual == expected, f"Unexpected split counts: {actual}"

train = df[df["split"] == "train"].copy()
val = df[df["split"] == "validation"].copy()
test = df[df["split"] == "test"].copy()

y_train = train["target"].astype(int)
y_val = val["target"].astype(int)
y_test = test["target"].astype(int)

# --------------------------------------------------
# Define feature groups
# --------------------------------------------------

structural_features = [
    "file_size_bytes",
    "elf_type",
    "abi_version",
    "endianness",
    "entry_point",
    "program_headers",
    "load_segments",
    "rwx_segments",
    "has_interp",
    "has_dynamic",
    "has_gnu_stack",
    "has_relro",
    "num_libraries",
    "is_statically_linked",
    "num_uncommon_libraries",
    "num_imported_functions",
    "num_exported_functions",
    "import_export_ratio",
    "num_dynamic_symbols",
    "num_static_symbols",
    "is_stripped",
    "num_suspicious_imports",
    "num_sections",
    "section_headers_present",
    "num_nonstandard_sections",
    "num_executable_sections",
    "text_size_ratio",
    "overlay_size_bytes",
    "has_overlay",
]

entropy_string_features = [
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

experiments = {
    "A_all_features": feature_columns,

    "B_without_program_headers": [
        c for c in feature_columns
        if c != "program_headers"
    ],

    "C_structural_only": structural_features,

    "D_entropy_and_strings_only": entropy_string_features,
}

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

def prepare_features(columns):

    columns = [
        c for c in columns
        if c in feature_columns
        and c not in dropped_constants
    ]

    if not columns:
        raise ValueError("No usable features in experiment")

    X_train = train[columns].copy()
    X_val = val[columns].copy()
    X_test = test[columns].copy()

    cats = [c for c in categorical_columns if c in columns]

    for c in columns:
        if c in cats:
            levels = pd.Index(
                X_train[c].dropna().unique()
            )

            for X in (X_train, X_val, X_test):
                X[c] = pd.Categorical(
                    X[c],
                    categories=levels
                )
        else:
            for X in (X_train, X_val, X_test):
                X[c] = pd.to_numeric(
                    X[c], errors="coerce"
                ).replace([np.inf, -np.inf], np.nan)

    return X_train, X_val, X_test, cats, columns


# --------------------------------------------------
# Validation threshold selection
# --------------------------------------------------

def select_threshold(y_true, probabilities):

    candidates = np.unique(np.concatenate([
        probabilities,
        np.array([0.0, 0.5, 1.0])
    ]))

    results = []

    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        results.append((score, threshold))

    # Max F1, then closest to 0.5, then higher threshold
    results.sort(
        key=lambda x: (x[0], -abs(x[1] - 0.5), x[1]),
        reverse=True
    )

    return results[0][1], results[0][0]


# --------------------------------------------------
# Run experiments
# --------------------------------------------------

results = []

for experiment_name, requested_features in experiments.items():

    print("\n" + "=" * 65)
    print("EXPERIMENT:", experiment_name)
    print("=" * 65)

    X_train, X_val, X_test, cats, used_features = (
        prepare_features(requested_features)
    )

    print("Features:", len(used_features))
    print("Categorical:", cats)

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        class_weight=None,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        colsample_bytree=0.9,
        subsample=0.9,
        subsample_freq=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
    )

    model.fit(
        X_train,
        y_train,
        categorical_feature=cats
    )

    # Select threshold using validation only
    val_prob = model.predict_proba(X_val)[:, 1]

    threshold, val_f1 = select_threshold(
        y_val,
        val_prob
    )

    # Evaluate held-out test set
    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)

    metrics = {
        "experiment": experiment_name,
        "feature_count": len(used_features),
        "threshold": float(threshold),
        "validation_f1": float(val_f1),
        "test_accuracy": float(
            accuracy_score(y_test, test_pred)
        ),
        "test_precision": float(
            precision_score(y_test, test_pred, zero_division=0)
        ),
        "test_recall": float(
            recall_score(y_test, test_pred, zero_division=0)
        ),
        "test_f1": float(
            f1_score(y_test, test_pred, zero_division=0)
        ),
        "test_roc_auc": float(
            roc_auc_score(y_test, test_prob)
        ),
        "test_pr_auc": float(
            average_precision_score(y_test, test_prob)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, test_pred
        ).tolist(),
    }

    results.append(metrics)

    print(json.dumps(metrics, indent=2))

    # Save individual experiment model
    model.booster_.save_model(
        str(OUT / f"{experiment_name}.txt")
    )

# --------------------------------------------------
# Save comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUT / "ablation_results.csv",
    index=False
)

with open(OUT / "ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n\nFINAL ABLATION COMPARISON")
print("=" * 90)

print(
    results_df[
        [
            "experiment",
            "feature_count",
            "threshold",
            "validation_f1",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_roc_auc",
            "test_pr_auc",
        ]
    ].to_string(index=False)
)

print("\nSaved results to:", OUT)
