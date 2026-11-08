
#!/usr/bin/env python3
"""
Adversarial retraining for the 12-feature GA LightGBM model.

- Preserves the existing train/validation/test split.
- Uses only GA-evasive variants from original training malware.
- Reconstructs model features using GA explainability records.
- Verifies reconstructed predictions against the original GA scores.
- Caps variants per source to reduce over-representation.
- Selects the new threshold on clean validation data.
- Evaluates the untouched clean test set.

Supports either:
    --ga-root reports/ga/full_dataset_run3
or:
    --ga-zip ga-full_dataset_run.zip
"""

import argparse
import json
import zipfile
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


EXPECTED_SPLITS = {
    "train": 699,
    "validation": 150,
    "test": 150,
}

EXPECTED_CLASSES = {
    "train": {"benign": 349, "malware": 350},
    "validation": {"benign": 75, "malware": 75},
    "test": {"benign": 75, "malware": 75},
}


def read_ga_reports(ga_root=None, ga_zip=None):
    """Yield (report_name, report_json) from a directory or ZIP."""

    if ga_zip:
        with zipfile.ZipFile(ga_zip, "r") as archive:
            names = [
                n for n in archive.namelist()
                if n.endswith("/ga_results.json")
            ]

            for name in names:
                try:
                    data = json.loads(archive.read(name))
                    yield name, data
                except Exception as exc:
                    print(f"[WARN] Could not read {name}: {exc}")

    else:
        root = Path(ga_root)
        if not root.is_dir():
            raise FileNotFoundError(root)

        for path in root.rglob("ga_results.json"):
            try:
                yield str(path), json.loads(
                    path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                print(f"[WARN] Could not read {path}: {exc}")


def choose_threshold(y_true, probabilities):
    """Maximize validation F1, with deterministic tie-breaking."""

    candidates = np.unique(
        np.concatenate([
            np.array([0.0, 0.5, 1.0]),
            np.asarray(probabilities, dtype=float),
        ])
    )

    best = None

    for threshold in candidates:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        candidate = (
            score,
            -abs(float(threshold) - 0.5),
            float(threshold),
        )

        if best is None or candidate > best:
            best = candidate

    return best[2], best[0]


def evaluate(y_true, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true, predictions, zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(
                y_true, probabilities
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["benign", "malware"],
            output_dict=True,
            zero_division=0,
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features",
        required=True,
        help="Original 999-row tabular feature CSV",
    )

    source = parser.add_mutually_exclusive_group(required=True)

    source.add_argument("--ga-root")
    source.add_argument("--ga-zip")

    parser.add_argument(
        "--baseline-model",
        required=True,
    )

    parser.add_argument(
        "--baseline-metadata",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default="models/final_n1_adv",
    )

    parser.add_argument(
        "--max-variants-per-source",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--score-tolerance",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Reconstruct and audit candidates without training",
    )

    args = parser.parse_args()

    if args.max_variants_per_source < 1:
        raise ValueError(
            "--max-variants-per-source must be at least 1"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Load original dataset and verify fixed split
    # ---------------------------------------------------------

    print("\n[1] Loading original feature dataset")

    df = pd.read_csv(args.features)

    required = {
        "sample_id",
        "label",
        "source",
        "split",
        "binary_sha256",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["label"] = (
        df["label"].astype(str).str.lower().str.strip()
    )

    df["split"] = (
        df["split"].astype(str).str.lower().str.strip()
    )

    df["binary_sha256"] = (
        df["binary_sha256"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    if df["binary_sha256"].duplicated().any():
        raise ValueError("Duplicate binary SHA-256 values.")

    observed_splits = df["split"].value_counts().to_dict()

    if observed_splits != EXPECTED_SPLITS:
        raise ValueError(
            f"Unexpected split counts: {observed_splits}"
        )

    for split, expected in EXPECTED_CLASSES.items():
        observed = (
            df[df["split"] == split]["label"]
            .value_counts()
            .to_dict()
        )

        if observed != expected:
            raise ValueError(
                f"{split} class counts mismatch: {observed}"
            )

    print("Original dataset:", len(df))
    print("Split counts:", observed_splits)

    # ---------------------------------------------------------
    # 2. Load original GA LightGBM model and metadata
    # ---------------------------------------------------------

    print("\n[2] Loading original GA LightGBM model")

    metadata = json.loads(
        Path(args.baseline_metadata).read_text(
            encoding="utf-8"
        )
    )

    feature_columns = list(metadata["feature_columns"])
    original_threshold = float(metadata["threshold"])

    baseline_model = lgb.Booster(
        model_file=args.baseline_model
    )

    if baseline_model.num_feature() != len(feature_columns):
        raise ValueError(
            "Model and metadata feature counts differ."
        )

    if baseline_model.feature_name() != feature_columns:
        raise ValueError(
            "Model feature order differs from metadata."
        )

    print("Features:", feature_columns)
    print("Original threshold:", original_threshold)

    missing_features = set(feature_columns) - set(df.columns)

    if missing_features:
        raise ValueError(
            f"Dataset missing model features: {missing_features}"
        )

    # Convert model inputs to numeric values.
    for col in feature_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    indexed = df.set_index("binary_sha256", drop=False)

    # ---------------------------------------------------------
    # 3. Reconstruct GA-evasive feature rows
    # ---------------------------------------------------------

    print("\n[3] Reading GA results and reconstructing variants")

    audit = []
    candidates = []
    source_cache = {}

    counters = {
        "reports": 0,
        "evasive_records": 0,
        "source_not_in_dataset": 0,
        "non_training_source": 0,
        "non_malware_source": 0,
        "baseline_mismatch": 0,
        "variant_mismatch": 0,
        "validated": 0,
    }

    for report_name, report in read_ga_reports(
        ga_root=args.ga_root,
        ga_zip=args.ga_zip,
    ):
        counters["reports"] += 1

        baseline = report.get("baseline", {})
        source_hash = str(
            baseline.get("sha256", "")
        ).lower().strip()

        if not source_hash:
            source_hash = Path(report_name).parent.name.lower()

        if source_hash not in indexed.index:
            counters["source_not_in_dataset"] += 1
            continue

        source_row = indexed.loc[source_hash]

        if isinstance(source_row, pd.DataFrame):
            source_row = source_row.iloc[0]

        if source_row["split"] != "train":
            counters["non_training_source"] += 1
            continue

        if source_row["label"] != "malware":
            counters["non_malware_source"] += 1
            continue

        # Verify original source prediction against GA.
        if source_hash not in source_cache:
            base_vector = pd.DataFrame(
                [[source_row[c] for c in feature_columns]],
                columns=feature_columns,
            )

            baseline_score = float(
                baseline_model.predict(base_vector)[0]
            )

            ga_baseline_score = float(
                baseline["lightgbm_score"]
            )

            if abs(
                baseline_score - ga_baseline_score
            ) > args.score_tolerance:
                source_cache[source_hash] = None
                counters["baseline_mismatch"] += 1
                continue

            source_cache[source_hash] = source_row

        for result in report.get("results", []):
            fb = result.get("fitness_breakdown") or {}

            if fb.get("lgbm_evasion") is not True:
                continue

            counters["evasive_records"] += 1

            variant_path = str(
                result.get("variant_path", "")
            )

            explainability = (
                fb.get("explainability") or {}
            )

            changed_features = (
                explainability.get("changed_features") or []
            )

            # Start with the original source's 12 features.
            variant = {
                feature: float(source_row[feature])
                for feature in feature_columns
            }

            # Apply all retained changes that affect the GA model.
            for change in changed_features:
                feature = change.get("feature")

                if feature not in feature_columns:
                    continue

                try:
                    variant[feature] = float(
                        change["variant"]
                    )
                except (KeyError, TypeError, ValueError):
                    continue

            variant_df = pd.DataFrame(
                [[variant[c] for c in feature_columns]],
                columns=feature_columns,
            )

            reconstructed_score = float(
                baseline_model.predict(variant_df)[0]
            )

            try:
                ga_variant_score = float(
                    fb["variant_lgbm"]
                )
            except (KeyError, TypeError, ValueError):
                counters["variant_mismatch"] += 1
                continue

            # A score match is required because GA explainability
            # retains only its top changed features.
            if abs(
                reconstructed_score - ga_variant_score
            ) > args.score_tolerance:
                counters["variant_mismatch"] += 1

                audit.append({
                    "source_sha256": source_hash,
                    "variant_path": variant_path,
                    "status": "reconstruction_score_mismatch",
                    "ga_score": ga_variant_score,
                    "reconstructed_score": reconstructed_score,
                })

                continue

            counters["validated"] += 1

            try:
                modification_cost = float(
                    fb.get("modification_cost", np.inf)
                )
            except (TypeError, ValueError):
                modification_cost = np.inf

            candidates.append({
                "source_sha256": source_hash,
                "variant_path": variant_path,
                "features": variant,
                "ga_score": ga_variant_score,
                "modification_cost": modification_cost,
                "changed_feature_count": (
                    explainability.get("changed_feature_count")
                ),
            })

    # ---------------------------------------------------------
    # 4. Deduplicate and cap candidates per source
    # ---------------------------------------------------------

    print("\n[4] Deduplicating and limiting variants")

    unique = {}

    for candidate in candidates:
        key = (
            candidate["source_sha256"],
            candidate["variant_path"],
        )

        unique[key] = candidate

    candidates = list(unique.values())

    candidates.sort(
        key=lambda x: (
            x["source_sha256"],
            x["ga_score"],
            x["modification_cost"],
            x["variant_path"],
        )
    )

    selected = []
    source_counts = {}

    for candidate in candidates:
        source = candidate["source_sha256"]

        count = source_counts.get(source, 0)

        if count < args.max_variants_per_source:
            selected.append(candidate)
            source_counts[source] = count + 1

            status = "selected_for_training"
        else:
            status = "excluded_source_cap"

        audit.append({
            "source_sha256": source,
            "variant_path": candidate["variant_path"],
            "status": status,
            "ga_score": candidate["ga_score"],
            "modification_cost": candidate["modification_cost"],
        })

    if not selected:
        raise RuntimeError(
            "No validated training-split adversarial rows found."
        )

    # ---------------------------------------------------------
    # 5. Save reconstructed adversarial training rows
    # ---------------------------------------------------------

    adversarial_rows = []

    for candidate in selected:
        row = dict(candidate["features"])

        row.update({
            "label": "malware",
            "split": "train",
            "source_sha256": candidate["source_sha256"],
            "variant_path": candidate["variant_path"],
            "ga_score": candidate["ga_score"],
            "modification_cost": candidate["modification_cost"],
        })

        adversarial_rows.append(row)

    adv_df = pd.DataFrame(adversarial_rows)

    adv_csv = output_dir / "adversarial_train_rows.csv"
    adv_df.to_csv(adv_csv, index=False)

    audit_csv = output_dir / "candidate_audit.csv"
    pd.DataFrame(audit).to_csv(audit_csv, index=False)

    print("\nGA reconstruction counters:")
    print(json.dumps(counters, indent=2))

    print("Validated unique candidates:", len(candidates))
    print("Selected adversarial rows:", len(selected))
    print("Unique source binaries:", len(source_counts))
    print("Saved:", adv_csv)
    print("Saved:", audit_csv)

    if args.dry_run:
        print("\nDRY RUN COMPLETE. Model was NOT trained.")
        return

    # ---------------------------------------------------------
    # 6. Prepare clean and adversarial training data
    # ---------------------------------------------------------

    print("\n[5] Preparing training data")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()

    X_train_clean = train_df[feature_columns].copy()
    X_val = val_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()

    y_train_clean = (
        train_df["label"]
        .map({"benign": 0, "malware": 1})
        .astype(int)
        .to_numpy()
    )

    y_val = (
        val_df["label"]
        .map({"benign": 0, "malware": 1})
        .astype(int)
        .to_numpy()
    )

    y_test = (
        test_df["label"]
        .map({"benign": 0, "malware": 1})
        .astype(int)
        .to_numpy()
    )

    X_adv = adv_df[feature_columns].copy()
    y_adv = np.ones(len(X_adv), dtype=int)

    X_train = pd.concat(
        [X_train_clean, X_adv],
        ignore_index=True,
    )

    y_train = np.concatenate(
        [y_train_clean, y_adv]
    )

    print("Clean training rows:", len(X_train_clean))
    print("Adversarial rows:", len(X_adv))
    print("Combined training rows:", len(X_train))
    print("Validation rows:", len(X_val))
    print("Test rows:", len(X_test))

    # ---------------------------------------------------------
    # 7. Train same LightGBM architecture
    # ---------------------------------------------------------

    print("\n[6] Training adversarially hardened LightGBM")

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 8. Select new threshold on clean validation data
    # ---------------------------------------------------------

    print("\n[7] Selecting threshold using clean validation data")

    val_prob = model.predict_proba(X_val)[:, 1]

    new_threshold, validation_f1 = choose_threshold(
        y_val,
        val_prob,
    )

    print("New threshold:", new_threshold)
    print("Validation F1:", validation_f1)

    # ---------------------------------------------------------
    # 9. Evaluate untouched clean test set
    # ---------------------------------------------------------

    print("\n[8] Evaluating untouched clean test set")

    baseline_test_prob = baseline_model.predict(X_test)

    baseline_metrics = evaluate(
        y_test,
        baseline_test_prob,
        original_threshold,
    )

    hardened_test_prob = model.predict_proba(X_test)[:, 1]

    hardened_metrics = evaluate(
        y_test,
        hardened_test_prob,
        new_threshold,
    )

    print("\nBASELINE TEST METRICS")
    print(json.dumps(baseline_metrics, indent=2))

    print("\nADVERSARIAL MODEL TEST METRICS")
    print(json.dumps(hardened_metrics, indent=2))

    # ---------------------------------------------------------
    # 10. Save model and experiment artifacts
    # ---------------------------------------------------------

    print("\n[9] Saving model and reports")

    model_path = output_dir / "lightgbm_adversarial.txt"
    joblib_path = output_dir / "lightgbm_adversarial.joblib"
    metadata_path = output_dir / "metadata.json"
    metrics_path = output_dir / "metrics.json"
    importance_path = output_dir / "feature_importance.csv"

    model.booster_.save_model(str(model_path))
    joblib.dump(model, joblib_path)

    importance = pd.DataFrame({
        "feature": feature_columns,
        "gain": model.booster_.feature_importance(
            importance_type="gain"
        ),
        "split": model.booster_.feature_importance(
            importance_type="split"
        ),
    }).sort_values("gain", ascending=False)

    importance.to_csv(importance_path, index=False)

    metrics = {
        "baseline": baseline_metrics,
        "adversarial_model": hardened_metrics,
        "validation_f1": float(validation_f1),
        "original_threshold": original_threshold,
        "new_threshold": new_threshold,
        "clean_training_rows": len(X_train_clean),
        "adversarial_training_rows": len(X_adv),
        "combined_training_rows": len(X_train),
        "unique_adversarial_sources": len(source_counts),
        "ga_reconstruction_counters": counters,
        "model_parameters": {
            "n_estimators": 800,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
        },
    }

    metadata_out = {
        "experiment": "lightgbm_adversarial_retraining",
        "model_file": model_path.name,
        "feature_columns": feature_columns,
        "threshold": float(new_threshold),
        "original_threshold": original_threshold,
        "metrics_file": metrics_path.name,
        "training_rows": len(X_train),
        "adversarial_rows": len(X_adv),
        "unique_adversarial_sources": len(source_counts),
        "split_counts": EXPECTED_SPLITS,
        "notes": [
            "Only original training-split malware sources were used.",
            "Validation and test samples remained clean.",
            "Threshold selected on clean validation data.",
            "Test set was not used for threshold selection.",
            "GA reconstruction required prediction-score agreement.",
        ],
    }

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    metadata_path.write_text(
        json.dumps(metadata_out, indent=2),
        encoding="utf-8",
    )

    print("\nRETRAINING COMPLETE")
    print("Model:", model_path)
    print("Metadata:", metadata_path)
    print("Metrics:", metrics_path)
    print("Feature importance:", importance_path)
    print("\nUse this metadata.json with the GA model loader.")


if __name__ == "__main__":
    main()
