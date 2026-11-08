"""
LightGBM-only, offline feature-space GA pilot.

This runner operates ONLY on the 12 numeric LightGBM metadata features.
It does not modify or generate ELF files. Results must be described as
abstract feature-space robustness/evasion simulations, not executable
malware evasion.

The current generic NSGA-II engine requires two risk values. For this
single-detector experiment this adapter supplies the SAME LightGBM risk
twice. This does not introduce a second detector, but it duplicates that
objective in NSGA-II crowding calculations. The output explicitly records
this limitation. For publication-quality multi-objective results, the
engine should later be generalized to support a single risk objective
plus modification cost directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GA_DIR = PROJECT_ROOT / "src" / "ga"
sys.path.insert(0, str(GA_DIR))

from nsga2_feature_search import NSGA2Config, run_nsga2, save_result
from nsga2_model_interface import ModelInterface


INTEGER_FEATURES = {
    "num_strings",
    "max_string_len",
    "num_ip_strings",
    "num_url_strings",
    "num_suspicious_strings",
    "num_high_entropy_strings",
}

COUNT_FEATURES = {
    "num_strings",
    "num_ip_strings",
    "num_url_strings",
    "num_suspicious_strings",
    "num_high_entropy_strings",
}

REQUIRED_BASELINE_COLUMNS = {
    "sample_id", "baseline_probability", "baseline_prediction"
}


def repair_candidate(candidate, feature_names, lower, upper):
    """Project a continuous GA vector onto basic feature-domain constraints."""
    x = np.asarray(candidate, dtype=float).copy()
    x = np.clip(x, lower, upper)
    index = {name: i for i, name in enumerate(feature_names)}

    for name in INTEGER_FEATURES:
        if name in index:
            i = index[name]
            x[i] = np.rint(x[i])

    # Counts cannot be negative, and subordinate string counts cannot
    # exceed the total number of strings.
    for name in COUNT_FEATURES:
        if name in index:
            x[index[name]] = max(0, x[index[name]])

    if "num_strings" in index:
        total = x[index["num_strings"]]
        for name in (
            "num_ip_strings", "num_url_strings",
            "num_suspicious_strings", "num_high_entropy_strings"
        ):
            if name in index:
                x[index[name]] = min(x[index[name]], total)

    if "avg_string_len" in index and "max_string_len" in index:
        avg_i, max_i = index["avg_string_len"], index["max_string_len"]
        # Do not invent a larger maximum: cap average length to the
        # repaired maximum so the basic relationship remains valid.
        x[avg_i] = min(x[avg_i], x[max_i])

    if "mean_section_entropy" in index and "max_section_entropy" in index:
        mean_i = index["mean_section_entropy"]
        max_i = index["max_section_entropy"]
        x[mean_i] = min(x[mean_i], x[max_i])

    return np.clip(x, lower, upper)


def load_inputs():
    features_path = PROJECT_ROOT / "data/features/tabular_n1/tabular_feature.csv"
    baseline_path = PROJECT_ROOT / "reports/ga/lightgbm_validation_baseline.csv"
    train_manifest_path = PROJECT_ROOT / "data/splits/final_n1/train.csv"

    for path in (features_path, baseline_path, train_manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    features = pd.read_csv(features_path)
    baseline = pd.read_csv(baseline_path)
    train_manifest = pd.read_csv(train_manifest_path)

    missing = REQUIRED_BASELINE_COLUMNS - set(baseline.columns)
    if missing:
        raise ValueError(
            f"Baseline CSV is missing columns {sorted(missing)}. "
            "Expected sample_id, baseline_probability, baseline_prediction."
        )

    if "sample_id" not in features or "split" not in features:
        raise ValueError("Tabular feature CSV must contain sample_id and split.")

    return features, baseline, train_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=10,
        help="Number of correctly detected validation malware samples. "
             "Use 0 to process all eligible samples (default: 10)."
    )
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", default="reports/ga/lightgbm_feature_space"
    )
    args = parser.parse_args()

    if args.limit < 0:
        raise ValueError("--limit must be >= 0")

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    features_df, baseline_df, train_manifest = load_inputs()
    model = ModelInterface()
    feature_names = list(model.lgbm_features)

    missing_features = set(feature_names) - set(features_df.columns)
    if missing_features:
        raise ValueError(f"Feature table missing model features: {sorted(missing_features)}")

    # Verify validation alignment using the split column and sample IDs.
    val_features = features_df.loc[features_df["split"] == "validation"].copy()
    if val_features["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id values in validation feature rows.")

    # Keep model features exclusively from the canonical tabular feature table.
    # Drop overlapping feature columns from baseline metadata before merging.
    feature_source = val_features[["sample_id", "label"] + feature_names].copy()
    baseline_metadata = baseline_df.drop(columns=[c for c in feature_names if c in baseline_df.columns], errors="ignore",).copy()

    joined = baseline_metadata.merge(
        feature_source,
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )

    if len(joined) != len(baseline_df):
        raise ValueError(
            f"Baseline-to-validation join mismatch: "
            f"{len(joined)} joined from {len(baseline_df)} baseline rows."
        )

    missing_after_merge = set(feature_names) - set(joined.columns)
    if missing_after_merge:
        raise ValueError(
            f"Missing model features after merge: {sorted(missing_after_merge)}")

    # Normalize labels but never modify ground truth.
    joined["label_norm"] = joined["label"].astype(str).str.lower()

    malware_mask = joined["label_norm"].isin({"malware", "1", "true"})
    eligible = joined.loc[
        malware_mask & (joined["baseline_prediction"].astype(int) == 1)
    ].copy()

    already_missed = int((malware_mask & (joined["baseline_prediction"].astype(int) == 0)).sum())
    if eligible.empty:
        raise RuntimeError("No baseline-detected validation malware samples found.")

    eligible = eligible.sort_values(
        "baseline_probability", ascending=False
    ).reset_index(drop=True)
    if args.limit:
        eligible = eligible.head(args.limit)

    # Training-only percentile bounds. No validation/test rows contribute.
    train_rows = features_df.loc[features_df["split"] == "train"].copy()
    if train_rows.empty:
        raise ValueError("No rows with split='train' in tabular feature CSV.")

    lower = train_rows[feature_names].min().to_numpy(dtype=float)
    upper = train_rows[feature_names].max().to_numpy(dtype=float)

    # Ensure integer features have valid integer endpoints and nonzero ranges.
    for i, name in enumerate(feature_names):
        if name in INTEGER_FEATURES:
            lower[i] = np.floor(lower[i])
            upper[i] = np.ceil(upper[i])
        if not np.isfinite(lower[i]) or not np.isfinite(upper[i]) or lower[i] >= upper[i]:
            raise ValueError(f"Invalid training-derived bounds for {name}: {lower[i]}, {upper[i]}")

    # Every selected baseline must lie inside the optimization box. If not,
    # abort instead of silently moving the original sample.
    config = NSGA2Config(
        population_size=args.population,
        generations=args.generations,
        seed=args.seed,
    )

    sample_rows = []
    pareto_rows = []
    generation_rows = []

    for sample_number, row in eligible.iterrows():
        sample_id = str(row["sample_id"])
        baseline_vector = row.loc[feature_names].to_numpy(dtype=float)

        # Keep original input intact. Optimization is only run when the
        # baseline lies within training-derived 1st-99th percentile bounds.
        if np.any(baseline_vector < lower) or np.any(baseline_vector > upper):
            sample_rows.append({
                "sample_id": sample_id,
                "status": "skipped_baseline_outside_training_p01_p99",
                "baseline_probability": float(row["baseline_probability"]),
                "candidate_probability": np.nan,
                "label_flip": False,
                "note": "Baseline lies outside training-derived percentile bounds."
            })
            continue

        baseline_repaired = repair_candidate(
            baseline_vector, feature_names, lower, upper
        )

        def evaluate(candidate):
            repaired = repair_candidate(candidate, feature_names, lower, upper)
            feature_dict = dict(zip(feature_names, repaired.tolist()))
            scored = model.score_lightgbm(feature_dict)
            risk = float(scored["malware_probability"])
            # Compatibility adapter: NSGA-II API currently requires two
            # risks. Duplicate risk is explicitly documented in JSON output.
            return risk, risk

        result = run_nsga2(
            baseline=baseline_repaired,
            lower_bounds=lower,
            upper_bounds=upper,
            evaluate_fn=evaluate,
            config=NSGA2Config(
                population_size=args.population,
                generations=args.generations,
                seed=args.seed + sample_number,
            ),
        )

        # Re-repair and re-score every final Pareto candidate so exported
        # vectors, predictions and costs refer to the constrained vector.
        scored_front = []
        for item in result["pareto_front"]:
            candidate = repair_candidate(
                item["candidate"], feature_names, lower, upper
            )
            feature_dict = dict(zip(feature_names, candidate.tolist()))
            scored = model.score_lightgbm(feature_dict)
            cost = float(np.mean(
                np.abs(candidate - baseline_repaired) / (upper - lower)
            ))
            scored_front.append({
                "candidate": candidate,
                "probability": float(scored["malware_probability"]),
                "prediction": int(scored["prediction"]),
                "cost": cost,
            })

        if not scored_front:
            raise RuntimeError(f"No Pareto candidates returned for {sample_id}")

        # Choose the lowest-risk Pareto candidate for the headline flip test.
        # Cost remains reported so the probability reduction is not presented
        # without its associated perturbation magnitude.
        best = min(scored_front, key=lambda x: (x["probability"], x["cost"]))
        flip = bool(best["prediction"] == 0)

        sample_rows.append({
            "sample_id": sample_id,
            "status": "completed",
            "baseline_probability": float(row["baseline_probability"]),
            "candidate_probability": best["probability"],
            "threshold": float(model.lgbm_threshold),
            "baseline_prediction": int(row["baseline_prediction"]),
            "candidate_prediction": best["prediction"],
            "label_flip": flip,
            "normalized_l1_cost": best["cost"],
            "pareto_candidate_count": len(scored_front),
        })

        for rank, item in enumerate(scored_front):
            record = {
                "sample_id": sample_id,
                "pareto_index": rank,
                "probability": item["probability"],
                "prediction": item["prediction"],
                "normalized_l1_cost": item["cost"],
            }
            record.update({
                f"feature_{name}": float(value)
                for name, value in zip(feature_names, item["candidate"])
            })
            pareto_rows.append(record)

        for entry in result["history"]:
            generation_rows.append({
                "sample_id": sample_id,
                **entry
            })

        # Save the per-sample engine result for audit/reproducibility.
        save_result(
            result,
            str(output_dir / f"{sample_id}_nsga2_engine_result.json")
        )

        print(
            f"[{sample_number + 1}/{len(eligible)}] {sample_id[:12]} "
            f"baseline={row['baseline_probability']:.6f} "
            f"best={best['probability']:.6f} flip={flip}"
        )

    results_df = pd.DataFrame(sample_rows)
    results_df.to_csv(output_dir / "sample_results.csv", index=False)
    pd.DataFrame(pareto_rows).to_csv(output_dir / "pareto_candidates.csv", index=False)
    pd.DataFrame(generation_rows).to_csv(output_dir / "generation_history.csv", index=False)

    completed = results_df[results_df["status"] == "completed"]
    flips = int(completed["label_flip"].sum()) if not completed.empty else 0
    summary = {
        "experiment": "LightGBM-only constrained abstract feature-space GA",
        "feature_columns": feature_names,
        "threshold": float(model.lgbm_threshold),
        "validation_malware_total": int(malware_mask.sum()),
        "already_missed_at_baseline": already_missed,
        "baseline_detected_eligible": int(len(eligible)),
        "completed": int(len(completed)),
        "skipped": int((results_df["status"] != "completed").sum()),
        "ga_induced_label_flips": flips,
        "flip_rate_among_completed": (
            float(flips / len(completed)) if len(completed) else None
        ),
        "population_size": args.population,
        "generations": args.generations,
        "seed": args.seed,
        "bounds": "training partition 1st-99th percentiles",
        "objective_adapter": (
            "The current generic NSGA-II interface requires two risk values. "
            "This single-model runner supplies the LightGBM risk twice. "
            "There is no Siamese evaluation in this experiment; duplicated "
            "risk can affect crowding-distance weighting."
        ),
        "scope_limitation": (
            "Candidates are constrained abstract metadata feature vectors. "
            "They are not modified ELF files and are not evidence of "
            "executable-level malware evasion."
        ),
        "outputs": [
            "sample_results.csv",
            "pareto_candidates.csv",
            "generation_history.csv",
            "*_nsga2_engine_result.json"
        ],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== GA SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved outputs to: {output_dir}")

if __name__ == "__main__":
    main()
