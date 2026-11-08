#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from collections import Counter

from genetic_algorithm_mutator import SiameseSVMScorer


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def score_file(scorer, path):
    path = Path(path).resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    result = scorer.predict_details(str(path))

    margin = float(result["malware_margin"])
    prediction = int(result["prediction"])

    # Independently verify the implementation's decision rule.
    expected_prediction = int(margin > 0.0)

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "prediction": prediction,
        "expected_prediction": expected_prediction,
        "margin": margin,
        "benign_score": float(
            result["benign_same_class_score"]
        ),
        "malware_score": float(
            result["malware_same_class_score"]
        ),
        "support_ids": result["support_ids"],
        "selected_k": int(result["selected_k"]),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--matrix",
        default="data/features/tfidf_n1/X_train.npz"
    )
    parser.add_argument(
        "--labels",
        default="data/features/tfidf_n1/train_samples.csv"
    )
    parser.add_argument(
        "--pairs",
        required=True,
        help="CSV with baseline_path,candidate_path columns"
    )
    parser.add_argument(
        "--output",
        default="svm_label_flip_audit.json"
    )

    args = parser.parse_args()

    print("[1/5] Loading canonical Siamese-SVM scorer...")

    scorer = SiameseSVMScorer(
        model_path=args.model,
        matrix_path=args.matrix,
        labels_path=args.labels,
        random_state=42,
    )

    print("[2/5] Reading evaluation pairs...")

    with open(args.pairs, newline="", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))

    required = {"baseline_path", "candidate_path"}

    if not pairs:
        raise RuntimeError("The pairs CSV contains no rows.")

    if not required.issubset(pairs[0].keys()):
        raise ValueError(
            "CSV must contain baseline_path and candidate_path"
        )

    records = []
    errors = []
    counts = Counter()

    print(f"[3/5] Evaluating {len(pairs)} pairs...")

    for index, pair in enumerate(pairs, start=1):
        baseline_path = pair["baseline_path"]
        candidate_path = pair["candidate_path"]

        print(
            f"\nPair {index}/{len(pairs)}"
        )

        try:
            baseline = score_file(scorer, baseline_path)
            candidate = score_file(scorer, candidate_path)

            # Repeat inference to check deterministic behavior.
            baseline_repeat = score_file(scorer, baseline_path)
            candidate_repeat = score_file(scorer, candidate_path)

            baseline_deterministic = (
                baseline["prediction"]
                == baseline_repeat["prediction"]
                and abs(
                    baseline["margin"] - baseline_repeat["margin"]
                ) < 1e-10
            )

            candidate_deterministic = (
                candidate["prediction"]
                == candidate_repeat["prediction"]
                and abs(
                    candidate["margin"] - candidate_repeat["margin"]
                ) < 1e-10
            )

            support_bank_stable = (
                baseline["support_ids"] == candidate["support_ids"]
            )

            rule_consistent = (
                baseline["prediction"]
                == baseline["expected_prediction"]
                and candidate["prediction"]
                == candidate["expected_prediction"]
            )

            # Strictly count only malware -> benign.
            label_flip = (
                baseline["prediction"] == 1
                and candidate["prediction"] == 0
            )

            valid_measurement = (
                baseline_deterministic
                and candidate_deterministic
                and support_bank_stable
                and rule_consistent
            )

            if not valid_measurement:
                counts["measurement_integrity_failures"] += 1

            if label_flip and valid_measurement:
                counts["verified_malware_to_benign_flips"] += 1

            if baseline["prediction"] == 0:
                counts["baseline_already_benign"] += 1

            if baseline["prediction"] == 1:
                counts["baseline_malware"] += 1

            if candidate["prediction"] == 0:
                counts["candidate_benign"] += 1

            record = {
                "pair_index": index,
                "baseline": baseline,
                "candidate": candidate,
                "baseline_deterministic": baseline_deterministic,
                "candidate_deterministic": candidate_deterministic,
                "support_bank_stable": support_bank_stable,
                "decision_rule_consistent": rule_consistent,
                "malware_to_benign_flip": label_flip,
                "valid_measurement": valid_measurement,
            }

            records.append(record)

            print(
                "Baseline:",
                baseline["prediction"],
                "margin:",
                round(baseline["margin"], 8)
            )
            print(
                "Candidate:",
                candidate["prediction"],
                "margin:",
                round(candidate["margin"], 8)
            )
            print("Label flip:", label_flip)
            print("Measurement valid:", valid_measurement)

        except Exception as exc:
            counts["evaluation_errors"] += 1
            errors.append({
                "pair_index": index,
                "baseline_path": baseline_path,
                "candidate_path": candidate_path,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

            print("[ERROR]", type(exc).__name__, str(exc))

    print("\n[4/5] Creating audit report...")

    report = {
        "model_path": str(Path(args.model).resolve()),
        "matrix_path": str(Path(args.matrix).resolve()),
        "labels_path": str(Path(args.labels).resolve()),
        "pairs_evaluated": len(pairs),
        "counts": dict(counts),
        "records": records,
        "errors": errors,
    }

    output_path = Path(args.output).resolve()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("[5/5] Finished.")
    print(json.dumps(dict(counts), indent=2))
    print("Report:", output_path)


if __name__ == "__main__":
    main()
