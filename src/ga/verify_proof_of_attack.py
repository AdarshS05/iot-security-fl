#!/usr/bin/env python3
"""
verify_proof_of_attack.py

Independent, out-of-process proof that GA-produced adversarial variants
actually flip the detector's label -- and aggregate label-flip metrics
across one run or a whole batch of ga_results.json files.

Why this exists: the GA's own [VARIANT]/[FINAL RANKING] logs report scores
computed INSIDE the GA process, using in-memory baseline values. That is
not proof an attack "works" -- it only proves the GA's internal bookkeeping
is self-consistent. This script re-loads each saved .elf variant from disk,
in a fresh process, through the exact same loader functions the GA itself
uses (FitnessEvaluator.default_feature_extractor / default_lightgbm_predict,
SiameseSVMScorer), and re-derives baseline and variant predictions from
scratch. Only a flip that survives this independent re-check counts as a
"proof of attack".

Place this file in the same directory as genetic_algorithm_mutator.py
(src/ga/), since it imports that module directly.

Usage (single run):
    python3 verify_proof_of_attack.py \
        --ga-results reports/ga/binary_mutator_test_instr/ga_results.json \
        --feature-extractor-module src/tabular/elf_feature_extractor.py \
        --lightgbm-model models/final_n1/entropy_strings_final \
        --siamese-model models/siamese_svm_n1/siamese_svm_fewshot.joblib \
        --tfidf-matrix data/features/tfidf_n1/X_train.npz \
        --tfidf-labels data/features/tfidf_n1/train_samples.csv \
        --output-dir reports/ga/binary_mutator_test_instr/proof

Usage (batch -- scan every ga_results.json under a directory tree):
    python3 verify_proof_of_attack.py \
        --ga-results-glob "reports/ga/batch_instr/**/ga_results.json" \
        --feature-extractor-module src/tabular/elf_feature_extractor.py \
        --lightgbm-model models/final_n1/entropy_strings_final \
        --siamese-model models/siamese_svm_n1/siamese_svm_fewshot.joblib \
        --tfidf-matrix data/features/tfidf_n1/X_train.npz \
        --tfidf-labels data/features/tfidf_n1/train_samples.csv \
        --output-dir reports/ga/batch_instr/proof_aggregate
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import the GA module directly (must be colocated).
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import genetic_algorithm_mutator as gam  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_fresh_scorers(args):
    """Loads feature extractor, LightGBM predictor, and Siamese scorer
    using the SAME code paths the GA itself uses -- fresh, out of process,
    no reuse of any GA in-memory state."""
    extract_fn = gam.FitnessEvaluator.default_feature_extractor(
        args.feature_extractor_module
    )
    lgbm_fn = gam.FitnessEvaluator.default_lightgbm_predict(
        args.lightgbm_model
    )
    siamese_scorer = gam.SiameseSVMScorer(
        model_path=args.siamese_model,
        matrix_path=args.tfidf_matrix,
        labels_path=args.tfidf_labels,
    )
    return extract_fn, lgbm_fn, siamese_scorer


def classify(binary_path: str, extract_fn, lgbm_fn, siamese_scorer) -> Dict[str, Any]:
    """Re-derives LightGBM probability/prediction and Siamese score/prediction
    for one binary, entirely from disk, with no assumptions from any prior run."""
    if not os.path.isfile(binary_path):
        return {"error": f"file not found: {binary_path}"}

    features = extract_fn(binary_path)
    lgbm_prob = float(lgbm_fn(features))
    threshold = getattr(lgbm_fn, "best_threshold", 0.5) or 0.5
    lgbm_pred = int(lgbm_prob >= threshold)

    siamese_details = siamese_scorer.predict_details(binary_path)

    return {
        "sha256": sha256_file(Path(binary_path)),
        "lgbm_probability": lgbm_prob,
        "lgbm_threshold": threshold,
        "lgbm_prediction": lgbm_pred,          # 1 = malware, 0 = benign
        "siamese_score": float(siamese_details["score"]),
        "siamese_prediction": int(siamese_details["prediction"]),  # 1=malware-like
    }


def verify_one_run(ga_results_path: Path, extract_fn, lgbm_fn, siamese_scorer) -> Dict[str, Any]:
    with open(ga_results_path, "r", encoding="utf-8") as f:
        ga_results = json.load(f)

    source_binary = ga_results["source_binary"]

    # --- re-verify baseline independently ---
    baseline_fresh = classify(source_binary, extract_fn, lgbm_fn, siamese_scorer)
    recorded_baseline = ga_results.get("baseline", {})

    baseline_consistency = {
        "lgbm_probability_matches": (
            "error" not in baseline_fresh
            and abs(
                baseline_fresh["lgbm_probability"]
                - float(recorded_baseline.get("lightgbm_score", float("nan")))
            ) < 1e-6
        ),
        "siamese_score_matches": (
            "error" not in baseline_fresh
            and abs(
                baseline_fresh["siamese_score"]
                - float(recorded_baseline.get("siamese_svm_score", float("nan")))
            ) < 1e-6
        ),
    }

    # --- re-verify every reported variant independently ---
    variant_proofs = []
    for chromosome in ga_results.get("results", []):
        variant_path = chromosome.get("variant_path")
        if not variant_path:
            continue

        fresh = classify(variant_path, extract_fn, lgbm_fn, siamese_scorer)
        if "error" in fresh:
            variant_proofs.append({
                "variant_path": variant_path,
                "verified": False,
                "reason": fresh["error"],
            })
            continue

        lgbm_flip = (
            baseline_fresh.get("lgbm_prediction") == 1
            and fresh["lgbm_prediction"] == 0
        )
        svm_flip = (
            baseline_fresh.get("siamese_prediction") == 1
            and fresh["siamese_prediction"] == 0
        )
        joint_flip = lgbm_flip and svm_flip

        gene_operators = [
            gene.get("operator") for gene in chromosome.get("genes", [])
        ]
        non_padding_operator_present = any(
            op != gam.Operator.PADDING.value for op in gene_operators
        )

        variant_proofs.append({
            "variant_path": variant_path,
            "verified": True,
            "sha256": fresh["sha256"],
            "gene_operators": gene_operators,
            "non_padding_operator_present": non_padding_operator_present,
            "modification_cost": chromosome.get("modification_cost"),
            "fresh_lgbm_probability": fresh["lgbm_probability"],
            "fresh_lgbm_prediction": fresh["lgbm_prediction"],
            "fresh_siamese_score": fresh["siamese_score"],
            "fresh_siamese_prediction": fresh["siamese_prediction"],
            "lgbm_label_flip": lgbm_flip,
            "svm_label_flip": svm_flip,
            "joint_label_flip": joint_flip,
            # Cross-check against what the GA itself recorded at evaluation time,
            # to catch any drift between in-process and out-of-process scoring.
            "matches_ga_recorded_result": (
                chromosome.get("fitness_breakdown", {}).get("variant_lgbm_prediction")
                == fresh["lgbm_prediction"]
                and chromosome.get("fitness_breakdown", {}).get("variant_svm_prediction")
                == fresh["siamese_prediction"]
            ),
        })

    return {
        "source_binary": source_binary,
        "ga_results_path": str(ga_results_path),
        "baseline_prediction": {
            "lgbm_prediction": baseline_fresh.get("lgbm_prediction"),
            "siamese_prediction": baseline_fresh.get("siamese_prediction"),
            "independently_reproduced": baseline_consistency,
        },
        "variants": variant_proofs,
    }


def aggregate_metrics(run_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_variants = 0
    lgbm_flips = 0
    svm_flips = 0
    joint_flips = 0
    joint_flips_with_structural_gene = 0
    mismatches_vs_ga = 0

    operator_in_joint_flips = Counter()
    per_binary = []

    for run in run_reports:
        variants = run["variants"]
        n = len(variants)
        n_lgbm = sum(1 for v in variants if v.get("lgbm_label_flip"))
        n_svm = sum(1 for v in variants if v.get("svm_label_flip"))
        n_joint = sum(1 for v in variants if v.get("joint_label_flip"))
        n_mismatch = sum(
            1 for v in variants
            if v.get("verified") and not v.get("matches_ga_recorded_result", True)
        )

        total_variants += n
        lgbm_flips += n_lgbm
        svm_flips += n_svm
        joint_flips += n_joint
        mismatches_vs_ga += n_mismatch

        for v in variants:
            if v.get("joint_label_flip"):
                if v.get("non_padding_operator_present"):
                    joint_flips_with_structural_gene += 1
                for op in v.get("gene_operators", []):
                    operator_in_joint_flips[op] += 1

        per_binary.append({
            "source_binary": run["source_binary"],
            "variants_checked": n,
            "lgbm_flip_rate": round(n_lgbm / n, 4) if n else None,
            "svm_flip_rate": round(n_svm / n, 4) if n else None,
            "joint_flip_rate": round(n_joint / n, 4) if n else None,
            "ga_recorded_mismatches": n_mismatch,
        })

    return {
        "binaries_checked": len(run_reports),
        "total_variants_checked": total_variants,
        "lgbm_flip_rate_overall": (
            round(lgbm_flips / total_variants, 4) if total_variants else None
        ),
        "svm_flip_rate_overall": (
            round(svm_flips / total_variants, 4) if total_variants else None
        ),
        "joint_flip_rate_overall": (
            round(joint_flips / total_variants, 4) if total_variants else None
        ),
        "joint_flips_using_only_padding": (
            joint_flips - joint_flips_with_structural_gene
        ),
        "joint_flips_using_structural_gene": joint_flips_with_structural_gene,
        "operator_frequency_in_joint_flips": dict(operator_in_joint_flips),
        "ga_recorded_vs_independent_mismatches": mismatches_vs_ga,
        "per_binary": per_binary,
    }


def write_markdown_proof(aggregate: Dict[str, Any], run_reports: List[Dict[str, Any]], out_path: Path):
    lines = []
    lines.append("# Proof of Attack -- Independent Verification Report\n")
    lines.append(
        "Every result below was re-derived from the saved variant .elf file, "
        "loaded fresh and re-scored outside the GA process, using the same "
        "loader functions the GA itself uses (`FitnessEvaluator.default_feature_extractor`, "
        "`default_lightgbm_predict`, `SiameseSVMScorer`).\n"
    )
    lines.append("## Aggregate label-flip metrics\n")
    lines.append(f"- Binaries checked: **{aggregate['binaries_checked']}**")
    lines.append(f"- Variants checked: **{aggregate['total_variants_checked']}**")
    lines.append(f"- LightGBM flip rate: **{aggregate['lgbm_flip_rate_overall']}**")
    lines.append(f"- Siamese SVM flip rate: **{aggregate['svm_flip_rate_overall']}**")
    lines.append(f"- Joint (both detectors) flip rate: **{aggregate['joint_flip_rate_overall']}**")
    lines.append(
        f"- Joint flips achieved WITHOUT any structural gene (padding/overlay only): "
        f"**{aggregate['joint_flips_using_only_padding']}**"
    )
    lines.append(
        f"- Joint flips that included a structural gene (instruction/code-cave/"
        f"trampoline/opaque-predicate): **{aggregate['joint_flips_using_structural_gene']}**"
    )
    lines.append(
        f"- Mismatches between GA-recorded prediction and independent re-check: "
        f"**{aggregate['ga_recorded_vs_independent_mismatches']}** "
        "(should be 0; a nonzero value means the GA's in-process scoring drifted "
        "from a fresh re-load -- investigate before trusting any result)\n"
    )

    lines.append("## Per-binary breakdown\n")
    lines.append("| Binary | Variants | LGBM flip | SVM flip | Joint flip | GA mismatches |")
    lines.append("|---|---|---|---|---|---|")
    for row in aggregate["per_binary"]:
        lines.append(
            f"| `{Path(row['source_binary']).name}` | {row['variants_checked']} | "
            f"{row['lgbm_flip_rate']} | {row['svm_flip_rate']} | {row['joint_flip_rate']} | "
            f"{row['ga_recorded_mismatches']} |"
        )

    lines.append("\n## Individually proven joint-evasion variants\n")
    any_proof = False
    for run in run_reports:
        for v in run["variants"]:
            if v.get("joint_label_flip"):
                any_proof = True
                lines.append(f"### `{Path(v['variant_path']).name}`")
                lines.append(f"- source binary: `{run['source_binary']}`")
                lines.append(f"- variant sha256: `{v['sha256']}`")
                lines.append(f"- genes: {', '.join(v['gene_operators'])}")
                lines.append(
                    f"- structural (non-padding) gene present: "
                    f"**{v['non_padding_operator_present']}**"
                )
                lines.append(f"- modification cost: {v['modification_cost']}")
                lines.append(
                    f"- fresh LightGBM: prob={v['fresh_lgbm_probability']:.6f} "
                    f"-> prediction={v['fresh_lgbm_prediction']} (0=benign)"
                )
                lines.append(
                    f"- fresh Siamese SVM: score={v['fresh_siamese_score']:.6f} "
                    f"-> prediction={v['fresh_siamese_prediction']} (0=benign)"
                )
                lines.append(
                    f"- matches GA-recorded result at evaluation time: "
                    f"**{v['matches_ga_recorded_result']}**\n"
                )
    if not any_proof:
        lines.append("_No independently-confirmed joint label flips found in this run._\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ga-results", help="Path to one ga_results.json")
    parser.add_argument(
        "--ga-results-glob",
        help="Glob pattern matching several ga_results.json files for batch verification "
             "(e.g. 'reports/ga/batch_instr/**/ga_results.json')",
    )
    parser.add_argument("--feature-extractor-module", required=True)
    parser.add_argument("--lightgbm-model", required=True)
    parser.add_argument("--siamese-model", required=True)
    parser.add_argument("--tfidf-matrix", required=True)
    parser.add_argument("--tfidf-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if not args.ga_results and not args.ga_results_glob:
        parser.error("Provide either --ga-results or --ga-results-glob")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[PROOF] Loading fresh scorers (independent of any GA process)...")
    extract_fn, lgbm_fn, siamese_scorer = load_fresh_scorers(args)

    paths = (
        [Path(args.ga_results)]
        if args.ga_results
        else [Path(p) for p in glob.glob(args.ga_results_glob, recursive=True)]
    )
    if not paths:
        raise SystemExit("No ga_results.json files matched.")

    run_reports = []
    for path in paths:
        print(f"[PROOF] Verifying {path} ...")
        run_reports.append(verify_one_run(path, extract_fn, lgbm_fn, siamese_scorer))

    aggregate = aggregate_metrics(run_reports)

    with open(out_dir / "label_flip_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {"aggregate": aggregate, "runs": run_reports},
            f,
            indent=2,
        )

    write_markdown_proof(aggregate, run_reports, out_dir / "proof_of_attack.md")

    print("\n" + "=" * 70)
    print("PROOF OF ATTACK -- SUMMARY")
    print("=" * 70)
    print(f"Binaries checked        : {aggregate['binaries_checked']}")
    print(f"Variants checked        : {aggregate['total_variants_checked']}")
    print(f"LightGBM flip rate      : {aggregate['lgbm_flip_rate_overall']}")
    print(f"Siamese SVM flip rate   : {aggregate['svm_flip_rate_overall']}")
    print(f"Joint flip rate         : {aggregate['joint_flip_rate_overall']}")
    print(f"Joint flips (padding only)     : {aggregate['joint_flips_using_only_padding']}")
    print(f"Joint flips (structural gene)  : {aggregate['joint_flips_using_structural_gene']}")
    print(f"GA-vs-independent mismatches   : {aggregate['ga_recorded_vs_independent_mismatches']}")
    print(f"\nFull metrics : {out_dir / 'label_flip_metrics.json'}")
    print(f"Proof report : {out_dir / 'proof_of_attack.md'}")


if __name__ == "__main__":
    main()
