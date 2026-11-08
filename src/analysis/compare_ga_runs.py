import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COMMON_THRESHOLD = 0.9419649314154088
UPDATED_THRESHOLD = 0.5


def read_run(run_dir, run_name):
    records = []
    json_files = list(Path(run_dir).rglob("ga_results.json"))

    for file in json_files:
        try:
            data = json.loads(file.read_text())
        except Exception as exc:
            print(f"[SKIP] {file}: {exc}")
            continue

        baseline = data.get("baseline", {})
        source = Path(data.get("source_binary", "")).name

        baseline_lgbm = baseline.get("lightgbm_score")

        for result in data.get("results", []):
            fb = result.get("fitness_breakdown", {})

            base_score = fb.get("baseline_lgbm", baseline_lgbm)
            variant_score = fb.get("variant_lgbm")

            if base_score is None or variant_score is None:
                continue

            validation = result.get("validation", {})
            if validation and not validation.get("valid", True):
                continue

            # Recorded prediction transitions.
            base_svm_pred = fb.get("baseline_svm_prediction")
            variant_svm_pred = fb.get("variant_svm_prediction")

            base_malware = float(base_score) >= COMMON_THRESHOLD

            common_flip = (float(base_score) >= COMMON_THRESHOLD and float(variant_score) < COMMON_THRESHOLD)

            operational_flip = (float(base_score) >= UPDATED_THRESHOLD and float(variant_score) < UPDATED_THRESHOLD)

            svm_flip = (
                base_svm_pred == 1
                and variant_svm_pred == 0
            )

            records.append({
                "run": run_name,
                "source": source,
                "variant_path": result.get("variant_path", ""),
                "baseline_lgbm": float(base_score),
                "variant_lgbm": float(variant_score),
                "common_threshold_flip": common_flip,
                "operational_0.5_flip": operational_flip,
                "svm_flip": svm_flip,
                "joint_flip": common_flip and svm_flip,
                "fitness": result.get("fitness"),
                "modification_cost": result.get("modification_cost"),
            })

    return records, len(json_files)


def summarise(df):
    rows = []

    for run, group in df.groupby("run"):
        eligible = group[group["baseline_lgbm"] >= COMMON_THRESHOLD]

        rows.append({
            "run": run,
            "source_files": group["source"].nunique(),
            "candidate_variants": len(group),
            "eligible_variants": len(eligible),
            "LGBM_flips_common": int(
                eligible["common_threshold_flip"].sum()
            ),
            "LGBM_common_flip_rate_%": (
                100 * eligible["common_threshold_flip"].mean()
                if len(eligible) else 0
            ),
            "LGBM_flips_at_0.5": int(
                eligible["operational_0.5_flip"].sum()
            ),
            "LGBM_operational_flip_rate_%": (
                100 * eligible["operational_0.5_flip"].mean()
                if len(eligible) else 0
            ),
            "SVM_flips": int(group["svm_flip"].sum()),
            "Joint_flips_common": int(group["joint_flip"].sum()),
            "sources_with_LGBM_evasion": int(
                eligible.groupby("source")["common_threshold_flip"].any().sum()
            ),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", default="results/ga_comparison")
    args = parser.parse_args()

    before, before_files = read_run(args.before, "Original")
    after, after_files = read_run(args.after, "Retrained")

    all_records = before + after
    if not all_records:
        raise RuntimeError(
            "No scored candidates found. Check that completed source "
            "directories contain ga_results.json files."
        )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    detail = pd.DataFrame(all_records)
    summary = summarise(detail)

    detail.to_csv(out_dir / "ga_variant_comparison.csv", index=False)
    summary.to_csv(out_dir / "ga_evasion_summary.csv", index=False)

    print("\nGA RUN COMPARISON")
    print(summary.to_string(index=False))
    print("\nCompleted source JSON files:")
    print("Original:", before_files)
    print("Retrained:", after_files)

    # Plot common-threshold comparison and updated operational result.
    plot_data = summary.set_index("run")

    labels = list(plot_data.index)
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - width,
        plot_data["LGBM_common_flip_rate_%"],
        width,
        label=f"LightGBM @ {COMMON_THRESHOLD:.3f}",
    )

    ax.bar(
        x,
        plot_data["SVM_flips"] / plot_data["candidate_variants"] * 100,
        width,
        label="Siamese SVM",
    )

    if "Retrained" in plot_data.index:
        ax.bar(
            x + width,
            plot_data["LGBM_operational_flip_rate_%"],
            width,
            label="Retrained LightGBM @ 0.5",
        )

    ax.set_ylabel("Evasion rate (%)")
    ax.set_title("GA Evasion Before and After Adversarial Retraining")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "ga_evasion_comparison.png", dpi=300)
    plt.close(fig)

    print("\nSaved results to:", out_dir)


if __name__ == "__main__":
    main()
