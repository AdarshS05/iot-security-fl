
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_model_data(path):
    data = load_json(path)
    metrics = data.get("metrics", data)

    counts = data.get("split_counts", {})

    return {
        "train_samples": counts.get("train", np.nan),
        "validation_samples": counts.get("validation", np.nan),
        "test_samples": counts.get("test", np.nan),
        "decision_threshold": metrics.get(
            "threshold_selected_on_validation",
            data.get("threshold", np.nan)
        ),
        "accuracy": metrics.get("test_accuracy", np.nan),
        "precision": metrics.get("test_precision", np.nan),
        "recall": metrics.get("test_recall", np.nan),
        "f1": metrics.get("test_f1", np.nan),
        "roc_auc": metrics.get("test_roc_auc", np.nan),
        "pr_auc": metrics.get("test_pr_auc", np.nan),
    }


def get_mpd_data(directory):
    directory = Path(directory)

    thresholds = load_json(directory / "thresholds.json")
    test = pd.read_csv(directory / "lgbm_clean_test_mpd.csv")

    # run_all.py stores LightGBM thresholds under "lgbm"
    lgbm_thresholds = thresholds.get("lgbm", thresholds)

    mpd = test["mpd"]

    return {
        "mpd_test_samples": len(test),
        "mpd_T90": lgbm_thresholds.get("T90", np.nan),
        "mpd_T95": lgbm_thresholds.get("T95", np.nan),
        "mpd_T99": lgbm_thresholds.get("T99", np.nan),
        "mpd_mean": mpd.mean(),
        "mpd_median": mpd.median(),
        "mpd_std": mpd.std(),
        "mpd_T95_flagged": (
            test["flag_t95"].sum()
            if "flag_t95" in test else np.nan
        ),
        "mpd_T99_flagged": (
            test["flag_t99"].sum()
            if "flag_t99" in test else np.nan
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--before-model", required=True)
    parser.add_argument("--after-model", required=True)
    parser.add_argument("--before-mpd", required=True)
    parser.add_argument("--after-mpd", required=True)
    parser.add_argument("--ga-summary", required=True)
    parser.add_argument("--output", default="results/final_comparison")

    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    before = {
        **get_model_data(args.before_model),
        **get_mpd_data(args.before_mpd),
    }

    after = {
        **get_model_data(args.after_model),
        **get_mpd_data(args.after_mpd),
    }

    # Read GA comparison results
    ga = pd.read_csv(args.ga_summary).set_index("run")

    for label, row in [("Original", before), ("Retrained", after)]:
        if label in ga.index:
            g = ga.loc[label]
            row.update({
                "GA_source_files": g["source_files"],
                "GA_candidate_variants": g["candidate_variants"],
                "GA_eligible_variants": g["eligible_variants"],
                "GA_flips_common": g["LGBM_flips_common"],
                "GA_evasion_common_pct": g[
                    "LGBM_common_flip_rate_%"
                ],
                "GA_flips_operational": g["LGBM_flips_at_0.5"],
                "GA_evasion_operational_pct": g[
                    "LGBM_operational_flip_rate_%"
                ],
            })

    # Build final comparison table
    table = pd.DataFrame(
        [before, after],
        index=["Original", "Retrained"]
    )
    table.index.name = "Model"

    table.to_csv(output / "final_model_comparison.csv")

    print("\nFINAL MODEL COMPARISON")
    print("=" * 80)
    print(table.T.to_string())

    # Publication plot: GA evasion rates
    labels = ["Original", "Retrained"]

    common = [
        before.get("GA_evasion_common_pct", np.nan),
        after.get("GA_evasion_common_pct", np.nan),
    ]

    operational = [
        before.get("GA_evasion_operational_pct", np.nan),
        after.get("GA_evasion_operational_pct", np.nan),
    ]

    x = np.arange(2)
    width = 0.34

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        x - width / 2,
        common,
        width,
        label="Common threshold (0.942)"
    )

    ax.bar(
        x + width / 2,
        operational,
        width,
        label="Operational threshold"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("GA evasion rate (%)")
    ax.set_title("GA Evasion Before and After Adversarial Retraining")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.2f%%", padding=3)

    fig.tight_layout()
    fig.savefig(
        output / "final_ga_evasion_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    print("\nSaved:")
    print(output / "final_model_comparison.csv")
    print(output / "final_ga_evasion_comparison.png")


if __name__ == "__main__":
    main()
