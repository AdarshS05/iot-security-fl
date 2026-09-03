"""
Script to generate FL training metrics dashboard image from docs/metrics_log.csv.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def generate_dashboard(csv_path: str = "docs/metrics_log.csv", output_path: str = "docs/metrics_dashboard.png"):
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Metrics CSV file not found at {csv_path}. Run FL training first.")

    df = pd.read_csv(csv_file)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("FL Training Metrics — Dummy Baseline (Feed-Forward Placeholder)", fontsize=16, fontweight="bold", y=0.98)

    rounds = df["round_number"]

    axes[0, 0].plot(rounds, df["accuracy"], marker="o", color="#1f77b4", linewidth=2.5, label="Accuracy")
    axes[0, 0].set_title("Accuracy vs Round", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Round", fontsize=10)
    axes[0, 0].set_ylabel("Accuracy", fontsize=10)
    axes[0, 0].set_xticks(rounds)
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)
    axes[0, 0].set_ylim(0, 1.05)

    axes[0, 1].plot(rounds, df["loss"], marker="s", color="#d62728", linewidth=2.5, label="Loss")
    axes[0, 1].set_title("Loss vs Round", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Round", fontsize=10)
    axes[0, 1].set_ylabel("Binary Cross-Entropy Loss", fontsize=10)
    axes[0, 1].set_xticks(rounds)
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)

    axes[1, 0].plot(rounds, df["precision"], marker="^", color="#2ca02c", linewidth=2.5, label="Precision")
    axes[1, 0].plot(rounds, df["recall"], marker="v", color="#ff7f0e", linewidth=2.5, label="Recall")
    axes[1, 0].set_title("Precision & Recall vs Round", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Round", fontsize=10)
    axes[1, 0].set_ylabel("Score", fontsize=10)
    axes[1, 0].set_xticks(rounds)
    axes[1, 0].legend(loc="best", frameon=True)
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)
    axes[1, 0].set_ylim(0, 1.05)

    axes[1, 1].plot(rounds, df["f1_score"], marker="D", color="#9467bd", linewidth=2.5, label="F1 Score")
    axes[1, 1].set_title("F1 Score vs Round", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Round", fontsize=10)
    axes[1, 1].set_ylabel("F1 Score", fontsize=10)
    axes[1, 1].set_xticks(rounds)
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)
    axes[1, 1].set_ylim(0, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Metrics dashboard successfully saved to {output_file.resolve()}")


if __name__ == "__main__":
    generate_dashboard()
