import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from common.partitioner import generate_dummy_labels, partition_dirichlet

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "partition_config.yaml"
OUTPUT_IMG_PATH = Path(__file__).resolve().parent.parent / "docs" / "partition_distribution.png"


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config(CONFIG_PATH)

    num_clients = config.get("num_clients", 10)
    num_classes = config.get("num_classes", 2)
    dirichlet_alpha = config.get("dirichlet_alpha", 0.5)
    random_seed = config.get("random_seed", 42)
    num_samples = config.get("num_samples", 1000)

    labels = generate_dummy_labels(num_samples=num_samples, num_classes=num_classes, seed=random_seed)
    client_partitions = partition_dirichlet(
        labels=labels,
        num_clients=num_clients,
        alpha=dirichlet_alpha,
        seed=random_seed,
    )

    client_ids = [f"client_{i}" for i in range(num_clients)]
    benign_counts = []
    malware_counts = []

    summary_rows = []

    for client_id in client_ids:
        indices = client_partitions[client_id]
        client_labels = labels[indices]
        total = len(client_labels)
        b_count = int(np.sum(client_labels == 0))
        m_count = int(np.sum(client_labels == 1))

        benign_counts.append(b_count)
        malware_counts.append(m_count)

        pct_benign = (b_count / total * 100) if total > 0 else 0.0
        pct_malware = (m_count / total * 100) if total > 0 else 0.0

        summary_rows.append((client_id, total, pct_benign, pct_malware))

    header = f"{'Client ID':<12} | {'Total Samples':<13} | {'% Benign (0)':<14} | {'% Malware (1)':<14}"
    divider = "-" * len(header)
    print("\n" + divider)
    print(" Federated Client Data Partitioning Summary (Dirichlet alpha = {})".format(dirichlet_alpha))
    print(divider)
    print(header)
    print(divider)
    for cid, total, p_benign, p_malware in summary_rows:
        print(f"{cid:<12} | {total:<13} | {p_benign:>13.1f}% | {p_malware:>13.1f}%")
    print(divider + "\n")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(num_clients)
    width = 0.55

    color_benign = "#4C72B0"
    color_malware = "#C44E52"

    p1 = ax.bar(x, benign_counts, width, label="Benign (Class 0)", color=color_benign, edgecolor="black", linewidth=0.5)
    p2 = ax.bar(x, malware_counts, width, bottom=benign_counts, label="Malware (Class 1)", color=color_malware, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Client Identifier", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Number of Samples", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(
        f"Non-IID Client Data Distribution (Dirichlet α = {dirichlet_alpha})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i}" for i in range(num_clients)], fontsize=10)
    ax.legend(title="Data Class", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()

    OUTPUT_IMG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_IMG_PATH, dpi=300)
    plt.close(fig)

    print(f"Distribution plot saved to: {OUTPUT_IMG_PATH}\n")


if __name__ == "__main__":
    main()
