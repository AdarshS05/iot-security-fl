"""
Script to generate model choice rationale flow diagram as PNG image.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def generate_diagram(output_path: str = "docs/model_choice_diagram.png"):
    """Generate model choice rationale flow diagram as PNG image."""
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(
        5,
        5.6,
        "FL Model Architecture Rationale & Federation Pathways",
        fontsize=15,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8", edgecolor="#cbd5e1", lw=1.5),
    )

    blue_box = dict(boxstyle="round,pad=0.6", facecolor="#e0f2fe", edgecolor="#0284c7", lw=2)
    green_box = dict(boxstyle="round,pad=0.6", facecolor="#dcfce7", edgecolor="#16a34a", lw=2)
    amber_box = dict(boxstyle="round,pad=0.6", facecolor="#fef3c7", edgecolor="#d97706", lw=2)
    purple_box = dict(boxstyle="round,pad=0.6", facecolor="#f3e8ff", edgecolor="#9333ea", lw=2)

    ax.text(
        2.2,
        4.1,
        "Feed-Forward NN\n(Current Placeholder)",
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=blue_box,
    )

    ax.annotate(
        "",
        xy=(4.6, 4.1),
        xytext=(3.3, 4.1),
        arrowprops=dict(arrowstyle="->", color="#0284c7", lw=2.5),
    )
    ax.text(
        3.95,
        4.35,
        "swap-in",
        fontsize=10,
        fontstyle="italic",
        fontweight="bold",
        color="#0369a1",
        ha="center",
        va="center",
    )

    ax.text(
        6.8,
        4.1,
        "Siamese SVM Embedding Layer\n(Federates cleanly, same as placeholder)",
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=green_box,
    )

    ax.text(
        2.2,
        1.8,
        "LightGBM\n(Tree-Based Model)",
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=amber_box,
    )

    ax.text(
        7.2,
        1.8,
        "FedAvg Aggregation\n(Weight Averaging)",
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=purple_box,
    )

    ax.annotate(
        "",
        xy=(7.2, 2.35),
        xytext=(7.2, 3.4),
        arrowprops=dict(arrowstyle="->", color="#16a34a", lw=2.5),
    )
    ax.text(
        7.8,
        2.85,
        "direct weight\naggregation",
        fontsize=9,
        color="#15803d",
        ha="left",
        va="center",
    )

    ax.annotate(
        "",
        xy=(6.3, 2.3),
        xytext=(2.8, 3.5),
        arrowprops=dict(arrowstyle="->", color="#0284c7", lw=1.8, linestyle=":"),
    )

    ax.annotate(
        "",
        xy=(6.1, 1.8),
        xytext=(3.4, 1.8),
        arrowprops=dict(arrowstyle="->", color="#d97706", lw=2.5, linestyle="--"),
    )
    ax.text(
        4.75,
        2.15,
        "needs federated-boosting OR embed-only\nOR differentiable conversion",
        fontsize=9.5,
        fontweight="bold",
        color="#b45309",
        ha="center",
        va="center",
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Model choice diagram successfully saved to {output_file.resolve()}")


if __name__ == "__main__":
    generate_diagram()
