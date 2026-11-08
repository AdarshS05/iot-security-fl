import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "results" / "training" / "lightgbm_bootstrap" / "clean_mpd_baseline_v2.csv"
OUT_PNG = REPO_ROOT / "fl-orchestration" / "docs" / "mpd_clean_baseline_improved.png"

def main():
    if not CSV_PATH.exists():
        sys.exit(f"Error: {CSV_PATH} not found.")

    df = pd.read_csv(CSV_PATH)
    mpd_all = df["mpd_score"].values
    mpd_benign = df[df["label"] == 0]["mpd_score"].values
    mpd_malware = df[df["label"] == 1]["mpd_score"].values

    p90 = np.percentile(mpd_all, 90)
    p95 = np.percentile(mpd_all, 95)
    p99 = np.percentile(mpd_all, 99)

    # Set up matplotlib aesthetic style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # -------------------------------------------------------------------------
    # Panel (a): Log-Scale Binned Histogram (Benign vs Malware)
    # -------------------------------------------------------------------------
    # Avoid log(0) by clipping minimum to 1e-16
    log_mpd_benign = np.log10(np.clip(mpd_benign, 1e-16, 0.5))
    log_mpd_malware = np.log10(np.clip(mpd_malware, 1e-16, 0.5))

    bins = np.linspace(-16, 0, 33)  # 0.5 log unit per bin

    ax1.hist(
        log_mpd_benign,
        bins=bins,
        alpha=0.65,
        color="#27ae60",
        edgecolor="#1e8449",
        linewidth=0.8,
        label="Benign (n=500)"
    )
    ax1.hist(
        log_mpd_malware,
        bins=bins,
        alpha=0.65,
        color="#e74c3c",
        edgecolor="#b03a2e",
        linewidth=0.8,
        label="Malware (n=500)"
    )

    ax1.set_xlabel(r"$\log_{10}(\mathrm{MPD\ Score})$", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Sample Count", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Log-Scale Clean MPD Score Distribution", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlim(-16.5, 0.5)
    ax1.set_xticks(np.arange(-16, 1, 2))
    ax1.grid(True)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, fontsize=10)

    # Annotations on Panel (a)
    ax1.annotate(
        "High Confidence\n(90% samples < 10⁻⁶)",
        xy=(-12, 180),
        xytext=(-15, 230),
        arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.2),
        fontsize=9.5,
        color="#2c3e50",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ebf5fb", edgecolor="#aed6f1", alpha=0.9)
    )

    ax1.annotate(
        "Hard / Boundary Tail\n(5% samples > 0.002)",
        xy=(-2, 25),
        xytext=(-7, 100),
        arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.2),
        fontsize=9.5,
        color="#8e44ad",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5eeed", edgecolor="#d7bde2", alpha=0.9)
    )

    # -------------------------------------------------------------------------
    # Panel (b): ECDF & Key Threshold Markers
    # -------------------------------------------------------------------------
    sorted_mpd = np.sort(mpd_all)
    ecdf = np.arange(1, len(sorted_mpd) + 1) / len(sorted_mpd)

    ax2.plot(sorted_mpd, ecdf, color="#2980b9", linewidth=2.5, label="Overall Clean ECDF")

    # Add threshold lines
    ax2.axvline(p95, color="#e67e22", linestyle="--", linewidth=1.5, label=f"95th Pct = {p95:.4f}")
    ax2.axvline(p99, color="#c0392b", linestyle=":", linewidth=1.5, label=f"99th Pct = {p99:.4f}")

    ax2.set_xscale("log")
    ax2.set_xlabel("MPD Score (Log Scale)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cumulative Fraction of Samples (ECDF)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Empirical Cumulative Distribution & Calibrated Thresholds", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlim(1e-16, 0.6)
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, which="both")
    ax2.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, fontsize=10)

    # Stats Summary Box on Panel (b)
    stats_text = (
        r"$\mathbf{Clean\ Baseline\ Stats:}$" + "\n"
        f"• Total Samples: {len(df)}\n"
        f"• Median MPD: {np.median(mpd_all):.2e}\n"
        f"• Mean MPD: {np.mean(mpd_all):.4f}\n"
        f"• 90th Pct: {p90:.2e}\n"
        f"• 95th Pct: {p95:.4f}\n"
        f"• 99th Pct: {p99:.4f}"
    )
    ax2.text(
        0.03, 0.95,
        stats_text,
        transform=ax2.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9f9", edgecolor="#bdc3c7", alpha=0.95)
    )

    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)
    print(f"Successfully generated improved visualization: {OUT_PNG}")

if __name__ == "__main__":
    main()
