import os
import re
import ast
import glob
import json
from collections import Counter
import matplotlib.pyplot as plt

def parse_rejection_reasons_from_files(search_dirs):
    """
    Scans subdirectories for stage report files (.json, .log, or .txt)
    and extracts accumulated rejection reason counts.
    """
    aggregate_counts = Counter()
    
    # Locate all report/log files in specified stage directories
    files_to_scan = []
    for d in search_dirs:
        for root, _, files in os.walk(d):
            for file in files:
                if file.endswith(('.txt', '.log', '.json')):
                    files_to_scan.append(os.path.join(root, file))
    
    # Fallback to current directory if no stage subdirectories found
    if not files_to_scan:
        files_to_scan = glob.glob("*.txt") + glob.glob("*.log") + glob.glob("*.json")

    for filepath in set(files_to_scan):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 1. Parse string dicts logged in stdout/log lines:
            # e.g., [gen 02] rejection_reasons={'multiple_segment_growth_genes': 3, 'empty': 1}
            log_matches = re.findall(r'rejection_reasons\s*=\s*(\{.*?\})', content)
            stage_max = Counter()
            for match in log_matches:
                try:
                    reason_dict = ast.literal_eval(match)
                    for k, v in reason_dict.items():
                        stage_max[k] = max(stage_max[k], int(v))
                except Exception:
                    pass
            
            aggregate_counts.update(stage_max)

            # 2. Parse JSON reports if structured json is present
            if filepath.endswith('.json'):
                try:
                    data = json.loads(content)
                    if isinstance(data, dict) and "rejection_reasons" in data:
                        aggregate_counts.update(data["rejection_reasons"])
                except Exception:
                    pass

        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")

    # Remove zero-count categories
    return {k: v for k, v in aggregate_counts.items() if v > 0}


def plot_rejection_reasons(rejection_data, output_filename="rejection_reason_breakdown.png"):
    if not rejection_data:
        print("No non-zero rejection reasons found in logs.")
        return

    # Sort ascending so the largest category appears at the top of horizontal bar chart
    sorted_data = dict(sorted(rejection_data.items(), key=lambda item: item[1]))
    
    categories = list(sorted_data.keys())
    counts = list(sorted_data.values())

    # Styling for slide visuals
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(categories, counts, color='#2B5C8F', edgecolor='#1A3859', height=0.55)

    # Title & Labels
    ax.set_title(
        "Rejection Reason Breakdown\nConstraint System Is Actively Filtering Invalid Chromosomes",
        fontsize=13, fontweight='bold', pad=15, color='#111111'
    )
    ax.set_xlabel("Occurrences", fontsize=11, labelpad=10)
    ax.set_xlim(0, max(counts) + max(2, int(max(counts) * 0.2)))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    # Annotate bar values
    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f'{width}',
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            ha='left', va='center',
            fontsize=10, fontweight='bold', color='#222222'
        )

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Chart saved to {output_filename}")


if __name__ == "__main__":
    # Stage directories to search
    target_stage_dirs = ["./ga_output"]
    
    rejection_summary = parse_rejection_reasons_from_files(target_stage_dirs)
    print("Aggregate Rejection Summary:", rejection_summary)
    
    plot_rejection_reasons(rejection_summary)
