"""
Metrics logging utility to record round-level global FL metrics to CSV.
"""

from pathlib import Path
import csv
from typing import Dict, Any

HEADER = [
    "round_number",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "loss",
    "auc_roc",
    "num_clients",
    "mean_weight",
    "std_weight",
]


def log_global_metrics(metrics_dict: Dict[str, Any], csv_path: str = "docs/metrics_log.csv") -> None:
    """
    Append a round's global metrics row to docs/metrics_log.csv.
    If round_number is 1 or file does not exist, initialize file with CSV header.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    round_number = metrics_dict.get("round_number", 1)
    file_exists = path.exists()

    mode = "w" if (round_number == 1 or not file_exists) else "a"

    with open(path, mode, newline="") as csvfile:
        writer = csv.writer(csvfile)
        if mode == "w":
            writer.writerow(HEADER)

        row = [
            int(metrics_dict.get("round_number", 0)),
            f"{float(metrics_dict.get('accuracy', 0.0)):.6f}",
            f"{float(metrics_dict.get('precision', 0.0)):.6f}",
            f"{float(metrics_dict.get('recall', 0.0)):.6f}",
            f"{float(metrics_dict.get('f1_score', 0.0)):.6f}",
            f"{float(metrics_dict.get('loss', 0.0)):.6f}",
            f"{float(metrics_dict.get('auc_roc', 0.0)):.6f}",
            int(metrics_dict.get("num_clients", 0)),
            f"{float(metrics_dict.get('mean_weight', 0.0)):.6f}",
            f"{float(metrics_dict.get('std_weight', 0.0)):.6f}",
        ]
        writer.writerow(row)
