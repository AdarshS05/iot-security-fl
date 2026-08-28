"""
Metrics computation module for Federated Learning evaluation.
"""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    roc_auc_score,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, Any]:
    """
    Compute binary classification metrics for local or global FL evaluation.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary class labels (0 or 1).
        y_proba: Predicted probability of positive class (or (N, 2) probability array).

    Returns:
        Dict containing accuracy, precision, recall, f1_score, loss, auc_roc, and confusion matrix counts (tp, tn, fp, fn).
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    y_proba_arr = np.asarray(y_proba, dtype=float)

    if y_proba_arr.ndim == 2 and y_proba_arr.shape[1] == 2:
        y_proba_1d = y_proba_arr[:, 1]
    else:
        y_proba_1d = y_proba_arr

    y_proba_clipped = np.clip(y_proba_1d, 1e-15, 1 - 1e-15)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    try:
        loss_val = float(log_loss(y_true_arr, y_proba_clipped, labels=[0, 1]))
    except Exception:
        loss_val = float(-np.mean(y_true_arr * np.log(y_proba_clipped) + (1 - y_true_arr) * np.log(1 - y_proba_clipped)))

    try:
        auc_val = float(roc_auc_score(y_true_arr, y_proba_1d))
    except ValueError:
        auc_val = 0.5

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "loss": loss_val,
        "auc_roc": auc_val,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
