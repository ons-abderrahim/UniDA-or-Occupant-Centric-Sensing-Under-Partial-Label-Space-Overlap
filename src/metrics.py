from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)
from typing import Dict


UNKNOWN_LABEL = -1


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unknown_label: int = UNKNOWN_LABEL,
) -> Dict[str, float]:
    """Compute all five UniDA evaluation metrics.

    Args:
        y_true       : ground-truth labels; unknown samples have label == unknown_label.
        y_pred       : predicted labels;    rejected samples have label == unknown_label.
        unknown_label: sentinel value for the unknown / target-private class.

    Returns:
        dict with keys: accuracy, precision, recall, f1, ucdr  (all in [0, 1]).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    # UCDR  [Eq. 12]
    # UT = unknown samples correctly identified as unknown
    # UF = unknown samples misclassified as known
    mask_unknown = y_true == unknown_label
    if mask_unknown.sum() == 0:
        ucdr = np.nan
    else:
        UT = np.sum((y_pred == unknown_label) & mask_unknown)
        UF = np.sum((y_pred != unknown_label) & mask_unknown)
        ucdr = float(UT) / (float(UT) + float(UF) + 1e-9)

    #  Known-class metrics  [Eq. 8-11] 
    # Restrict to samples that are truly *known*
    mask_known = ~mask_unknown
    yt_k = y_true[mask_known]
    yp_k = y_pred[mask_known]

    # Treat predictions of 'unknown_label' on known samples as wrong
    # (count as misclassification into a dummy class)
    yp_k_clean = np.where(yp_k == unknown_label, -999, yp_k)

    if len(yt_k) == 0:
        return dict(accuracy=0.0, precision=0.0, recall=0.0, f1=0.0, ucdr=ucdr)

    acc  = accuracy_score(yt_k, yp_k_clean)
    prec = precision_score(yt_k, yp_k_clean, average="weighted", zero_division=0)
    rec  = recall_score   (yt_k, yp_k_clean, average="weighted", zero_division=0)
    f1   = f1_score       (yt_k, yp_k_clean, average="weighted", zero_division=0)

    return dict(
        accuracy=float(acc),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        ucdr=float(ucdr) if not np.isnan(ucdr) else 0.0,
    )


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """Pretty-print the metrics dict."""
    tag = f"[{prefix}] " if prefix else ""
    print(
        f"{tag}"
        f"Acc={metrics['accuracy']:.4f}  "
        f"Prec={metrics['precision']:.4f}  "
        f"Rec={metrics['recall']:.4f}  "
        f"F1={metrics['f1']:.4f}  "
        f"UCDR={metrics['ucdr']:.4f}"
    )
