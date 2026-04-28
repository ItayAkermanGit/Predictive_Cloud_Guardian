"""Forecasting metrics.

The proposal lists three KPIs:

    Recall              — the *operationally* important one. We compute it
                          per-metric over threshold-crossing events. High
                          recall = few missed failures = AsymmetricLoss
                          working as intended.
    False Positive rate — paired with recall to form the precision/recall
                          trade-off curve.
    MTTR (Mean Time To  — out of scope for the forecaster itself; that is
    Repair)               an end-to-end alerting metric measured by the
                          controller.

This module exposes plain functions plus a ``ForecastMetrics`` aggregate
that stores them together so callers (training loop, tests) can pass one
object around.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


# --------------------------------------------------------------------- #
# Per-metric scalar functions
# --------------------------------------------------------------------- #

def mae(y_pred: Tensor, y_true: Tensor) -> float:
    """Mean Absolute Error. Lower is better; 0 means perfect."""
    _check_shape(y_pred, y_true)
    return torch.mean(torch.abs(y_pred - y_true)).item()


def rmse(y_pred: Tensor, y_true: Tensor) -> float:
    """Root Mean Squared Error.

    Penalizes large mistakes more than MAE — useful as a sanity check
    that the asymmetric loss is not letting outlier mispredictions slip
    through.
    """
    _check_shape(y_pred, y_true)
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()


def threshold_recall(
    y_pred: Tensor,
    y_true: Tensor,
    threshold: float,
) -> float:
    """Recall over threshold-crossing events.

    Definitions used here:
        positive event   — any timestep where y_true exceeds ``threshold``.
        true positive    — the model also predicted > threshold there.
        false negative   — the model predicted ≤ threshold there.

    Returns ``TP / (TP + FN)``. If there are no positives in y_true the
    function returns 1.0 (vacuously perfect: nothing to miss). This is
    the convention scikit-learn uses.

    This is THE metric the proposal optimizes for via AsymmetricLoss
    (alpha=10) — high recall on real failures.
    """
    _check_shape(y_pred, y_true)
    actual_pos = y_true > threshold
    pred_pos = y_pred > threshold
    tp = (actual_pos & pred_pos).sum().item()
    fn = (actual_pos & ~pred_pos).sum().item()
    if tp + fn == 0:
        return 1.0
    return tp / (tp + fn)


def threshold_false_positive_rate(
    y_pred: Tensor,
    y_true: Tensor,
    threshold: float,
) -> float:
    """FP / (FP + TN) — proportion of safe moments mislabeled as failures.

    Used alongside recall to monitor the cost paid for the asymmetric
    loss: pushing recall up usually pushes FP up too. Watching both
    keeps the trade-off honest.
    """
    _check_shape(y_pred, y_true)
    actual_neg = y_true <= threshold
    pred_pos = y_pred > threshold
    fp = (actual_neg & pred_pos).sum().item()
    tn = (actual_neg & ~pred_pos).sum().item()
    if fp + tn == 0:
        return 0.0
    return fp / (fp + tn)


# --------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class ForecastMetrics:
    """Bundle of metrics emitted by the training/eval loop.

    ``recall`` and ``fpr`` are computed against a per-metric threshold
    that the caller supplies — typically a [0,1] cutoff matching the
    operational threshold in ``configs/thresholds.yaml`` after
    normalization.
    """

    mae: float
    rmse: float
    recall: float
    fpr: float

    @classmethod
    def from_tensors(
        cls,
        y_pred: Tensor,
        y_true: Tensor,
        threshold: float,
    ) -> "ForecastMetrics":
        return cls(
            mae=mae(y_pred, y_true),
            rmse=rmse(y_pred, y_true),
            recall=threshold_recall(y_pred, y_true, threshold),
            fpr=threshold_false_positive_rate(y_pred, y_true, threshold),
        )


# --------------------------------------------------------------------- #
# Internal
# --------------------------------------------------------------------- #

def _check_shape(y_pred: Tensor, y_true: Tensor) -> None:
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"shape mismatch: y_pred={tuple(y_pred.shape)} vs "
            f"y_true={tuple(y_true.shape)}"
        )
