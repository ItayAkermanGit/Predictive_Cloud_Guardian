# Forecasting metrics: MAE, RMSE, threshold-based recall and FPR.

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def mae(y_pred: Tensor, y_true: Tensor) -> float:
    # Mean Absolute Error.
    _check_shape(y_pred, y_true)
    return torch.mean(torch.abs(y_pred - y_true)).item()


def rmse(y_pred: Tensor, y_true: Tensor) -> float:
    # Root Mean Squared Error.
    _check_shape(y_pred, y_true)
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()


def threshold_recall(
    y_pred: Tensor,
    y_true: Tensor,
    threshold: float,
) -> float:
    # Recall over threshold-crossing events.
    # positive    : y_true > threshold
    # TP          : both pred and true above threshold
    # FN          : true above threshold, pred not
    # If there are no positives in y_true returns 1.0 (sklearn convention).
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
    # FP / (FP + TN) — fraction of safe moments mislabeled as breach.
    _check_shape(y_pred, y_true)
    actual_neg = y_true <= threshold
    pred_pos = y_pred > threshold
    fp = (actual_neg & pred_pos).sum().item()
    tn = (actual_neg & ~pred_pos).sum().item()
    if fp + tn == 0:
        return 0.0
    return fp / (fp + tn)


@dataclass(frozen=True)
class ForecastMetrics:
    # Bundle of metrics emitted by the training/eval loop.

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


def _check_shape(y_pred: Tensor, y_true: Tensor) -> None:
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"shape mismatch: y_pred={tuple(y_pred.shape)} vs "
            f"y_true={tuple(y_true.shape)}"
        )
