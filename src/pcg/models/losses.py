# Asymmetric loss that punishes false negatives much harder than false positives.
#
# In a critical monitoring system, missing a real failure is far worse
# than a false alarm. So we weight residuals where y_true > y_pred
# (under-prediction = potential missed failure) by alpha=10, and the
# opposite case by 1. This pushes Adam to bias predictions upward.
#
# Regression form:
#   residual = y_true - y_pred
#   weight   = alpha if residual > 0 else 1
#   L = mean(weight * residual^2)
#
# Binary form (for a probability output):
#   L = -[ alpha * y * log(p) + (1-y) * log(1-p) ]

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from ..core.constants import ASYMMETRIC_LOSS_ALPHA


class AsymmetricLoss(nn.Module):
    # Custom asymmetric loss for the forecaster.
    # alpha must be > 0; alpha=1 collapses to plain MSE / BCE.

    def __init__(
        self,
        alpha: float = ASYMMETRIC_LOSS_ALPHA,
        mode: Literal["regression", "binary"] = "regression",
        eps: float = 1e-7,
        reduction: Literal["mean", "sum"] = "mean",
    ) -> None:
        super().__init__()
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if mode not in ("regression", "binary"):
            raise ValueError(f"unknown mode {mode!r}")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"unknown reduction {reduction!r}")
        self.alpha = float(alpha)
        self.mode = mode
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # Compute the asymmetric loss between predictions and targets.
        # y_pred and y_true must have the same shape.
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"shape mismatch: y_pred={tuple(y_pred.shape)} "
                f"vs y_true={tuple(y_true.shape)}"
            )

        if self.mode == "regression":
            elementwise = self._regression_loss(y_pred, y_true)
        else:
            elementwise = self._binary_loss(y_pred, y_true)

        return elementwise.mean() if self.reduction == "mean" else elementwise.sum()

    def _regression_loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # Asymmetric MSE.
        # residual > 0 -> under-prediction (FN-like) -> weight=alpha.
        # residual <= 0 -> over-prediction (FP-like) -> weight=1.
        residual = y_true - y_pred
        weight = torch.where(
            residual > 0,
            torch.full_like(residual, self.alpha),
            torch.ones_like(residual),
        )
        return weight * residual.pow(2)

    def _binary_loss(self, p_pred: Tensor, y_true: Tensor) -> Tensor:
        # Asymmetric binary cross-entropy on probabilities in [0, 1].
        # alpha multiplies the y=1 term to penalize missed positives.
        p = p_pred.clamp(min=self.eps, max=1.0 - self.eps)
        positive = y_true * torch.log(p)
        negative = (1.0 - y_true) * torch.log(1.0 - p)
        return -(self.alpha * positive + negative)
