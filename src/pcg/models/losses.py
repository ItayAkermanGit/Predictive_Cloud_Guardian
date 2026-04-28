"""Asymmetric loss functions (proposal Problem 3).

Defense intuition (read this before the oral exam):

    A standard regression loss like MSE treats two kinds of errors as
    equally costly:

        FP — model predicts a HIGH value when the truth is low.
             ("the system shouted *wolf*; nothing happened")  → annoying.

        FN — model predicts a LOW value when the truth is high.
             ("the system stayed quiet; the server crashed")  → disastrous.

    In a critical-infrastructure AIOps setting these costs are wildly
    different. The proposal calls FP a `מטרד תפעולי` (operational nuisance)
    and FN a `הרסנית` (destructive) outcome. To make the model behave
    accordingly, we hand-craft a loss that is mathematically
    α-times more expensive when the residual is positive
    (y_true > y_pred → we under-predicted → FN territory).

Mathematical form (regression mode):

    residual_i  = y_true_i - y_pred_i
    weight_i    = α   if residual_i > 0   (under-prediction → FN-ish)
                  1   otherwise            (over-prediction → FP-ish)
    L           = mean_i ( weight_i * residual_i^2 )

Why this works (gradient view):

    ∂L/∂y_pred_i = -2 * weight_i * residual_i / N

    When the model under-predicts, weight_i = α and the gradient is α-times
    stronger than the symmetric MSE. Adam therefore takes much larger
    descent steps that pull predictions UPWARDS until residual_i ≤ 0.
    The optimization process literally rewires the network to bias
    upward — exactly the "high recall on failures" outcome the proposal
    asks for.

Classification mode (used when the model emits a probability for a binary
threshold-cross event rather than a raw value):

    L_BCE_asym = -[ α * y * log(p) + (1-y) * log(1-p) ]

    The alpha multiplier on the y=1 term punishes false negatives the
    same way as in regression mode. We expose this via ``mode='binary'``
    so future phases (e.g. a binary 'will_breach' classifier head) can
    reuse the same module without re-deriving the math.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from ..core.constants import ASYMMETRIC_LOSS_ALPHA


class AsymmetricLoss(nn.Module):
    """Custom asymmetric loss honoring proposal Problem 3.

    Parameters
    ----------
    alpha:
        Multiplier applied to the false-negative residual / log term.
        Defaults to ``ASYMMETRIC_LOSS_ALPHA`` (=10), as the proposal
        specifies. Must be > 0; alpha=1 collapses to symmetric MSE/BCE.
    mode:
        ``"regression"`` — asymmetric MSE (default).
        ``"binary"``     — asymmetric binary cross-entropy on probabilities
                          in [0, 1].
    eps:
        Numerical floor for the log() in binary mode.
    reduction:
        ``"mean"`` (default) or ``"sum"``. ``"none"`` is intentionally not
        supported — every consumer in PCG wants a scalar loss.
    """

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

    # ----- core API --------------------------------------------------- #

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Compute the asymmetric loss between predictions and targets.

        Shapes are arbitrary as long as ``y_pred.shape == y_true.shape``.
        Defense note: the same module handles ``(B, 15, N)`` forecasts and
        plain ``(B,)`` scalars without any reshape — element-wise math
        is shape-agnostic.
        """
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

    # ----- mode implementations -------------------------------------- #

    def _regression_loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Asymmetric MSE.

        residual > 0  ⇒  prediction was too low (FN-like) ⇒ weight=alpha.
        residual ≤ 0  ⇒  prediction was too high (FP-like) ⇒ weight=1.

        We use ``torch.where`` to keep the operation differentiable; the
        weights are constants w.r.t. residual sign (a step function with
        zero gradient at the boundary), which is fine because the squared
        residual provides the smooth signal that Adam follows.
        """
        residual = y_true - y_pred
        # weight is a constant tensor (alpha / 1.0); detach is unnecessary
        # because torch.where's branches are not differentiated through.
        weight = torch.where(
            residual > 0,
            torch.full_like(residual, self.alpha),
            torch.ones_like(residual),
        )
        return weight * residual.pow(2)

    def _binary_loss(self, p_pred: Tensor, y_true: Tensor) -> Tensor:
        """Asymmetric binary cross-entropy.

        Inputs are probabilities in [0, 1] (use sigmoid upstream). The
        ``alpha`` factor scales the y=1 term (true-positive log-likelihood),
        which is the term controlling whether the model misses a real
        threshold breach.
        """
        p = p_pred.clamp(min=self.eps, max=1.0 - self.eps)
        positive = y_true * torch.log(p)            # y=1 contributes here
        negative = (1.0 - y_true) * torch.log(1.0 - p)
        return -(self.alpha * positive + negative)
