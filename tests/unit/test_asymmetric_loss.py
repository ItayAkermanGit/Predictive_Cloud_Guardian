# Unit tests for AsymmetricLoss.

from __future__ import annotations

import math

import pytest
import torch

from pcg.core.constants import ASYMMETRIC_LOSS_ALPHA
from pcg.models.losses import AsymmetricLoss


# regression mode

def test_alpha_one_equals_mse() -> None:
    # With alpha=1 the loss must collapse to plain MSE.
    torch.manual_seed(0)
    y_pred = torch.randn(8, 15, 4)
    y_true = torch.randn(8, 15, 4)
    custom = AsymmetricLoss(alpha=1.0)(y_pred, y_true)
    mse = torch.nn.functional.mse_loss(y_pred, y_true)
    assert torch.allclose(custom, mse, atol=1e-6)


def test_false_negative_costs_alpha_times_more_than_false_positive() -> None:
    # Two equal-magnitude residuals, opposite signs:
    # FN (under-prediction) loss should be alpha * FP (over-prediction).
    alpha = 10.0
    loss_fn = AsymmetricLoss(alpha=alpha)

    fp_pred = torch.tensor([1.0])  # over-predicting
    fp_true = torch.tensor([0.0])
    fn_pred = torch.tensor([0.0])  # under-predicting
    fn_true = torch.tensor([1.0])

    fp_loss = loss_fn(fp_pred, fp_true).item()
    fn_loss = loss_fn(fn_pred, fn_true).item()

    assert fn_loss == pytest.approx(alpha * fp_loss, rel=1e-6)


def test_default_alpha_matches_constants() -> None:
    # Default alpha must equal the project-wide constant (=10).
    assert AsymmetricLoss().alpha == ASYMMETRIC_LOSS_ALPHA


def test_zero_residual_zero_loss() -> None:
    y = torch.linspace(0.0, 1.0, 60).reshape(1, 60, 1)
    loss = AsymmetricLoss(alpha=10.0)(y, y).item()
    assert loss == pytest.approx(0.0)


def test_gradient_pushes_predictions_upward_under_under_prediction() -> None:
    # When the model under-predicts the gradient on y_pred must be negative
    # so optimizer step y_pred -= lr*grad pushes y_pred up.
    y_pred = torch.zeros(1, requires_grad=True)
    y_true = torch.ones(1)
    AsymmetricLoss(alpha=10.0)(y_pred, y_true).backward()
    # dL/dy_pred = -2 * alpha * (y_true - y_pred) = -2 * 10 * 1 = -20
    assert y_pred.grad.item() == pytest.approx(-20.0)


def test_gradient_for_over_prediction_uses_weight_one() -> None:
    # Symmetric to above; weight=1, so grad = -2 * (-1) = +2.
    y_pred = torch.ones(1, requires_grad=True)
    y_true = torch.zeros(1)
    AsymmetricLoss(alpha=10.0)(y_pred, y_true).backward()
    assert y_pred.grad.item() == pytest.approx(2.0)


def test_invalid_alpha_raises() -> None:
    with pytest.raises(ValueError):
        AsymmetricLoss(alpha=0.0)
    with pytest.raises(ValueError):
        AsymmetricLoss(alpha=-1.0)


def test_shape_mismatch_raises() -> None:
    loss_fn = AsymmetricLoss()
    with pytest.raises(ValueError):
        loss_fn(torch.zeros(2, 3), torch.zeros(2, 4))


def test_reduction_sum_returns_total() -> None:
    y_pred = torch.zeros(4)
    y_true = torch.tensor([1.0, 1.0, 1.0, 1.0])
    mean_loss = AsymmetricLoss(alpha=10.0, reduction="mean")(y_pred, y_true).item()
    sum_loss = AsymmetricLoss(alpha=10.0, reduction="sum")(y_pred, y_true).item()
    assert sum_loss == pytest.approx(mean_loss * 4)


# binary mode

def test_binary_mode_alpha_one_equals_bce() -> None:
    p = torch.tensor([0.2, 0.7, 0.9])
    y = torch.tensor([0.0, 1.0, 1.0])
    custom = AsymmetricLoss(alpha=1.0, mode="binary")(p, y).item()
    expected = torch.nn.functional.binary_cross_entropy(p, y).item()
    assert custom == pytest.approx(expected, rel=1e-5)


def test_binary_mode_penalizes_missed_positives_more() -> None:
    # Same probability error magnitude on both classes; the y=1 case
    # contributes alpha-times more to the loss.
    alpha = 10.0
    loss_fn = AsymmetricLoss(alpha=alpha, mode="binary")

    fn_loss = loss_fn(torch.tensor([0.1]), torch.tensor([1.0])).item()
    fp_loss = loss_fn(torch.tensor([0.9]), torch.tensor([0.0])).item()

    # FN log-prob is alpha * log(0.1); FP log-prob is log(0.1). Ratio = alpha.
    assert fn_loss == pytest.approx(alpha * fp_loss, rel=1e-6)


def test_binary_mode_clamps_avoid_inf() -> None:
    p = torch.tensor([0.0, 1.0])
    y = torch.tensor([1.0, 0.0])
    loss = AsymmetricLoss(alpha=10.0, mode="binary")(p, y).item()
    # Should be a large but finite number, not inf.
    assert math.isfinite(loss)


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        AsymmetricLoss(mode="exotic")  # type: ignore[arg-type]
