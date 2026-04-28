"""Unit tests for LSTMForecaster and AdditiveAttention."""

from __future__ import annotations

import pytest
import torch

from pcg.core.constants import HORIZON_MINUTES, N_METRICS, WINDOW_SIZE
from pcg.models.lstm_attention import AdditiveAttention, LSTMForecaster


# ----- AdditiveAttention -------------------------------------------- #

def test_attention_returns_correct_shapes() -> None:
    attn = AdditiveAttention(hidden_size=32)
    h = torch.randn(2, 60, 32)
    context, weights = attn(h)
    assert context.shape == (2, 32)
    assert weights.shape == (2, 60)


def test_attention_weights_are_a_distribution_over_time() -> None:
    """Softmax along the time axis must produce non-negative weights
    that sum to 1 within each batch element."""
    torch.manual_seed(0)
    attn = AdditiveAttention(hidden_size=16)
    h = torch.randn(4, 60, 16)
    _, weights = attn(h)
    # All non-negative
    assert torch.all(weights >= 0)
    # Sum to 1 along time
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(4), atol=1e-5)


def test_attention_rejects_non_3d_input() -> None:
    attn = AdditiveAttention(hidden_size=8)
    with pytest.raises(ValueError):
        attn(torch.randn(2, 8))  # 2-D, not (B, T, H)


def test_attention_invalid_hidden_size_raises() -> None:
    with pytest.raises(ValueError):
        AdditiveAttention(hidden_size=0)


# ----- LSTMForecaster ----------------------------------------------- #

def test_forward_pass_shapes() -> None:
    model = LSTMForecaster()
    x = torch.zeros(3, WINDOW_SIZE, N_METRICS)
    forecast, attn = model(x)
    assert forecast.shape == (3, HORIZON_MINUTES, N_METRICS)
    assert attn.shape == (3, WINDOW_SIZE)


def test_forward_dtype_is_float32() -> None:
    model = LSTMForecaster()
    x = torch.zeros(1, WINDOW_SIZE, N_METRICS)
    forecast, attn = model(x)
    assert forecast.dtype == torch.float32
    assert attn.dtype == torch.float32


def test_forward_rejects_wrong_feature_dim() -> None:
    model = LSTMForecaster()
    bad = torch.zeros(2, WINDOW_SIZE, N_METRICS + 1)
    with pytest.raises(ValueError):
        model(bad)


def test_forward_rejects_non_3d_input() -> None:
    model = LSTMForecaster()
    bad = torch.zeros(WINDOW_SIZE, N_METRICS)  # missing batch axis
    with pytest.raises(ValueError):
        model(bad)


def test_backward_runs_and_updates_parameters() -> None:
    """A single optimization step must lower the loss on a constant target."""
    torch.manual_seed(0)
    model = LSTMForecaster(hidden_size=16)
    x = torch.randn(8, WINDOW_SIZE, N_METRICS)
    target = torch.ones(8, HORIZON_MINUTES, N_METRICS) * 0.5

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.functional.mse_loss

    initial_pred, _ = model(x)
    loss_before = loss_fn(initial_pred, target).item()

    for _ in range(5):
        optimizer.zero_grad()
        pred, _ = model(x)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()

    final_pred, _ = model(x)
    loss_after = loss_fn(final_pred, target).item()
    assert loss_after < loss_before


def test_invalid_constructor_args_raise() -> None:
    with pytest.raises(ValueError):
        LSTMForecaster(n_metrics=0)
    with pytest.raises(ValueError):
        LSTMForecaster(hidden_size=0)
    with pytest.raises(ValueError):
        LSTMForecaster(num_layers=0)
    with pytest.raises(ValueError):
        LSTMForecaster(horizon=0)


def test_attention_weights_remain_a_distribution_after_lstm() -> None:
    """End-to-end check that the softmax invariant survives the LSTM stack."""
    torch.manual_seed(0)
    model = LSTMForecaster()
    x = torch.randn(2, WINDOW_SIZE, N_METRICS)
    _, attn = model(x)
    sums = attn.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(2), atol=1e-5)
    assert torch.all(attn >= 0)


def test_two_inputs_yield_different_forecasts() -> None:
    """Sanity check that the model is responsive to its input — a frozen
    output across different inputs would silently mask a bug."""
    torch.manual_seed(0)
    model = LSTMForecaster(hidden_size=16)
    x_a = torch.randn(1, WINDOW_SIZE, N_METRICS)
    x_b = torch.randn(1, WINDOW_SIZE, N_METRICS)
    pred_a, _ = model(x_a)
    pred_b, _ = model(x_b)
    assert not torch.allclose(pred_a, pred_b)
