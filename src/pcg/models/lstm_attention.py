# Multivariate LSTM forecaster with additive attention.
#
# Input:  (B, WINDOW_SIZE=60, N_METRICS=4)
# Output: (B, HORIZON_MINUTES=15, N_METRICS=4) and (B, WINDOW_SIZE) attention weights.
#
# The LSTM has the standard forget/input/output gates; we use nn.LSTM
# directly for speed. Attention learns a per-timestep weight so the
# model can highlight which past minutes mattered most for the forecast.
#
# Attention scoring (Bahdanau-style additive):
#   e_t = v^T * tanh(W * h_t)
#   alpha = softmax(e)
#   context = sum_t alpha_t * h_t
# The context vector is then projected to (HORIZON * N_METRICS) outputs
# and reshaped.

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..core.constants import HORIZON_MINUTES, N_METRICS, WINDOW_SIZE


class AdditiveAttention(nn.Module):
    # Bahdanau-style attention with a learnable global query.

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {hidden_size}")
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs: Tensor) -> Tuple[Tensor, Tensor]:
        # encoder_outputs: (B, T, H).
        # Returns context (B, H) and softmax weights (B, T).
        if encoder_outputs.dim() != 3:
            raise ValueError(
                f"expected (B, T, H), got shape {tuple(encoder_outputs.shape)}"
            )

        # Scoring head: (B,T,H) -> (B,T,H) -> (B,T,1) -> (B,T).
        scores = self.v(torch.tanh(self.W(encoder_outputs))).squeeze(-1)
        weights = F.softmax(scores, dim=-1)

        # Weighted sum: (B,1,T) bmm (B,T,H) -> (B,1,H) -> (B,H).
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class LSTMForecaster(nn.Module):
    # LSTM + attention forecaster with multi-step output.

    def __init__(
        self,
        n_metrics: int = N_METRICS,
        hidden_size: int = 64,
        num_layers: int = 1,
        horizon: int = HORIZON_MINUTES,
        window_size: int = WINDOW_SIZE,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if n_metrics <= 0:
            raise ValueError(f"n_metrics must be > 0, got {n_metrics}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {hidden_size}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")

        self.n_metrics = n_metrics
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon
        self.window_size = window_size

        # PyTorch only applies LSTM dropout between stacked layers.
        layer_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=n_metrics,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=layer_dropout,
        )
        self.attention = AdditiveAttention(hidden_size)

        # One linear projection from the context vector to all
        # horizon * n_metrics outputs, then reshape.
        self.projection = nn.Linear(hidden_size, horizon * n_metrics)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # Returns forecast (B, horizon, n_metrics) and attention (B, window_size).
        self._validate_input(x)

        encoder_outputs, _ = self.lstm(x)        # (B, T, H)
        context, attn_weights = self.attention(encoder_outputs)  # (B, H), (B, T)
        flat = self.projection(context)          # (B, horizon * n_metrics)
        forecast = flat.view(-1, self.horizon, self.n_metrics)
        return forecast, attn_weights

    def _validate_input(self, x: Tensor) -> None:
        if x.dim() != 3:
            raise ValueError(
                f"expected 3-D input (B, T, F), got shape {tuple(x.shape)}"
            )
        if x.size(2) != self.n_metrics:
            raise ValueError(
                f"expected last dim = {self.n_metrics}, got {x.size(2)}"
            )
