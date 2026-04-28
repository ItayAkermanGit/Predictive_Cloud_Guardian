"""Multivariate LSTM forecaster with additive attention (Problem 1).

What this module produces
-------------------------
``LSTMForecaster(x)`` consumes a ``(B, WINDOW_SIZE=60, N_METRICS=4)``
tensor — the canonical Phase-2 output — and emits two values:

    forecast       — ``(B, HORIZON_MINUTES=15, N_METRICS=4)``
                     The next 15 minutes of every metric.
    attn_weights   — ``(B, WINDOW_SIZE)``
                     A normalized vector ``α_t`` over the input timesteps
                     describing which past minutes the model leaned on.

Why an LSTM
-----------
The LSTM cell maintains a long-term memory that can survive across
dozens of timesteps without vanishing — exactly what the proposal calls
"זיכרון ארוך-טווח קצר-טווח". Each cell uses three gates:

    forget gate (f)  — decides which parts of the previous cell state to
                       drop. Critical for shedding stale context after a
                       deployment or an operational shift.
    input gate (i)   — decides which parts of the current input deserve
                       to enter long-term memory. This is where rare
                       early-warning patterns get encoded.
    output gate (o)  — decides which slice of the cell state contributes
                       to the immediate output / next prediction.

We use ``nn.LSTM`` directly rather than re-implementing these gates: the
cuDNN-backed implementation is one to two orders of magnitude faster,
and the gating semantics are identical.

Why attention on top of the LSTM
--------------------------------
A 60-step LSTM hidden state still tends to over-emphasize the *last*
timestep — the most recent value dominates the cell state. An attention
layer learns a per-timestep weight ``α_t`` so the forecaster can flag,
e.g., "the slope from minute 40 to 50 was the early warning, more than
the values at minute 59". This is the proposal's argument
("מנגנון הקשב מאפשר למודל להקצות משקל משתנה לכל נקודת זמן בנתוני העבר.
המודל מזהה אילו חלקים ברצף הקלט הם קריטיים לביצוע החיזוי הרגעי") in code.

Implementation choice — additive (Bahdanau) attention with a learnable
global query — is the simplest variant that performs well on small
multivariate time-series and remains trivially explainable to an
examiner. The score formula is:

    e_t      = v^T · tanh(W · h_t)            (W ∈ R^{H×H}, v ∈ R^{H})
    α_t      = softmax_t(e_t)
    context  = Σ_t  α_t · h_t                 (∈ R^{H})

The context vector is then projected linearly to
``HORIZON_MINUTES × N_METRICS`` outputs, reshaped, and returned.

We deliberately do NOT use auto-regressive decoding: predicting all 15
future steps from a single context vector is faster and avoids the
compounding-error problem of step-by-step rollouts in a small model.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..core.constants import HORIZON_MINUTES, N_METRICS, WINDOW_SIZE


class AdditiveAttention(nn.Module):
    """Bahdanau-style attention with a learnable, input-independent query.

    A score is produced for each input timestep, softmax-normalized to a
    probability distribution α, and used to compute a weighted sum of
    the encoder hidden states.

    Defense note: the trainable matrix ``W`` is the projection that
    transforms hidden states before scoring; the vector ``v`` is the
    learnable query that picks out informative directions in that space.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {hidden_size}")
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs: Tensor) -> Tuple[Tensor, Tensor]:
        """Score → softmax → weighted sum.

        Args:
            encoder_outputs: ``(B, T, H)`` LSTM outputs across time.

        Returns:
            context : ``(B, H)`` — attention-weighted summary.
            weights : ``(B, T)`` — softmax probabilities (sum to 1 along T).
        """
        if encoder_outputs.dim() != 3:
            raise ValueError(
                f"expected (B, T, H), got shape {tuple(encoder_outputs.shape)}"
            )

        # Project then non-linearity then linear scoring head.
        # Shape walk:  (B,T,H) → (B,T,H) → (B,T,H) → (B,T,1) → (B,T)
        scores = self.v(torch.tanh(self.W(encoder_outputs))).squeeze(-1)
        weights = F.softmax(scores, dim=-1)

        # Weighted sum:  (B,1,T) · (B,T,H)  →  (B,1,H)  →  (B,H)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class LSTMForecaster(nn.Module):
    """Multivariate LSTM + attention 15-minute forecaster.

    Defaults are tuned for prototype scale: hidden_size=64, single layer.
    These are sized so the network trains in seconds on a CPU but already
    fits the diurnal+noise pattern of the synthetic generator from
    ``pcg.data.synthetic``.
    """

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

        # PyTorch's LSTM dropout only applies *between* stacked layers,
        # so we silently zero it out for single-layer setups to avoid the
        # noisy "dropout has no effect" warning.
        layer_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=n_metrics,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=layer_dropout,
        )
        self.attention = AdditiveAttention(hidden_size)

        # The decoder is a single linear projection from the attention
        # context to all `horizon * n_metrics` output values. Reshaping
        # afterwards rather than running 15 separate heads keeps the
        # parameter count low and lets the model learn cross-step
        # correlations in one weight matrix.
        self.projection = nn.Linear(hidden_size, horizon * n_metrics)

    # ----- forward pass --------------------------------------------- #

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Predict the next ``horizon`` steps.

        Args:
            x: ``(B, window_size, n_metrics)`` normalized input window.

        Returns:
            forecast: ``(B, horizon, n_metrics)`` predictions.
            attn:     ``(B, window_size)`` attention weights over the past.
        """
        self._validate_input(x)

        # encoder_outputs: (B, T, H) — every timestep gets a hidden state.
        # We discard (h_n, c_n) because attention reads `encoder_outputs`,
        # not the final state.
        encoder_outputs, _ = self.lstm(x)

        context, attn_weights = self.attention(encoder_outputs)
        # context: (B, H)

        flat = self.projection(context)
        # flat: (B, horizon * n_metrics)

        forecast = flat.view(-1, self.horizon, self.n_metrics)
        return forecast, attn_weights

    # ----- helpers --------------------------------------------------- #

    def _validate_input(self, x: Tensor) -> None:
        if x.dim() != 3:
            raise ValueError(
                f"expected 3-D input (B, T, F), got shape {tuple(x.shape)}"
            )
        if x.size(2) != self.n_metrics:
            raise ValueError(
                f"expected last dim = {self.n_metrics}, got {x.size(2)}"
            )
