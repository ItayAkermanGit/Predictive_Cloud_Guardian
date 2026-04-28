"""Neural-network architectures used across PCG.

Phase 3 contributes:
    losses.py        — AsymmetricLoss (proposal Problem 3, alpha=10)
    lstm_attention.py — LSTM + additive attention forecaster (Problem 1)

Future phases add the autoencoder (Problem 2) and the cold-start generic
model (Problem 8). All architectures consume the canonical tensor shape
defined by ``pcg.data.windowing`` — ``(B, WINDOW_SIZE, N_METRICS)``.
"""

from .losses import AsymmetricLoss
from .lstm_attention import AdditiveAttention, LSTMForecaster

__all__ = [
    "AsymmetricLoss",
    "AdditiveAttention",
    "LSTMForecaster",
]
