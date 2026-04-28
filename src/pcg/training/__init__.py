"""Training pipelines for PCG models.

Phase 3 ships:
    dataset.py    — pairs (60-min input, next-15-min target) for the LSTM.
    metrics.py    — MAE, RMSE, threshold recall (Recall is the proposal KPI).
    train_lstm.py — training loop with AsymmetricLoss(alpha=10).

Future phases add the autoencoder trainer and the drift-triggered
retraining pipeline.
"""

from .dataset import ForecastingDataset
from .metrics import mae, rmse, threshold_recall, ForecastMetrics
from .train_lstm import TrainConfig, train_forecaster

__all__ = [
    "ForecastingDataset",
    "mae",
    "rmse",
    "threshold_recall",
    "ForecastMetrics",
    "TrainConfig",
    "train_forecaster",
]
