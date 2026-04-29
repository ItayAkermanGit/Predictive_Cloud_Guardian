# End-to-end integration tests for the forecasting pipeline.
# Trains a small LSTM forecaster on synthetic data for a few epochs
# and checks: loss decreases, inference shapes match, denorm is invertible.

from __future__ import annotations

from datetime import datetime

import numpy as np
import torch

from pcg.core.constants import (
    HORIZON_MINUTES,
    METRIC_ORDER,
    N_METRICS,
    WINDOW_SIZE,
)
from pcg.data.normalizer import MinMaxNormalizer
from pcg.data.pipeline import DataPipeline
from pcg.data.synthetic import SyntheticConfig, SyntheticMetricGenerator
from pcg.data.tsdb_client import InMemoryTSDBClient
from pcg.inference.forecaster import Forecaster, ForecastOutput
from pcg.models.lstm_attention import LSTMForecaster
from pcg.training.dataset import ForecastingDataset
from pcg.training.train_lstm import TrainConfig, train_forecaster


def _build_environment(minutes: int = 60 * 24):
    # Generate synthetic data, fit a normalizer, and split train/val.
    frame = SyntheticMetricGenerator(
        SyntheticConfig(seed=11, missing_rate=0.02)
    ).generate(datetime(2026, 1, 1), minutes)
    split = int(0.8 * len(frame))
    train_frame = frame.iloc[:split]
    val_frame = frame.iloc[split:]
    normalizer = MinMaxNormalizer().fit(train_frame.dropna())
    return frame, train_frame, val_frame, normalizer


def test_training_loss_decreases_over_epochs() -> None:
    # A modest training run on synthetic data must reduce training loss.
    _, train_frame, val_frame, norm = _build_environment()
    train_ds = ForecastingDataset(train_frame, norm)
    val_ds = ForecastingDataset(val_frame, norm)

    model = LSTMForecaster(hidden_size=32)
    history = train_forecaster(
        model,
        train_ds,
        val_ds,
        TrainConfig(epochs=3, batch_size=64, learning_rate=2e-3, seed=0),
    )

    assert len(history.train_loss) == 3
    assert len(history.val_loss) == 3
    assert history.train_loss[-1] < history.train_loss[0]


def test_inference_returns_canonical_output() -> None:
    frame, train_frame, _, norm = _build_environment()
    train_ds = ForecastingDataset(train_frame, norm)

    model = LSTMForecaster(hidden_size=32)
    train_forecaster(
        model,
        train_ds,
        None,
        TrainConfig(epochs=1, batch_size=64, seed=0),
    )

    client = InMemoryTSDBClient()
    client.upsert("srv-1", frame)
    pipeline = DataPipeline(client, norm)
    end = frame.index[-1].to_pydatetime()
    prepared = pipeline.prepare_window("srv-1", end)

    forecaster = Forecaster(model, norm)
    out = forecaster.predict(prepared)

    assert isinstance(out, ForecastOutput)
    assert out.predicted_normalized.shape == (HORIZON_MINUTES, N_METRICS)
    assert out.predicted_original.shape == (HORIZON_MINUTES, N_METRICS)
    assert out.attention_weights.shape == (WINDOW_SIZE,)
    assert len(out.forecast_timestamps) == HORIZON_MINUTES
    assert out.metric_order == METRIC_ORDER


def test_attention_weights_sum_to_one_after_inference() -> None:
    frame, train_frame, _, norm = _build_environment()
    train_ds = ForecastingDataset(train_frame, norm)
    model = LSTMForecaster(hidden_size=32)
    train_forecaster(model, train_ds, None, TrainConfig(epochs=1, seed=0))

    client = InMemoryTSDBClient()
    client.upsert("srv-1", frame)
    pipeline = DataPipeline(client, norm)
    end = frame.index[-1].to_pydatetime()
    prepared = pipeline.prepare_window("srv-1", end)

    out = Forecaster(model, norm).predict(prepared)
    assert abs(float(out.attention_weights.sum()) - 1.0) <= 1e-4
    assert np.all(out.attention_weights >= -1e-7)


def test_predicted_original_values_match_invertible_normalization() -> None:
    # Manually invert the [0, 1] prediction and confirm it equals
    # predicted_original from the Forecaster wrapper.
    frame, train_frame, _, norm = _build_environment()
    train_ds = ForecastingDataset(train_frame, norm)
    model = LSTMForecaster(hidden_size=32)
    train_forecaster(model, train_ds, None, TrainConfig(epochs=1, seed=0))

    client = InMemoryTSDBClient()
    client.upsert("srv-1", frame)
    pipeline = DataPipeline(client, norm)
    end = frame.index[-1].to_pydatetime()
    prepared = pipeline.prepare_window("srv-1", end)

    out = Forecaster(model, norm).predict(prepared)

    manual = np.empty_like(out.predicted_normalized)
    for i, metric in enumerate(METRIC_ORDER):
        mn = float(norm.mins[metric])
        mx = float(norm.maxs[metric])
        manual[:, i] = out.predicted_normalized[:, i] * (mx - mn) + mn

    np.testing.assert_allclose(manual, out.predicted_original, rtol=1e-4)
