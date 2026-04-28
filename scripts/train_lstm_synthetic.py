"""End-to-end demo: generate synthetic data, fit, train, predict.

Run this from the project root with:

    python scripts/train_lstm_synthetic.py

Outputs:
    * training & validation losses per epoch on stdout
    * a one-step forecast preview (normalized + original units)
    * the attention-weight distribution over the input window

This script doubles as a smoke test for the integration of every Phase-2
and Phase-3 component, and as the demo we can run live during the oral
project defense.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Make `pcg` importable when running this script directly without
# `pip install -e .`. Adds the project's `src/` to sys.path.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from pcg.core.constants import (
    HORIZON_MINUTES,
    METRIC_ORDER,
    SAMPLE_INTERVAL_SECONDS,
    WINDOW_SIZE,
)
from pcg.data.normalizer import MinMaxNormalizer
from pcg.data.pipeline import DataPipeline
from pcg.data.synthetic import (
    FailureInjection,
    SyntheticConfig,
    SyntheticMetricGenerator,
)
from pcg.data.tsdb_client import InMemoryTSDBClient
from pcg.inference.forecaster import Forecaster
from pcg.models.lstm_attention import LSTMForecaster
from pcg.training.dataset import ForecastingDataset
from pcg.training.metrics import ForecastMetrics
from pcg.training.train_lstm import TrainConfig, train_forecaster


def main() -> None:
    # ---------------------------------------------------------------- #
    # 1. Generate three days of synthetic per-minute metrics
    # ---------------------------------------------------------------- #
    base = datetime(2026, 1, 1, 0, 0)
    minutes = 60 * 24 * 3  # three days
    cfg = SyntheticConfig(
        seed=11,
        diurnal_amplitude=0.20,
        noise_std=0.03,
        missing_rate=0.02,
        # One late-night CPU spike to give the threshold-recall metric
        # something to chew on during validation.
        failures=[
            FailureInjection(
                metric="cpu_util",
                start_offset_minutes=60 * 24 + 60 * 3,  # day-2 03:00
                duration_minutes=30,
                magnitude=0.5,
            )
        ],
    )
    full_frame = SyntheticMetricGenerator(cfg).generate(base, minutes)

    # ---------------------------------------------------------------- #
    # 2. Train/val split + normalizer fit on training only
    # ---------------------------------------------------------------- #
    split_idx = int(0.8 * len(full_frame))
    train_frame = full_frame.iloc[:split_idx]
    val_frame = full_frame.iloc[split_idx:]

    normalizer = MinMaxNormalizer().fit(train_frame.dropna())
    print("Fitted normalizer ranges:")
    for metric in METRIC_ORDER:
        print(
            f"  {metric:<10s} min={normalizer.mins[metric]:+.3f} "
            f"max={normalizer.maxs[metric]:+.3f}"
        )

    train_ds = ForecastingDataset(train_frame, normalizer)
    val_ds = ForecastingDataset(val_frame, normalizer)
    print(
        f"\nDataset pairs — train: {len(train_ds):>5d}  "
        f"val: {len(val_ds):>5d}"
    )

    # ---------------------------------------------------------------- #
    # 3. Train
    # ---------------------------------------------------------------- #
    model = LSTMForecaster(hidden_size=64, num_layers=1)
    history = train_forecaster(
        model,
        train_ds,
        val_ds,
        TrainConfig(epochs=8, batch_size=64, learning_rate=1e-3),
    )

    print("\nTraining curve (asymmetric loss, alpha=10):")
    for epoch, (tl, vl) in enumerate(zip(history.train_loss, history.val_loss)):
        print(f"  epoch {epoch:>2d} — train {tl:.5f}   val {vl:.5f}")

    # ---------------------------------------------------------------- #
    # 4. Validation metrics
    # ---------------------------------------------------------------- #
    import torch
    from torch.utils.data import DataLoader

    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            yhat, _ = model(x)
            preds.append(yhat)
            targets.append(y)
    y_pred = torch.cat(preds, dim=0)
    y_true = torch.cat(targets, dim=0)

    # CPU column threshold = 0.85 in normalized units → mid-yellow alert.
    metrics = ForecastMetrics.from_tensors(y_pred, y_true, threshold=0.85)
    print(
        "\nValidation metrics — "
        f"MAE={metrics.mae:.4f}  RMSE={metrics.rmse:.4f}  "
        f"Recall@0.85={metrics.recall:.3f}  FPR@0.85={metrics.fpr:.3f}"
    )

    # ---------------------------------------------------------------- #
    # 5. End-to-end inference on the most recent window
    # ---------------------------------------------------------------- #
    client = InMemoryTSDBClient()
    client.upsert("srv-demo", full_frame)
    pipeline = DataPipeline(client, normalizer)

    end = full_frame.index[-1].to_pydatetime()
    prepared = pipeline.prepare_window("srv-demo", end)

    forecaster = Forecaster(model, normalizer)
    out = forecaster.predict(prepared)

    cpu_idx = METRIC_ORDER.index("cpu_util")
    print(f"\nNext-{HORIZON_MINUTES}-minute CPU forecast (original units):")
    for i, t in enumerate(out.forecast_timestamps):
        print(f"  +{i + 1:>2d} min  {t.strftime('%H:%M')}  "
              f"cpu={out.predicted_original[i, cpu_idx]:+.3f}")

    print("\nAttention weight summary over the 60-min input:")
    aw = out.attention_weights
    print(f"  min={aw.min():.4f}  max={aw.max():.4f}  "
          f"argmax={int(np.argmax(aw))} (most-attended past minute)")


if __name__ == "__main__":
    main()
