# Online forecaster wrapper around a trained LSTMForecaster.
# Adds eval mode + no_grad, denormalizes the output to original units,
# and returns a small ForecastOutput dataclass.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import torch

from ..core.constants import (
    HORIZON_MINUTES,
    METRIC_ORDER,
    N_METRICS,
    SAMPLE_INTERVAL_SECONDS,
    WINDOW_SIZE,
)
from ..data.normalizer import MinMaxNormalizer
from ..data.pipeline import PreparedWindow
from ..models.lstm_attention import LSTMForecaster


@dataclass(frozen=True)
class ForecastOutput:
    # Result of a single forecast call.
    # predicted_normalized:  (HORIZON_MINUTES, N_METRICS) in [0, 1].
    # predicted_original:    (HORIZON_MINUTES, N_METRICS) in original units.
    # attention_weights:     (WINDOW_SIZE,) softmax weights over the past.
    # forecast_timestamps:   wall-clock timestamps for each forecast row.

    server_id: str
    window_end: datetime
    predicted_normalized: np.ndarray
    predicted_original: np.ndarray
    attention_weights: np.ndarray
    forecast_timestamps: list[datetime]
    metric_order: tuple[str, ...] = METRIC_ORDER

    def __post_init__(self) -> None:
        expected = (HORIZON_MINUTES, N_METRICS)
        if self.predicted_normalized.shape != expected:
            raise ValueError(
                f"predicted_normalized shape "
                f"{self.predicted_normalized.shape} != {expected}"
            )
        if self.predicted_original.shape != expected:
            raise ValueError(
                f"predicted_original shape "
                f"{self.predicted_original.shape} != {expected}"
            )
        if self.attention_weights.shape != (WINDOW_SIZE,):
            raise ValueError(
                f"attention_weights shape "
                f"{self.attention_weights.shape} != ({WINDOW_SIZE},)"
            )
        if len(self.forecast_timestamps) != HORIZON_MINUTES:
            raise ValueError(
                f"forecast_timestamps len {len(self.forecast_timestamps)} "
                f"!= {HORIZON_MINUTES}"
            )


class Forecaster:
    # Bind a trained model + its normalizer for online use.

    def __init__(
        self,
        model: LSTMForecaster,
        normalizer: MinMaxNormalizer,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device or "cpu")
        # eval() disables dropout/BN; no_grad on forward freezes parameters.
        self.model = model.to(self.device).eval()
        self.normalizer = normalizer

    @torch.no_grad()
    def predict(self, prepared: PreparedWindow) -> ForecastOutput:
        # Forecast the next HORIZON_MINUTES from a prepared window.
        x = prepared.tensor.to(self.device)
        forecast, attn = self.model(x)
        # forecast: (1, HORIZON_MINUTES, N_METRICS)
        # attn:     (1, WINDOW_SIZE)

        normalized = forecast.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        attention = attn.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        original = self._invert_normalizer(normalized)

        timestamps = [
            prepared.window_end + timedelta(seconds=SAMPLE_INTERVAL_SECONDS * (i + 1))
            for i in range(HORIZON_MINUTES)
        ]

        return ForecastOutput(
            server_id=prepared.server_id,
            window_end=prepared.window_end,
            predicted_normalized=normalized,
            predicted_original=original,
            attention_weights=attention,
            forecast_timestamps=timestamps,
        )

    def _invert_normalizer(self, normalized: np.ndarray) -> np.ndarray:
        # Map [0, 1] predictions back to original metric units.
        # Per metric column i: x * (max - min) + min.
        out = np.empty_like(normalized, dtype=np.float32)
        for i, metric in enumerate(METRIC_ORDER):
            mn = float(self.normalizer.mins[metric])
            mx = float(self.normalizer.maxs[metric])
            out[:, i] = normalized[:, i] * (mx - mn) + mn
        return out
