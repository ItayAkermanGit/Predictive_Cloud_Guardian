"""Online forecaster — runs a trained ``LSTMForecaster`` on prepared windows.

What this layer adds over calling ``model(...)`` directly
--------------------------------------------------------
1. Switches the module into eval mode and disables grad — the proposal's
   per-minute latency budget can't afford autograd bookkeeping.
2. Inverts the min-max normalization so callers see the prediction in
   the SAME units they originally fed in (e.g. CPU 87.3% rather than
   a [0,1] float). This is what makes the alert payload human-readable.
3. Wraps everything in a small, immutable ``ForecastOutput`` schema so
   downstream code (the Phase-4 controller) doesn't have to remember
   numpy axis orderings.
"""

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


# --------------------------------------------------------------------- #
# Output schema
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class ForecastOutput:
    """Result of a single forecast call.

    Attributes:
        server_id:           Echoed from the input PreparedWindow.
        window_end:          Wall-clock end of the input window.
        predicted_normalized:
                             Shape ``(HORIZON_MINUTES, N_METRICS)``, in [0,1].
                             Used by anomaly/threshold logic that operates
                             on normalized units.
        predicted_original:
                             Shape ``(HORIZON_MINUTES, N_METRICS)``, in the
                             original metric units. Used for human-readable
                             alerts.
        attention_weights:
                             Shape ``(WINDOW_SIZE,)``. Useful for the
                             defense demo: shows which past minutes the
                             model "looked at" most.
        forecast_timestamps:
                             Wall-clock timestamps of each predicted step,
                             aligned with ``predicted_*`` rows.
        metric_order:
                             Column ordering on axis 1 of the prediction
                             arrays. Always ``METRIC_ORDER``.
    """

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


# --------------------------------------------------------------------- #
# Inference wrapper
# --------------------------------------------------------------------- #

class Forecaster:
    """Bind a trained model + its normalizer for online use.

    The class is deliberately stateless beyond holding the model and the
    normalizer — it is safe to share a single Forecaster instance across
    threads as long as the underlying PyTorch model is not being trained
    elsewhere (no module-state mutation happens here).
    """

    def __init__(
        self,
        model: LSTMForecaster,
        normalizer: MinMaxNormalizer,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device or "cpu")
        # eval() disables dropout/BN updates; we keep parameters frozen
        # by wrapping forward() in torch.no_grad().
        self.model = model.to(self.device).eval()
        self.normalizer = normalizer

    @torch.no_grad()
    def predict(self, prepared: PreparedWindow) -> ForecastOutput:
        """Forecast the next ``HORIZON_MINUTES`` from a prepared window."""
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

    # ------------------------------------------------------------- #
    # helpers
    # ------------------------------------------------------------- #

    def _invert_normalizer(self, normalized: np.ndarray) -> np.ndarray:
        """Map [0,1] predictions back to original metric units.

        For each metric column ``i``, we apply ``x * (max - min) + min``
        with the per-metric ``min``/``max`` stored on the normalizer
        artifact. Predictions outside [0,1] are kept (they signal an
        out-of-distribution forecast and we don't want to hide it).
        """
        out = np.empty_like(normalized, dtype=np.float32)
        for i, metric in enumerate(METRIC_ORDER):
            mn = float(self.normalizer.mins[metric])
            mx = float(self.normalizer.maxs[metric])
            out[:, i] = normalized[:, i] * (mx - mn) + mn
        return out
