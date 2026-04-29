# End-to-end data pipeline.
# Order: query TSDB -> interpolate -> smooth -> normalize -> window -> tensor.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import torch

from ..core.constants import (
    METRIC_ORDER,
    N_METRICS,
    SAMPLE_INTERVAL_SECONDS,
    SMOOTHING_WINDOW,
    WINDOW_SIZE,
)
from ..core.exceptions import InsufficientHistoryError
from .interpolation import interpolate_missing
from .normalizer import MinMaxNormalizer
from .smoothing import smooth
from .tsdb_client import TSDBClient
from .windowing import build_inference_window, to_inference_tensor


@dataclass
class PreparedWindow:
    # Container produced by DataPipeline.prepare_window().
    # tensor.shape == (1, WINDOW_SIZE, N_METRICS).

    server_id: str
    window_end: datetime
    tensor: torch.Tensor
    metric_order: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = (1, WINDOW_SIZE, N_METRICS)
        if tuple(self.tensor.shape) != expected:
            raise ValueError(
                f"tensor shape {tuple(self.tensor.shape)} != {expected}"
            )


class DataPipeline:
    # Glue between the TSDB and the model layer.
    # We over-fetch a few extra minutes so the moving-average warm-up
    # is primed on real data.
    _EXTRA_LOOKBACK = SMOOTHING_WINDOW + 5

    def __init__(self, client: TSDBClient, normalizer: MinMaxNormalizer) -> None:
        self.client = client
        self.normalizer = normalizer

    def prepare_window(self, server_id: str, end: datetime) -> PreparedWindow:
        # Return a model-ready PreparedWindow for the given server/time.
        # Raises InsufficientHistoryError if there are not enough rows.
        lookback_minutes = WINDOW_SIZE + self._EXTRA_LOOKBACK
        start = end - timedelta(seconds=SAMPLE_INTERVAL_SECONDS * lookback_minutes)

        raw = self.client.query(server_id, start, end)
        if raw.empty:
            raise InsufficientHistoryError(
                f"TSDB returned 0 rows for server {server_id!r} between "
                f"{start.isoformat()} and {end.isoformat()}"
            )

        clean = interpolate_missing(raw)
        smoothed = smooth(clean)
        normalized = self.normalizer.transform(smoothed)

        # Defensive cleanup: drop rows that still contain NaN.
        normalized = normalized.dropna(subset=list(METRIC_ORDER))

        window = build_inference_window(normalized)
        tensor = to_inference_tensor(window)

        return PreparedWindow(
            server_id=server_id,
            window_end=end,
            tensor=tensor,
            metric_order=METRIC_ORDER,
        )
