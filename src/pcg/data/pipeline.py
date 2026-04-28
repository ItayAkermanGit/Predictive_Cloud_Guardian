"""End-to-end data pipeline orchestration.

Composes the four pre-processing steps in the canonical order:

    1. interpolate  — fill missing samples (Problem 5)
    2. smooth       — 5-sample trailing mean to suppress noise (Problem 5)
    3. normalize    — min-max to [0, 1] using fitted parameters (Problem 1)
    4. window       — take the last 60 rows and shape them into a tensor

Order rationale (defense-relevant):
    * Interpolate first — every later step is undefined on NaN. A rolling
      mean of NaN is NaN; `(NaN - min) / (max - min)` is NaN.
    * Smooth before normalize — we want the [0,1] range to reflect the
      operational signal, not the high-frequency jitter that smoothing
      strips out.
    * Normalize last — the model expects [0,1]; do this immediately
      before tensor construction so we never pass un-scaled values into
      the windowing layer.

Defense note — symmetry between training and inference:
    The training script applies steps 1–3 to a long history then calls
    `build_training_windows` to generate (B, 60, N) batches. The inference
    path here applies the SAME 1–3 to a short tail and calls
    `build_inference_window` for a single (1, 60, N). Identical
    pre-processing is what keeps train/serve skew at zero.
"""

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
    """Container produced by ``DataPipeline.prepare_window``.

    `tensor` carries the canonical shape ``(1, WINDOW_SIZE, N_METRICS)``;
    callers use the surrounding fields (server_id, window_end, metric_order)
    to correlate predictions back to their source without inspecting
    tensor data.
    """

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
    """Glue between TSDB and the model layer.

    The pipeline is intentionally stateless apart from holding a TSDB
    client and a fitted normalizer. That makes it cheap to construct on
    every request and trivial to share between the inference API and the
    drift monitor (which both need the same pre-processing).
    """

    # We over-fetch a few extra minutes so the trailing moving-average
    # warm-up is primed on real data, not on shorter, noisier means.
    _EXTRA_LOOKBACK = SMOOTHING_WINDOW + 5

    def __init__(self, client: TSDBClient, normalizer: MinMaxNormalizer) -> None:
        self.client = client
        self.normalizer = normalizer

    def prepare_window(self, server_id: str, end: datetime) -> PreparedWindow:
        """Return a model-ready ``PreparedWindow`` for the given server/time.

        Raises ``InsufficientHistoryError`` if the TSDB does not yield
        WINDOW_SIZE rows after pre-processing — the caller (controller)
        translates that into a cold-start fallback.
        """
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

        # Defensive cleanup: a metric column that was entirely NaN at the
        # edges and couldn't be interpolated will still produce NaN here.
        # Drop those rows rather than feed NaN into the model.
        normalized = normalized.dropna(subset=list(METRIC_ORDER))

        window = build_inference_window(normalized)
        tensor = to_inference_tensor(window)

        return PreparedWindow(
            server_id=server_id,
            window_end=end,
            tensor=tensor,
            metric_order=METRIC_ORDER,
        )
