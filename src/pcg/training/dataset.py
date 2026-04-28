"""Pair builder for supervised forecasting training.

A forecasting dataset turns a single long time-series into many
(input, target) pairs of shape:

    input  : (WINDOW_SIZE=60, N_METRICS=4)
    target : (HORIZON_MINUTES=15, N_METRICS=4)

Generation logic
----------------
For each valid start index ``s``:
    input  = matrix[s          : s + 60]
    target = matrix[s + 60     : s + 60 + 15]

The number of pairs is ``(T - 60 - 15) // stride + 1`` for series length
``T``. With stride=1 we get every minute as a possible window start —
the densest usable signal.

Pre-processing parity with inference
------------------------------------
The dataset applies the exact same Phase-2 pipeline used at inference
(interpolate → smooth → normalizer.transform → matrix). That guarantees
a model trained here sees the same value distribution it will face in
production, eliminating one of the most common train/serve skew bugs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..core.constants import HORIZON_MINUTES, METRIC_ORDER, N_METRICS, WINDOW_SIZE
from ..data.interpolation import interpolate_missing
from ..data.normalizer import MinMaxNormalizer
from ..data.smoothing import smooth
from ..data.windowing import to_feature_matrix


class ForecastingDataset(Dataset):
    """Materializes (input_window, target_window) tensor pairs from a frame.

    Args:
        frame:       Raw DataFrame with at least the METRIC_ORDER columns
                     and a DatetimeIndex.
        normalizer:  A *fitted* MinMaxNormalizer (fit it on training data
                     once, then reuse for every dataset to keep distributions
                     consistent).
        window_size: Input length in samples. Defaults to WINDOW_SIZE (60).
        horizon:     Output length in samples. Defaults to HORIZON_MINUTES (15).
        stride:      Step between consecutive window starts.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        normalizer: MinMaxNormalizer,
        window_size: int = WINDOW_SIZE,
        horizon: int = HORIZON_MINUTES,
        stride: int = 1,
    ) -> None:
        if window_size <= 0 or horizon <= 0 or stride <= 0:
            raise ValueError("window_size, horizon, stride must be positive")

        # Apply the same preprocessing that inference uses, in the same
        # order. Skipping any of these steps here is an instant train/serve
        # skew bug.
        clean = interpolate_missing(frame)
        smoothed = smooth(clean)
        scaled = normalizer.transform(smoothed)

        matrix = to_feature_matrix(scaled)  # (T, N_METRICS)

        if matrix.shape[1] != N_METRICS:
            # Defensive — should never trip if METRIC_ORDER is honored.
            raise ValueError(
                f"feature matrix has {matrix.shape[1]} columns, expected {N_METRICS}"
            )

        self._matrix = matrix
        self._window_size = window_size
        self._horizon = horizon
        self._stride = stride

        usable = len(matrix) - window_size - horizon
        if usable < 0:
            raise ValueError(
                f"frame has {len(matrix)} samples; need at least "
                f"{window_size + horizon} for one (input, target) pair"
            )
        self._n_pairs = usable // stride + 1

    # --------------------------------------------------------------- #
    # PyTorch Dataset API
    # --------------------------------------------------------------- #

    def __len__(self) -> int:
        return self._n_pairs

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self._n_pairs:
            raise IndexError(f"index {idx} out of range [0, {self._n_pairs})")
        start = idx * self._stride
        x = self._matrix[start : start + self._window_size]
        y = self._matrix[
            start + self._window_size : start + self._window_size + self._horizon
        ]
        # Float32 to match the canonical inference tensor dtype.
        return (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).float(),
            torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).float(),
        )

    # --------------------------------------------------------------- #
    # Introspection helpers (used by tests, scripts, defense demos)
    # --------------------------------------------------------------- #

    @property
    def metric_order(self) -> tuple[str, ...]:
        """The column ordering encoded in every produced tensor."""
        return METRIC_ORDER

    @property
    def n_pairs(self) -> int:
        return self._n_pairs

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def horizon(self) -> int:
        return self._horizon
