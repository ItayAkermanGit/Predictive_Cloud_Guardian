# Builds (input_window, target_window) tensor pairs for the forecaster.
#
#   input  : (WINDOW_SIZE=60, N_METRICS=4)
#   target : (HORIZON_MINUTES=15, N_METRICS=4)
#
# For each start index s:
#   input  = matrix[s : s + 60]
#   target = matrix[s + 60 : s + 60 + 15]
#
# We apply the same preprocessing the inference path uses
# (interpolate -> smooth -> normalize) so the training distribution
# matches what the model will see in production.

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
    # Materializes (input_window, target_window) pairs from a DataFrame.

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

        # Same preprocessing the inference path uses, in the same order.
        clean = interpolate_missing(frame)
        smoothed = smooth(clean)
        scaled = normalizer.transform(smoothed)

        matrix = to_feature_matrix(scaled)  # (T, N_METRICS)

        if matrix.shape[1] != N_METRICS:
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
        return (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).float(),
            torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).float(),
        )

    @property
    def metric_order(self) -> tuple[str, ...]:
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
