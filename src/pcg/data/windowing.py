# Sliding-window construction and tensor preparation.
# All model inputs follow shape (B, WINDOW_SIZE=60, N_METRICS=4).
# Column order on the last axis is METRIC_ORDER from core.constants.

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..core.constants import METRIC_ORDER, N_METRICS, WINDOW_SIZE
from ..core.exceptions import InsufficientHistoryError, MetricSchemaError


def to_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    # Convert a DataFrame to (n_samples, N_METRICS) array in METRIC_ORDER.
    # Reorders columns automatically; raises if a metric is missing.
    missing = [m for m in METRIC_ORDER if m not in df.columns]
    if missing:
        raise MetricSchemaError(f"missing metric columns: {missing}")
    return df.loc[:, list(METRIC_ORDER)].to_numpy(dtype=np.float32, copy=True)


def build_inference_window(df: pd.DataFrame) -> np.ndarray:
    # Take the last WINDOW_SIZE rows. Shape: (WINDOW_SIZE, N_METRICS).
    # Raises InsufficientHistoryError if there are not enough rows
    # (controller catches this and uses the cold-start model).
    if len(df) < WINDOW_SIZE:
        raise InsufficientHistoryError(
            f"need {WINDOW_SIZE} samples, got {len(df)}"
        )
    tail = df.iloc[-WINDOW_SIZE:]
    matrix = to_feature_matrix(tail)
    if matrix.shape != (WINDOW_SIZE, N_METRICS):
        raise ValueError(f"unexpected window shape {matrix.shape}")
    return matrix


def build_training_windows(df: pd.DataFrame, stride: int = 1) -> np.ndarray:
    # Slice a long series into overlapping windows.
    # Shape: (n_windows, WINDOW_SIZE, N_METRICS).
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if len(df) < WINDOW_SIZE:
        return np.empty((0, WINDOW_SIZE, N_METRICS), dtype=np.float32)
    matrix = to_feature_matrix(df)
    n_windows = 1 + (len(matrix) - WINDOW_SIZE) // stride
    out = np.empty((n_windows, WINDOW_SIZE, N_METRICS), dtype=np.float32)
    for i in range(n_windows):
        start = i * stride
        out[i] = matrix[start : start + WINDOW_SIZE]
    return out


def to_inference_tensor(window: np.ndarray) -> torch.Tensor:
    # (WINDOW_SIZE, N_METRICS) -> (1, WINDOW_SIZE, N_METRICS) float32 tensor.
    if window.shape != (WINDOW_SIZE, N_METRICS):
        raise ValueError(
            f"expected ({WINDOW_SIZE}, {N_METRICS}), got {window.shape}"
        )
    contiguous = np.ascontiguousarray(window, dtype=np.float32)
    return torch.from_numpy(contiguous).unsqueeze(0).float()


def to_training_tensor(windows: np.ndarray) -> torch.Tensor:
    # (B, WINDOW_SIZE, N_METRICS) numpy array -> torch tensor.
    if windows.ndim != 3 or windows.shape[1:] != (WINDOW_SIZE, N_METRICS):
        raise ValueError(
            f"expected (B, {WINDOW_SIZE}, {N_METRICS}), got {windows.shape}"
        )
    contiguous = np.ascontiguousarray(windows, dtype=np.float32)
    return torch.from_numpy(contiguous).float()
