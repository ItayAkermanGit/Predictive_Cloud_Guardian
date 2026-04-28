"""Sliding-window construction and tensor preparation (Problems 1, 7).

Canonical tensor contract for ALL model heads in PCG:

    inference  →  torch.float32  shape (1, WINDOW_SIZE=60, N_METRICS=4)
    training   →  torch.float32  shape (B, WINDOW_SIZE=60, N_METRICS=4)

The column ordering on axis=2 is `core.constants.METRIC_ORDER`. Defending
this single, central definition is the entire reason this module exists:
when the LSTM forecaster expects column 0 to be `cpu_util`, every code
path producing tensors must agree, and "every code path" means *this
module only*.

Defense note — multivariate vs. univariate:
    Problem 7 in the proposal argues that single-metric monitoring is
    dangerous (a CPU spike alone is ambiguous; a CPU spike WITHOUT a
    matching network spike is the actionable correlation break). That is
    why the tensor's last axis carries all metrics simultaneously instead
    of being a univariate scalar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..core.constants import METRIC_ORDER, N_METRICS, WINDOW_SIZE
from ..core.exceptions import InsufficientHistoryError, MetricSchemaError


# --------------------------------------------------------------------- #
# Multivariate feature-vector construction (Problem 7)
# --------------------------------------------------------------------- #

def to_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Convert a DataFrame to an ``(n_samples, N_METRICS)`` array in METRIC_ORDER.

    Always re-orders columns according to METRIC_ORDER, so callers may pass
    DataFrames with any column ordering without changing tensor semantics.
    Raises ``MetricSchemaError`` if any required metric is missing.
    """
    missing = [m for m in METRIC_ORDER if m not in df.columns]
    if missing:
        raise MetricSchemaError(f"missing metric columns: {missing}")
    return df.loc[:, list(METRIC_ORDER)].to_numpy(dtype=np.float32, copy=True)


# --------------------------------------------------------------------- #
# Sliding window construction
# --------------------------------------------------------------------- #

def build_inference_window(df: pd.DataFrame) -> np.ndarray:
    """Take the most-recent ``WINDOW_SIZE`` rows.

    Output shape: ``(WINDOW_SIZE, N_METRICS)``.
    Raises ``InsufficientHistoryError`` if fewer than WINDOW_SIZE rows are
    available — the controller catches this to engage the cold-start
    generic model (Problem 8).
    """
    if len(df) < WINDOW_SIZE:
        raise InsufficientHistoryError(
            f"need {WINDOW_SIZE} samples, got {len(df)}"
        )
    tail = df.iloc[-WINDOW_SIZE:]
    matrix = to_feature_matrix(tail)
    if matrix.shape != (WINDOW_SIZE, N_METRICS):
        # Belt-and-braces: would only fire if to_feature_matrix() broke contract.
        raise ValueError(f"unexpected window shape {matrix.shape}")
    return matrix


def build_training_windows(df: pd.DataFrame, stride: int = 1) -> np.ndarray:
    """Slice a long series into overlapping windows for training.

    Output shape: ``(n_windows, WINDOW_SIZE, N_METRICS)``.
    With ``stride=1`` every consecutive minute starts a new window — this
    is the densest signal possible, which we want for a college-scale
    dataset where samples are precious.
    Returns an empty 3-D array if the input is too short to form even
    one window.
    """
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


# --------------------------------------------------------------------- #
# Tensor preparation
# --------------------------------------------------------------------- #

def to_inference_tensor(window: np.ndarray) -> torch.Tensor:
    """Convert a single window to the model-input tensor.

    ``(WINDOW_SIZE, N_METRICS)`` → ``(1, WINDOW_SIZE, N_METRICS)`` torch.float32.

    Adding the batch axis here (and NOT inside each model) keeps a single
    source of truth: every model in PCG starts with `B = x.size(0)`.
    """
    if window.shape != (WINDOW_SIZE, N_METRICS):
        raise ValueError(
            f"expected ({WINDOW_SIZE}, {N_METRICS}), got {window.shape}"
        )
    # `np.ascontiguousarray` guarantees a positive-stride buffer that
    # `torch.from_numpy` can wrap without copy. `.float()` enforces dtype.
    contiguous = np.ascontiguousarray(window, dtype=np.float32)
    return torch.from_numpy(contiguous).unsqueeze(0).float()


def to_training_tensor(windows: np.ndarray) -> torch.Tensor:
    """Convert a stack of windows ``(B, WINDOW_SIZE, N_METRICS)`` to a tensor.

    The dtype is forced to float32 to match `to_inference_tensor`; mixed
    fp16/fp32 tensors are a frequent source of "cuDNN error" surprises
    when the same model serves both training and inference paths.
    """
    if windows.ndim != 3 or windows.shape[1:] != (WINDOW_SIZE, N_METRICS):
        raise ValueError(
            f"expected (B, {WINDOW_SIZE}, {N_METRICS}), got {windows.shape}"
        )
    contiguous = np.ascontiguousarray(windows, dtype=np.float32)
    return torch.from_numpy(contiguous).float()
