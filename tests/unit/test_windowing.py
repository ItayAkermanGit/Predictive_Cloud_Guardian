# Unit tests for the windowing & tensor preparation layer.

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from pcg.core.constants import METRIC_ORDER, N_METRICS, WINDOW_SIZE
from pcg.core.exceptions import InsufficientHistoryError, MetricSchemaError
from pcg.data.windowing import (
    build_inference_window,
    build_training_windows,
    to_feature_matrix,
    to_inference_tensor,
    to_training_tensor,
)


def _frame(length: int) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=length, freq="1min")
    return pd.DataFrame(
        {m: np.linspace(0.0, 1.0, length) for m in METRIC_ORDER}, index=idx,
    )


# to_feature_matrix

def test_feature_matrix_enforces_metric_order() -> None:
    df = _frame(5)
    df = df[list(reversed(METRIC_ORDER))]  # scramble columns
    matrix = to_feature_matrix(df)
    assert matrix.shape == (5, N_METRICS)
    # Column 0 must be METRIC_ORDER[0] regardless of input order.
    assert matrix[0, 0] == pytest.approx(0.0)
    assert matrix[-1, 0] == pytest.approx(1.0)


def test_feature_matrix_missing_column_raises() -> None:
    df = _frame(5).drop(columns=[METRIC_ORDER[0]])
    with pytest.raises(MetricSchemaError):
        to_feature_matrix(df)


# build_inference_window

def test_inference_window_takes_tail() -> None:
    df = _frame(100)
    window = build_inference_window(df)
    assert window.shape == (WINDOW_SIZE, N_METRICS)
    expected_first = df[METRIC_ORDER[0]].iloc[-WINDOW_SIZE]
    assert window[0, 0] == pytest.approx(expected_first)
    expected_last = df[METRIC_ORDER[0]].iloc[-1]
    assert window[-1, 0] == pytest.approx(expected_last)


def test_inference_window_too_short_raises() -> None:
    df = _frame(WINDOW_SIZE - 1)
    with pytest.raises(InsufficientHistoryError):
        build_inference_window(df)


def test_inference_window_exact_length_succeeds() -> None:
    df = _frame(WINDOW_SIZE)
    window = build_inference_window(df)
    assert window.shape == (WINDOW_SIZE, N_METRICS)


# build_training_windows

def test_training_windows_count_with_stride_one() -> None:
    df = _frame(WINDOW_SIZE + 5)
    windows = build_training_windows(df, stride=1)
    assert windows.shape == (6, WINDOW_SIZE, N_METRICS)


def test_training_windows_count_with_stride_two() -> None:
    df = _frame(WINDOW_SIZE + 6)
    windows = build_training_windows(df, stride=2)
    # (66 - 60) // 2 + 1 = 4
    assert windows.shape == (4, WINDOW_SIZE, N_METRICS)


def test_training_windows_empty_when_too_short() -> None:
    df = _frame(WINDOW_SIZE - 1)
    windows = build_training_windows(df)
    assert windows.shape == (0, WINDOW_SIZE, N_METRICS)


def test_training_windows_invalid_stride() -> None:
    with pytest.raises(ValueError):
        build_training_windows(_frame(WINDOW_SIZE + 5), stride=0)


# to_inference_tensor / to_training_tensor

def test_inference_tensor_shape_and_dtype() -> None:
    arr = np.zeros((WINDOW_SIZE, N_METRICS), dtype=np.float32)
    tensor = to_inference_tensor(arr)
    assert isinstance(tensor, torch.Tensor)
    assert tuple(tensor.shape) == (1, WINDOW_SIZE, N_METRICS)
    assert tensor.dtype == torch.float32


def test_inference_tensor_wrong_shape_raises() -> None:
    with pytest.raises(ValueError):
        to_inference_tensor(np.zeros((30, N_METRICS), dtype=np.float32))


def test_training_tensor_shape_and_dtype() -> None:
    arr = np.zeros((4, WINDOW_SIZE, N_METRICS), dtype=np.float32)
    tensor = to_training_tensor(arr)
    assert tuple(tensor.shape) == (4, WINDOW_SIZE, N_METRICS)
    assert tensor.dtype == torch.float32


def test_training_tensor_wrong_shape_raises() -> None:
    with pytest.raises(ValueError):
        to_training_tensor(np.zeros((4, 30, N_METRICS), dtype=np.float32))
    with pytest.raises(ValueError):
        to_training_tensor(np.zeros((WINDOW_SIZE, N_METRICS), dtype=np.float32))


def test_inference_tensor_preserves_values() -> None:
    arr = np.arange(WINDOW_SIZE * N_METRICS, dtype=np.float32).reshape(
        WINDOW_SIZE, N_METRICS
    )
    tensor = to_inference_tensor(arr)
    np.testing.assert_array_equal(tensor.squeeze(0).numpy(), arr)
