# Unit tests for ForecastingDataset.

from __future__ import annotations

from datetime import datetime

import pytest
import torch

from pcg.core.constants import HORIZON_MINUTES, METRIC_ORDER, N_METRICS, WINDOW_SIZE
from pcg.data.normalizer import MinMaxNormalizer
from pcg.data.synthetic import SyntheticConfig, SyntheticMetricGenerator
from pcg.training.dataset import ForecastingDataset


def _frame(minutes: int):
    return SyntheticMetricGenerator(SyntheticConfig(seed=3)).generate(
        datetime(2026, 1, 1), minutes
    )


def _normalizer(frame):
    return MinMaxNormalizer().fit(frame.dropna())


def test_dataset_pair_shapes() -> None:
    frame = _frame(WINDOW_SIZE + HORIZON_MINUTES + 5)
    norm = _normalizer(frame)
    ds = ForecastingDataset(frame, norm)
    x, y = ds[0]
    assert x.shape == (WINDOW_SIZE, N_METRICS)
    assert y.shape == (HORIZON_MINUTES, N_METRICS)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_dataset_n_pairs_with_stride_one() -> None:
    minutes = WINDOW_SIZE + HORIZON_MINUTES + 10
    frame = _frame(minutes)
    norm = _normalizer(frame)
    ds = ForecastingDataset(frame, norm, stride=1)
    # 10 + 1 = 11 valid (input, target) pairs
    assert len(ds) == 11


def test_dataset_too_short_raises() -> None:
    frame = _frame(WINDOW_SIZE + HORIZON_MINUTES - 1)
    norm = _normalizer(frame)
    with pytest.raises(ValueError):
        ForecastingDataset(frame, norm)


def test_dataset_index_out_of_range_raises() -> None:
    frame = _frame(WINDOW_SIZE + HORIZON_MINUTES + 5)
    ds = ForecastingDataset(frame, _normalizer(frame))
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_dataset_metric_order_property() -> None:
    frame = _frame(WINDOW_SIZE + HORIZON_MINUTES + 5)
    ds = ForecastingDataset(frame, _normalizer(frame))
    assert ds.metric_order == METRIC_ORDER


def test_dataset_consecutive_pairs_are_offset_by_stride() -> None:
    # Pair i+1 must start one sample after pair i when stride=1.
    frame = _frame(WINDOW_SIZE + HORIZON_MINUTES + 10)
    ds = ForecastingDataset(frame, _normalizer(frame), stride=1)
    x0, _ = ds[0]
    x1, _ = ds[1]
    # Last 59 rows of x0 must equal first 59 rows of x1.
    assert torch.allclose(x0[1:], x1[:-1])
