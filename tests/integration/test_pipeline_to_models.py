# End-to-end tests for the data pipeline.
# Exercises TSDB query -> interpolate -> smooth -> normalize -> window -> tensor.

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import torch

from pcg.core.constants import (
    METRIC_ORDER,
    N_METRICS,
    WINDOW_SIZE,
)
from pcg.core.exceptions import InsufficientHistoryError
from pcg.data.normalizer import MinMaxNormalizer
from pcg.data.pipeline import DataPipeline, PreparedWindow
from pcg.data.synthetic import (
    FailureInjection,
    SyntheticConfig,
    SyntheticMetricGenerator,
)
from pcg.data.tsdb_client import InMemoryTSDBClient


def _seeded_environment(
    server_id: str = "srv-1",
    minutes: int = 200,
    missing_rate: float = 0.0,
    failures: list[FailureInjection] | None = None,
) -> Tuple[InMemoryTSDBClient, MinMaxNormalizer, pd.DataFrame]:
    cfg = SyntheticConfig(
        seed=11,
        missing_rate=missing_rate,
        failures=failures or [],
    )
    df = SyntheticMetricGenerator(cfg).generate(datetime(2026, 1, 1, 8), minutes)

    client = InMemoryTSDBClient()
    client.upsert(server_id, df)

    # Fit on a NaN-free copy so an entirely-missing column doesn't crash fit.
    norm = MinMaxNormalizer().fit(df.dropna())
    return client, norm, df


def test_pipeline_produces_canonical_tensor_shape() -> None:
    client, norm, df = _seeded_environment(minutes=200)
    pipeline = DataPipeline(client, norm)

    end = df.index[-1].to_pydatetime()
    prepared = pipeline.prepare_window("srv-1", end)

    assert isinstance(prepared, PreparedWindow)
    assert prepared.tensor.shape == (1, WINDOW_SIZE, N_METRICS)
    assert prepared.tensor.dtype == torch.float32
    assert prepared.metric_order == METRIC_ORDER
    assert prepared.server_id == "srv-1"


def test_pipeline_output_is_finite_with_missing_data() -> None:
    # Even with ~10% missing samples, the tensor must be NaN-free.
    client, norm, df = _seeded_environment(minutes=200, missing_rate=0.10)
    pipeline = DataPipeline(client, norm)
    end = df.index[-1].to_pydatetime()
    prepared = pipeline.prepare_window("srv-1", end)
    assert torch.all(torch.isfinite(prepared.tensor))


def test_pipeline_output_is_roughly_in_unit_range() -> None:
    # After normalization, healthy synthetic data sits within [0, 1] modulo
    # a small smoothing tail.
    client, norm, df = _seeded_environment(minutes=200)
    pipeline = DataPipeline(client, norm)
    end = df.index[-1].to_pydatetime()
    prepared = pipeline.prepare_window("srv-1", end)
    arr = prepared.tensor.squeeze(0).numpy()
    assert arr.min() >= -0.5
    assert arr.max() <= 1.5


def test_pipeline_raises_for_unknown_server() -> None:
    _, norm, _ = _seeded_environment()
    empty_client = InMemoryTSDBClient()
    pipeline = DataPipeline(empty_client, norm)
    with pytest.raises(InsufficientHistoryError):
        pipeline.prepare_window("ghost", datetime(2026, 1, 1, 12))


def test_pipeline_raises_when_history_is_too_short() -> None:
    # Less than WINDOW_SIZE rows is exactly the cold-start case.
    client, norm, df = _seeded_environment(minutes=30)
    pipeline = DataPipeline(client, norm)
    end = df.index[-1].to_pydatetime()
    with pytest.raises(InsufficientHistoryError):
        pipeline.prepare_window("srv-1", end)


def test_pipeline_reflects_injected_failure_in_tensor() -> None:
    # The series covers 200 minutes; the inference window pulls rows 140..199.
    # Inject the failure at the END so it lives at the end of the window.
    failures = [
        FailureInjection(
            metric="cpu_util",
            start_offset_minutes=180,
            duration_minutes=20,
            magnitude=0.6,
        )
    ]
    client, norm, df = _seeded_environment(minutes=200, failures=failures)
    pipeline = DataPipeline(client, norm)
    end = df.index[-1].to_pydatetime()
    prepared = pipeline.prepare_window("srv-1", end)

    cpu_idx = METRIC_ORDER.index("cpu_util")
    cpu_window = prepared.tensor.squeeze(0).numpy()[:, cpu_idx]
    # Last 20 minutes are the failure; first 20 are healthy.
    assert cpu_window[-20:].mean() - cpu_window[:20].mean() > 0.3


def test_pipeline_produces_deterministic_tensor_for_same_inputs() -> None:
    client, norm, df = _seeded_environment()
    pipeline = DataPipeline(client, norm)
    end = df.index[-1].to_pydatetime()
    a = pipeline.prepare_window("srv-1", end).tensor
    b = pipeline.prepare_window("srv-1", end).tensor
    np.testing.assert_array_equal(a.numpy(), b.numpy())
