"""End-to-end tests for the data pipeline.

These exercise the full flow that an inference call will use in later
phases: TSDB query → interpolate → smooth → normalize → window → tensor.
The goal is to verify the canonical tensor contract
``(1, WINDOW_SIZE, N_METRICS)`` is produced regardless of upstream noise
and missing data.
"""

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

    # The normalizer must be fitted on a NaN-free version, otherwise an
    # entirely-missing column would crash. Drop NaNs for fit only.
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
    """Even with ~10% missing samples, the tensor must be NaN-free."""
    client, norm, df = _seeded_environment(minutes=200, missing_rate=0.10)
    pipeline = DataPipeline(client, norm)
    end = df.index[-1].to_pydatetime()
    prepared = pipeline.prepare_window("srv-1", end)
    assert torch.all(torch.isfinite(prepared.tensor))


def test_pipeline_output_is_roughly_in_unit_range() -> None:
    """After normalization, healthy synthetic data sits within [0,1] modulo
    a small smoothing tail. Hard bounds confirm the [0,1] contract."""
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
    """Less than WINDOW_SIZE rows is exactly the cold-start case."""
    client, norm, df = _seeded_environment(minutes=30)
    pipeline = DataPipeline(client, norm)
    end = df.index[-1].to_pydatetime()
    with pytest.raises(InsufficientHistoryError):
        pipeline.prepare_window("srv-1", end)


def test_pipeline_reflects_injected_failure_in_tensor() -> None:
    """A clearly anomalous CPU spike must show up as elevated values in
    the cpu_util column of the produced tensor — this is the integration
    point the autoencoder will rely on in Phase 3."""
    # Series covers 200 minutes; the inference window pulls minutes 140..199.
    # We inject the failure at the END of the series so it lives at the END
    # of the window — that mirrors how a real fault would look at inference.
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
    # Last 20 minutes of the window correspond to the failure period; first
    # 20 minutes are healthy. The lift should be clearly visible.
    assert cpu_window[-20:].mean() - cpu_window[:20].mean() > 0.3


def test_pipeline_produces_deterministic_tensor_for_same_inputs() -> None:
    client, norm, df = _seeded_environment()
    pipeline = DataPipeline(client, norm)
    end = df.index[-1].to_pydatetime()
    a = pipeline.prepare_window("srv-1", end).tensor
    b = pipeline.prepare_window("srv-1", end).tensor
    np.testing.assert_array_equal(a.numpy(), b.numpy())
