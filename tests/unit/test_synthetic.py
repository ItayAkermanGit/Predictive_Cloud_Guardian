"""Unit tests for the synthetic metric generator."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from pcg.core.constants import METRIC_ORDER
from pcg.data.synthetic import (
    FailureInjection,
    SyntheticConfig,
    SyntheticMetricGenerator,
)


def test_generate_yields_correct_shape_and_columns() -> None:
    gen = SyntheticMetricGenerator()
    df = gen.generate(datetime(2026, 1, 1), 90)
    assert len(df) == 90
    for m in METRIC_ORDER:
        assert m in df.columns


def test_determinism_with_fixed_seed() -> None:
    a = SyntheticMetricGenerator(SyntheticConfig(seed=7)).generate(
        datetime(2026, 1, 1), 50
    )
    b = SyntheticMetricGenerator(SyntheticConfig(seed=7)).generate(
        datetime(2026, 1, 1), 50
    )
    np.testing.assert_array_equal(
        a.fillna(0).to_numpy(), b.fillna(0).to_numpy()
    )


def test_different_seeds_produce_different_streams() -> None:
    a = SyntheticMetricGenerator(SyntheticConfig(seed=1)).generate(
        datetime(2026, 1, 1), 50
    )
    b = SyntheticMetricGenerator(SyntheticConfig(seed=2)).generate(
        datetime(2026, 1, 1), 50
    )
    assert not np.array_equal(a.to_numpy(), b.to_numpy())


def test_missing_rate_introduces_nans() -> None:
    cfg = SyntheticConfig(seed=1, missing_rate=0.10)
    df = SyntheticMetricGenerator(cfg).generate(datetime(2026, 1, 1), 100)
    assert df.isna().any().any()


def test_no_missing_when_rate_zero() -> None:
    cfg = SyntheticConfig(seed=1, missing_rate=0.0)
    df = SyntheticMetricGenerator(cfg).generate(datetime(2026, 1, 1), 100)
    assert not df.isna().any().any()


def test_failure_injection_lifts_only_target_metric() -> None:
    cfg = SyntheticConfig(
        seed=1,
        failures=[
            FailureInjection(
                metric="cpu_util",
                start_offset_minutes=30,
                duration_minutes=10,
                magnitude=1.0,
            )
        ],
    )
    df = SyntheticMetricGenerator(cfg).generate(datetime(2026, 1, 1), 60)
    pre_cpu = df["cpu_util"].iloc[20:30].mean()
    during_cpu = df["cpu_util"].iloc[30:40].mean()
    assert during_cpu - pre_cpu > 0.5

    # Other metrics should NOT be affected.
    pre_mem = df["mem_util"].iloc[20:30].mean()
    during_mem = df["mem_util"].iloc[30:40].mean()
    assert abs(during_mem - pre_mem) < 0.2


def test_sample_interval_is_one_minute() -> None:
    df = SyntheticMetricGenerator().generate(datetime(2026, 1, 1), 5)
    diffs = df.index.to_series().diff().dropna()
    for delta in diffs:
        assert delta.total_seconds() == 60.0
