# Unit tests for moving-average smoothing.

from __future__ import annotations

import pandas as pd
import pytest

from pcg.core.constants import METRIC_ORDER
from pcg.data.smoothing import smooth


def _frame(length: int = 10, value: float = 1.0) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=length, freq="1min")
    return pd.DataFrame({m: [value] * length for m in METRIC_ORDER}, index=idx)


def test_constant_signal_is_unchanged() -> None:
    df = _frame(20, value=0.7)
    out = smooth(df)
    for m in METRIC_ORDER:
        assert out[m].to_numpy() == pytest.approx(0.7)


def test_single_spike_is_attenuated_by_window_5() -> None:
    df = _frame(10, value=0.0)
    df.iloc[5] = 1.0  # one-sample spike at index 5
    out = smooth(df, window=5)
    # Trailing window covers indices 1..5 -> mean = (0+0+0+0+1)/5 = 0.2.
    for m in METRIC_ORDER:
        assert out[m].iloc[5] == pytest.approx(0.2)


def test_min_periods_one_avoids_leading_nan() -> None:
    df = _frame(5, value=0.5)
    out = smooth(df, window=10)
    for m in METRIC_ORDER:
        assert not out[m].isna().any()


def test_input_is_not_mutated() -> None:
    df = _frame(10, value=0.5)
    snapshot = df.copy()
    smooth(df)
    pd.testing.assert_frame_equal(df, snapshot)


def test_invalid_window_raises() -> None:
    with pytest.raises(ValueError):
        smooth(_frame(5), window=0)


def test_no_metric_columns_returns_copy() -> None:
    df = pd.DataFrame({"unrelated": [1.0, 2.0, 3.0]},
                      index=pd.date_range("2026-01-01", periods=3, freq="1min"))
    out = smooth(df)
    pd.testing.assert_frame_equal(df, out)
    assert out is not df
