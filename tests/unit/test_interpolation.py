# Unit tests for linear interpolation.

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pcg.core.constants import METRIC_ORDER
from pcg.data.interpolation import interpolate_missing


def _frame(values_per_metric: dict[str, list[float]]) -> pd.DataFrame:
    n = len(next(iter(values_per_metric.values())))
    idx = pd.date_range("2026-01-01", periods=n, freq="1min")
    return pd.DataFrame(values_per_metric, index=idx)


def test_interior_nan_linearly_interpolated() -> None:
    data = {m: [0.0, np.nan, 1.0] for m in METRIC_ORDER}
    out = interpolate_missing(_frame(data))
    for m in METRIC_ORDER:
        assert out[m].iloc[1] == pytest.approx(0.5)


def test_multi_step_gap_interpolated_proportionally() -> None:
    # Three NaNs between 0 and 1 -> 0.25, 0.5, 0.75.
    data = {m: [0.0, np.nan, np.nan, np.nan, 1.0] for m in METRIC_ORDER}
    out = interpolate_missing(_frame(data))
    for m in METRIC_ORDER:
        assert out[m].iloc[1] == pytest.approx(0.25)
        assert out[m].iloc[2] == pytest.approx(0.50)
        assert out[m].iloc[3] == pytest.approx(0.75)


def test_leading_nan_filled_from_first_valid() -> None:
    data = {m: [np.nan, np.nan, 0.7, 0.8] for m in METRIC_ORDER}
    out = interpolate_missing(_frame(data))
    for m in METRIC_ORDER:
        assert not out[m].isna().any()
        assert out[m].iloc[0] == pytest.approx(0.7)
        assert out[m].iloc[1] == pytest.approx(0.7)


def test_trailing_nan_filled_from_last_valid() -> None:
    data = {m: [0.3, 0.4, np.nan, np.nan] for m in METRIC_ORDER}
    out = interpolate_missing(_frame(data))
    for m in METRIC_ORDER:
        assert not out[m].isna().any()
        assert out[m].iloc[-1] == pytest.approx(0.4)


def test_input_is_not_mutated() -> None:
    data = {m: [0.0, np.nan, 1.0] for m in METRIC_ORDER}
    df = _frame(data)
    snapshot = df.copy()
    interpolate_missing(df)
    pd.testing.assert_frame_equal(df, snapshot)


def test_extra_columns_preserved_unchanged() -> None:
    data = {m: [0.0, 0.5, 1.0] for m in METRIC_ORDER}
    df = _frame(data)
    df["host_label"] = ["a", "b", "c"]
    out = interpolate_missing(df)
    assert list(out["host_label"]) == ["a", "b", "c"]


def test_no_metric_columns_returns_copy() -> None:
    df = pd.DataFrame({"unrelated": [1, 2, 3]},
                      index=pd.date_range("2026-01-01", periods=3, freq="1min"))
    out = interpolate_missing(df)
    pd.testing.assert_frame_equal(df, out)
    assert out is not df
