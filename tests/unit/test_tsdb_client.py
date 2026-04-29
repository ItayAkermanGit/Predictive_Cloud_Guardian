# Unit tests for the in-memory TSDB client.

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from pcg.core.constants import METRIC_ORDER
from pcg.data.synthetic import SyntheticConfig, SyntheticMetricGenerator
from pcg.data.tsdb_client import InMemoryTSDBClient


def _make_frame(start: datetime, minutes: int) -> pd.DataFrame:
    return SyntheticMetricGenerator(SyntheticConfig(seed=3)).generate(start, minutes)


def test_query_unknown_server_returns_empty_schema() -> None:
    client = InMemoryTSDBClient()
    out = client.query("nope", datetime(2026, 1, 1), datetime(2026, 1, 2))
    assert out.empty
    for m in METRIC_ORDER:
        assert m in out.columns


def test_upsert_then_query_window() -> None:
    client = InMemoryTSDBClient()
    start = datetime(2026, 1, 1, 10)
    df = _make_frame(start, 120)
    client.upsert("srv-1", df)

    qstart = start + timedelta(minutes=30)
    qend = start + timedelta(minutes=90)
    out = client.query("srv-1", qstart, qend)
    # 60-minute span at 1-min cadence inclusive on both ends -> 61 rows.
    assert len(out) == 61
    for m in METRIC_ORDER:
        assert m in out.columns


def test_upsert_merges_and_deduplicates() -> None:
    client = InMemoryTSDBClient()
    start = datetime(2026, 1, 1, 10)
    a = _make_frame(start, 60)
    b = _make_frame(start + timedelta(minutes=30), 60)  # 30 min overlap
    client.upsert("srv-1", a)
    client.upsert("srv-1", b)

    out = client.query("srv-1", start, start + timedelta(minutes=90))
    # A covers 0..59 (60 rows), B covers 30..89 (60 rows), overlap 30..59
    # -> merged distinct timestamps span 0..89 = 90 rows.
    assert len(out) == 90
    assert out.index.is_unique


def test_count_contiguous_samples_on_full_history() -> None:
    client = InMemoryTSDBClient()
    start = datetime(2026, 1, 1, 10)
    df = _make_frame(start, 75)
    client.upsert("srv-1", df)
    n = client.count_contiguous_samples("srv-1", start + timedelta(minutes=74))
    assert n == 75


def test_count_contiguous_samples_on_unknown_server() -> None:
    client = InMemoryTSDBClient()
    n = client.count_contiguous_samples("nope", datetime(2026, 1, 1))
    assert n == 0


def test_count_contiguous_samples_breaks_on_gap() -> None:
    client = InMemoryTSDBClient()
    start = datetime(2026, 1, 1, 10)
    a = _make_frame(start, 30)
    # Big gap (5 hours), then more rows.
    b = _make_frame(start + timedelta(hours=5), 30)
    client.upsert("srv-1", a)
    client.upsert("srv-1", b)

    end = start + timedelta(hours=5, minutes=29)
    n = client.count_contiguous_samples("srv-1", end)
    # Walking back from `end`, only the 30 contiguous rows in `b` count.
    assert n == 30


def test_upsert_rejects_non_datetime_index() -> None:
    client = InMemoryTSDBClient()
    df = pd.DataFrame({m: [0.0] for m in METRIC_ORDER}, index=[0])
    with pytest.raises(TypeError):
        client.upsert("srv-1", df)
