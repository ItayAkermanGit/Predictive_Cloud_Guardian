"""Time Series Database (TSDB) ingestion abstraction.

Why an abstract base class:
    The proposal commits to a TSDB as the source of truth (see project
    description: "צינור נתונים רציף המעבד נתונים מבסיס נתונים של סדרות זמן
    (TSDB)"). For the prototype we don't depend on a real engine — instead
    every consumer talks to a small `TSDBClient` interface, and we ship one
    in-memory implementation that the synthetic generator and tests use.
    Swapping to InfluxDB / Prometheus / VictoriaMetrics later is a single
    new subclass; nothing downstream changes.

Contract for all clients:
    `query(server_id, start, end)` → ``pandas.DataFrame`` with
        * a `pd.DatetimeIndex` (UTC, monotonic increasing)
        * one column per metric in `core.constants.METRIC_ORDER`
        * NaN entries are allowed; the interpolation stage will fix them.

`count_contiguous_samples` is consumed by the cold-start logic (Problem 8)
to decide whether to use the dedicated model or the generic fallback.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from ..core.constants import METRIC_ORDER, SAMPLE_INTERVAL_SECONDS


class TSDBClient(ABC):
    """Abstract read interface for a time-series database."""

    @abstractmethod
    def query(
        self,
        server_id: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Return rows for `server_id` whose timestamps lie in [start, end]."""

    @abstractmethod
    def count_contiguous_samples(self, server_id: str, end: datetime) -> int:
        """Return the count of consecutive valid samples ending at `end`.

        Used by the cold-start controller: a server with fewer than
        COLD_START_MIN_SAMPLES (=60) is routed to the generic model.
        """


class InMemoryTSDBClient(TSDBClient):
    """TSDB-shaped wrapper around an in-memory dict of DataFrames.

    Designed for unit tests, the synthetic-data demo, and offline training.
    Not concurrency-safe — call sites are expected to be single-threaded.

    Defense note: the real TSDB would partition by server_id internally; we
    reproduce that mental model here so tests written against this client
    map cleanly to behavior against a production backend.
    """

    def __init__(self, frames: Optional[dict[str, pd.DataFrame]] = None) -> None:
        # Defensive copy so mutations to the input dict don't leak in.
        self._frames: dict[str, pd.DataFrame] = (
            {k: v.copy() for k, v in frames.items()} if frames else {}
        )

    def upsert(self, server_id: str, frame: pd.DataFrame) -> None:
        """Insert or merge new rows for a server. Last-write-wins on duplicate
        timestamps — matches typical TSDB semantics for a re-ingested point."""
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError("frame must be indexed by pd.DatetimeIndex")
        existing = self._frames.get(server_id)
        if existing is None:
            self._frames[server_id] = frame.sort_index().copy()
            return
        merged = pd.concat([existing, frame])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        self._frames[server_id] = merged

    def query(
        self,
        server_id: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        if server_id not in self._frames:
            # Empty frame with the correct schema — keeps downstream callers
            # from special-casing "server unknown" vs. "server has no data".
            return pd.DataFrame(columns=list(METRIC_ORDER))
        df = self._frames[server_id]
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        return df.loc[mask].copy()

    def count_contiguous_samples(self, server_id: str, end: datetime) -> int:
        if server_id not in self._frames:
            return 0
        df = self._frames[server_id]
        if df.empty:
            return 0

        # We walk back from `end` and count rows whose timestamps form an
        # unbroken chain at the SAMPLE_INTERVAL_SECONDS cadence. A 1.5×
        # tolerance absorbs jitter from the upstream collector.
        tolerance = timedelta(seconds=SAMPLE_INTERVAL_SECONDS * 1.5)
        end_ts = pd.Timestamp(end)
        # iterate timestamps ≤ end, newest first
        timestamps = [t for t in df.index if t <= end_ts]
        timestamps.sort(reverse=True)
        if not timestamps:
            return 0

        count = 0
        cursor = timestamps[0]
        for ts in timestamps:
            if cursor - ts <= tolerance:
                count += 1
                cursor = ts
            else:
                break
        return count
