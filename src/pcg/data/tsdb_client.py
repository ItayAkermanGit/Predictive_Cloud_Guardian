# Read interface for the time series database.

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from ..core.constants import METRIC_ORDER, SAMPLE_INTERVAL_SECONDS


class TSDBClient(ABC):
    # Abstract TSDB client. All implementations return rows indexed by time.

    @abstractmethod
    def query(
        self,
        server_id: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        # Return rows for a server between start and end (inclusive).
        ...

    @abstractmethod
    def count_contiguous_samples(self, server_id: str, end: datetime) -> int:
        # Count consecutive valid samples ending at `end`. Used by cold start.
        ...


class InMemoryTSDBClient(TSDBClient):
    # In-memory client backed by a dict of DataFrames per server_id.

    def __init__(self, frames: Optional[dict[str, pd.DataFrame]] = None) -> None:
        # Defensive copy so external mutations don't leak in.
        self._frames: dict[str, pd.DataFrame] = (
            {k: v.copy() for k, v in frames.items()} if frames else {}
        )

    def upsert(self, server_id: str, frame: pd.DataFrame) -> None:
        # Insert or merge rows for a server. Last write wins on duplicates.
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

        # Walk back from `end` and count rows that are spaced one minute
        # apart (with a small tolerance for jitter).
        tolerance = timedelta(seconds=SAMPLE_INTERVAL_SECONDS * 1.5)
        end_ts = pd.Timestamp(end)
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
