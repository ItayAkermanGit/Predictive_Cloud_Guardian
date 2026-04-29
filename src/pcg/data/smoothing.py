# Trailing moving-average smoothing to suppress short noise spikes.

from __future__ import annotations

import pandas as pd

from ..core.constants import METRIC_ORDER, SMOOTHING_WINDOW


def smooth(df: pd.DataFrame, window: int = SMOOTHING_WINDOW) -> pd.DataFrame:
    # Apply a trailing moving average to all metric columns.
    if window < 1:
        raise ValueError(f"smoothing window must be >= 1, got {window}")
    out = df.copy()
    cols = [c for c in METRIC_ORDER if c in out.columns]
    if not cols:
        return out
    out[cols] = out[cols].rolling(window=window, min_periods=1).mean()
    return out
