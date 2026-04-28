"""Moving-average smoothing to suppress short-term noise (Problem 5).

Why a 5-sample trailing mean:
    The proposal calls out "תנודתיות רגעית גבוהה (רעש)" — short-lived
    spikes from background processes (GC pauses, log rotation, brief
    network bursts) that should NOT trigger alerts. A 5-minute trailing
    mean is the proposal's chosen size; it removes sub-5-minute noise
    while preserving the multi-minute operational trends the LSTM and
    autoencoder care about.

Why TRAILING (causal) and not centered:
    At inference time we only have past samples. A centered window would
    require future values, leaking information that doesn't exist in
    production. Using `rolling(window).mean()` (which is left-anchored
    in pandas) keeps the training and inference paths identical.

Why min_periods=1:
    The first WINDOW-1 rows have fewer than `window` samples behind them.
    With `min_periods=1` we still emit a smoothed value (computed over
    however many samples are available) instead of leaking NaNs into the
    rest of the pipeline. The cost is a slightly noisier smoothed value
    at the very start of the series; this is acceptable because the
    pipeline always over-fetches by a few extra minutes (see pipeline.py)
    so the meaningful 60-minute window starts after the warm-up.
"""

from __future__ import annotations

import pandas as pd

from ..core.constants import METRIC_ORDER, SMOOTHING_WINDOW


def smooth(df: pd.DataFrame, window: int = SMOOTHING_WINDOW) -> pd.DataFrame:
    """Apply a trailing moving-average to all METRIC_ORDER columns.

    The original DataFrame is NOT mutated. Non-metric columns are passed
    through unchanged.
    """
    if window < 1:
        raise ValueError(f"smoothing window must be >= 1, got {window}")
    out = df.copy()
    cols = [c for c in METRIC_ORDER if c in out.columns]
    if not cols:
        return out
    out[cols] = out[cols].rolling(window=window, min_periods=1).mean()
    return out
