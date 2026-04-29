# Linear interpolation for missing samples.

from __future__ import annotations

import pandas as pd

from ..core.constants import METRIC_ORDER


def interpolate_missing(df: pd.DataFrame) -> pd.DataFrame:
    # Fill NaN values in metric columns using linear interpolation.
    # Boundary NaNs (start/end of the series) are filled with the nearest
    # valid sample. Returns a copy, the input is not mutated.
    out = df.copy()
    cols = [c for c in METRIC_ORDER if c in out.columns]
    if not cols:
        return out

    method = "time" if isinstance(out.index, pd.DatetimeIndex) else "linear"
    out[cols] = out[cols].interpolate(method=method, limit_direction="both")

    # Safety net for runs of leading/trailing NaNs.
    out[cols] = out[cols].bfill().ffill()
    return out
