"""Linear interpolation for missing samples (Problem 5).

Why linear interpolation, not forward-fill or constant imputation:
    The proposal explicitly mandates interpolation that preserves the
    local trend ("שמירה על הרצף והמגמה ההגיונית של הנתונים, ובכך נמנעת
    מחיקה או השלמה בערך קבוע שאינו מייצג את המציאות"). Forward-fill turns
    a climbing CPU curve into a step plateau; constant imputation would
    fabricate a flat value at zero or whatever the previous reading was.
    Linear interpolation respects the slope across the gap.

Boundary edge case:
    Linear interpolation needs anchors on both sides. If the FIRST or LAST
    rows of the queried window are NaN, there is no left/right anchor.
    We fall back to the nearest valid sample (bfill / ffill) for those
    rows because pretending we have a slope across an unknown endpoint is
    worse than acknowledging the boundary. In practice this is rare —
    the interior of a 60-minute window almost always has anchors.
"""

from __future__ import annotations

import pandas as pd

from ..core.constants import METRIC_ORDER


def interpolate_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaNs in metric columns using time-aware linear interpolation.

    - Interior NaNs: linearly interpolated using the DatetimeIndex distance
      so an unevenly-sampled gap (e.g. 1 missed minute vs 5) is weighted
      correctly.
    - Boundary NaNs: filled with the nearest valid sample.
    - Columns outside METRIC_ORDER are passed through unchanged.
    - The input DataFrame is NOT mutated; a copy is always returned.
    """
    out = df.copy()
    cols = [c for c in METRIC_ORDER if c in out.columns]
    if not cols:
        return out

    # Use time-aware interpolation when we have a DatetimeIndex (the normal
    # case). Falls back to position-based linear interpolation if not, which
    # also keeps the smoothing/normalization unit tests happy.
    method = "time" if isinstance(out.index, pd.DatetimeIndex) else "linear"
    out[cols] = out[cols].interpolate(method=method, limit_direction="both")

    # Defensive bfill/ffill: a run of leading or trailing NaNs can survive
    # `interpolate(limit_direction="both")` in some pandas versions. We make
    # sure no NaN reaches the model by chaining bfill then ffill.
    out[cols] = out[cols].bfill().ffill()
    return out
