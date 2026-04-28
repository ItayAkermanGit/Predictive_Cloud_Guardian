"""Synthetic metric generator for the prototype, demos, and tests.

Why we ship this:
    The proposal builds against TSDB metrics that we don't have access to
    in a college environment. A deterministic, well-shaped generator lets
    every Phase 2+ component run end-to-end (training, drift checks, alert
    storms) without external dependencies. The output schema is *exactly*
    what a real TSDB query returns — same DatetimeIndex, same columns —
    so swapping the source later changes nothing downstream.

The generator produces:
    * a configurable diurnal seasonal component (sin wave on a 1440-min
      period) — represents day/night load cycles mentioned in proposal
      Problem 1 ("עומס של 80% יכול להיות תקין בשעות שיא, אך קריטי בלילה"),
    * gaussian noise — represents the high-frequency jitter that the
      moving-average smoother (Problem 5) must tolerate,
    * optional missing samples — exercises the linear interpolation path
      (Problem 5),
    * optional failure injection — used by anomaly-detection tests to
      verify the autoencoder/forecaster react correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from ..core.constants import METRIC_ORDER, SAMPLE_INTERVAL_SECONDS


@dataclass
class FailureInjection:
    """Specification of a synthetic anomaly added on top of the baseline.

    Used by the anomaly-detection tests to verify that the autoencoder
    flags structurally abnormal windows (Problem 2) and that the forecaster
    catches threshold-crossing trends (Problem 1).
    """

    metric: str                   # must be one of METRIC_ORDER
    start_offset_minutes: int     # offset from the start of the generated series
    duration_minutes: int         # how long the anomaly lasts
    magnitude: float              # additive shift applied during the window


@dataclass
class SyntheticConfig:
    """Tuning knobs for the synthetic generator.

    `baselines` are the per-metric mean values around which the seasonal
    and noise components oscillate. They sit roughly in the middle of the
    [0,1] range produced after normalization, so generated data resembles
    realistic post-normalization tensors.
    """

    seed: int = 42
    diurnal_amplitude: float = 0.20
    noise_std: float = 0.03
    missing_rate: float = 0.0
    failures: list[FailureInjection] = field(default_factory=list)
    baselines: dict[str, float] = field(
        default_factory=lambda: {
            "cpu_util": 0.40,
            "mem_util": 0.55,
            "net_io": 0.30,
            "disk_io": 0.25,
        }
    )


class SyntheticMetricGenerator:
    """Generates a per-minute multivariate metric DataFrame.

    Defense note: the class holds the RNG as state so that two consecutive
    `generate(...)` calls draw distinct samples (mirroring real systems
    where each minute brings new noise). Determinism comes from the seed.
    """

    def __init__(self, config: Optional[SyntheticConfig] = None) -> None:
        self.config = config or SyntheticConfig()
        self._rng = np.random.default_rng(self.config.seed)

    def generate(self, start: datetime, minutes: int) -> pd.DataFrame:
        """Return `minutes` rows starting at `start` (1-minute spacing)."""
        if minutes <= 0:
            raise ValueError("minutes must be positive")

        index = pd.date_range(
            start=pd.Timestamp(start),
            periods=minutes,
            freq=f"{SAMPLE_INTERVAL_SECONDS}s",
        )

        # Diurnal phase walk: 1440 minutes per day → 2π full cycle.
        # Anchoring on (start hour + start minute) preserves phase across
        # consecutive `generate` calls so an "8 AM peak" looks the same
        # whether produced now or in tomorrow's call with the same seed.
        minute_of_day = (start.hour * 60 + start.minute + np.arange(minutes)) % 1440
        phase = 2.0 * np.pi * minute_of_day / 1440.0

        data: dict[str, np.ndarray] = {}
        for metric in METRIC_ORDER:
            baseline = self.config.baselines.get(metric, 0.5)
            seasonal = self.config.diurnal_amplitude * np.sin(phase)
            noise = self._rng.normal(0.0, self.config.noise_std, size=minutes)
            series = baseline + seasonal + noise

            # Apply any failure injections that target this metric.
            # We mutate `series` directly because each metric has its own copy.
            for failure in self.config.failures:
                if failure.metric != metric:
                    continue
                a = max(0, failure.start_offset_minutes)
                b = min(minutes, failure.start_offset_minutes + failure.duration_minutes)
                if a < b:
                    series[a:b] = series[a:b] + failure.magnitude

            data[metric] = series

        df = pd.DataFrame(data, index=index)
        df.index.name = "timestamp"

        # Optionally introduce missing samples. This deliberately exercises
        # `interpolation.interpolate_missing` — without holes in the data
        # the linear-interpolation path is never tested end-to-end.
        if self.config.missing_rate > 0.0:
            n_drop = int(minutes * self.config.missing_rate)
            if n_drop > 0:
                drop_idx = self._rng.choice(minutes, size=n_drop, replace=False)
                df.iloc[drop_idx] = np.nan

        return df
