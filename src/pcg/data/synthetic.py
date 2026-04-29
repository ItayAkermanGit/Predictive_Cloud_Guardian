# Synthetic per-minute metric generator used for tests and demos.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from ..core.constants import METRIC_ORDER, SAMPLE_INTERVAL_SECONDS


@dataclass
class FailureInjection:
    # Synthetic anomaly added on top of the baseline series.

    metric: str                   # must be in METRIC_ORDER
    start_offset_minutes: int     # offset from the start of the series
    duration_minutes: int
    magnitude: float              # additive shift during the failure


@dataclass
class SyntheticConfig:
    # Knobs for the generator.

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
    # Generates a multivariate metric DataFrame with one row per minute.

    def __init__(self, config: Optional[SyntheticConfig] = None) -> None:
        self.config = config or SyntheticConfig()
        self._rng = np.random.default_rng(self.config.seed)

    def generate(self, start: datetime, minutes: int) -> pd.DataFrame:
        # Return `minutes` rows starting at `start` (1-minute spacing).
        if minutes <= 0:
            raise ValueError("minutes must be positive")

        index = pd.date_range(
            start=pd.Timestamp(start),
            periods=minutes,
            freq=f"{SAMPLE_INTERVAL_SECONDS}s",
        )

        # Diurnal phase (1440 minutes per day).
        minute_of_day = (start.hour * 60 + start.minute + np.arange(minutes)) % 1440
        phase = 2.0 * np.pi * minute_of_day / 1440.0

        data: dict[str, np.ndarray] = {}
        for metric in METRIC_ORDER:
            baseline = self.config.baselines.get(metric, 0.5)
            seasonal = self.config.diurnal_amplitude * np.sin(phase)
            noise = self._rng.normal(0.0, self.config.noise_std, size=minutes)
            series = baseline + seasonal + noise

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

        # Optionally drop random samples to test interpolation.
        if self.config.missing_rate > 0.0:
            n_drop = int(minutes * self.config.missing_rate)
            if n_drop > 0:
                drop_idx = self._rng.choice(minutes, size=n_drop, replace=False)
                df.iloc[drop_idx] = np.nan

        return df
