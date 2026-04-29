# Min-max normalization to the [0, 1] range.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import pandas as pd

from ..core.constants import METRIC_ORDER
from ..core.exceptions import MetricSchemaError, NormalizerNotFittedError


@dataclass
class MinMaxNormalizer:
    # Per-metric min-max scaler with JSON persistence.

    mins: dict[str, float] = field(default_factory=dict)
    maxs: dict[str, float] = field(default_factory=dict)
    _fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "MinMaxNormalizer":
        # Compute per-metric min and max from training data.
        self._require_columns(df)
        for metric in METRIC_ORDER:
            col = df[metric].dropna()
            if col.empty:
                raise ValueError(
                    f"cannot fit normalizer on empty column {metric!r}"
                )
            mn, mx = float(col.min()), float(col.max())
            if mx <= mn:
                # Constant column: widen by epsilon so transform doesn't divide by zero.
                mx = mn + 1e-6
            self.mins[metric] = mn
            self.maxs[metric] = mx
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Apply the fitted min/max scaling. Returns a new DataFrame.
        if not self._fitted:
            raise NormalizerNotFittedError(
                "MinMaxNormalizer.transform called before fit() — "
                "load saved parameters with .load(path) before serving."
            )
        self._require_columns(df)
        out = df.copy()
        for metric in METRIC_ORDER:
            mn = self.mins[metric]
            mx = self.maxs[metric]
            out[metric] = (out[metric] - mn) / (mx - mn)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, path: Union[str, Path]) -> None:
        # Save parameters as JSON: {"mins": {...}, "maxs": {...}}.
        payload = {"mins": self.mins, "maxs": self.maxs}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MinMaxNormalizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        instance = cls(mins=dict(payload["mins"]), maxs=dict(payload["maxs"]))
        # Validate that the loaded artifact matches the current metric set.
        missing = [m for m in METRIC_ORDER if m not in instance.mins]
        if missing:
            raise MetricSchemaError(
                f"normalizer artifact at {path} missing metrics {missing}"
            )
        instance._fitted = True
        return instance

    @staticmethod
    def _require_columns(df: pd.DataFrame) -> None:
        missing = [m for m in METRIC_ORDER if m not in df.columns]
        if missing:
            raise MetricSchemaError(
                f"dataframe missing required metric columns: {missing}"
            )
