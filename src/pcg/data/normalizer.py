"""Min-max normalization to [0, 1] (Problem 1).

Why [0, 1] specifically:
    1. LSTM training stability — large input magnitudes saturate the
       sigmoid/tanh gates inside the LSTM cell ("שערים אלו מאפשרים למודל
       לנהל את הזיכרון") and stall gradient descent.
    2. Autoencoder error interpretability — reconstruction MSE on a [0,1]
       scale gives errors directly comparable across CPU, memory, network,
       disk. This is what makes the per-metric error decomposition in
       root-cause analysis (Problem 6) meaningful.

Why we PERSIST the fitted ranges to disk:
    A model trained on min/max from January cannot be deployed in March
    using March's min/max — that would silently shift the input
    distribution. The min/max ARE part of the model artifact and must
    travel with the weights.

Why we DON'T clip out-of-range values at inference:
    A CPU reading of 110% (thermal throttle) at inference is genuinely
    new information. Clipping it back to 1.0 would HIDE that anomaly
    from the autoencoder, which is exactly the case we want to detect
    (Problem 2). Letting the value exceed 1.0 produces a high
    reconstruction error — the desired behaviour.
"""

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
    """Per-metric min-max scaler with JSON persistence.

    State:
        mins, maxs — per-metric scalars populated by ``fit`` and consumed
        by ``transform``. Saved/loaded via JSON so the file is human-
        readable, diff-friendly, and language-portable.
    """

    mins: dict[str, float] = field(default_factory=dict)
    maxs: dict[str, float] = field(default_factory=dict)
    _fitted: bool = False

    # ----- fitting ---------------------------------------------------- #

    def fit(self, df: pd.DataFrame) -> "MinMaxNormalizer":
        """Compute and store the per-metric min and max from training data."""
        self._require_columns(df)
        for metric in METRIC_ORDER:
            col = df[metric].dropna()
            if col.empty:
                raise ValueError(
                    f"cannot fit normalizer on empty column {metric!r}"
                )
            mn, mx = float(col.min()), float(col.max())
            if mx <= mn:
                # Degenerate (constant) column: widen by an epsilon so
                # later transforms don't divide by zero. The scaled output
                # for a constant-column training set will be ~0; that's
                # the correct behaviour — nothing varies, nothing learns.
                mx = mn + 1e-6
            self.mins[metric] = mn
            self.maxs[metric] = mx
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted min/max scaling. Returns a NEW DataFrame."""
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

    # ----- persistence ------------------------------------------------ #

    def save(self, path: Union[str, Path]) -> None:
        """Persist parameters as JSON. Schema: ``{"mins": {...}, "maxs": {...}}``."""
        payload = {"mins": self.mins, "maxs": self.maxs}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MinMaxNormalizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        instance = cls(mins=dict(payload["mins"]), maxs=dict(payload["maxs"]))
        # Validate that the loaded artifact matches the current metric set —
        # silently mismatching schemas is one of the worst MLOps bugs.
        missing = [m for m in METRIC_ORDER if m not in instance.mins]
        if missing:
            raise MetricSchemaError(
                f"normalizer artifact at {path} missing metrics {missing}"
            )
        instance._fitted = True
        return instance

    # ----- helpers ---------------------------------------------------- #

    @staticmethod
    def _require_columns(df: pd.DataFrame) -> None:
        missing = [m for m in METRIC_ORDER if m not in df.columns]
        if missing:
            raise MetricSchemaError(
                f"dataframe missing required metric columns: {missing}"
            )
