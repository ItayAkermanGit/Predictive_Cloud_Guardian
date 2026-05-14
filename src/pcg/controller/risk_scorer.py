# Combined risk scorer: merges forecasting risk and anomaly score into
# a single unified risk signal fed to the alert decision layer.
#
# Two independent signals are produced by different models:
#   1. Forecast risk  — how far is the predicted metric trajectory from
#      safe thresholds? Derived from LSTMForecaster output.
#   2. Anomaly score  — how far is the current window from the learned
#      normal manifold? Derived from ConvAutoencoder reconstruction error.
#
# Combined risk:
#   risk = w_forecast * forecast_risk + w_anomaly * anomaly_risk
#
# Both component risks are first normalized to [0, 1] before weighting:
#   forecast_risk  = clamp(max_metric_breach_margin / breach_scale, 0, 1)
#   anomaly_risk   = clamp(anomaly_index / n_sigma_clip, 0, 1)
#
# where:
#   breach_margin   = (predicted_value - safe_threshold) / safe_threshold
#                     for each metric; zero when below threshold.
#   anomaly_index   = (reconstruction_MSE - mu_normal) / sigma_normal
#
# The combined risk lives in [0, 1]. Values above risk_threshold trigger
# the alert decision layer.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..anomaly.anomaly_api import AnomalyReport
from ..core.constants import HORIZON_MINUTES, METRIC_ORDER, N_METRICS
from ..inference.forecaster import ForecastOutput


# Default safe-operation thresholds (fraction of metric scale, i.e. normalized).
# A forecast that stays below these values carries zero forecast risk.
DEFAULT_SAFE_THRESHOLDS: dict[str, float] = {
    "cpu_util": 0.80,
    "mem_util": 0.85,
    "net_io":   0.90,
    "disk_io":  0.90,
}


@dataclass
class RiskConfig:
    # Weights and thresholds for the combined risk scorer.

    # Blend weights — must sum to 1.0.
    w_forecast: float = 0.5
    w_anomaly: float = 0.5

    # Fraction of sigma above normal at which anomaly_risk is clamped to 1.
    # E.g. n_sigma_clip=6 means anomaly_index >= 6σ → risk=1.0.
    n_sigma_clip: float = 6.0

    # Scale for forecast breach margin normalization.
    # A breach of `breach_scale` above threshold maps to risk=1.0.
    breach_scale: float = 0.20

    # Per-metric safe thresholds (normalized [0,1] space).
    safe_thresholds: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SAFE_THRESHOLDS)
    )

    # Overall risk threshold above which an alert is warranted.
    risk_threshold: float = 0.5

    def __post_init__(self) -> None:
        if abs(self.w_forecast + self.w_anomaly - 1.0) > 1e-6:
            raise ValueError(
                f"w_forecast + w_anomaly must equal 1.0, "
                f"got {self.w_forecast} + {self.w_anomaly}"
            )
        if self.n_sigma_clip <= 0:
            raise ValueError("n_sigma_clip must be > 0")
        if self.breach_scale <= 0:
            raise ValueError("breach_scale must be > 0")


@dataclass(frozen=True)
class RiskAssessment:
    # Output of one combined risk evaluation.

    server_id: str

    # Combined weighted risk in [0, 1].
    combined_risk: float

    # Component risks, each in [0, 1].
    forecast_risk: float
    anomaly_risk: float

    # True when combined_risk >= risk_threshold.
    should_alert: bool

    # Which metrics are forecast to breach their safe thresholds.
    # metric -> expected breach margin (positive = above threshold).
    forecast_breaches: dict[str, float]

    # From AnomalyReport — which metric has highest reconstruction error.
    anomaly_root_cause: Optional[str]

    # Anomaly index (sigma above normal) before clamping.
    anomaly_index: float

    def summary(self) -> str:
        alert = "ALERT" if self.should_alert else "ok"
        breaches = ", ".join(
            f"{m}+{v:.2f}" for m, v in self.forecast_breaches.items()
        ) or "none"
        return (
            f"[{alert}] server={self.server_id} "
            f"risk={self.combined_risk:.3f} "
            f"(forecast={self.forecast_risk:.3f} anomaly={self.anomaly_risk:.3f}) "
            f"breaches={breaches} rca={self.anomaly_root_cause or 'n/a'}"
        )


class CombinedRiskScorer:
    # Merges ForecastOutput and AnomalyReport into a single RiskAssessment.

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        self.cfg = config or RiskConfig()

    def assess(
        self,
        forecast: ForecastOutput,
        anomaly: AnomalyReport,
    ) -> RiskAssessment:
        forecast_risk, breaches = self._forecast_risk(forecast)
        anomaly_risk = self._anomaly_risk(anomaly.anomaly_index)

        combined = (
            self.cfg.w_forecast * forecast_risk
            + self.cfg.w_anomaly * anomaly_risk
        )
        combined = float(np.clip(combined, 0.0, 1.0))

        return RiskAssessment(
            server_id=forecast.server_id,
            combined_risk=combined,
            forecast_risk=forecast_risk,
            anomaly_risk=anomaly_risk,
            should_alert=combined >= self.cfg.risk_threshold,
            forecast_breaches=breaches,
            anomaly_root_cause=anomaly.root_cause_metric,
            anomaly_index=anomaly.anomaly_index,
        )

    def _forecast_risk(
        self, forecast: ForecastOutput
    ) -> tuple[float, dict[str, float]]:
        # For each metric, compute the maximum breach margin over the horizon.
        # breach_margin[m] = max over t of:
        #   (predicted_normalized[t, m] - safe_threshold[m]) / safe_threshold[m]
        # Negative margins are zeroed — only above-threshold values carry risk.
        # forecast_risk = max breach margin across all metrics, clamped to [0, 1].
        breaches: dict[str, float] = {}
        metric_risks: list[float] = []

        for i, metric in enumerate(METRIC_ORDER[:N_METRICS]):
            threshold = self.cfg.safe_thresholds.get(metric, 0.85)
            predicted = forecast.predicted_normalized[:, i]  # (HORIZON,)

            # Margin: how far above the threshold does the forecast reach?
            margin = float(np.max(predicted) - threshold)
            if margin > 0:
                normalized_margin = margin / (threshold * self.cfg.breach_scale)
                breaches[metric] = round(margin, 4)
                metric_risks.append(float(np.clip(normalized_margin, 0.0, 1.0)))
            else:
                metric_risks.append(0.0)

        forecast_risk = float(max(metric_risks)) if metric_risks else 0.0
        return forecast_risk, breaches

    def _anomaly_risk(self, anomaly_index: float) -> float:
        # Map anomaly_index (sigma above normal) to [0, 1].
        # index <= 0  → risk = 0  (better than average normal)
        # index >= n_sigma_clip → risk = 1
        return float(np.clip(anomaly_index / self.cfg.n_sigma_clip, 0.0, 1.0))
