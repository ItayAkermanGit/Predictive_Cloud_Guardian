# Model drift monitor.
#
# "Drift" here means the live scoring distribution has shifted away from the
# distribution observed during model calibration. Two forms are tracked:
#
# 1. Score drift (data drift proxy):
#    The rolling mean and std of combined_risk scores are compared to the
#    baseline mean/std captured at calibration time (fit()). If the running
#    mean deviates by more than `mean_drift_threshold` sigma from baseline,
#    drift is flagged.
#
# 2. Alert rate drift (concept drift proxy):
#    The rolling fraction of assessments that triggered an alert is compared
#    to the baseline alert rate. If the live rate exceeds `rate_multiplier`
#    times the baseline rate, it suggests the model is now over-alerting —
#    either because real incidents spiked or because the model no longer fits
#    the current normal behavior (concept drift).
#
# Both checks use an exponential moving average (EMA) so recent observations
# carry more weight without requiring a fixed-size buffer.
#
# EMA update:  ema_new = alpha * x + (1 - alpha) * ema_old
# alpha = 2 / (window + 1)  — standard EMA formula for window-size decay.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class DriftConfig:
    # How sensitive the drift detector is.

    # EMA window size in number of observations.
    ema_window: int = 100

    # Flag score drift when |ema_mean - baseline_mean| > mean_drift_threshold * baseline_std.
    mean_drift_threshold: float = 2.0

    # Flag alert rate drift when ema_alert_rate > rate_multiplier * baseline_alert_rate.
    rate_multiplier: float = 3.0

    # Minimum observations before drift can be reported (avoids cold-start false positives).
    min_observations: int = 50

    def __post_init__(self) -> None:
        if self.ema_window < 1:
            raise ValueError("ema_window must be >= 1")
        if self.mean_drift_threshold <= 0:
            raise ValueError("mean_drift_threshold must be > 0")
        if self.rate_multiplier <= 1:
            raise ValueError("rate_multiplier must be > 1")


@dataclass(frozen=True)
class DriftReport:
    # Snapshot of drift monitor state at one point in time.

    observed_at: datetime
    n_observations: int

    # Score drift.
    ema_mean_risk: float
    baseline_mean_risk: float
    mean_drift_sigmas: float       # how many baseline-sigma the mean has shifted
    score_drift_detected: bool

    # Alert rate drift.
    ema_alert_rate: float
    baseline_alert_rate: float
    rate_ratio: float              # ema_alert_rate / baseline_alert_rate
    rate_drift_detected: bool

    @property
    def any_drift(self) -> bool:
        return self.score_drift_detected or self.rate_drift_detected

    def summary(self) -> str:
        flags = []
        if self.score_drift_detected:
            flags.append(f"score_drift({self.mean_drift_sigmas:.1f}σ)")
        if self.rate_drift_detected:
            flags.append(f"rate_drift({self.rate_ratio:.1f}x)")
        status = ", ".join(flags) if flags else "ok"
        return (
            f"[drift={status}] n={self.n_observations} "
            f"ema_risk={self.ema_mean_risk:.3f} "
            f"ema_alert_rate={self.ema_alert_rate:.3f}"
        )


class DriftMonitor:
    # Tracks rolling score statistics and compares them to a calibration baseline.

    def __init__(self, config: Optional[DriftConfig] = None) -> None:
        self.cfg = config or DriftConfig()
        self._alpha = 2.0 / (self.cfg.ema_window + 1)

        # Baseline — set by fit().
        self._baseline_mean: Optional[float] = None
        self._baseline_std:  Optional[float] = None
        self._baseline_alert_rate: Optional[float] = None
        self._fitted = False

        # Live EMA state.
        self._ema_mean:       Optional[float] = None
        self._ema_alert_rate: Optional[float] = None
        self._n_observations: int = 0

    def fit(self, scores: list[float], alert_flags: list[bool]) -> "DriftMonitor":
        # Calibrate baseline statistics from a representative normal period.
        if len(scores) == 0:
            raise ValueError("scores must be non-empty")
        if len(scores) != len(alert_flags):
            raise ValueError("scores and alert_flags must have equal length")

        import statistics
        self._baseline_mean = statistics.mean(scores)
        self._baseline_std  = statistics.pstdev(scores) or 1e-8
        self._baseline_alert_rate = sum(alert_flags) / len(alert_flags)
        self._fitted = True
        return self

    def update(self, risk_score: float, alerted: bool) -> None:
        # Feed one new observation into the EMA.
        self._n_observations += 1

        alert_val = 1.0 if alerted else 0.0
        if self._ema_mean is None:
            self._ema_mean       = risk_score
            self._ema_alert_rate = alert_val
        else:
            self._ema_mean       = self._alpha * risk_score + (1 - self._alpha) * self._ema_mean
            self._ema_alert_rate = self._alpha * alert_val  + (1 - self._alpha) * self._ema_alert_rate

    def report(self, now: Optional[datetime] = None) -> DriftReport:
        self._require_fitted()
        if now is None:
            now = datetime.utcnow()

        ema_mean = self._ema_mean if self._ema_mean is not None else self._baseline_mean
        ema_rate = self._ema_alert_rate if self._ema_alert_rate is not None else self._baseline_alert_rate

        ready = self._n_observations >= self.cfg.min_observations

        # Score drift: deviation in sigma units from baseline mean.
        drift_sigmas = abs(ema_mean - self._baseline_mean) / self._baseline_std  # type: ignore
        score_drift = ready and drift_sigmas >= self.cfg.mean_drift_threshold

        # Alert rate drift: live rate vs baseline rate.
        safe_baseline = self._baseline_alert_rate or 1e-6  # type: ignore
        rate_ratio = ema_rate / safe_baseline  # type: ignore
        rate_drift = ready and rate_ratio >= self.cfg.rate_multiplier

        return DriftReport(
            observed_at=now,
            n_observations=self._n_observations,
            ema_mean_risk=float(ema_mean),
            baseline_mean_risk=float(self._baseline_mean),  # type: ignore
            mean_drift_sigmas=float(drift_sigmas),
            score_drift_detected=score_drift,
            ema_alert_rate=float(ema_rate),
            baseline_alert_rate=float(self._baseline_alert_rate),  # type: ignore
            rate_ratio=float(rate_ratio),
            rate_drift_detected=rate_drift,
        )

    def reset(self) -> None:
        # Reset live EMA state while keeping the calibration baseline.
        self._ema_mean = None
        self._ema_alert_rate = None
        self._n_observations = 0

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "DriftMonitor.fit() must be called before report()."
            )
