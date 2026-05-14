# Alert decision layer: converts RiskAssessment into actionable Alert objects.
#
# Responsibilities:
#   1. Cold-start guard   — suppress alerts until the server has enough history.
#   2. Alert decision     — fire when combined_risk >= threshold.
#   3. Alert storm guard  — deduplicate/group alerts that arrive within a short
#                           window for the same server. Without this, a single
#                           sustained anomaly produces hundreds of identical alerts.
#
# Cold-start logic:
#   Each server tracks how many samples it has seen. Until it reaches
#   COLD_START_MIN_SAMPLES (60), every assessment returns ColdStartResult
#   instead of an Alert so that random initialization noise doesn't page anyone.
#
# Storm grouping logic:
#   The controller keeps a per-server timestamp of the last fired alert.
#   A new alert is only emitted when:
#     - No alert has fired for that server in the last GROUPING_WINDOW_SECONDS, OR
#     - The new alert's severity is strictly higher than the last one.
#   This means one alert per server per minute at most (given the default 60s window).

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Union

from ..core.constants import COLD_START_MIN_SAMPLES, GROUPING_WINDOW_SECONDS
from .risk_scorer import RiskAssessment


class AlertSeverity(str, Enum):
    LOW      = "low"       # 0.50 <= risk < 0.65
    MEDIUM   = "medium"    # 0.65 <= risk < 0.80
    HIGH     = "high"      # 0.80 <= risk < 0.92
    CRITICAL = "critical"  # risk >= 0.92


def _severity_from_risk(risk: float) -> AlertSeverity:
    if risk >= 0.92:
        return AlertSeverity.CRITICAL
    if risk >= 0.80:
        return AlertSeverity.HIGH
    if risk >= 0.65:
        return AlertSeverity.MEDIUM
    return AlertSeverity.LOW


def _severity_rank(s: AlertSeverity) -> int:
    return {
        AlertSeverity.LOW: 0,
        AlertSeverity.MEDIUM: 1,
        AlertSeverity.HIGH: 2,
        AlertSeverity.CRITICAL: 3,
    }[s]


@dataclass(frozen=True)
class Alert:
    server_id: str
    fired_at: datetime
    severity: AlertSeverity
    combined_risk: float
    forecast_risk: float
    anomaly_risk: float
    forecast_breaches: dict[str, float]
    anomaly_root_cause: Optional[str]
    anomaly_index: float
    suppressed_by_storm: bool = False   # True when a lower-severity repeat was swallowed

    def as_dict(self) -> dict:
        return {
            "server_id": self.server_id,
            "fired_at": self.fired_at.isoformat(),
            "severity": self.severity.value,
            "combined_risk": round(self.combined_risk, 4),
            "forecast_risk": round(self.forecast_risk, 4),
            "anomaly_risk": round(self.anomaly_risk, 4),
            "forecast_breaches": self.forecast_breaches,
            "anomaly_root_cause": self.anomaly_root_cause,
            "anomaly_index": round(self.anomaly_index, 3),
        }


@dataclass(frozen=True)
class ColdStartResult:
    server_id: str
    samples_seen: int
    samples_needed: int

    @property
    def warmup_fraction(self) -> float:
        return min(self.samples_seen / self.samples_needed, 1.0)


@dataclass(frozen=True)
class SuppressedAlert:
    # Emitted when storm grouping swallows a repeat alert.
    server_id: str
    suppressed_at: datetime
    reason: str   # "storm_window" or "lower_severity"


# Union type returned by AlertDecisionController.evaluate().
DecisionResult = Union[Alert, ColdStartResult, SuppressedAlert, None]


@dataclass
class _ServerState:
    # Per-server mutable state tracked by the controller.
    samples_seen: int = 0
    last_alert_at: Optional[datetime] = None
    last_alert_severity: Optional[AlertSeverity] = None


class AlertDecisionController:
    # Stateful per-server alert decision maker.

    def __init__(
        self,
        cold_start_min_samples: int = COLD_START_MIN_SAMPLES,
        grouping_window_seconds: int = GROUPING_WINDOW_SECONDS,
    ) -> None:
        if cold_start_min_samples < 0:
            raise ValueError("cold_start_min_samples must be >= 0")
        if grouping_window_seconds < 0:
            raise ValueError("grouping_window_seconds must be >= 0")

        self._cold_start_min = cold_start_min_samples
        self._grouping_window = timedelta(seconds=grouping_window_seconds)
        self._states: dict[str, _ServerState] = {}

    def evaluate(
        self,
        assessment: RiskAssessment,
        now: Optional[datetime] = None,
    ) -> DecisionResult:
        # Process one RiskAssessment and return the appropriate result.
        # Returns:
        #   ColdStartResult  — server still in warmup phase.
        #   None             — assessment below risk threshold, no action.
        #   SuppressedAlert  — above threshold but storm guard swallowed it.
        #   Alert            — a genuine new alert to be acted upon.
        if now is None:
            now = datetime.utcnow()

        state = self._states.setdefault(assessment.server_id, _ServerState())
        state.samples_seen += 1

        # Cold-start guard: no alerts until minimum history is accumulated.
        if state.samples_seen < self._cold_start_min:
            return ColdStartResult(
                server_id=assessment.server_id,
                samples_seen=state.samples_seen,
                samples_needed=self._cold_start_min,
            )

        if not assessment.should_alert:
            return None

        severity = _severity_from_risk(assessment.combined_risk)

        # Storm guard: check if we recently fired for this server.
        if state.last_alert_at is not None:
            elapsed = now - state.last_alert_at
            if elapsed < self._grouping_window:
                # Within the dedup window — only escalate if severity increased.
                if (
                    state.last_alert_severity is not None
                    and _severity_rank(severity)
                    <= _severity_rank(state.last_alert_severity)
                ):
                    return SuppressedAlert(
                        server_id=assessment.server_id,
                        suppressed_at=now,
                        reason="storm_window",
                    )

        alert = Alert(
            server_id=assessment.server_id,
            fired_at=now,
            severity=severity,
            combined_risk=assessment.combined_risk,
            forecast_risk=assessment.forecast_risk,
            anomaly_risk=assessment.anomaly_risk,
            forecast_breaches=assessment.forecast_breaches,
            anomaly_root_cause=assessment.anomaly_root_cause,
            anomaly_index=assessment.anomaly_index,
        )
        state.last_alert_at = now
        state.last_alert_severity = severity
        return alert

    def reset_server(self, server_id: str) -> None:
        # Clear accumulated state for a server (e.g. after maintenance).
        self._states.pop(server_id, None)

    def samples_seen(self, server_id: str) -> int:
        return self._states.get(server_id, _ServerState()).samples_seen

    def is_warmed_up(self, server_id: str) -> bool:
        return self.samples_seen(server_id) >= self._cold_start_min
