# Root Cause Analysis (RCA) layer.
#
# Given an Alert and the supporting AnomalyReport + RiskAssessment, this
# module produces a structured RCAReport that answers:
#   "What failed, why, and what is likely the underlying cause?"
#
# Evidence ranking:
#   Each candidate metric is scored by a weighted combination of:
#     1. reconstruction_error_rank  — autoencoder per-metric MSE rank
#        (highest MSE = strongest evidence the metric is anomalous).
#     2. forecast_breach_rank       — whether the LSTM forecast predicts
#        this metric will breach its safe threshold.
#     3. correlation_rank           — whether this metric was flagged as
#        correlated with the primary anomalous metric by CorrelationDetector.
#
# The metric with the highest combined evidence score is the primary cause.
# Secondary metrics form the "contributing factors" list.
#
# Confidence:
#   confidence = primary_score / sum(all_scores)
#   Values near 1.0 mean one metric dominates; values near 1/N mean the
#   evidence is spread evenly and the root cause is uncertain.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..anomaly.anomaly_api import AnomalyReport
from ..core.constants import METRIC_ORDER
from .alert_decision import Alert
from .risk_scorer import RiskAssessment


@dataclass(frozen=True)
class RCAReport:
    # Structured root-cause analysis for one alert.

    alert_id: str                    # server_id + timestamp string, for tracing
    primary_cause: str               # metric with highest evidence score
    confidence: float                # fraction of total evidence on primary cause
    contributing_factors: list[str]  # other metrics with elevated evidence
    evidence: dict[str, float]       # per-metric raw evidence scores
    narrative: str                   # human-readable one-line explanation

    def as_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "primary_cause": self.primary_cause,
            "confidence": round(self.confidence, 3),
            "contributing_factors": self.contributing_factors,
            "evidence": {k: round(v, 4) for k, v in self.evidence.items()},
            "narrative": self.narrative,
        }


# Evidence weights — tune without changing the algorithm.
_W_RECONSTRUCTION = 0.50   # autoencoder MSE is the strongest direct signal
_W_FORECAST       = 0.30   # forecast breach confirms the metric is trending bad
_W_CORRELATION    = 0.20   # being flagged as correlated adds supporting evidence


class RootCauseAnalyzer:

    def __init__(
        self,
        metric_order: tuple[str, ...] = METRIC_ORDER,
        min_contributing_evidence: float = 0.10,
    ) -> None:
        # min_contributing_evidence: a metric is listed as a contributing factor
        # only when its evidence score is at least this fraction of the primary score.
        self.metric_order = metric_order
        self.min_contributing_evidence = min_contributing_evidence

    def analyze(
        self,
        alert: Alert,
        anomaly: AnomalyReport,
        assessment: RiskAssessment,
    ) -> RCAReport:
        alert_id = f"{alert.server_id}@{alert.fired_at.isoformat()}"
        evidence = self._build_evidence(anomaly, assessment)

        if not evidence:
            return RCAReport(
                alert_id=alert_id,
                primary_cause="unknown",
                confidence=0.0,
                contributing_factors=[],
                evidence={},
                narrative="Insufficient evidence to determine root cause.",
            )

        total = sum(evidence.values())
        ranked = sorted(evidence.items(), key=lambda kv: kv[1], reverse=True)

        primary_metric, primary_score = ranked[0]
        confidence = primary_score / total if total > 0 else 0.0

        threshold = primary_score * self.min_contributing_evidence
        contributing = [
            m for m, s in ranked[1:]
            if s >= threshold
        ]

        narrative = self._build_narrative(
            primary_metric, confidence, contributing,
            alert, anomaly, assessment,
        )

        return RCAReport(
            alert_id=alert_id,
            primary_cause=primary_metric,
            confidence=confidence,
            contributing_factors=contributing,
            evidence=evidence,
            narrative=narrative,
        )

    def _build_evidence(
        self,
        anomaly: AnomalyReport,
        assessment: RiskAssessment,
    ) -> dict[str, float]:
        evidence: dict[str, float] = {m: 0.0 for m in self.metric_order}

        # --- Reconstruction error evidence ---
        # Normalize each metric's MSE by the total MSE so scores sum to 1.
        total_mse = sum(anomaly.metric_errors.values()) or 1.0
        for metric, mse in anomaly.metric_errors.items():
            if metric in evidence:
                evidence[metric] += _W_RECONSTRUCTION * (mse / total_mse)

        # --- Forecast breach evidence ---
        # Binary: does the forecast predict a breach for this metric?
        # Each breaching metric gets equal weight; shared over all breaches.
        if assessment.forecast_breaches:
            breach_share = _W_FORECAST / len(assessment.forecast_breaches)
            for metric in assessment.forecast_breaches:
                if metric in evidence:
                    evidence[metric] += breach_share

        # --- Correlation evidence ---
        # Metrics flagged by CorrelationDetector as co-anomalous get a boost.
        if anomaly.correlated_metrics:
            corr_share = _W_CORRELATION / len(anomaly.correlated_metrics)
            for metric in anomaly.correlated_metrics:
                if metric in evidence:
                    evidence[metric] += corr_share

        # Remove metrics with zero evidence to keep output clean.
        return {m: v for m, v in evidence.items() if v > 0}

    def _build_narrative(
        self,
        primary: str,
        confidence: float,
        contributing: list[str],
        alert: Alert,
        anomaly: AnomalyReport,
        assessment: RiskAssessment,
    ) -> str:
        conf_str = f"{confidence:.0%}"
        sev = alert.severity.value.upper()
        parts = [
            f"{sev} alert on {alert.server_id}.",
            f"Primary anomalous metric: {primary} ({conf_str} confidence).",
        ]
        if contributing:
            parts.append(f"Contributing factors: {', '.join(contributing)}.")
        if assessment.forecast_breaches:
            breach_list = ", ".join(
                f"{m} (+{v:.1%})" for m, v in assessment.forecast_breaches.items()
            )
            parts.append(f"Forecast predicts threshold breach in: {breach_list}.")
        if anomaly.anomaly_index > 0:
            parts.append(
                f"Anomaly index: {anomaly.anomaly_index:.1f}σ above normal baseline."
            )
        return " ".join(parts)
