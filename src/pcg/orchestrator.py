# Main orchestration runner — wires every module into one processing loop.
#
# For each server tick the pipeline executes in order:
#   1. Data pipeline  — fetch raw metrics, interpolate, smooth, normalize, window.
#   2. Forecaster     — LSTM+attention predicts the next HORIZON_MINUTES.
#   3. Anomaly        — ConvAutoencoder reconstruction error + correlation RCA.
#   4. Risk scorer    — weighted blend of forecast risk + anomaly index.
#   5. Alert decision — cold-start guard, storm dedup, severity assignment.
#   6. RCA            — evidence ranking when an alert fires.
#   7. Persistence    — alert written to SQLite; drift monitor updated.
#
# The orchestrator is intentionally synchronous and single-threaded.
# Each call to `tick()` processes one (server_id, timestamp) pair.
# Callers drive the loop (APScheduler, a while-loop in the demo, or a test).

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import torch

from .anomaly.anomaly_api import AnomalyDetectionAPI, AnomalyReport
from .anomaly.anomaly_scorer import AnomalyScorer
from .anomaly.correlation_detector import CorrelationDetector
from .controller.alert_decision import (
    Alert,
    AlertDecisionController,
    ColdStartResult,
    DecisionResult,
    SuppressedAlert,
)
from .controller.drift_monitor import DriftMonitor
from .controller.feedback_store import FeedbackStore
from .controller.rca import RCAReport, RootCauseAnalyzer
from .controller.risk_scorer import CombinedRiskScorer, RiskAssessment, RiskConfig
from .data.normalizer import MinMaxNormalizer
from .data.pipeline import DataPipeline, PreparedWindow
from .inference.forecaster import Forecaster, ForecastOutput
from .models.lstm_attention import LSTMForecaster


@dataclass
class TickResult:
    # Full result of processing one (server_id, timestamp) pair.

    server_id: str
    timestamp: datetime
    prepared_window: PreparedWindow
    forecast: ForecastOutput
    anomaly: AnomalyReport
    assessment: RiskAssessment
    decision: DecisionResult        # Alert | SuppressedAlert | ColdStartResult | None
    rca: Optional[RCAReport]
    alert_db_id: Optional[int]      # set when an Alert was persisted

    @property
    def alerted(self) -> bool:
        return isinstance(self.decision, Alert)

    def summary(self) -> str:
        if isinstance(self.decision, ColdStartResult):
            return (
                f"[{self.server_id}] warming up "
                f"({self.decision.samples_seen}/{self.decision.samples_needed})"
            )
        if self.decision is None:
            return (
                f"[{self.server_id}] ok "
                f"risk={self.assessment.combined_risk:.3f}"
            )
        if isinstance(self.decision, SuppressedAlert):
            return f"[{self.server_id}] suppressed ({self.decision.reason})"
        if isinstance(self.decision, Alert):
            rca_str = ""
            if self.rca:
                rca_str = f" rca={self.rca.primary_cause}({self.rca.confidence:.0%})"
            return (
                f"[{self.server_id}] ALERT {self.decision.severity.value.upper()} "
                f"risk={self.assessment.combined_risk:.3f}{rca_str}"
            )
        return f"[{self.server_id}] unknown decision"


class PCGOrchestrator:
    # Central coordinator — instantiated once, then driven by repeated tick() calls.

    def __init__(
        self,
        pipeline: DataPipeline,
        forecaster: Forecaster,
        anomaly_api: AnomalyDetectionAPI,
        risk_scorer: CombinedRiskScorer,
        alert_controller: AlertDecisionController,
        rca_analyzer: RootCauseAnalyzer,
        drift_monitor: DriftMonitor,
        feedback_store: FeedbackStore,
    ) -> None:
        self._pipeline       = pipeline
        self._forecaster     = forecaster
        self._anomaly_api    = anomaly_api
        self._risk_scorer    = risk_scorer
        self._alert_ctrl     = alert_controller
        self._rca            = rca_analyzer
        self._drift          = drift_monitor
        self._store          = feedback_store

    def tick(
        self,
        server_id: str,
        now: Optional[datetime] = None,
    ) -> TickResult:
        # Process one observation tick for a server. All exceptions propagate —
        # callers that want fault isolation should wrap this in try/except.
        if now is None:
            now = datetime.utcnow()

        # Step 1: data pipeline.
        prepared = self._pipeline.prepare_window(server_id, end=now)

        # Step 2: forecast.
        forecast = self._forecaster.predict(prepared)

        # Step 3: anomaly detection (channels-first tensor required).
        # prepared.tensor is (1, WINDOW_SIZE, N_METRICS) — transpose to (N_METRICS, WINDOW_SIZE).
        window_cf = prepared.tensor.squeeze(0).T  # (N_METRICS, WINDOW_SIZE)
        anomaly = self._anomaly_api.analyze(window_cf)

        # Step 4: combined risk.
        assessment = self._risk_scorer.assess(forecast, anomaly)

        # Step 5: alert decision.
        decision = self._alert_ctrl.evaluate(assessment, now=now)

        # Step 6: RCA (only when an alert fires).
        rca: Optional[RCAReport] = None
        if isinstance(decision, Alert):
            rca = self._rca.analyze(decision, anomaly, assessment)

        # Step 7: persistence + drift update.
        alert_db_id: Optional[int] = None
        if isinstance(decision, Alert):
            alert_db_id = self._store.insert_alert(
                server_id=decision.server_id,
                fired_at=decision.fired_at,
                severity=decision.severity.value,
                combined_risk=decision.combined_risk,
                forecast_risk=decision.forecast_risk,
                anomaly_risk=decision.anomaly_risk,
                anomaly_index=decision.anomaly_index,
                root_cause=rca.primary_cause if rca else decision.anomaly_root_cause,
                breaches=decision.forecast_breaches,
            )

        self._drift.update(
            risk_score=assessment.combined_risk,
            alerted=isinstance(decision, Alert),
        )

        return TickResult(
            server_id=server_id,
            timestamp=now,
            prepared_window=prepared,
            forecast=forecast,
            anomaly=anomaly,
            assessment=assessment,
            decision=decision,
            rca=rca,
            alert_db_id=alert_db_id,
        )


def build_orchestrator(
    pipeline: DataPipeline,
    forecaster_model: LSTMForecaster,
    normalizer: MinMaxNormalizer,
    anomaly_scorer: AnomalyScorer,
    correlation_detector: Optional[CorrelationDetector] = None,
    risk_config: Optional[RiskConfig] = None,
    db_path: str = "pcg_feedback.db",
    device: str = "cpu",
) -> PCGOrchestrator:
    # Convenience factory: assembles all components and returns a ready orchestrator.
    forecaster = Forecaster(forecaster_model, normalizer, device=device)
    anomaly_api = AnomalyDetectionAPI(anomaly_scorer, correlation_detector)
    risk_scorer = CombinedRiskScorer(risk_config)
    alert_ctrl = AlertDecisionController()
    rca = RootCauseAnalyzer()

    drift_monitor = DriftMonitor()
    # Baseline for drift: treat a 10% alert rate as normal.
    drift_monitor.fit(
        scores=[0.3] * 200,
        alert_flags=[False] * 180 + [True] * 20,
    )

    store = FeedbackStore(db_path=db_path)

    return PCGOrchestrator(
        pipeline=pipeline,
        forecaster=forecaster,
        anomaly_api=anomaly_api,
        risk_scorer=risk_scorer,
        alert_controller=alert_ctrl,
        rca_analyzer=rca,
        drift_monitor=drift_monitor,
        feedback_store=store,
    )
