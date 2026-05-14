# Minimal FastAPI endpoints for the Decision Controller layer.
#
# Endpoints:
#   POST /assess             — submit a pre-computed risk + anomaly result and get a decision.
#   POST /feedback/{alert_id} — submit operator feedback (TP/FP/FN) for a filed alert.
#   GET  /feedback/queue      — list pending false-positive retraining queue items.
#   POST /feedback/queue/process — run one batch of the retraining queue processor.
#   GET  /drift               — current drift monitor report.
#   GET  /shadow              — current shadow deployment report.
#   GET  /alerts              — list recent alerts (optionally filtered by server).
#   GET  /health              — liveness check.
#
# The app is instantiated with pre-built component instances so that the
# caller controls model loading, DB path, config, etc.
# In production, wire it up via a lifespan function or dependency injection.

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .alert_decision import Alert, AlertDecisionController, ColdStartResult, SuppressedAlert
from .drift_monitor import DriftMonitor, DriftReport
from .feedback_store import FeedbackStore, FeedbackLabel
from .retraining_queue import RetrainingQueueProcessor, ThresholdAdjustment
from .rca import RCAReport, RootCauseAnalyzer
from .risk_scorer import RiskAssessment, RiskConfig
from .shadow_deployment import ShadowDeployment, ShadowReport


# ─── Request / Response models ───────────────────────────────────────────────

class AssessRequest(BaseModel):
    server_id: str
    combined_risk: float = Field(ge=0.0, le=1.0)
    forecast_risk: float = Field(ge=0.0, le=1.0)
    anomaly_risk: float = Field(ge=0.0, le=1.0)
    forecast_breaches: dict[str, float] = Field(default_factory=dict)
    anomaly_root_cause: Optional[str] = None
    anomaly_index: float = 0.0
    should_alert: bool = False
    # Optional: anomaly metric errors for RCA.
    metric_errors: dict[str, float] = Field(default_factory=dict)
    correlated_metrics: list[str] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    label: FeedbackLabel
    notes: Optional[str] = None
    submitted_by: str = "operator"


class AssessResponse(BaseModel):
    decision: str          # "alert" | "suppressed" | "cold_start" | "ok"
    server_id: str
    combined_risk: float
    severity: Optional[str] = None
    alert_id: Optional[int] = None
    rca: Optional[dict] = None
    message: Optional[str] = None


class ProcessQueueResponse(BaseModel):
    items_processed: int
    adjustments: list[dict]


# ─── App factory ─────────────────────────────────────────────────────────────

def create_app(
    alert_controller: AlertDecisionController,
    feedback_store: FeedbackStore,
    queue_processor: RetrainingQueueProcessor,
    drift_monitor: DriftMonitor,
    shadow_deployment: ShadowDeployment,
    rca_analyzer: RootCauseAnalyzer,
    risk_config: RiskConfig,
) -> FastAPI:
    app = FastAPI(
        title="Predictive Cloud Guardian — Decision Controller",
        version="0.1.0",
        description="Intelligence layer: combines forecast + anomaly signals into alerts.",
    )

    # ─── Health ──────────────────────────────────────────────────────────────

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

    # ─── Assess ──────────────────────────────────────────────────────────────

    @app.post("/assess", response_model=AssessResponse)
    def assess(req: AssessRequest) -> AssessResponse:
        # Build a RiskAssessment from the submitted scores.
        assessment = RiskAssessment(
            server_id=req.server_id,
            combined_risk=req.combined_risk,
            forecast_risk=req.forecast_risk,
            anomaly_risk=req.anomaly_risk,
            should_alert=req.should_alert,
            forecast_breaches=req.forecast_breaches,
            anomaly_root_cause=req.anomaly_root_cause,
            anomaly_index=req.anomaly_index,
        )

        # Feed into drift monitor.
        drift_monitor.update(req.combined_risk, alerted=req.should_alert)

        # Alert decision (cold start / storm guard / fire).
        result = alert_controller.evaluate(assessment)

        if isinstance(result, ColdStartResult):
            return AssessResponse(
                decision="cold_start",
                server_id=req.server_id,
                combined_risk=req.combined_risk,
                message=f"warming up ({result.samples_seen}/{result.samples_needed})",
            )

        if result is None:
            return AssessResponse(
                decision="ok",
                server_id=req.server_id,
                combined_risk=req.combined_risk,
            )

        if isinstance(result, SuppressedAlert):
            return AssessResponse(
                decision="suppressed",
                server_id=req.server_id,
                combined_risk=req.combined_risk,
                message=result.reason,
            )

        # result is Alert — persist it and run RCA.
        assert isinstance(result, Alert)
        alert_id = feedback_store.insert_alert(
            server_id=result.server_id,
            fired_at=result.fired_at,
            severity=result.severity.value,
            combined_risk=result.combined_risk,
            forecast_risk=result.forecast_risk,
            anomaly_risk=result.anomaly_risk,
            anomaly_index=result.anomaly_index,
            root_cause=result.anomaly_root_cause,
            breaches=result.forecast_breaches,
        )

        # Build lightweight RCA from the submitted metric_errors.
        rca_dict: Optional[dict] = None
        if req.metric_errors:
            from ..anomaly.anomaly_api import AnomalyReport as AR
            anomaly_report = AR(
                is_anomaly=True,
                score=req.anomaly_risk,
                anomaly_index=req.anomaly_index,
                root_cause_metric=req.anomaly_root_cause,
                metric_errors=req.metric_errors,
                correlated_metrics=req.correlated_metrics,
            )
            rca_report = rca_analyzer.analyze(result, anomaly_report, assessment)
            rca_dict = rca_report.as_dict()

        return AssessResponse(
            decision="alert",
            server_id=req.server_id,
            combined_risk=req.combined_risk,
            severity=result.severity.value,
            alert_id=alert_id,
            rca=rca_dict,
        )

    # ─── Feedback ────────────────────────────────────────────────────────────

    @app.post("/feedback/{alert_id}")
    def submit_feedback(alert_id: int, req: FeedbackRequest) -> dict:
        stored = feedback_store.get_alert(alert_id)
        if stored is None:
            raise HTTPException(status_code=404, detail=f"alert {alert_id} not found")
        feedback_id = feedback_store.submit_feedback(
            alert_id=alert_id,
            label=req.label,
            notes=req.notes,
            submitted_by=req.submitted_by,
        )
        queued = req.label == "false_positive"
        return {
            "feedback_id": feedback_id,
            "alert_id": alert_id,
            "label": req.label,
            "queued_for_retraining": queued,
        }

    @app.get("/feedback/queue")
    def list_queue(limit: int = 50) -> dict:
        items = feedback_store.pending_queue_items(limit=limit)
        return {
            "pending": feedback_store.queue_length("pending"),
            "items": [
                {
                    "id": it.id,
                    "alert_id": it.alert_id,
                    "server_id": it.server_id,
                    "fired_at": it.fired_at,
                    "combined_risk": it.combined_risk,
                    "reason": it.reason,
                    "enqueued_at": it.enqueued_at,
                }
                for it in items
            ],
        }

    @app.post("/feedback/queue/process", response_model=ProcessQueueResponse)
    def process_queue() -> ProcessQueueResponse:
        adjustments = queue_processor.process_batch(
            current_threshold=risk_config.risk_threshold
        )
        return ProcessQueueResponse(
            items_processed=queue_processor.fp_score_count("global"),
            adjustments=[a.__dict__ | {"computed_at": a.computed_at.isoformat()} for a in adjustments],
        )

    # ─── Drift ───────────────────────────────────────────────────────────────

    @app.get("/drift")
    def drift() -> dict:
        try:
            report: DriftReport = drift_monitor.report()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        return {
            "any_drift": report.any_drift,
            "score_drift": report.score_drift_detected,
            "rate_drift": report.rate_drift_detected,
            "n_observations": report.n_observations,
            "ema_mean_risk": round(report.ema_mean_risk, 4),
            "ema_alert_rate": round(report.ema_alert_rate, 4),
            "mean_drift_sigmas": round(report.mean_drift_sigmas, 3),
            "rate_ratio": round(report.rate_ratio, 3),
            "summary": report.summary(),
        }

    # ─── Shadow ──────────────────────────────────────────────────────────────

    @app.get("/shadow")
    def shadow() -> dict:
        report: ShadowReport = shadow_deployment.report()
        return {
            "ready_to_promote": report.ready_to_promote,
            "n_evaluations": report.n_evaluations,
            "agreement_rate": round(report.agreement_rate, 4),
            "candidate_only_rate": round(report.candidate_only_rate, 4),
            "production_only_rate": round(report.production_only_rate, 4),
            "risk_mae": round(report.risk_mae, 5),
            "promotion_blockers": report.promotion_blockers,
            "summary": report.summary(),
        }

    # ─── Alerts ──────────────────────────────────────────────────────────────

    @app.get("/alerts")
    def list_alerts(server_id: Optional[str] = None, limit: int = 50) -> dict:
        alerts = feedback_store.list_alerts(server_id=server_id, limit=limit)
        return {
            "count": len(alerts),
            "alerts": [
                {
                    "id": a.id,
                    "server_id": a.server_id,
                    "fired_at": a.fired_at,
                    "severity": a.severity,
                    "combined_risk": a.combined_risk,
                    "root_cause": a.root_cause,
                    "breaches": a.breaches,
                }
                for a in alerts
            ],
        }

    return app
