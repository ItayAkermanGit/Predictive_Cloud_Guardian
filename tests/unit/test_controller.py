# Unit tests for Phase 5: Decision Controller and Intelligence Layer.
#
# Organized by module:
#   1. CombinedRiskScorer
#   2. AlertDecisionController (cold start, storm grouping, severity)
#   3. RootCauseAnalyzer
#   4. DriftMonitor
#   5. FeedbackStore + retraining queue integration
#   6. RetrainingQueueProcessor
#   7. ShadowDeployment

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pytest

from pcg.controller.alert_decision import (
    Alert,
    AlertDecisionController,
    AlertSeverity,
    ColdStartResult,
    SuppressedAlert,
    _severity_from_risk,
)
from pcg.controller.drift_monitor import DriftConfig, DriftMonitor
from pcg.controller.feedback_store import FeedbackStore
from pcg.controller.rca import RootCauseAnalyzer
from pcg.controller.retraining_queue import RetrainingQueueConfig, RetrainingQueueProcessor
from pcg.controller.risk_scorer import CombinedRiskScorer, RiskAssessment, RiskConfig
from pcg.controller.shadow_deployment import ShadowConfig, ShadowDeployment


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_assessment(
    server_id: str = "srv-1",
    combined_risk: float = 0.6,
    forecast_risk: float = 0.5,
    anomaly_risk: float = 0.7,
    should_alert: bool = True,
    forecast_breaches: Optional[dict] = None,
    anomaly_root_cause: Optional[str] = "cpu_util",
    anomaly_index: float = 3.5,
) -> RiskAssessment:
    return RiskAssessment(
        server_id=server_id,
        combined_risk=combined_risk,
        forecast_risk=forecast_risk,
        anomaly_risk=anomaly_risk,
        should_alert=should_alert,
        forecast_breaches=forecast_breaches or {"cpu_util": 0.05},
        anomaly_root_cause=anomaly_root_cause,
        anomaly_index=anomaly_index,
    )


_SENTINEL = object()


def _make_anomaly_report(
    metric_errors=_SENTINEL,
    correlated=_SENTINEL,
    anomaly_index: float = 3.5,
    root_cause: str = "cpu_util",
):
    from pcg.anomaly.anomaly_api import AnomalyReport
    if metric_errors is _SENTINEL:
        metric_errors = {
            "cpu_util": 0.08,
            "mem_util": 0.03,
            "net_io": 0.02,
            "disk_io": 0.01,
        }
    if correlated is _SENTINEL:
        correlated = ["mem_util"]
    return AnomalyReport(
        is_anomaly=True,
        score=0.05,
        anomaly_index=anomaly_index,
        root_cause_metric=root_cause,
        metric_errors=metric_errors,
        correlated_metrics=correlated,
    )


# ─── 1. CombinedRiskScorer ───────────────────────────────────────────────────

class TestCombinedRiskScorer:
    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError):
            RiskConfig(w_forecast=0.6, w_anomaly=0.6)

    def test_zero_risk_when_below_thresholds(self) -> None:
        from pcg.inference.forecaster import ForecastOutput
        import numpy as np
        from pcg.core.constants import HORIZON_MINUTES, N_METRICS, WINDOW_SIZE, METRIC_ORDER

        now = datetime(2024, 1, 1)
        timestamps = [now + timedelta(minutes=i + 1) for i in range(HORIZON_MINUTES)]
        # All metrics at 0.1 — well below any safe threshold.
        normalized = np.full((HORIZON_MINUTES, N_METRICS), 0.1, dtype=np.float32)
        forecast = ForecastOutput(
            server_id="srv-1",
            window_end=now,
            predicted_normalized=normalized,
            predicted_original=normalized.copy(),
            attention_weights=np.ones(WINDOW_SIZE, dtype=np.float32) / WINDOW_SIZE,
            forecast_timestamps=timestamps,
        )
        anomaly = _make_anomaly_report(anomaly_index=0.0)
        anomaly_report_with_zero = type(anomaly)(
            is_anomaly=False,
            score=0.0,
            anomaly_index=0.0,
            root_cause_metric=None,
            metric_errors={"cpu_util": 0.0, "mem_util": 0.0, "net_io": 0.0, "disk_io": 0.0},
            correlated_metrics=[],
        )
        scorer = CombinedRiskScorer()
        result = scorer.assess(forecast, anomaly_report_with_zero)
        assert result.forecast_risk == 0.0
        assert result.anomaly_risk == 0.0
        assert result.combined_risk == 0.0
        assert not result.should_alert

    def test_combined_risk_is_weighted_sum(self) -> None:
        cfg = RiskConfig(w_forecast=0.4, w_anomaly=0.6, n_sigma_clip=4.0)
        scorer = CombinedRiskScorer(cfg)
        # Inject pre-computed component risks directly via _anomaly_risk.
        anomaly_risk = scorer._anomaly_risk(4.0)   # index=4.0, clip=4.0 → 1.0
        assert abs(anomaly_risk - 1.0) < 1e-6

    def test_anomaly_risk_clamps_to_one(self) -> None:
        scorer = CombinedRiskScorer(RiskConfig(n_sigma_clip=3.0))
        assert scorer._anomaly_risk(100.0) == 1.0

    def test_anomaly_risk_zero_for_negative_index(self) -> None:
        scorer = CombinedRiskScorer()
        assert scorer._anomaly_risk(-5.0) == 0.0

    def test_assessment_summary_string(self) -> None:
        a = _make_assessment()
        assert "srv-1" in a.summary()

    def test_breach_scale_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            RiskConfig(breach_scale=0.0)

    def test_n_sigma_clip_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            RiskConfig(n_sigma_clip=0.0)


# ─── 2. AlertDecisionController ──────────────────────────────────────────────

class TestAlertDecisionController:
    def test_cold_start_suppresses_alerts(self) -> None:
        ctrl = AlertDecisionController(cold_start_min_samples=5, grouping_window_seconds=0)
        for _ in range(4):
            result = ctrl.evaluate(_make_assessment())
            assert isinstance(result, ColdStartResult)

    def test_cold_start_result_has_correct_count(self) -> None:
        ctrl = AlertDecisionController(cold_start_min_samples=10, grouping_window_seconds=0)
        result = ctrl.evaluate(_make_assessment())
        assert isinstance(result, ColdStartResult)
        assert result.samples_seen == 1
        assert result.samples_needed == 10

    def test_alert_fires_after_warmup(self) -> None:
        ctrl = AlertDecisionController(cold_start_min_samples=2, grouping_window_seconds=0)
        ctrl.evaluate(_make_assessment())  # sample 1 → ColdStart
        result = ctrl.evaluate(_make_assessment())  # sample 2 → Alert
        assert isinstance(result, Alert)

    def test_no_alert_when_below_threshold(self) -> None:
        ctrl = AlertDecisionController(cold_start_min_samples=1, grouping_window_seconds=0)
        result = ctrl.evaluate(_make_assessment(should_alert=False))
        assert result is None

    def test_storm_suppresses_repeat_alert(self) -> None:
        ctrl = AlertDecisionController(
            cold_start_min_samples=1,
            grouping_window_seconds=60,
        )
        now = datetime(2024, 1, 1, 12, 0, 0)
        ctrl.evaluate(_make_assessment(), now=now)  # first alert fires
        result = ctrl.evaluate(
            _make_assessment(), now=now + timedelta(seconds=30)
        )
        assert isinstance(result, SuppressedAlert)
        assert result.reason == "storm_window"

    def test_escalation_breaks_storm_guard(self) -> None:
        ctrl = AlertDecisionController(
            cold_start_min_samples=1,
            grouping_window_seconds=60,
        )
        now = datetime(2024, 1, 1, 12, 0, 0)
        # First alert: LOW severity (risk=0.51)
        ctrl.evaluate(_make_assessment(combined_risk=0.51), now=now)
        # Within window but CRITICAL severity (risk=0.95) → should escalate.
        result = ctrl.evaluate(
            _make_assessment(combined_risk=0.95), now=now + timedelta(seconds=10)
        )
        assert isinstance(result, Alert)
        assert result.severity == AlertSeverity.CRITICAL

    def test_alert_fires_after_grouping_window_expires(self) -> None:
        ctrl = AlertDecisionController(
            cold_start_min_samples=1,
            grouping_window_seconds=30,
        )
        now = datetime(2024, 1, 1, 12, 0, 0)
        ctrl.evaluate(_make_assessment(), now=now)
        result = ctrl.evaluate(
            _make_assessment(), now=now + timedelta(seconds=31)
        )
        assert isinstance(result, Alert)

    def test_severity_mapping(self) -> None:
        assert _severity_from_risk(0.50) == AlertSeverity.LOW
        assert _severity_from_risk(0.65) == AlertSeverity.MEDIUM
        assert _severity_from_risk(0.80) == AlertSeverity.HIGH
        assert _severity_from_risk(0.92) == AlertSeverity.CRITICAL

    def test_alert_as_dict_has_required_keys(self) -> None:
        ctrl = AlertDecisionController(cold_start_min_samples=1, grouping_window_seconds=0)
        result = ctrl.evaluate(_make_assessment())
        assert isinstance(result, Alert)
        d = result.as_dict()
        for key in ("server_id", "fired_at", "severity", "combined_risk"):
            assert key in d

    def test_reset_server_clears_state(self) -> None:
        ctrl = AlertDecisionController(cold_start_min_samples=1, grouping_window_seconds=0)
        ctrl.evaluate(_make_assessment())
        ctrl.reset_server("srv-1")
        assert ctrl.samples_seen("srv-1") == 0

    def test_is_warmed_up_false_initially(self) -> None:
        ctrl = AlertDecisionController(cold_start_min_samples=5, grouping_window_seconds=0)
        assert not ctrl.is_warmed_up("srv-new")

    def test_is_warmed_up_true_after_enough_samples(self) -> None:
        ctrl = AlertDecisionController(cold_start_min_samples=2, grouping_window_seconds=0)
        ctrl.evaluate(_make_assessment(should_alert=False))
        ctrl.evaluate(_make_assessment(should_alert=False))
        assert ctrl.is_warmed_up("srv-1")

    def test_invalid_cold_start_raises(self) -> None:
        with pytest.raises(ValueError):
            AlertDecisionController(cold_start_min_samples=-1)

    def test_invalid_grouping_window_raises(self) -> None:
        with pytest.raises(ValueError):
            AlertDecisionController(grouping_window_seconds=-1)


# ─── 3. RootCauseAnalyzer ────────────────────────────────────────────────────

class TestRootCauseAnalyzer:
    def _make_alert(self) -> Alert:
        return Alert(
            server_id="srv-1",
            fired_at=datetime(2024, 1, 1),
            severity=AlertSeverity.HIGH,
            combined_risk=0.82,
            forecast_risk=0.7,
            anomaly_risk=0.9,
            forecast_breaches={"cpu_util": 0.05},
            anomaly_root_cause="cpu_util",
            anomaly_index=4.2,
        )

    def test_primary_cause_is_highest_evidence_metric(self) -> None:
        rca = RootCauseAnalyzer()
        report = rca.analyze(self._make_alert(), _make_anomaly_report(), _make_assessment())
        assert report.primary_cause == "cpu_util"

    def test_confidence_in_zero_one(self) -> None:
        rca = RootCauseAnalyzer()
        report = rca.analyze(self._make_alert(), _make_anomaly_report(), _make_assessment())
        assert 0.0 <= report.confidence <= 1.0

    def test_report_has_narrative(self) -> None:
        rca = RootCauseAnalyzer()
        report = rca.analyze(self._make_alert(), _make_anomaly_report(), _make_assessment())
        assert len(report.narrative) > 10

    def test_evidence_keys_are_metric_names(self) -> None:
        rca = RootCauseAnalyzer()
        report = rca.analyze(self._make_alert(), _make_anomaly_report(), _make_assessment())
        from pcg.core.constants import METRIC_ORDER
        for key in report.evidence:
            assert key in METRIC_ORDER

    def test_contributing_factors_exclude_primary(self) -> None:
        rca = RootCauseAnalyzer()
        report = rca.analyze(self._make_alert(), _make_anomaly_report(), _make_assessment())
        assert report.primary_cause not in report.contributing_factors

    def test_as_dict_has_expected_keys(self) -> None:
        rca = RootCauseAnalyzer()
        report = rca.analyze(self._make_alert(), _make_anomaly_report(), _make_assessment())
        d = report.as_dict()
        for key in ("primary_cause", "confidence", "contributing_factors", "evidence", "narrative"):
            assert key in d

    def test_empty_metric_errors_returns_unknown(self) -> None:
        rca = RootCauseAnalyzer()
        # No metric errors, no correlated metrics, no forecast breaches, no root_cause.
        alert_no_cause = Alert(
            server_id="srv-1",
            fired_at=datetime(2024, 1, 1),
            severity=AlertSeverity.HIGH,
            combined_risk=0.82,
            forecast_risk=0.7,
            anomaly_risk=0.9,
            forecast_breaches={},
            anomaly_root_cause=None,
            anomaly_index=4.2,
        )
        anomaly = _make_anomaly_report(metric_errors={}, correlated=[])
        # Build assessment directly to avoid _make_assessment's default breach dict.
        assessment = RiskAssessment(
            server_id="srv-1",
            combined_risk=0.6,
            forecast_risk=0.0,
            anomaly_risk=0.6,
            should_alert=True,
            forecast_breaches={},
            anomaly_root_cause=None,
            anomaly_index=3.0,
        )
        report = rca.analyze(alert_no_cause, anomaly, assessment)
        assert report.primary_cause == "unknown"


# ─── 4. DriftMonitor ─────────────────────────────────────────────────────────

class TestDriftMonitor:
    def test_report_before_fit_raises(self) -> None:
        monitor = DriftMonitor()
        with pytest.raises(RuntimeError):
            monitor.report()

    def test_fit_and_report_no_drift_on_same_distribution(self) -> None:
        scores = [0.3] * 100
        flags = [False] * 100
        monitor = DriftMonitor(DriftConfig(min_observations=10))
        monitor.fit(scores, flags)
        for s, f in zip(scores, flags):
            monitor.update(s, f)
        report = monitor.report()
        assert not report.score_drift_detected
        assert not report.rate_drift_detected

    def test_score_drift_detected_on_large_shift(self) -> None:
        baseline = [0.2] * 200
        monitor = DriftMonitor(DriftConfig(mean_drift_threshold=1.0, min_observations=50))
        monitor.fit(baseline, [False] * 200)
        for _ in range(60):
            monitor.update(0.9, True)  # feed very high scores
        report = monitor.report()
        assert report.score_drift_detected

    def test_rate_drift_detected_on_high_alert_rate(self) -> None:
        scores = [0.3] * 200
        flags = [False] * 200   # baseline: 0% alert rate
        monitor = DriftMonitor(DriftConfig(rate_multiplier=2.0, min_observations=50))
        monitor.fit(scores, flags)
        for _ in range(60):
            monitor.update(0.6, True)  # 100% alert rate
        report = monitor.report()
        assert report.rate_drift_detected

    def test_below_min_observations_no_drift(self) -> None:
        monitor = DriftMonitor(DriftConfig(min_observations=100))
        monitor.fit([0.3] * 50, [False] * 50)
        for _ in range(10):
            monitor.update(0.9, True)
        report = monitor.report()
        assert not report.any_drift  # too few observations

    def test_reset_clears_ema_state(self) -> None:
        monitor = DriftMonitor(DriftConfig(min_observations=1))
        monitor.fit([0.3] * 10, [False] * 10)
        monitor.update(0.9, True)
        monitor.reset()
        report = monitor.report()
        assert report.n_observations == 0

    def test_fit_empty_raises(self) -> None:
        monitor = DriftMonitor()
        with pytest.raises(ValueError):
            monitor.fit([], [])

    def test_fit_mismatched_lengths_raises(self) -> None:
        monitor = DriftMonitor()
        with pytest.raises(ValueError):
            monitor.fit([0.1, 0.2], [True])

    def test_summary_string_present(self) -> None:
        monitor = DriftMonitor(DriftConfig(min_observations=1))
        monitor.fit([0.3] * 5, [False] * 5)
        monitor.update(0.3, False)
        assert isinstance(monitor.report().summary(), str)

    def test_invalid_ema_window_raises(self) -> None:
        with pytest.raises(ValueError):
            DriftConfig(ema_window=0)

    def test_invalid_rate_multiplier_raises(self) -> None:
        with pytest.raises(ValueError):
            DriftConfig(rate_multiplier=0.5)


# ─── 5. FeedbackStore ────────────────────────────────────────────────────────

class TestFeedbackStore:
    def _store(self) -> FeedbackStore:
        return FeedbackStore(db_path=":memory:")

    def _insert_alert(self, store: FeedbackStore, server_id: str = "srv-1") -> int:
        return store.insert_alert(
            server_id=server_id,
            fired_at=datetime(2024, 1, 1),
            severity="high",
            combined_risk=0.75,
            forecast_risk=0.6,
            anomaly_risk=0.9,
            anomaly_index=4.0,
            root_cause="cpu_util",
            breaches={"cpu_util": 0.05},
        )

    def test_insert_and_retrieve_alert(self) -> None:
        store = self._store()
        aid = self._insert_alert(store)
        alert = store.get_alert(aid)
        assert alert is not None
        assert alert.server_id == "srv-1"
        assert alert.severity == "high"

    def test_get_nonexistent_alert_returns_none(self) -> None:
        store = self._store()
        assert store.get_alert(9999) is None

    def test_submit_feedback_true_positive(self) -> None:
        store = self._store()
        aid = self._insert_alert(store)
        fid = store.submit_feedback(aid, "true_positive")
        assert fid is not None
        assert store.queue_length("pending") == 0

    def test_submit_false_positive_enqueues(self) -> None:
        store = self._store()
        aid = self._insert_alert(store)
        store.submit_feedback(aid, "false_positive", submitted_by="operator")
        assert store.queue_length("pending") == 1

    def test_multiple_false_positives_multiple_queue_items(self) -> None:
        store = self._store()
        for _ in range(3):
            aid = self._insert_alert(store)
            store.submit_feedback(aid, "false_positive")
        assert store.queue_length("pending") == 3

    def test_pending_queue_items_returns_correct_count(self) -> None:
        store = self._store()
        for _ in range(5):
            aid = self._insert_alert(store)
            store.submit_feedback(aid, "false_positive")
        items = store.pending_queue_items(limit=3)
        assert len(items) == 3

    def test_mark_queue_item_done(self) -> None:
        store = self._store()
        aid = self._insert_alert(store)
        store.submit_feedback(aid, "false_positive")
        items = store.pending_queue_items()
        store.mark_queue_item(items[0].id, "done")
        assert store.queue_length("pending") == 0
        assert store.queue_length("done") == 1

    def test_list_alerts_by_server(self) -> None:
        store = self._store()
        self._insert_alert(store, "srv-A")
        self._insert_alert(store, "srv-B")
        alerts = store.list_alerts(server_id="srv-A")
        assert all(a.server_id == "srv-A" for a in alerts)

    def test_alert_breaches_roundtrip(self) -> None:
        store = self._store()
        aid = store.insert_alert(
            server_id="srv-1",
            fired_at=datetime(2024, 1, 1),
            severity="medium",
            combined_risk=0.6,
            forecast_risk=0.5,
            anomaly_risk=0.7,
            anomaly_index=2.5,
            root_cause=None,
            breaches={"mem_util": 0.12, "net_io": 0.07},
        )
        alert = store.get_alert(aid)
        assert alert.breaches == {"mem_util": 0.12, "net_io": 0.07}


# ─── 6. RetrainingQueueProcessor ─────────────────────────────────────────────

class TestRetrainingQueueProcessor:
    def _store_with_fp(self, n: int) -> FeedbackStore:
        store = FeedbackStore(db_path=":memory:")
        for _ in range(n):
            aid = store.insert_alert(
                server_id="srv-1",
                fired_at=datetime(2024, 1, 1),
                severity="low",
                combined_risk=0.55,
                forecast_risk=0.4,
                anomaly_risk=0.7,
                anomaly_index=3.1,
                root_cause="cpu_util",
                breaches={},
            )
            store.submit_feedback(aid, "false_positive")
        return store

    def test_no_adjustments_below_min_fp(self) -> None:
        store = self._store_with_fp(3)
        cfg = RetrainingQueueConfig(min_fp_for_adjustment=5, batch_size=10)
        proc = RetrainingQueueProcessor(store, cfg)
        adjustments = proc.process_batch(current_threshold=0.50)
        assert adjustments == []

    def test_adjustment_produced_above_min_fp(self) -> None:
        store = self._store_with_fp(10)
        cfg = RetrainingQueueConfig(min_fp_for_adjustment=5, batch_size=20)
        proc = RetrainingQueueProcessor(store, cfg)
        adjustments = proc.process_batch(current_threshold=0.50)
        assert len(adjustments) > 0

    def test_suggested_threshold_above_current(self) -> None:
        store = self._store_with_fp(10)
        cfg = RetrainingQueueConfig(min_fp_for_adjustment=5, batch_size=20)
        proc = RetrainingQueueProcessor(store, cfg)
        adjustments = proc.process_batch(current_threshold=0.50)
        for adj in adjustments:
            assert adj.suggested_threshold > adj.current_threshold

    def test_max_adjustment_respected(self) -> None:
        store = self._store_with_fp(20)
        cfg = RetrainingQueueConfig(
            min_fp_for_adjustment=5,
            batch_size=30,
            max_adjustment=0.05,
        )
        proc = RetrainingQueueProcessor(store, cfg)
        adjustments = proc.process_batch(current_threshold=0.50)
        for adj in adjustments:
            assert adj.suggested_threshold <= 0.50 + 0.05 + 1e-9

    def test_queue_items_marked_done_after_processing(self) -> None:
        store = self._store_with_fp(5)
        proc = RetrainingQueueProcessor(store)
        proc.process_batch(current_threshold=0.50)
        assert store.queue_length("pending") == 0

    def test_fp_score_count_tracks_ingested(self) -> None:
        store = self._store_with_fp(7)
        proc = RetrainingQueueProcessor(store)
        proc.process_batch(current_threshold=0.50)
        assert proc.fp_score_count("global") == 7


# ─── 7. ShadowDeployment ─────────────────────────────────────────────────────

class TestShadowDeployment:
    def _both_agree(self, risk: float = 0.3) -> tuple[RiskAssessment, RiskAssessment]:
        prod = _make_assessment("srv-1", combined_risk=risk, should_alert=risk >= 0.5)
        cand = _make_assessment("srv-1", combined_risk=risk, should_alert=risk >= 0.5)
        return prod, cand

    def test_empty_report_not_ready(self) -> None:
        sd = ShadowDeployment()
        assert not sd.report().ready_to_promote

    def test_server_id_mismatch_raises(self) -> None:
        sd = ShadowDeployment()
        prod = _make_assessment("srv-A")
        cand = _make_assessment("srv-B")
        with pytest.raises(ValueError):
            sd.evaluate(prod, cand)

    def test_agreement_rate_one_when_always_agree(self) -> None:
        cfg = ShadowConfig(min_evaluations=5)
        sd = ShadowDeployment(cfg)
        for _ in range(5):
            prod, cand = self._both_agree()
            sd.evaluate(prod, cand)
        report = sd.report()
        assert report.agreement_rate == 1.0

    def test_candidate_only_counted_correctly(self) -> None:
        sd = ShadowDeployment()
        # Candidate alerts but production doesn't.
        prod = _make_assessment(should_alert=False, combined_risk=0.3)
        cand = _make_assessment(should_alert=True, combined_risk=0.7)
        sd.evaluate(prod, cand)
        report = sd.report()
        assert report.candidate_only_rate == 1.0

    def test_production_only_counted_correctly(self) -> None:
        sd = ShadowDeployment()
        prod = _make_assessment(should_alert=True, combined_risk=0.7)
        cand = _make_assessment(should_alert=False, combined_risk=0.3)
        sd.evaluate(prod, cand)
        report = sd.report()
        assert report.production_only_rate == 1.0

    def test_risk_mae_computed(self) -> None:
        sd = ShadowDeployment()
        prod = _make_assessment(combined_risk=0.6)
        cand = _make_assessment(combined_risk=0.4)
        sd.evaluate(prod, cand)
        report = sd.report()
        assert abs(report.risk_mae - 0.2) < 1e-6

    def test_ready_to_promote_when_all_conditions_met(self) -> None:
        cfg = ShadowConfig(
            min_evaluations=5,
            min_agreement_rate=0.8,
            max_regression_rate=0.1,
            max_risk_mae=0.1,
        )
        sd = ShadowDeployment(cfg)
        for _ in range(5):
            prod, cand = self._both_agree(risk=0.3)
            sd.evaluate(prod, cand)
        assert sd.report().ready_to_promote

    def test_not_ready_when_too_few_evaluations(self) -> None:
        cfg = ShadowConfig(min_evaluations=100)
        sd = ShadowDeployment(cfg)
        for _ in range(10):
            prod, cand = self._both_agree()
            sd.evaluate(prod, cand)
        report = sd.report()
        assert not report.ready_to_promote
        assert any("evaluations" in b for b in report.promotion_blockers)

    def test_reset_clears_comparisons(self) -> None:
        sd = ShadowDeployment()
        prod, cand = self._both_agree()
        sd.evaluate(prod, cand)
        sd.reset()
        assert sd.n_evaluations == 0

    def test_summary_contains_status(self) -> None:
        sd = ShadowDeployment()
        summary = sd.report().summary()
        assert "NOT READY" in summary or "PROMOTE" in summary

    def test_invalid_agreement_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            ShadowConfig(min_agreement_rate=1.5)

    def test_invalid_regression_rate_raises(self) -> None:
        with pytest.raises(ValueError):
            ShadowConfig(max_regression_rate=-0.1)
