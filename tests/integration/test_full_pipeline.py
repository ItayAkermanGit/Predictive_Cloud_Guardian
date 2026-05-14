# Integration tests: full pipeline from raw metrics to alert decision.
#
# These tests wire real module instances together (no mocks) and verify
# that the end-to-end data flow works correctly. They use small synthetic
# datasets and minimal training epochs to keep runtime under 60 s.

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
import torch

from pcg.anomaly.anomaly_scorer import AnomalyScorer
from pcg.anomaly.autoencoder import ConvAutoencoder
from pcg.anomaly.correlation_detector import CorrelationDetector
from pcg.anomaly.synthetic_normal import NormalWindowDataset, build_normal_dataframe
from pcg.anomaly.train_autoencoder import AutoencoderTrainConfig, train_autoencoder
from pcg.controller.alert_decision import Alert, AlertDecisionController, ColdStartResult
from pcg.controller.drift_monitor import DriftMonitor
from pcg.controller.feedback_store import FeedbackStore
from pcg.controller.rca import RootCauseAnalyzer
from pcg.controller.risk_scorer import CombinedRiskScorer, RiskConfig
from pcg.core.constants import METRIC_ORDER, N_METRICS, WINDOW_SIZE
from pcg.data.normalizer import MinMaxNormalizer
from pcg.data.pipeline import DataPipeline
from pcg.data.synthetic import FailureInjection, SyntheticConfig, SyntheticMetricGenerator
from pcg.data.tsdb_client import InMemoryTSDBClient
from pcg.models.lstm_attention import LSTMForecaster
from pcg.orchestrator import PCGOrchestrator, TickResult, build_orchestrator
from pcg.training.dataset import ForecastingDataset
from pcg.training.train_lstm import TrainConfig, train_forecaster


# ─── Shared fixtures ──────────────────────────────────────────────────────────

START = datetime(2024, 1, 1, 0, 0, 0)
SERVER = "test-server-01"
TRAIN_MINUTES = 250   # enough for WINDOW_SIZE=60 + training set
DEMO_MINUTES  = 300


@pytest.fixture(scope="module")
def normal_df():
    cfg = SyntheticConfig(seed=0, noise_std=0.03, failures=[])
    return SyntheticMetricGenerator(cfg).generate(START, DEMO_MINUTES)


@pytest.fixture(scope="module")
def normalizer(normal_df):
    n = MinMaxNormalizer()
    n.fit(normal_df)
    return n


@pytest.fixture(scope="module")
def trained_lstm(normal_df, normalizer):
    ds = ForecastingDataset(normal_df, normalizer, stride=3)
    model = LSTMForecaster(n_metrics=N_METRICS, hidden_size=32, num_layers=1)
    train_forecaster(model, ds, config=TrainConfig(epochs=3, batch_size=32, seed=0))
    return model


@pytest.fixture(scope="module")
def trained_ae_and_scorer(normal_df, normalizer):
    ds = NormalWindowDataset(normal_df, normalizer, stride=5)
    model = ConvAutoencoder(n_metrics=N_METRICS, latent_dim=8)
    train_autoencoder(model, ds, config=AutoencoderTrainConfig(epochs=3, batch_size=32, seed=0))
    scorer = AnomalyScorer(model, n_sigma=3.0)
    scorer.fit_threshold(ds.all_windows)
    return model, scorer


@pytest.fixture(scope="module")
def corr_detector(normal_df, normalizer):
    ds = NormalWindowDataset(normal_df, normalizer, stride=5)
    det = CorrelationDetector(corr_threshold=0.5)
    det.fit(ds.all_windows)
    return det


def _make_tsdb(df, server_id: str) -> InMemoryTSDBClient:
    client = InMemoryTSDBClient()
    client.upsert(server_id, df)
    return client


@pytest.fixture(scope="module")
def orchestrator(normal_df, normalizer, trained_lstm, trained_ae_and_scorer, corr_detector):
    _, scorer = trained_ae_and_scorer
    tsdb = _make_tsdb(normal_df, SERVER)
    pipeline = DataPipeline(tsdb, normalizer)
    orch = build_orchestrator(
        pipeline=pipeline,
        forecaster_model=trained_lstm,
        normalizer=normalizer,
        anomaly_scorer=scorer,
        correlation_detector=corr_detector,
        risk_config=RiskConfig(risk_threshold=0.5),
        db_path=":memory:",
    )
    return orch


# ─── Data pipeline integration ────────────────────────────────────────────────

class TestDataPipelineIntegration:
    def test_prepare_window_returns_correct_tensor_shape(
        self, normal_df, normalizer
    ) -> None:
        tsdb = _make_tsdb(normal_df, SERVER)
        pipeline = DataPipeline(tsdb, normalizer)
        now = START + timedelta(minutes=WINDOW_SIZE + 5)
        window = pipeline.prepare_window(SERVER, end=now)
        assert tuple(window.tensor.shape) == (1, WINDOW_SIZE, N_METRICS)

    def test_prepare_window_values_in_zero_one(
        self, normal_df, normalizer
    ) -> None:
        tsdb = _make_tsdb(normal_df, SERVER)
        pipeline = DataPipeline(tsdb, normalizer)
        now = START + timedelta(minutes=WINDOW_SIZE + 5)
        window = pipeline.prepare_window(SERVER, end=now)
        assert window.tensor.min().item() >= -0.1   # small margin for smoothing edge
        assert window.tensor.max().item() <= 1.1


# ─── Forecasting integration ──────────────────────────────────────────────────

class TestForecastingIntegration:
    def test_forecast_output_shapes(
        self, normal_df, normalizer, trained_lstm
    ) -> None:
        from pcg.inference.forecaster import Forecaster
        from pcg.core.constants import HORIZON_MINUTES

        tsdb = _make_tsdb(normal_df, SERVER)
        pipeline = DataPipeline(tsdb, normalizer)
        forecaster = Forecaster(trained_lstm, normalizer)
        now = START + timedelta(minutes=WINDOW_SIZE + 5)
        window = pipeline.prepare_window(SERVER, end=now)
        output = forecaster.predict(window)
        assert output.predicted_normalized.shape == (HORIZON_MINUTES, N_METRICS)
        assert output.attention_weights.shape == (WINDOW_SIZE,)

    def test_forecast_attention_sums_to_one(
        self, normal_df, normalizer, trained_lstm
    ) -> None:
        import numpy as np
        from pcg.inference.forecaster import Forecaster

        tsdb = _make_tsdb(normal_df, SERVER)
        pipeline = DataPipeline(tsdb, normalizer)
        forecaster = Forecaster(trained_lstm, normalizer)
        now = START + timedelta(minutes=WINDOW_SIZE + 5)
        window = pipeline.prepare_window(SERVER, end=now)
        output = forecaster.predict(window)
        assert abs(output.attention_weights.sum() - 1.0) < 1e-4


# ─── Anomaly detection integration ───────────────────────────────────────────

class TestAnomalyDetectionIntegration:
    def test_normal_window_low_anomaly_score(
        self, normal_df, normalizer, trained_ae_and_scorer
    ) -> None:
        ds = NormalWindowDataset(normal_df, normalizer, stride=5)
        _, scorer = trained_ae_and_scorer
        # Normal window should score well below the threshold on average.
        scores = [scorer.score(ds[i]).score for i in range(min(20, len(ds)))]
        mean_score = sum(scores) / len(scores)
        assert mean_score < scorer.threshold * 2

    def test_anomalous_window_higher_score_than_normal(
        self, normal_df, normalizer, trained_ae_and_scorer
    ) -> None:
        from pcg.data.synthetic import FailureInjection, SyntheticConfig, SyntheticMetricGenerator
        _, scorer = trained_ae_and_scorer

        # Build a clearly anomalous window (large spike).
        normal_window = NormalWindowDataset(
            normal_df, normalizer, stride=5
        ).all_windows[0]  # (N_METRICS, WINDOW_SIZE)

        # Spike: add 0.5 to cpu_util channel.
        anomalous_window = normal_window.clone()
        anomalous_window[0, :] = (anomalous_window[0, :] + 0.5).clamp(0, 1)

        normal_score = scorer.score(normal_window).score
        anomaly_score = scorer.score(anomalous_window).score
        assert anomaly_score > normal_score

    def test_metric_errors_keys_match_metric_order(
        self, normal_df, normalizer, trained_ae_and_scorer
    ) -> None:
        ds = NormalWindowDataset(normal_df, normalizer, stride=5)
        _, scorer = trained_ae_and_scorer
        result = scorer.score(ds[0])
        assert set(result.metric_errors.keys()) == set(METRIC_ORDER)

    def test_correlation_detector_fitted(self, corr_detector) -> None:
        mat = corr_detector.correlation_matrix()
        assert mat.shape == (N_METRICS, N_METRICS)
        import numpy as np
        np.testing.assert_allclose(mat.diagonal(), 1.0, atol=1e-5)


# ─── Risk scoring integration ─────────────────────────────────────────────────

class TestRiskScoringIntegration:
    def test_combined_risk_in_zero_one(
        self, normal_df, normalizer, trained_lstm, trained_ae_and_scorer
    ) -> None:
        from pcg.inference.forecaster import Forecaster
        from pcg.anomaly.anomaly_api import AnomalyDetectionAPI

        tsdb = _make_tsdb(normal_df, SERVER)
        pipeline = DataPipeline(tsdb, normalizer)
        forecaster = Forecaster(trained_lstm, normalizer)
        _, scorer = trained_ae_and_scorer
        anomaly_api = AnomalyDetectionAPI(scorer)
        risk_scorer = CombinedRiskScorer()

        now = START + timedelta(minutes=WINDOW_SIZE + 5)
        window = pipeline.prepare_window(SERVER, end=now)
        forecast = forecaster.predict(window)
        window_cf = window.tensor.squeeze(0).T
        anomaly = anomaly_api.analyze(window_cf)
        assessment = risk_scorer.assess(forecast, anomaly)

        assert 0.0 <= assessment.combined_risk <= 1.0
        assert 0.0 <= assessment.forecast_risk <= 1.0
        assert 0.0 <= assessment.anomaly_risk <= 1.0


# ─── Orchestrator integration ─────────────────────────────────────────────────

class TestOrchestratorIntegration:
    def test_tick_returns_tick_result(self, orchestrator) -> None:
        now = START + timedelta(minutes=WINDOW_SIZE + 5)
        result = orchestrator.tick(SERVER, now=now)
        assert isinstance(result, TickResult)

    def test_tick_result_has_all_fields(self, orchestrator) -> None:
        now = START + timedelta(minutes=WINDOW_SIZE + 10)
        result = orchestrator.tick(SERVER, now=now)
        assert result.server_id == SERVER
        assert result.forecast is not None
        assert result.anomaly is not None
        assert result.assessment is not None

    def test_cold_start_suppresses_early_alerts(self, orchestrator) -> None:
        # A server that has never been seen has 0 accumulated samples → not warmed up.
        assert not orchestrator._alert_ctrl.is_warmed_up("brand-new-server-xyz")

    def test_multiple_ticks_accumulate_drift_observations(
        self, orchestrator
    ) -> None:
        start_n = orchestrator._drift._n_observations
        for i in range(5):
            now = START + timedelta(minutes=WINDOW_SIZE + 20 + i)
            orchestrator.tick(SERVER, now=now)
        assert orchestrator._drift._n_observations >= start_n + 5

    def test_tick_summary_is_string(self, orchestrator) -> None:
        now = START + timedelta(minutes=WINDOW_SIZE + 30)
        result = orchestrator.tick(SERVER, now=now)
        assert isinstance(result.summary(), str)
        assert SERVER in result.summary()


# ─── Feedback loop integration ────────────────────────────────────────────────

class TestFeedbackLoopIntegration:
    def test_false_positive_feedback_enqueues(self) -> None:
        store = FeedbackStore(":memory:")
        aid = store.insert_alert(
            server_id=SERVER,
            fired_at=datetime(2024, 1, 1),
            severity="high",
            combined_risk=0.75,
            forecast_risk=0.6,
            anomaly_risk=0.9,
            anomaly_index=4.0,
            root_cause="cpu_util",
            breaches={"cpu_util": 0.05},
        )
        store.submit_feedback(aid, "false_positive", submitted_by="integration_test")
        assert store.queue_length("pending") == 1

    def test_retraining_processor_drains_queue(self) -> None:
        from pcg.controller.retraining_queue import RetrainingQueueProcessor

        store = FeedbackStore(":memory:")
        for _ in range(8):
            aid = store.insert_alert(
                server_id=SERVER,
                fired_at=datetime(2024, 1, 1),
                severity="medium",
                combined_risk=0.60,
                forecast_risk=0.5,
                anomaly_risk=0.7,
                anomaly_index=3.5,
                root_cause="mem_util",
                breaches={},
            )
            store.submit_feedback(aid, "false_positive")

        proc = RetrainingQueueProcessor(store)
        proc.process_batch(current_threshold=0.50)
        assert store.queue_length("pending") == 0

    def test_drift_monitor_detects_shift(self) -> None:
        monitor = DriftMonitor()
        monitor.fit([0.2] * 200, [False] * 200)
        for _ in range(100):
            monitor.update(0.85, True)
        report = monitor.report()
        assert report.any_drift


# ─── RCA integration ──────────────────────────────────────────────────────────

class TestRCAIntegration:
    def test_rca_primary_cause_in_metric_order(
        self, normal_df, normalizer, trained_lstm, trained_ae_and_scorer, corr_detector
    ) -> None:
        from pcg.controller.alert_decision import Alert, AlertSeverity
        from pcg.anomaly.anomaly_api import AnomalyDetectionAPI
        from pcg.inference.forecaster import Forecaster

        tsdb = _make_tsdb(normal_df, SERVER)
        pipeline = DataPipeline(tsdb, normalizer)
        forecaster = Forecaster(trained_lstm, normalizer)
        _, scorer = trained_ae_and_scorer
        anomaly_api = AnomalyDetectionAPI(scorer, corr_detector)
        risk_scorer = CombinedRiskScorer(RiskConfig(risk_threshold=0.0))  # always alert

        now = START + timedelta(minutes=WINDOW_SIZE + 5)
        window = pipeline.prepare_window(SERVER, end=now)
        forecast = forecaster.predict(window)
        window_cf = window.tensor.squeeze(0).T
        anomaly = anomaly_api.analyze(window_cf)
        assessment = risk_scorer.assess(forecast, anomaly)

        alert = Alert(
            server_id=SERVER,
            fired_at=now,
            severity=AlertSeverity.HIGH,
            combined_risk=assessment.combined_risk,
            forecast_risk=assessment.forecast_risk,
            anomaly_risk=assessment.anomaly_risk,
            forecast_breaches=assessment.forecast_breaches,
            anomaly_root_cause=assessment.anomaly_root_cause,
            anomaly_index=assessment.anomaly_index,
        )

        rca = RootCauseAnalyzer()
        report = rca.analyze(alert, anomaly, assessment)
        assert report.primary_cause in METRIC_ORDER or report.primary_cause == "unknown"
        assert 0.0 <= report.confidence <= 1.0
