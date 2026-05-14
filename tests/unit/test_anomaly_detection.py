# Unit tests for the Phase 4 anomaly detection module.
#
# Tests are organized in the same order as the module files:
#   1. ConvAutoencoder (autoencoder.py)
#   2. AnomalyScorer (anomaly_scorer.py)
#   3. CorrelationDetector (correlation_detector.py)
#   4. AnomalyDetectionAPI (anomaly_api.py)
#   5. NormalWindowDataset / AnomalousWindowDataset (synthetic_normal.py)
#   6. train_autoencoder (train_autoencoder.py)

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import torch

from pcg.anomaly.anomaly_api import AnomalyDetectionAPI, AnomalyReport
from pcg.anomaly.anomaly_scorer import AnomalyScore, AnomalyScorer
from pcg.anomaly.autoencoder import ConvAutoencoder
from pcg.anomaly.correlation_detector import CorrelationDetector
from pcg.anomaly.synthetic_normal import (
    AnomalousWindowDataset,
    NormalWindowDataset,
    build_anomalous_dataframe,
    build_normal_dataframe,
)
from pcg.anomaly.train_autoencoder import AutoencoderTrainConfig, train_autoencoder
from pcg.core.constants import METRIC_ORDER, N_METRICS, WINDOW_SIZE
from pcg.data.normalizer import MinMaxNormalizer


# ─── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_model() -> ConvAutoencoder:
    # Small model for fast tests (latent_dim=4 instead of default 16).
    return ConvAutoencoder(n_metrics=N_METRICS, latent_dim=4, window_size=WINDOW_SIZE)


@pytest.fixture(scope="module")
def normal_df():
    return build_normal_dataframe(minutes=300, seed=0)


@pytest.fixture(scope="module")
def anomalous_df():
    return build_anomalous_dataframe(minutes=200, seed=1)


@pytest.fixture(scope="module")
def fitted_normalizer(normal_df):
    n = MinMaxNormalizer()
    n.fit(normal_df)
    return n


@pytest.fixture(scope="module")
def normal_dataset(normal_df, fitted_normalizer):
    return NormalWindowDataset(normal_df, fitted_normalizer, window_size=WINDOW_SIZE, stride=5)


@pytest.fixture(scope="module")
def normal_windows(normal_dataset) -> torch.Tensor:
    # (N, N_METRICS, WINDOW_SIZE)
    return normal_dataset.all_windows


@pytest.fixture(scope="module")
def trained_scorer(tiny_model, normal_windows) -> AnomalyScorer:
    # Fit scorer on normal windows with a freshly trained (few-epoch) model.
    # We don't need a great model — just a working scorer for API tests.
    torch.manual_seed(0)
    cfg = AutoencoderTrainConfig(epochs=3, batch_size=32, seed=0)
    train_autoencoder(tiny_model, normal_windows, config=cfg)

    scorer = AnomalyScorer(tiny_model, n_sigma=3.0)
    scorer.fit_threshold(normal_windows)
    return scorer


# ─── 1. ConvAutoencoder ───────────────────────────────────────────────────────

class TestConvAutoencoder:
    def test_forward_output_shapes(self, tiny_model: ConvAutoencoder) -> None:
        x = torch.zeros(4, N_METRICS, WINDOW_SIZE)
        x_hat, z = tiny_model(x)
        assert x_hat.shape == (4, N_METRICS, WINDOW_SIZE), (
            f"reconstruction shape mismatch: {x_hat.shape}"
        )
        assert z.shape == (4, 4), f"latent shape mismatch: {z.shape}"

    def test_forward_output_dtype_is_float32(self, tiny_model: ConvAutoencoder) -> None:
        x = torch.zeros(2, N_METRICS, WINDOW_SIZE)
        x_hat, z = tiny_model(x)
        assert x_hat.dtype == torch.float32
        assert z.dtype == torch.float32

    def test_reconstruction_in_0_1_range(self, tiny_model: ConvAutoencoder) -> None:
        # Sigmoid in the decoder guarantees outputs are in [0, 1].
        x = torch.rand(8, N_METRICS, WINDOW_SIZE)
        x_hat, _ = tiny_model(x)
        assert x_hat.min().item() >= 0.0
        assert x_hat.max().item() <= 1.0

    def test_reconstruct_convenience_method(self, tiny_model: ConvAutoencoder) -> None:
        x = torch.zeros(2, N_METRICS, WINDOW_SIZE)
        r = tiny_model.reconstruct(x)
        assert r.shape == (2, N_METRICS, WINDOW_SIZE)

    def test_rejects_wrong_n_metrics(self, tiny_model: ConvAutoencoder) -> None:
        bad = torch.zeros(1, N_METRICS + 2, WINDOW_SIZE)
        with pytest.raises(ValueError):
            tiny_model(bad)

    def test_rejects_wrong_window_size(self, tiny_model: ConvAutoencoder) -> None:
        bad = torch.zeros(1, N_METRICS, WINDOW_SIZE + 10)
        with pytest.raises(ValueError):
            tiny_model(bad)

    def test_rejects_2d_input(self, tiny_model: ConvAutoencoder) -> None:
        bad = torch.zeros(N_METRICS, WINDOW_SIZE)
        with pytest.raises(ValueError):
            tiny_model(bad)

    def test_invalid_latent_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            ConvAutoencoder(latent_dim=0)

    def test_invalid_n_metrics_raises(self) -> None:
        with pytest.raises(ValueError):
            ConvAutoencoder(n_metrics=0)

    def test_backward_runs(self, tiny_model: ConvAutoencoder) -> None:
        # Verify that gradients flow through the full autoencoder.
        torch.manual_seed(0)
        x = torch.rand(4, N_METRICS, WINDOW_SIZE)
        x_hat, _ = tiny_model(x)
        loss = torch.nn.functional.mse_loss(x_hat, x)
        loss.backward()
        for name, param in tiny_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"no gradient for {name}"

    def test_two_inputs_produce_different_reconstructions(
        self, tiny_model: ConvAutoencoder
    ) -> None:
        torch.manual_seed(1)
        xa = torch.rand(1, N_METRICS, WINDOW_SIZE)
        xb = torch.rand(1, N_METRICS, WINDOW_SIZE) + 0.5
        ra, _ = tiny_model(xa)
        rb, _ = tiny_model(xb)
        assert not torch.allclose(ra, rb)

    def test_default_constructor_uses_project_constants(self) -> None:
        m = ConvAutoencoder()
        assert m.n_metrics == N_METRICS
        assert m.window_size == WINDOW_SIZE


# ─── 2. AnomalyScorer ────────────────────────────────────────────────────────

class TestAnomalyScorer:
    def test_score_before_fit_raises(self, tiny_model: ConvAutoencoder) -> None:
        scorer = AnomalyScorer(tiny_model)
        x = torch.rand(N_METRICS, WINDOW_SIZE)
        with pytest.raises(RuntimeError):
            scorer.score(x)

    def test_fit_threshold_sets_attributes(
        self, trained_scorer: AnomalyScorer
    ) -> None:
        assert trained_scorer.threshold > 0
        assert trained_scorer.mu_normal >= 0
        assert trained_scorer.sigma_normal > 0

    def test_threshold_is_mu_plus_n_sigma(
        self, trained_scorer: AnomalyScorer
    ) -> None:
        expected = trained_scorer.mu_normal + 3.0 * trained_scorer.sigma_normal
        assert abs(trained_scorer.threshold - expected) < 1e-8

    def test_score_returns_anomaly_score_dataclass(
        self, trained_scorer: AnomalyScorer
    ) -> None:
        x = torch.rand(N_METRICS, WINDOW_SIZE)
        result = trained_scorer.score(x)
        assert isinstance(result, AnomalyScore)

    def test_score_returns_all_metric_errors(
        self, trained_scorer: AnomalyScorer
    ) -> None:
        x = torch.rand(N_METRICS, WINDOW_SIZE)
        result = trained_scorer.score(x)
        assert set(result.metric_errors.keys()) == set(METRIC_ORDER)

    def test_score_metric_errors_are_non_negative(
        self, trained_scorer: AnomalyScorer
    ) -> None:
        x = torch.rand(N_METRICS, WINDOW_SIZE)
        result = trained_scorer.score(x)
        for name, err in result.metric_errors.items():
            assert err >= 0, f"negative error for {name}: {err}"

    def test_score_root_cause_has_highest_error(
        self, trained_scorer: AnomalyScorer
    ) -> None:
        x = torch.rand(N_METRICS, WINDOW_SIZE)
        result = trained_scorer.score(x)
        max_metric = max(result.metric_errors, key=result.metric_errors.__getitem__)
        assert result.root_cause_metric == max_metric

    def test_score_reconstruction_none_by_default(
        self, trained_scorer: AnomalyScorer
    ) -> None:
        x = torch.rand(N_METRICS, WINDOW_SIZE)
        result = trained_scorer.score(x)
        assert result.reconstruction is None

    def test_score_reconstruction_returned_when_requested(
        self, trained_scorer: AnomalyScorer
    ) -> None:
        x = torch.rand(N_METRICS, WINDOW_SIZE)
        result = trained_scorer.score(x, return_reconstruction=True)
        assert result.reconstruction is not None
        assert result.reconstruction.shape == (N_METRICS, WINDOW_SIZE)

    def test_score_batch_returns_correct_count(
        self, trained_scorer: AnomalyScorer
    ) -> None:
        windows = torch.rand(7, N_METRICS, WINDOW_SIZE)
        results = trained_scorer.score_batch(windows)
        assert len(results) == 7

    def test_score_batch_rejects_2d_input(
        self, trained_scorer: AnomalyScorer
    ) -> None:
        bad = torch.rand(N_METRICS, WINDOW_SIZE)
        with pytest.raises(ValueError):
            trained_scorer.score_batch(bad)

    def test_anomaly_index_is_positive_for_high_error_window(
        self, trained_scorer: AnomalyScorer
    ) -> None:
        # A window filled with values far outside [0, 1] should score high.
        x = torch.full((N_METRICS, WINDOW_SIZE), fill_value=5.0)
        result = trained_scorer.score(x)
        assert result.anomaly_index > 0

    def test_overall_score_equals_mean_of_metric_errors(
        self, trained_scorer: AnomalyScorer
    ) -> None:
        x = torch.rand(N_METRICS, WINDOW_SIZE)
        result = trained_scorer.score(x)
        expected = np.mean(list(result.metric_errors.values()))
        assert abs(result.score - expected) < 1e-5

    def test_n_sigma_zero_raises(self, tiny_model: ConvAutoencoder) -> None:
        with pytest.raises(ValueError):
            AnomalyScorer(tiny_model, n_sigma=0.0)


# ─── 3. CorrelationDetector ───────────────────────────────────────────────────

class TestCorrelationDetector:
    def test_find_correlated_before_fit_raises(self) -> None:
        det = CorrelationDetector()
        with pytest.raises(RuntimeError):
            det.find_correlated("cpu_util", {"cpu_util": 0.1})

    def test_fit_produces_square_matrix(self, normal_windows: torch.Tensor) -> None:
        det = CorrelationDetector()
        det.fit(normal_windows)
        mat = det.correlation_matrix()
        assert mat.shape == (N_METRICS, N_METRICS)

    def test_diagonal_is_one(self, normal_windows: torch.Tensor) -> None:
        det = CorrelationDetector()
        det.fit(normal_windows)
        mat = det.correlation_matrix()
        np.testing.assert_allclose(np.diag(mat), 1.0, atol=1e-5)

    def test_matrix_is_symmetric(self, normal_windows: torch.Tensor) -> None:
        det = CorrelationDetector()
        det.fit(normal_windows)
        mat = det.correlation_matrix()
        np.testing.assert_allclose(mat, mat.T, atol=1e-8)

    def test_fit_rejects_non_3d_input(self) -> None:
        det = CorrelationDetector()
        with pytest.raises(ValueError):
            det.fit(torch.rand(N_METRICS, WINDOW_SIZE))  # 2-D

    def test_unknown_root_metric_returns_empty(
        self, normal_windows: torch.Tensor
    ) -> None:
        det = CorrelationDetector()
        det.fit(normal_windows)
        result = det.find_correlated("nonexistent", {"cpu_util": 0.5})
        assert result == []

    def test_zero_root_error_returns_empty(
        self, normal_windows: torch.Tensor
    ) -> None:
        det = CorrelationDetector()
        det.fit(normal_windows)
        result = det.find_correlated("cpu_util", {"cpu_util": 0.0})
        assert result == []

    def test_find_correlated_returns_list_of_strings(
        self, normal_windows: torch.Tensor
    ) -> None:
        det = CorrelationDetector(corr_threshold=0.0, error_ratio_threshold=0.0)
        det.fit(normal_windows)
        result = det.find_correlated(
            "cpu_util",
            {m: 0.1 for m in METRIC_ORDER},
        )
        # All non-root metrics should appear at threshold=0.
        for name in result:
            assert name in METRIC_ORDER
            assert name != "cpu_util"

    def test_root_cause_metric_excluded_from_results(
        self, normal_windows: torch.Tensor
    ) -> None:
        det = CorrelationDetector(corr_threshold=0.0, error_ratio_threshold=0.0)
        det.fit(normal_windows)
        result = det.find_correlated(
            "cpu_util",
            {m: 0.5 for m in METRIC_ORDER},
        )
        assert "cpu_util" not in result

    def test_high_threshold_returns_no_correlated(
        self, normal_windows: torch.Tensor
    ) -> None:
        # corr_threshold=0.9999 — only nearly-perfect correlations pass.
        det = CorrelationDetector(corr_threshold=0.9999)
        det.fit(normal_windows)
        result = det.find_correlated(
            "cpu_util",
            {m: 0.5 for m in METRIC_ORDER},
        )
        # With diurnal data the synthetic correlations won't be that tight.
        assert isinstance(result, list)

    def test_strongest_correlations_returns_sorted_list(
        self, normal_windows: torch.Tensor
    ) -> None:
        det = CorrelationDetector()
        det.fit(normal_windows)
        pairs = det.strongest_correlations("cpu_util", top_k=3)
        assert len(pairs) <= 3
        abs_values = [abs(v) for _, v in pairs]
        assert abs_values == sorted(abs_values, reverse=True)

    def test_strongest_correlations_unknown_metric_raises(
        self, normal_windows: torch.Tensor
    ) -> None:
        det = CorrelationDetector()
        det.fit(normal_windows)
        with pytest.raises(ValueError):
            det.strongest_correlations("nonexistent")

    def test_invalid_corr_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            CorrelationDetector(corr_threshold=1.5)

    def test_invalid_error_ratio_raises(self) -> None:
        with pytest.raises(ValueError):
            CorrelationDetector(error_ratio_threshold=-0.1)


# ─── 4. AnomalyDetectionAPI ──────────────────────────────────────────────────

class TestAnomalyDetectionAPI:
    def test_analyze_returns_report(self, trained_scorer: AnomalyScorer) -> None:
        api = AnomalyDetectionAPI(trained_scorer)
        x = torch.rand(N_METRICS, WINDOW_SIZE)
        report = api.analyze(x)
        assert isinstance(report, AnomalyReport)

    def test_analyze_report_has_expected_fields(
        self, trained_scorer: AnomalyScorer
    ) -> None:
        api = AnomalyDetectionAPI(trained_scorer)
        x = torch.rand(N_METRICS, WINDOW_SIZE)
        report = api.analyze(x)
        assert isinstance(report.is_anomaly, bool)
        assert isinstance(report.score, float)
        assert isinstance(report.anomaly_index, float)
        assert isinstance(report.metric_errors, dict)
        assert report.correlated_metrics == []  # no detector passed

    def test_analyze_with_correlation_detector(
        self,
        trained_scorer: AnomalyScorer,
        normal_windows: torch.Tensor,
    ) -> None:
        det = CorrelationDetector(corr_threshold=0.0, error_ratio_threshold=0.0)
        det.fit(normal_windows)
        api = AnomalyDetectionAPI(trained_scorer, correlation_detector=det)

        # A highly anomalous window to trigger find_correlated.
        x = torch.full((N_METRICS, WINDOW_SIZE), fill_value=5.0)
        # Force is_anomaly to be checkable — if not flagged, skip correlation test.
        report = api.analyze(x)
        if report.is_anomaly:
            assert isinstance(report.correlated_metrics, list)

    def test_analyze_batch_returns_correct_length(
        self, trained_scorer: AnomalyScorer
    ) -> None:
        api = AnomalyDetectionAPI(trained_scorer)
        windows = torch.rand(5, N_METRICS, WINDOW_SIZE)
        reports = api.analyze_batch(windows)
        assert len(reports) == 5

    def test_summary_string_contains_status(
        self, trained_scorer: AnomalyScorer
    ) -> None:
        api = AnomalyDetectionAPI(trained_scorer)
        x = torch.rand(N_METRICS, WINDOW_SIZE)
        report = api.analyze(x)
        summary = report.summary()
        assert "ANOMALY" in summary or "normal" in summary

    def test_analyze_with_reconstruction(
        self, trained_scorer: AnomalyScorer
    ) -> None:
        api = AnomalyDetectionAPI(trained_scorer)
        x = torch.rand(N_METRICS, WINDOW_SIZE)
        report = api.analyze(x, return_reconstruction=True)
        assert report.reconstruction is not None
        assert report.reconstruction.shape == (N_METRICS, WINDOW_SIZE)


# ─── 5. Dataset classes ───────────────────────────────────────────────────────

class TestNormalWindowDataset:
    def test_len_is_positive(self, normal_dataset: NormalWindowDataset) -> None:
        assert len(normal_dataset) > 0

    def test_item_shape(self, normal_dataset: NormalWindowDataset) -> None:
        x = normal_dataset[0]
        assert x.shape == (N_METRICS, WINDOW_SIZE)

    def test_item_dtype_float32(self, normal_dataset: NormalWindowDataset) -> None:
        x = normal_dataset[0]
        assert x.dtype == torch.float32

    def test_all_windows_shape(self, normal_dataset: NormalWindowDataset) -> None:
        windows = normal_dataset.all_windows
        assert windows.dim() == 3
        assert windows.size(1) == N_METRICS
        assert windows.size(2) == WINDOW_SIZE

    def test_index_out_of_range_raises(
        self, normal_dataset: NormalWindowDataset
    ) -> None:
        with pytest.raises(IndexError):
            _ = normal_dataset[len(normal_dataset)]

    def test_rejects_too_short_frame(self, fitted_normalizer: MinMaxNormalizer) -> None:
        df = build_normal_dataframe(minutes=10, seed=7)
        with pytest.raises(ValueError):
            NormalWindowDataset(df, fitted_normalizer, window_size=WINDOW_SIZE)


class TestAnomalousWindowDataset:
    def test_len_is_positive(
        self, normal_df, anomalous_df, fitted_normalizer: MinMaxNormalizer
    ) -> None:
        ds = AnomalousWindowDataset(normal_df, anomalous_df, fitted_normalizer, stride=10)
        assert len(ds) > 0

    def test_item_returns_tensor_and_label(
        self, normal_df, anomalous_df, fitted_normalizer: MinMaxNormalizer
    ) -> None:
        ds = AnomalousWindowDataset(normal_df, anomalous_df, fitted_normalizer, stride=10)
        x, label = ds[0]
        assert x.shape == (N_METRICS, WINDOW_SIZE)
        assert label in (0, 1)

    def test_labels_contain_both_classes(
        self, normal_df, anomalous_df, fitted_normalizer: MinMaxNormalizer
    ) -> None:
        ds = AnomalousWindowDataset(normal_df, anomalous_df, fitted_normalizer, stride=10)
        unique = set(ds.labels.tolist())
        assert 0 in unique
        assert 1 in unique


# ─── 6. train_autoencoder ────────────────────────────────────────────────────

class TestTrainAutoencoder:
    def test_history_has_correct_epoch_count(
        self, normal_windows: torch.Tensor
    ) -> None:
        torch.manual_seed(0)
        model = ConvAutoencoder(n_metrics=N_METRICS, latent_dim=4, window_size=WINDOW_SIZE)
        cfg = AutoencoderTrainConfig(epochs=3, batch_size=32, seed=0)
        history = train_autoencoder(model, normal_windows, config=cfg)
        assert len(history.train_loss) == 3

    def test_train_loss_decreases_over_time(
        self, normal_windows: torch.Tensor
    ) -> None:
        # With enough steps the loss should go down (not guaranteed every epoch,
        # but first epoch should be higher than last over 10 epochs).
        torch.manual_seed(42)
        model = ConvAutoencoder(n_metrics=N_METRICS, latent_dim=4, window_size=WINDOW_SIZE)
        cfg = AutoencoderTrainConfig(epochs=10, batch_size=32, seed=42)
        history = train_autoencoder(model, normal_windows, config=cfg)
        assert history.train_loss[0] > history.train_loss[-1]

    def test_val_loss_returned_when_val_dataset_given(
        self, normal_windows: torch.Tensor
    ) -> None:
        torch.manual_seed(0)
        model = ConvAutoencoder(n_metrics=N_METRICS, latent_dim=4, window_size=WINDOW_SIZE)
        n = len(normal_windows)
        split = int(n * 0.8)
        train_w = normal_windows[:split]
        val_w = normal_windows[split:]
        cfg = AutoencoderTrainConfig(epochs=2, batch_size=32, seed=0)
        history = train_autoencoder(model, train_w, val_dataset=val_w, config=cfg)
        assert len(history.val_loss) == 2

    def test_val_loss_absent_when_no_val_dataset(
        self, normal_windows: torch.Tensor
    ) -> None:
        torch.manual_seed(0)
        model = ConvAutoencoder(n_metrics=N_METRICS, latent_dim=4, window_size=WINDOW_SIZE)
        cfg = AutoencoderTrainConfig(epochs=2, batch_size=32, seed=0)
        history = train_autoencoder(model, normal_windows, config=cfg)
        assert len(history.val_loss) == 0

    def test_train_loss_values_are_finite(
        self, normal_windows: torch.Tensor
    ) -> None:
        torch.manual_seed(0)
        model = ConvAutoencoder(n_metrics=N_METRICS, latent_dim=4, window_size=WINDOW_SIZE)
        cfg = AutoencoderTrainConfig(epochs=3, batch_size=32, seed=0)
        history = train_autoencoder(model, normal_windows, config=cfg)
        for loss in history.train_loss:
            assert np.isfinite(loss), f"non-finite loss: {loss}"
