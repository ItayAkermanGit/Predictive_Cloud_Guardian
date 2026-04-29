# Unit tests for forecasting metrics.

from __future__ import annotations

import pytest
import torch

from pcg.training.metrics import (
    ForecastMetrics,
    mae,
    rmse,
    threshold_false_positive_rate,
    threshold_recall,
)


# mae / rmse

def test_mae_zero_for_identical_inputs() -> None:
    y = torch.linspace(0.0, 1.0, 60)
    assert mae(y, y) == pytest.approx(0.0)


def test_mae_basic_case() -> None:
    y_pred = torch.tensor([0.0, 0.0, 0.0, 0.0])
    y_true = torch.tensor([1.0, 1.0, 1.0, 1.0])
    assert mae(y_pred, y_true) == pytest.approx(1.0)


def test_rmse_basic_case() -> None:
    y_pred = torch.tensor([0.0, 0.0, 0.0, 0.0])
    y_true = torch.tensor([2.0, 2.0, 2.0, 2.0])
    assert rmse(y_pred, y_true) == pytest.approx(2.0)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        mae(torch.zeros(3), torch.zeros(4))


# threshold_recall

def test_recall_perfect_when_all_breaches_predicted() -> None:
    y_pred = torch.tensor([0.95, 0.10, 0.96, 0.11])
    y_true = torch.tensor([0.96, 0.05, 0.97, 0.07])
    assert threshold_recall(y_pred, y_true, 0.9) == pytest.approx(1.0)


def test_recall_zero_when_all_breaches_missed() -> None:
    y_pred = torch.tensor([0.10, 0.20, 0.30])
    y_true = torch.tensor([0.95, 0.96, 0.99])
    assert threshold_recall(y_pred, y_true, 0.9) == pytest.approx(0.0)


def test_recall_returns_one_when_no_positives_in_truth() -> None:
    # No true positives -> recall is vacuously 1 (sklearn convention).
    y_pred = torch.tensor([0.5, 0.6])
    y_true = torch.tensor([0.1, 0.2])
    assert threshold_recall(y_pred, y_true, 0.9) == pytest.approx(1.0)


def test_recall_partial_hit_rate() -> None:
    y_pred = torch.tensor([0.95, 0.10, 0.20, 0.96])
    y_true = torch.tensor([0.99, 0.05, 0.97, 0.99])
    # actual_pos at 0, 2, 3; pred_pos at 0, 3 -> TP=2, FN=1 -> recall=2/3.
    assert threshold_recall(y_pred, y_true, 0.9) == pytest.approx(2 / 3)


# threshold_false_positive_rate

def test_fpr_zero_when_no_predicted_positives() -> None:
    y_pred = torch.tensor([0.1, 0.2, 0.3])
    y_true = torch.tensor([0.1, 0.2, 0.3])
    assert threshold_false_positive_rate(y_pred, y_true, 0.9) == pytest.approx(0.0)


def test_fpr_when_safe_inputs_misclassified() -> None:
    y_pred = torch.tensor([0.95, 0.95, 0.10])
    y_true = torch.tensor([0.10, 0.20, 0.30])
    # All actuals negative; 2 of 3 predicted as positive -> FPR = 2/3.
    fpr = threshold_false_positive_rate(y_pred, y_true, 0.9)
    assert fpr == pytest.approx(2 / 3)


# ForecastMetrics aggregate

def test_forecast_metrics_aggregate() -> None:
    y_pred = torch.tensor([0.95, 0.10, 0.20, 0.96])
    y_true = torch.tensor([0.99, 0.05, 0.97, 0.99])
    metrics = ForecastMetrics.from_tensors(y_pred, y_true, threshold=0.9)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.mae > 0.0
    assert metrics.rmse > 0.0
    assert 0.0 <= metrics.fpr <= 1.0
