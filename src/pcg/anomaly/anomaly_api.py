# Public anomaly-detection API consumed by other modules (inference, alerting).
#
# This is the single entry point that downstream modules should call.
# It composes AnomalyScorer (per-window scoring) and
# CorrelationDetector (cross-metric correlation analysis) into one
# unified result object so callers never need to import internal classes.
#
# Usage pattern:
#   api = AnomalyDetectionAPI(scorer, correlation_detector)
#   result = api.analyze(window_tensor)
#   if result.is_anomaly:
#       alert(result.root_cause_metric, result.correlated_metrics)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from torch import Tensor

from .anomaly_scorer import AnomalyScore, AnomalyScorer

if TYPE_CHECKING:
    from .correlation_detector import CorrelationDetector


@dataclass(frozen=True)
class AnomalyReport:
    # Complete anomaly detection result for one server window.

    # Primary anomaly flag and score from autoencoder reconstruction error.
    is_anomaly: bool
    score: float          # raw MSE
    anomaly_index: float  # sigma above normal mean — human-readable severity

    # RCA: which metric drove the anomaly.
    root_cause_metric: Optional[str]

    # Per-metric reconstruction MSE breakdown for detailed RCA.
    # {"cpu_util": 0.04, "mem_util": 0.01, ...}
    metric_errors: dict[str, float]

    # Metrics that are correlated with the root-cause metric and also
    # show elevated reconstruction error. Empty when no correlation was run.
    correlated_metrics: list[str] = field(default_factory=list)

    # Full reconstruction array for plotting/debugging.
    # Shape (N_METRICS, WINDOW_SIZE) or None.
    reconstruction: Optional[np.ndarray] = None

    def summary(self) -> str:
        # One-line human-readable summary.
        status = "ANOMALY" if self.is_anomaly else "normal"
        idx = f"{self.anomaly_index:.2f}σ"
        rca = self.root_cause_metric or "n/a"
        corr = ", ".join(self.correlated_metrics) if self.correlated_metrics else "none"
        return (
            f"[{status}] score={self.score:.5f} ({idx}) "
            f"root_cause={rca} correlated={corr}"
        )


class AnomalyDetectionAPI:
    # Unified anomaly detection interface for downstream modules.

    def __init__(
        self,
        scorer: AnomalyScorer,
        correlation_detector: Optional["CorrelationDetector"] = None,
    ) -> None:
        self._scorer = scorer
        self._correlation_detector = correlation_detector

    def analyze(
        self,
        window: Tensor,
        return_reconstruction: bool = False,
    ) -> AnomalyReport:
        # Analyze a single window and return a full AnomalyReport.
        # window: (N_METRICS, WINDOW_SIZE) — normalized, channels-first.
        score: AnomalyScore = self._scorer.score(
            window, return_reconstruction=return_reconstruction
        )

        correlated: list[str] = []
        if (
            score.is_anomaly
            and self._correlation_detector is not None
            and score.root_cause_metric is not None
        ):
            # Only run correlation analysis when an anomaly is detected —
            # avoids unnecessary computation on the majority of normal windows.
            correlated = self._correlation_detector.find_correlated(
                root_metric=score.root_cause_metric,
                metric_errors=score.metric_errors,
            )

        return AnomalyReport(
            is_anomaly=score.is_anomaly,
            score=score.score,
            anomaly_index=score.anomaly_index,
            root_cause_metric=score.root_cause_metric,
            metric_errors=score.metric_errors,
            correlated_metrics=correlated,
            reconstruction=score.reconstruction,
        )

    def analyze_batch(
        self,
        windows: Tensor,
        return_reconstruction: bool = False,
    ) -> list[AnomalyReport]:
        # Analyze a batch of windows.
        # windows: (N, N_METRICS, WINDOW_SIZE).
        scores: list[AnomalyScore] = self._scorer.score_batch(
            windows, return_reconstruction=return_reconstruction
        )

        reports: list[AnomalyReport] = []
        for score in scores:
            correlated: list[str] = []
            if (
                score.is_anomaly
                and self._correlation_detector is not None
                and score.root_cause_metric is not None
            ):
                correlated = self._correlation_detector.find_correlated(
                    root_metric=score.root_cause_metric,
                    metric_errors=score.metric_errors,
                )

            reports.append(
                AnomalyReport(
                    is_anomaly=score.is_anomaly,
                    score=score.score,
                    anomaly_index=score.anomaly_index,
                    root_cause_metric=score.root_cause_metric,
                    metric_errors=score.metric_errors,
                    correlated_metrics=correlated,
                    reconstruction=score.reconstruction,
                )
            )
        return reports
