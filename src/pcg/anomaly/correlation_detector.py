# Correlated anomaly detection for multi-metric Root Cause Analysis.
#
# Problem: when cpu_util spikes, mem_util and net_io often spike together
# because they share the same underlying cause (e.g., a runaway process).
# A naive per-metric scorer would flag all three independently, making it
# hard to identify the original root cause.
#
# This module answers: "given that metric X is anomalous, which other
# metrics are also anomalous AND historically correlated with X?"
#
# Two-stage approach
# ──────────────────
# Stage 1 — Static correlation matrix (fitted on normal data):
#   Pearson correlation over N_METRICS metric time-series computed from
#   the training windows. Stored as a (M, M) matrix.
#   Two metrics are "structurally correlated" if |r| > corr_threshold.
#
# Stage 2 — Dynamic error elevation check (at inference time):
#   For each structurally-correlated metric, check whether its
#   reconstruction error exceeds error_ratio_threshold * error_of_root_cause.
#   This filters out metrics that are normally correlated but not currently
#   showing elevated error, reducing noise in the RCA output.
#
# Result: correlated_metrics = metrics that are BOTH structurally
# correlated with root_cause AND currently showing elevated error.

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor

from ..core.constants import METRIC_ORDER


class CorrelationDetector:
    # Learns metric-to-metric correlations from normal windows and uses
    # them to identify secondary anomalous metrics during inference.

    def __init__(
        self,
        corr_threshold: float = 0.7,
        error_ratio_threshold: float = 0.3,
        metric_order: tuple[str, ...] = METRIC_ORDER,
    ) -> None:
        # corr_threshold: minimum |Pearson r| to consider two metrics correlated.
        # error_ratio_threshold: a correlated metric is "also anomalous" when
        #   its error >= error_ratio_threshold * root_cause_error.
        #   0.3 means "at least 30% as bad as the root-cause metric".
        if not 0 <= corr_threshold < 1:
            raise ValueError(f"corr_threshold must be in [0, 1), got {corr_threshold}")
        if error_ratio_threshold < 0:
            raise ValueError(
                f"error_ratio_threshold must be >= 0, got {error_ratio_threshold}"
            )

        self.corr_threshold = corr_threshold
        self.error_ratio_threshold = error_ratio_threshold
        self.metric_order = metric_order
        self.n_metrics = len(metric_order)

        # Filled by fit(); shape (M, M), values in [-1, 1].
        self._corr_matrix: Optional[np.ndarray] = None
        self._fitted = False

    # ─── Fitting ─────────────────────────────────────────────────────────────

    def fit(self, normal_windows: Tensor) -> "CorrelationDetector":
        # Compute the Pearson correlation matrix from normal training windows.
        # normal_windows: (N, M, T) — N windows, M metrics, T timesteps.
        #
        # We concatenate all windows along the time axis to get a long
        # (M, N*T) matrix, then compute pairwise Pearson correlations.
        # This gives a stable estimate of the structural co-movement between metrics.
        if normal_windows.dim() != 3:
            raise ValueError(
                f"expected (N, M, T), got shape {tuple(normal_windows.shape)}"
            )
        n, m, t = normal_windows.shape
        if m != self.n_metrics:
            raise ValueError(
                f"expected {self.n_metrics} metrics, got {m}"
            )

        # Reshape to (M, N*T) for correlation computation.
        # Each row is the full time series for one metric across all windows.
        flat = normal_windows.permute(1, 0, 2).reshape(m, n * t).numpy()  # (M, N*T)

        # Pearson correlation: corr(i, j) = cov(i,j) / (std_i * std_j).
        # np.corrcoef handles the normalization automatically.
        self._corr_matrix = np.corrcoef(flat)  # (M, M)

        # Replace NaN (constant series) with 0 — no correlation assumed.
        self._corr_matrix = np.nan_to_num(self._corr_matrix, nan=0.0)

        self._fitted = True
        return self

    # ─── Inference ───────────────────────────────────────────────────────────

    def find_correlated(
        self,
        root_metric: str,
        metric_errors: dict[str, float],
    ) -> list[str]:
        # Return metrics that are correlated with root_metric AND showing
        # elevated reconstruction error.
        #
        # Steps:
        #   1. Look up the root metric's index in metric_order.
        #   2. Read the correlation row for that metric.
        #   3. Filter to metrics where |r| > corr_threshold (structural correlation).
        #   4. Among those, keep only metrics where
        #        error[candidate] >= error_ratio_threshold * error[root_metric].
        #      This is the dynamic check — prevents flagging metrics that are
        #      structurally correlated but currently normal.
        self._require_fitted()

        if root_metric not in self.metric_order:
            return []

        root_idx = self.metric_order.index(root_metric)
        root_error = metric_errors.get(root_metric, 0.0)

        if root_error == 0.0:
            # Root cause has zero error — nothing to compare against.
            return []

        corr_row = self._corr_matrix[root_idx]  # type: ignore[index]

        correlated: list[str] = []
        for j, metric_name in enumerate(self.metric_order):
            if metric_name == root_metric:
                continue

            # Stage 1: structural correlation check.
            if abs(corr_row[j]) < self.corr_threshold:
                continue

            # Stage 2: dynamic error elevation check.
            candidate_error = metric_errors.get(metric_name, 0.0)
            if candidate_error >= self.error_ratio_threshold * root_error:
                correlated.append(metric_name)

        return correlated

    def correlation_matrix(self) -> np.ndarray:
        # Return the fitted (M, M) Pearson correlation matrix.
        self._require_fitted()
        return self._corr_matrix.copy()  # type: ignore[union-attr]

    def strongest_correlations(self, metric: str, top_k: int = 3) -> list[tuple[str, float]]:
        # Return the top-k most correlated metrics for `metric` (excluding itself).
        self._require_fitted()
        if metric not in self.metric_order:
            raise ValueError(f"unknown metric {metric!r}")
        idx = self.metric_order.index(metric)
        row = self._corr_matrix[idx]  # type: ignore[index]
        pairs = [
            (self.metric_order[j], float(row[j]))
            for j in range(self.n_metrics)
            if j != idx
        ]
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        return pairs[:top_k]

    # ─── Internal ────────────────────────────────────────────────────────────

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "CorrelationDetector.fit() must be called before find_correlated()."
            )
