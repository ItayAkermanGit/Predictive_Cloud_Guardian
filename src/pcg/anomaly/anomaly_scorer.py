# Reconstruction-error anomaly scorer with adaptive threshold logic.
#
# How the anomaly score is calculated
# ─────────────────────────────────────────────────────────────────────
# Given a trained autoencoder AE and an input window x ∈ ℝ^(M×T):
#
#   1. Forward pass:  x_hat = AE(x)            (reconstruction)
#   2. Per-element squared error:
#        err[m, t] = (x[m, t] - x_hat[m, t])^2
#   3. Per-metric error (averaged over time):
#        metric_error[m] = mean_t err[m, t]
#      This is used for Root Cause Analysis — the metric with the
#      highest error is the most anomalous dimension.
#   4. Overall anomaly score (averaged over all metrics and time):
#        score = mean_m metric_error[m]
#              = mean_{m,t} err[m, t]
#      This is the standard MSE between input and reconstruction.
#
# Threshold calibration (fit_threshold)
# ─────────────────────────────────────────────────────────────────────
# The scorer is fitted on a held-out set of NORMAL windows:
#   mu    = mean(scores_normal)
#   sigma = std(scores_normal)
#   threshold = mu + n_sigma * sigma
#
# Any future window whose score exceeds `threshold` is flagged anomalous.
# n_sigma=3 corresponds to ~99.7% of the normal distribution, giving a
# false-positive rate < 0.3% on a Gaussian score distribution.
#
# Scores are also z-normalized into a [0, ∞) "anomaly index":
#   index = (score - mu) / sigma
# index > n_sigma  →  anomalous.  index provides a human-readable severity.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from ..core.constants import ANOMALY_SIGMA, METRIC_ORDER, N_METRICS, WINDOW_SIZE
from .autoencoder import ConvAutoencoder


@dataclass
class AnomalyScore:
    # Result of scoring a single window.

    # Overall scalar reconstruction MSE — primary anomaly signal.
    score: float

    # z-score relative to the normal distribution seen during calibration.
    # index = (score - mu_normal) / sigma_normal
    # index > n_sigma is anomalous.
    anomaly_index: float

    # True when score exceeds the calibrated threshold.
    is_anomaly: bool

    # Per-metric MSE breakdown for RCA.
    # metric_errors["cpu_util"] = mean squared error for that metric over the window.
    metric_errors: dict[str, float] = field(default_factory=dict)

    # The metric with the highest reconstruction error — primary RCA candidate.
    root_cause_metric: Optional[str] = None

    # Raw reconstruction tensor for downstream use.
    # Shape: (N_METRICS, WINDOW_SIZE) — None if return_reconstruction=False.
    reconstruction: Optional[np.ndarray] = None


class AnomalyScorer:
    # Wraps a trained ConvAutoencoder and calibrated threshold.

    def __init__(
        self,
        model: ConvAutoencoder,
        n_sigma: float = ANOMALY_SIGMA,
        device: Optional[str] = None,
    ) -> None:
        if n_sigma <= 0:
            raise ValueError(f"n_sigma must be > 0, got {n_sigma}")
        self.model = model.to(torch.device(device or "cpu")).eval()
        self.device = torch.device(device or "cpu")
        self.n_sigma = n_sigma

        # Set by fit_threshold(); required before score() can classify.
        self._mu: Optional[float] = None
        self._sigma: Optional[float] = None
        self._threshold: Optional[float] = None
        self._fitted: bool = False

    # ─── Calibration ─────────────────────────────────────────────────────────

    def fit_threshold(self, normal_windows: Tensor) -> "AnomalyScorer":
        # Calibrate mu, sigma, and threshold from a set of normal windows.
        # normal_windows: (N, N_METRICS, WINDOW_SIZE) — only normal traffic.
        if normal_windows.dim() != 3:
            raise ValueError(
                f"expected (N, M, T) normal windows, got shape "
                f"{tuple(normal_windows.shape)}"
            )

        scores = self._batch_scores(normal_windows)  # (N,) numpy array

        self._mu = float(np.mean(scores))
        self._sigma = float(np.std(scores)) or 1e-8  # guard against zero-std

        # threshold = mu + n_sigma * sigma
        # Any score above this is outside the normal distribution at
        # the chosen confidence level (default 3σ ≈ 99.7%).
        self._threshold = self._mu + self.n_sigma * self._sigma
        self._fitted = True
        return self

    @property
    def threshold(self) -> float:
        self._require_fitted()
        return self._threshold  # type: ignore[return-value]

    @property
    def mu_normal(self) -> float:
        self._require_fitted()
        return self._mu  # type: ignore[return-value]

    @property
    def sigma_normal(self) -> float:
        self._require_fitted()
        return self._sigma  # type: ignore[return-value]

    # ─── Scoring ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def score(
        self,
        x: Tensor,
        return_reconstruction: bool = False,
    ) -> AnomalyScore:
        # Score a single window x of shape (N_METRICS, WINDOW_SIZE).
        # Adds the batch dim internally, removes it before returning.
        self._require_fitted()
        if x.dim() != 2:
            raise ValueError(
                f"expected 2-D input (N_METRICS, WINDOW_SIZE), got {tuple(x.shape)}"
            )

        x_batch = x.unsqueeze(0).to(self.device)        # (1, M, T)
        x_hat_batch, _ = self.model(x_batch)             # (1, M, T)

        # Squared error per element: (1, M, T)
        sq_err = (x_batch - x_hat_batch).pow(2)

        # Per-metric MSE: mean over time axis → (1, M)
        metric_mse = sq_err.mean(dim=2)                  # (1, M)

        # Overall MSE: mean over metrics → scalar
        overall_mse = float(metric_mse.mean().item())

        # Anomaly index: how many sigma above the normal mean.
        anomaly_index = (overall_mse - self._mu) / self._sigma  # type: ignore

        # Build per-metric error dict for RCA.
        metric_errors: dict[str, float] = {}
        m_arr = metric_mse.squeeze(0).cpu().numpy()      # (M,)
        for i, name in enumerate(METRIC_ORDER[: self.model.n_metrics]):
            metric_errors[name] = float(m_arr[i])

        root_cause = max(metric_errors, key=metric_errors.__getitem__)

        reconstruction: Optional[np.ndarray] = None
        if return_reconstruction:
            reconstruction = (
                x_hat_batch.squeeze(0).cpu().numpy().astype(np.float32)
            )

        return AnomalyScore(
            score=overall_mse,
            anomaly_index=float(anomaly_index),
            is_anomaly=overall_mse > self._threshold,  # type: ignore
            metric_errors=metric_errors,
            root_cause_metric=root_cause,
            reconstruction=reconstruction,
        )

    @torch.no_grad()
    def score_batch(
        self,
        windows: Tensor,
        return_reconstruction: bool = False,
    ) -> list[AnomalyScore]:
        # Score a batch of windows of shape (N, N_METRICS, WINDOW_SIZE).
        self._require_fitted()
        if windows.dim() != 3:
            raise ValueError(
                f"expected (N, M, T), got shape {tuple(windows.shape)}"
            )

        n = windows.size(0)
        x = windows.to(self.device)
        x_hat, _ = self.model(x)                         # (N, M, T)

        sq_err = (x - x_hat).pow(2)                      # (N, M, T)
        metric_mse = sq_err.mean(dim=2).cpu().numpy()     # (N, M)
        overall_mse = metric_mse.mean(axis=1)            # (N,)

        results: list[AnomalyScore] = []
        for i in range(n):
            score_val = float(overall_mse[i])
            anomaly_index = (score_val - self._mu) / self._sigma  # type: ignore
            metric_errors = {
                name: float(metric_mse[i, j])
                for j, name in enumerate(METRIC_ORDER[: self.model.n_metrics])
            }
            root_cause = max(metric_errors, key=metric_errors.__getitem__)

            reconstruction: Optional[np.ndarray] = None
            if return_reconstruction:
                reconstruction = x_hat[i].cpu().numpy().astype(np.float32)

            results.append(
                AnomalyScore(
                    score=score_val,
                    anomaly_index=float(anomaly_index),
                    is_anomaly=score_val > self._threshold,  # type: ignore
                    metric_errors=metric_errors,
                    root_cause_metric=root_cause,
                    reconstruction=reconstruction,
                )
            )
        return results

    # ─── Internal helpers ────────────────────────────────────────────────────

    @torch.no_grad()
    def _batch_scores(self, windows: Tensor) -> np.ndarray:
        # Returns raw MSE scores (N,) without threshold or index.
        # Used internally during fit_threshold.
        x = windows.to(self.device)
        x_hat, _ = self.model(x)
        sq_err = (x - x_hat).pow(2)         # (N, M, T)
        # overall MSE per sample
        return sq_err.mean(dim=(1, 2)).cpu().numpy()

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "AnomalyScorer.fit_threshold() must be called before scoring."
            )
