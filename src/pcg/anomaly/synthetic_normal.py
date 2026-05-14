# Sample training dataset for the autoencoder anomaly detector.
#
# Generates windows of NORMAL cloud behavior (no injected failures).
# The autoencoder trains exclusively on these windows to learn what
# "normal" looks like. Anomalous windows are only used at evaluation.
#
# NormalWindowDataset yields tensors of shape (N_METRICS, WINDOW_SIZE)
# in channels-first format, which is what ConvAutoencoder expects.
#
# AnomalousWindowDataset yields the same shape but with injected spikes,
# used only for evaluating the scorer (not for training the autoencoder).

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..core.constants import METRIC_ORDER, N_METRICS, WINDOW_SIZE
from ..data.interpolation import interpolate_missing
from ..data.normalizer import MinMaxNormalizer
from ..data.smoothing import smooth
from ..data.synthetic import FailureInjection, SyntheticConfig, SyntheticMetricGenerator
from ..data.windowing import to_feature_matrix


def build_normal_dataframe(
    minutes: int = 2000,
    seed: int = 42,
    start: Optional[datetime] = None,
) -> pd.DataFrame:
    # Generate a long normal-behavior DataFrame (no failures).
    # minutes: total length of the generated time series.
    config = SyntheticConfig(seed=seed, missing_rate=0.0, failures=[])
    gen = SyntheticMetricGenerator(config)
    return gen.generate(start=start or datetime(2024, 1, 1, 0, 0), minutes=minutes)


def build_anomalous_dataframe(
    minutes: int = 500,
    seed: int = 99,
    start: Optional[datetime] = None,
) -> pd.DataFrame:
    # Generate a DataFrame that contains injected failure spikes in every metric.
    # Used ONLY for evaluation — never for training the autoencoder.
    failures = [
        FailureInjection(
            metric="cpu_util",
            start_offset_minutes=100,
            duration_minutes=30,
            magnitude=0.45,    # spike +45% above normal
        ),
        FailureInjection(
            metric="mem_util",
            start_offset_minutes=105,
            duration_minutes=25,
            magnitude=0.40,
        ),
        FailureInjection(
            metric="net_io",
            start_offset_minutes=110,
            duration_minutes=20,
            magnitude=0.50,
        ),
        FailureInjection(
            metric="disk_io",
            start_offset_minutes=115,
            duration_minutes=15,
            magnitude=0.35,
        ),
    ]
    config = SyntheticConfig(seed=seed, missing_rate=0.0, failures=failures)
    gen = SyntheticMetricGenerator(config)
    return gen.generate(start=start or datetime(2024, 6, 1, 0, 0), minutes=minutes)


class NormalWindowDataset(Dataset):
    # PyTorch Dataset of normal-behavior windows for autoencoder training.
    # Each item is a tensor of shape (N_METRICS, WINDOW_SIZE).

    def __init__(
        self,
        frame: pd.DataFrame,
        normalizer: MinMaxNormalizer,
        window_size: int = WINDOW_SIZE,
        stride: int = 1,
    ) -> None:
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size and stride must be positive")

        clean = interpolate_missing(frame)
        smoothed = smooth(clean)
        scaled = normalizer.transform(smoothed)

        # to_feature_matrix returns (T, M) — we transpose to (M, T) for Conv1d.
        matrix = to_feature_matrix(scaled)  # (T, M) numpy float32

        if matrix.shape[0] < window_size:
            raise ValueError(
                f"frame has {matrix.shape[0]} samples; need at least {window_size}"
            )
        if matrix.shape[1] != N_METRICS:
            raise ValueError(
                f"expected {N_METRICS} metrics, got {matrix.shape[1]}"
            )

        # Pre-build all windows as a contiguous tensor for fast __getitem__.
        # Shape after stack: (N_windows, M, T) — channels-first.
        n_windows = (matrix.shape[0] - window_size) // stride + 1
        windows = np.stack(
            [
                matrix[i * stride : i * stride + window_size].T  # (M, T)
                for i in range(n_windows)
            ],
            axis=0,
        )  # (N_windows, M, T)

        self._windows = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32))
        self._n = n_windows

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Tensor:
        # Returns (N_METRICS, WINDOW_SIZE).
        if idx < 0 or idx >= self._n:
            raise IndexError(f"index {idx} out of range [0, {self._n})")
        return self._windows[idx]

    @property
    def all_windows(self) -> Tensor:
        return self._windows


class AnomalousWindowDataset(Dataset):
    # Evaluation-only dataset that mixes normal windows with anomalous ones.
    # Labels: 0 = normal, 1 = anomalous.
    # Each item: (tensor of shape (N_METRICS, WINDOW_SIZE), label int).

    def __init__(
        self,
        normal_frame: pd.DataFrame,
        anomalous_frame: pd.DataFrame,
        normalizer: MinMaxNormalizer,
        window_size: int = WINDOW_SIZE,
        stride: int = 5,    # wider stride to reduce dataset size for evaluation
    ) -> None:
        normal_ds = NormalWindowDataset(normal_frame, normalizer, window_size, stride)
        anomalous_ds = NormalWindowDataset(anomalous_frame, normalizer, window_size, stride)

        normal_windows = normal_ds.all_windows      # (N, M, T)
        anomalous_windows = anomalous_ds.all_windows  # (A, M, T)

        self._windows = torch.cat([normal_windows, anomalous_windows], dim=0)
        self._labels = torch.cat(
            [
                torch.zeros(len(normal_windows), dtype=torch.long),
                torch.ones(len(anomalous_windows), dtype=torch.long),
            ],
            dim=0,
        )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        return self._windows[idx], int(self._labels[idx].item())

    @property
    def all_windows(self) -> Tensor:
        return self._windows

    @property
    def labels(self) -> Tensor:
        return self._labels
