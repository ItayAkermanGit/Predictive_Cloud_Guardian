# Convolutional autoencoder for multivariate time-series anomaly detection.
#
# Architecture — Encoder:
#   Input  : (B, N_METRICS, WINDOW_SIZE)   [channels-first for Conv1d]
#   Conv1d layers progressively compress the temporal dimension.
#   Final flatten + Linear produces a latent vector of size `latent_dim`.
#
# Architecture — Decoder:
#   Linear expands latent vector back to a feature map.
#   ConvTranspose1d layers reconstruct the original temporal shape.
#   Output : (B, N_METRICS, WINDOW_SIZE)   same shape as input.
#
# Anomaly score per sample:
#   reconstruction_error[b] = mean over time of sum over metrics of (x - x_hat)^2
#   This is the per-sample MSE in the original (normalized) input space.
#
# Per-metric breakdown for RCA:
#   metric_error[b, m] = mean over time of (x[b, m, :] - x_hat[b, m, :])^2
#   The metric with the highest error is the root-cause candidate.

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..core.constants import N_METRICS, WINDOW_SIZE


class ConvEncoder(nn.Module):
    # Three-stage Conv1d encoder with LeakyReLU and BatchNorm.

    def __init__(self, n_metrics: int, latent_dim: int, window_size: int) -> None:
        super().__init__()
        if n_metrics <= 0:
            raise ValueError(f"n_metrics must be > 0, got {n_metrics}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {latent_dim}")
        if window_size < 8:
            raise ValueError(f"window_size must be >= 8, got {window_size}")

        # Each Conv1d halves the temporal length (stride=2).
        # channels: n_metrics -> 32 -> 64 -> 128
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_metrics, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
        )

        # Compressed temporal length after 3 stride-2 convolutions.
        # Formula: ceil(T / 2^3) for same-padding with stride 2.
        import math
        self._compressed_len = math.ceil(math.ceil(math.ceil(window_size / 2) / 2) / 2)
        self._flat_dim = 128 * self._compressed_len

        self.fc = nn.Linear(self._flat_dim, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, N_METRICS, T) -> latent: (B, latent_dim)
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = h.flatten(start_dim=1)  # (B, 128 * compressed_len)
        return self.fc(h)           # (B, latent_dim)

    @property
    def flat_dim(self) -> int:
        return self._flat_dim

    @property
    def compressed_len(self) -> int:
        return self._compressed_len


class ConvDecoder(nn.Module):
    # Mirror of ConvEncoder using ConvTranspose1d to upsample.

    def __init__(
        self,
        n_metrics: int,
        latent_dim: int,
        window_size: int,
        flat_dim: int,
        compressed_len: int,
    ) -> None:
        super().__init__()
        self._window_size = window_size
        self._compressed_len = compressed_len

        self.fc = nn.Linear(latent_dim, flat_dim)

        # Mirror of encoder: 128 -> 64 -> 32 -> n_metrics.
        # output_padding=1 corrects for floor division in stride-2 convs.
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
        )
        # Final layer uses Sigmoid to keep reconstructions in [0, 1] (normalized input).
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose1d(32, n_metrics, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: Tensor) -> Tensor:
        # z: (B, latent_dim) -> reconstruction: (B, N_METRICS, T)
        h = self.fc(z)                                       # (B, flat_dim)
        h = h.view(h.size(0), 128, self._compressed_len)    # (B, 128, compressed_len)
        h = self.deconv1(h)
        h = self.deconv2(h)
        h = self.deconv3(h)
        # Trim or pad to exact window_size in case ConvTranspose1d overshoots.
        return h[:, :, : self._window_size]


class ConvAutoencoder(nn.Module):
    # Full autoencoder = encoder + decoder.
    # Input/output shape: (B, N_METRICS, WINDOW_SIZE)  (channels-first).

    def __init__(
        self,
        n_metrics: int = N_METRICS,
        latent_dim: int = 16,
        window_size: int = WINDOW_SIZE,
    ) -> None:
        super().__init__()
        if n_metrics <= 0:
            raise ValueError(f"n_metrics must be > 0, got {n_metrics}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {latent_dim}")

        self.n_metrics = n_metrics
        self.latent_dim = latent_dim
        self.window_size = window_size

        self.encoder = ConvEncoder(n_metrics, latent_dim, window_size)
        self.decoder = ConvDecoder(
            n_metrics,
            latent_dim,
            window_size,
            flat_dim=self.encoder.flat_dim,
            compressed_len=self.encoder.compressed_len,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # Returns (reconstruction, latent).
        # reconstruction: (B, N_METRICS, WINDOW_SIZE)
        # latent:         (B, latent_dim)
        self._validate(x)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def reconstruct(self, x: Tensor) -> Tensor:
        # Convenience: returns reconstruction only.
        x_hat, _ = self(x)
        return x_hat

    def _validate(self, x: Tensor) -> None:
        if x.dim() != 3:
            raise ValueError(
                f"expected (B, N_METRICS, T), got shape {tuple(x.shape)}"
            )
        if x.size(1) != self.n_metrics:
            raise ValueError(
                f"expected dim-1 = {self.n_metrics} metrics, got {x.size(1)}"
            )
        if x.size(2) != self.window_size:
            raise ValueError(
                f"expected dim-2 = {self.window_size} timesteps, got {x.size(2)}"
            )
