# Training loop for the convolutional autoencoder on normal-behavior data.
#
# The autoencoder is trained exclusively on windows that represent normal
# cloud operation (no injected failures). After training, it learns a
# compact manifold of normal behavior. At inference time, anomalous windows
# produce higher reconstruction error because they lie off this manifold.
#
# Loss: plain MSE between input and reconstruction.
# Asymmetric loss is intentionally NOT used here — the autoencoder must
# learn normal patterns faithfully; asymmetry would bias the latent space.

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .autoencoder import ConvAutoencoder


@dataclass
class AutoencoderTrainConfig:
    # Hyperparameters for train_autoencoder.

    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5      # L2 regularization to keep latent space compact
    grad_clip_norm: Optional[float] = 1.0
    seed: int = 42
    device: str = "cpu"


@dataclass
class AutoencoderTrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)


def train_autoencoder(
    model: ConvAutoencoder,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    config: Optional[AutoencoderTrainConfig] = None,
) -> AutoencoderTrainHistory:
    # Train `model` on `train_dataset` using MSE reconstruction loss.
    # `train_dataset` must yield tensors of shape (N_METRICS, WINDOW_SIZE).
    cfg = config or AutoencoderTrainConfig()
    _seed_everything(cfg.seed)

    device = torch.device(cfg.device)
    model = model.to(device)

    # MSE between input and reconstruction — symmetric because we want the
    # autoencoder to faithfully represent all parts of normal behavior equally.
    loss_fn = nn.MSELoss(reduction="mean")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    # Reduce LR by half if val loss stops improving for 5 consecutive epochs.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader: Optional[DataLoader] = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    history = AutoencoderTrainHistory()

    for _ in range(cfg.epochs):
        model.train()
        running, n_seen = 0.0, 0

        for batch in train_loader:
            # Dataset yields either a single tensor x or a tuple (x, _).
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)

            optimizer.zero_grad()
            x_hat, _ = model(x)
            loss = loss_fn(x_hat, x)
            loss.backward()

            if cfg.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

            optimizer.step()
            running += float(loss.item()) * x.size(0)
            n_seen += x.size(0)

        epoch_train_loss = running / max(n_seen, 1)
        history.train_loss.append(epoch_train_loss)

        if val_loader is not None:
            val_loss = _eval_loss(model, val_loader, loss_fn, device)
            history.val_loss.append(val_loss)
            scheduler.step(val_loss)
        else:
            scheduler.step(epoch_train_loss)

    return history


def _eval_loss(
    model: ConvAutoencoder,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    running, n_seen = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            x_hat, _ = model(x)
            running += float(loss_fn(x_hat, x).item()) * x.size(0)
            n_seen += x.size(0)
    return running / max(n_seen, 1)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
