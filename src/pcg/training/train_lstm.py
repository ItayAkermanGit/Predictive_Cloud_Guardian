# Training loop for the LSTM+attention forecaster.
# Uses Adam + AsymmetricLoss(alpha=10) and gradient clipping.

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..core.constants import ASYMMETRIC_LOSS_ALPHA
from ..models.losses import AsymmetricLoss


@dataclass
class TrainConfig:
    # Hyperparameters for train_forecaster.

    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    alpha: float = ASYMMETRIC_LOSS_ALPHA
    grad_clip_norm: Optional[float] = 1.0   # None disables clipping
    seed: int = 42
    device: str = "cpu"


@dataclass
class TrainHistory:
    # Per-epoch losses returned by train_forecaster.

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)


def train_forecaster(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    config: Optional[TrainConfig] = None,
) -> TrainHistory:
    # Run the training loop and return per-epoch losses.
    cfg = config or TrainConfig()

    # Determinism for tests / repeatable demos.
    _seed_everything(cfg.seed)

    device = torch.device(cfg.device)
    model = model.to(device)
    loss_fn = AsymmetricLoss(alpha=cfg.alpha, mode="regression")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
        if val_dataset is not None
        else None
    )

    history = TrainHistory()

    for epoch in range(cfg.epochs):
        model.train()
        running, n_seen = 0.0, 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred, _attn = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()

            # Gradient clipping prevents the asymmetric loss's heavier
            # penalty from blowing up updates when residuals are large.
            if cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip_norm
                )
            optimizer.step()

            batch_size = x.size(0)
            running += float(loss.item()) * batch_size
            n_seen += batch_size

        train_loss = running / max(n_seen, 1)
        history.train_loss.append(train_loss)

        if val_loader is not None:
            history.val_loss.append(_eval_loss(model, val_loader, loss_fn, device))

    return history


def _eval_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    running, n_seen = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_pred, _ = model(x)
            running += float(loss_fn(y_pred, y).item()) * x.size(0)
            n_seen += x.size(0)
    return running / max(n_seen, 1)


def _seed_everything(seed: int) -> None:
    # Seed every RNG that affects training output we report.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
