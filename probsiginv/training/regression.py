"""Deterministic regression training backend."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class RegressionProcess:
    """Deterministic regression backend over the shared model(x_t, t, c) interface."""

    name = "regression"

    def __init__(self, device: torch.device | str = "cpu"):
        self.device = torch.device(device)

    def to(self, device: torch.device | str) -> "RegressionProcess":
        self.device = torch.device(device)
        return self

    def training_loss(
        self,
        model: torch.nn.Module,
        x_1: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        batch = x_1.shape[0]
        x_t = torch.zeros_like(x_1)
        t = torch.zeros(batch, device=x_1.device)
        pred = model(x_t, t, c)
        return F.mse_loss(pred, x_1)

    @torch.no_grad()
    def decode(
        self,
        model: torch.nn.Module,
        c: torch.Tensor,
        *,
        seq_len: int,
        path_dim: int = 1,
    ) -> torch.Tensor:
        batch = c.shape[0]
        if int(path_dim) < 1:
            raise ValueError(f"path_dim must be positive, got {path_dim}")
        shape = (batch, int(seq_len)) if int(path_dim) == 1 else (batch, int(path_dim), int(seq_len))
        x_t = torch.zeros(shape, device=c.device)
        t = torch.zeros(batch, device=c.device)
        return model(x_t, t, c)


__all__ = ["RegressionProcess"]
