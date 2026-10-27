"""Continuous flow-matching process backend."""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F


class AffineProbPath:
    """Simple affine interpolation path between Gaussian noise and data."""

    @staticmethod
    def sample_time(batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.rand(batch_size, device=device)

    @staticmethod
    def sample_source(shape: tuple[int, ...], device: torch.device | str) -> torch.Tensor:
        return torch.randn(shape, device=device)

    @staticmethod
    def sample_path(
        x_src: torch.Tensor,
        x_tgt: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_view = t.view(-1, *([1] * (x_tgt.ndim - 1)))
        x_t = (1.0 - t_view) * x_src + t_view * x_tgt
        dx_t = x_tgt - x_src
        return x_t, dx_t


class FlowMatchingProcess:
    """v1 continuous flow matching with affine probability paths."""

    name = "flow"

    def __init__(
        self,
        device: torch.device | str = "cpu",
        *,
        default_sampling_steps: int = 1000,
        prob_path: AffineProbPath | None = None,
    ):
        self.device = torch.device(device)
        self.default_sampling_steps = int(default_sampling_steps)
        self.prob_path = prob_path or AffineProbPath()

    def to(self, device: torch.device | str) -> "FlowMatchingProcess":
        self.device = torch.device(device)
        return self

    def training_loss(
        self,
        model: torch.nn.Module,
        x_1: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        batch = x_1.shape[0]
        t = self.prob_path.sample_time(batch, x_1.device)
        x_src = self.prob_path.sample_source(tuple(x_1.shape), x_1.device)
        x_t, dx_t = self.prob_path.sample_path(x_src, x_1, t)
        pred = model(x_t, t, c)
        return F.mse_loss(pred, dx_t)

    @torch.no_grad()
    def sample_loop(
        self,
        model: torch.nn.Module,
        shape: tuple[int, ...],
        c: torch.Tensor,
        sampling_steps: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        eta = kwargs.get("eta", 0.0)
        if eta not in (None, 0, 0.0):
            warnings.warn("eta is ignored for flow matching sampling", stacklevel=2)

        steps = self.default_sampling_steps if sampling_steps is None else int(sampling_steps)
        if steps <= 0:
            raise ValueError(f"sampling_steps must be positive, got {steps}")

        x = self.prob_path.sample_source(shape, c.device)
        dt = 1.0 / steps
        batch = shape[0]
        for step in range(steps):
            t = torch.full((batch,), step / steps, device=c.device)
            velocity = model(x, t, c)
            x = x + dt * velocity
        return x
