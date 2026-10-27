"""Generative-process factory utilities."""

from __future__ import annotations

from .diffusion import GaussianDiffusion
from .flow_matching import FlowMatchingProcess
from .regression import RegressionProcess


def normalize_training_method(training_method: str | None) -> str:
    """Normalize process aliases to the canonical method name."""
    if training_method is None:
        return "ddpm"

    method = str(training_method).strip().lower()
    if method in {"ddpm", "diffusion"}:
        return "ddpm"
    if method in {"flow", "flowmatching", "flow_matching", "fm"}:
        return "flow"
    if method in {"regression", "reg"}:
        return "regression"
    raise ValueError(f"Unknown training_method: {training_method!r}")



def build_process(
    training_method: str | None,
    *,
    device,
    diffusion_steps: int = 1000,
    clip_x0=None,
):
    """Construct the requested generative-process backend."""
    method = normalize_training_method(training_method)
    if method == "ddpm":
        return GaussianDiffusion(T=diffusion_steps, device=device, clip_x0=clip_x0)
    if method == "flow":
        return FlowMatchingProcess(device=device, default_sampling_steps=diffusion_steps)
    if method == "regression":
        return RegressionProcess(device=device)
    raise AssertionError(f"Unhandled training_method: {method!r}")


__all__ = [
    "build_process",
    "normalize_training_method",
]
