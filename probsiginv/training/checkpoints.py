"""Checkpoint helpers for training runs."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from probsiginv.utils.contract import (
    build_checkpoint_candidate_dir,
    build_conditioning_label,
    build_output_candidate_label,
    checkpoint_filename,
)
from probsiginv.utils.layout import build_dataset_tag


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return

    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)

    torch_state = state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state.cpu() if torch.is_tensor(torch_state) else torch_state)

    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() if torch.is_tensor(item) else item for item in cuda_state])


def resolve_training_checkpoint_layout(
    *,
    save_path: str | Path,
    data_path: str | Path,
    path_transform: str | None,
    arch: str,
    aug_type: str,
    sig_depth: int,
    sig_transform: str | None,
    sig_full: bool,
    training_method: str,
) -> dict[str, Path | str]:
    """Resolve checkpoint naming/layout for a training run."""
    dataset_tag = build_dataset_tag(data_path, path_transform=path_transform)
    conditioning = build_conditioning_label(aug_type, sig_depth)
    candidate_label = conditioning
    if training_method == "regression":
        candidate_label = build_output_candidate_label(
            "regression",
            conditioning=conditioning,
        )
    exp_path = build_checkpoint_candidate_dir(save_path, dataset_tag, candidate_label)
    ckpt_dir = exp_path
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dataset_tag": dataset_tag,
        "candidate_label": candidate_label,
        "exp_path": exp_path,
        "ckpt_dir": ckpt_dir,
    }


def build_training_checkpoint(
    *,
    args,
    epoch: int,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loss: float,
    sig_depth: int,
    sig_full: bool,
    training_method: str,
    path_dim: int = 1,
    sig_channels: int | None = None,
) -> dict[str, Any]:
    """Build the serialized checkpoint payload for a training epoch."""
    return {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_loss": float(train_loss),
        "val_loss": None,
        "rng_state": _capture_rng_state(),
        "args": {
            **vars(args),
            "sig_depth": int(sig_depth),
            "sig_full": bool(sig_full),
            "training_method": training_method,
            "path_dim": int(path_dim),
            "sig_channels": None if sig_channels is None else int(sig_channels),
        },
    }


def save_training_checkpoint(ckpt_dir: str | Path, checkpoint: dict[str, Any], epoch: int) -> Path:
    """Serialize a checkpoint under the canonical epoch filename."""
    ckpt_path = Path(ckpt_dir) / checkpoint_filename(epoch)
    torch.save(checkpoint, ckpt_path)
    return ckpt_path


def load_training_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a serialized training checkpoint on CPU."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _resolve_module_device(module: torch.nn.Module) -> torch.device:
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    return torch.device("cpu")


def _move_state_value_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _move_state_value_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_state_value_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_state_value_to_device(item, device) for item in value)
    return value


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            state[key] = _move_state_value_to_device(value, device)


def restore_training_state(
    checkpoint: dict[str, Any],
    *,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore model, EMA model, and optimizer state from a checkpoint."""
    model.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
    optimizer.load_state_dict(checkpoint["optimizer"])
    _move_optimizer_state_to_device(optimizer, _resolve_module_device(model))
    _restore_rng_state(checkpoint.get("rng_state"))
    return int(checkpoint.get("epoch", 0))


def should_save_checkpoint(epoch: int, total_epochs: int, save_every: int) -> bool:
    """Return whether this epoch should produce a checkpoint."""
    if save_every < 1:
        raise ValueError(f"save_every must be >= 1, got {save_every}")
    return (epoch % save_every == 0) or (epoch == total_epochs)
