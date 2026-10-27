"""Shared conditioning builders for training and inversion."""

from __future__ import annotations

import numpy as np

from probsiginv.utils.signatures import augment_path, compute_logsignatures
from probsiginv.utils.signature_config import load_level_indices
from probsiginv.utils.transforms import apply_sig_transform


def resolve_data_aug_type(aug_type: str) -> str:
    """Map logical conditioning aug types to on-disk dataset aug types."""
    return "TA" if aug_type == "LS" else aug_type


def _require_logsig(sig_type: str) -> str:
    """Validate that only logsig is used throughout the current code path."""
    value = str(sig_type)
    if value != "logsig":
        raise ValueError(f"Only sig_type=\"logsig\" is supported, got {value!r}")
    return value


def resolve_effective_sig_depth(sig_depth: int | None, dataset_sig_depth: int) -> int:
    """Resolve the conditioning depth against the dataset maximum depth."""
    if sig_depth is None:
        raise ValueError("sig_depth must be set explicitly; dataset fallback is not supported")
    effective_depth = int(sig_depth)
    if effective_depth < 0:
        raise ValueError(f"sig_depth must be non-negative, got {effective_depth}")
    if effective_depth > dataset_sig_depth:
        raise ValueError(
            f"sig_depth {effective_depth} exceeds dataset sig_depth {dataset_sig_depth}"
        )
    return effective_depth


def _select_signature_indices(
    effective_depth: int,
    level_indices: dict[int, list[int]] | None,
) -> list[int]:
    if effective_depth == 0:
        return []
    if level_indices is None:
        raise ValueError("level_indices is required when sig_depth > 0")
    return [
        idx
        for level in sorted(level_indices)
        if level <= effective_depth
        for idx in level_indices[level]
    ]


def build_conditioning_from_signatures(
    sigs: np.ndarray,
    *,
    aug_type: str,
    sig_type: str,
    sig_depth: int,
    sig_transform: str | None = None,
    sig_full: bool = False,
    level_indices: dict[int, list[int]] | None = None,
    sig_channels: int | None = None,
) -> tuple[np.ndarray, int, dict[int, list[int]] | None]:
    """Apply log-signature transforms and pruning to raw signature tensors."""
    sig_type = _require_logsig(sig_type)
    effective_depth = int(sig_depth)
    if effective_depth == 0:
        return sigs[:, :0].astype(np.float32, copy=False), effective_depth, None

    if level_indices is None:
        level_indices = load_level_indices(
            aug_type,
            sig_type,
            full=sig_full,
            min_depth=effective_depth,
            channels=sig_channels,
        )

    transformed = apply_sig_transform(sigs, sig_transform, level_indices)
    sig_indices = _select_signature_indices(effective_depth, level_indices)
    conditioned = transformed[:, sig_indices]
    return conditioned.astype(np.float32, copy=False), effective_depth, level_indices


def compute_raw_signatures_from_paths(
    paths: np.ndarray,
    time_grid: np.ndarray,
    *,
    aug_type: str,
    sig_depth: int,
    sig_type: str = "logsig",
) -> np.ndarray:
    """Compute raw full-depth log-signatures from paths."""
    _require_logsig(sig_type)
    if sig_depth == 0:
        return np.empty((paths.shape[0], 0), dtype=np.float32)

    augmented = augment_path(paths, time_grid, aug_type=aug_type)
    sigs = compute_logsignatures(augmented, depth=sig_depth)
    return sigs.astype(np.float32, copy=False)


def build_inversion_conditioning_from_paths(
    paths: np.ndarray,
    time_grid: np.ndarray,
    *,
    aug_type: str,
    sig_depth: int,
    sig_transform: str | None = None,
    sig_full: bool = False,
    sig_type: str = "logsig",
    level_indices: dict[int, list[int]] | None = None,
) -> tuple[np.ndarray, int, dict[int, list[int]] | None]:
    """Build inversion conditioning tensors from paths using the training replay path."""
    raw_sigs = compute_raw_signatures_from_paths(
        paths,
        time_grid,
        aug_type=aug_type,
        sig_depth=sig_depth,
        sig_type=sig_type,
    )
    sig_channels = int(augment_path(paths[:1], time_grid, aug_type=aug_type).shape[-1]) if sig_depth > 0 else None
    return build_conditioning_from_signatures(
        raw_sigs,
        aug_type=aug_type,
        sig_type=sig_type,
        sig_depth=sig_depth,
        sig_transform=sig_transform,
        sig_full=sig_full,
        level_indices=level_indices,
        sig_channels=sig_channels,
    )
