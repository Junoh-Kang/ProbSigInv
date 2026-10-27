"""Path/signature transform helpers."""

from __future__ import annotations

import math
import warnings

import numpy as np



def _level_scale(dimension: int, level_indices: dict[int, list[int]]) -> np.ndarray:
    scale = np.ones(dimension)
    for level, cols in level_indices.items():
        factor = math.factorial(level)
        for idx in cols:
            if idx < dimension:
                scale[idx] = factor
    return scale



def _warn_level_fallback(name: str) -> None:
    warnings.warn(f"{name} requested but level_indices is None; falling back to symlog")



def apply_sig_transform(
    sigs: np.ndarray,
    transform: str | None,
    level_indices: dict[int, list[int]] | None = None,
) -> np.ndarray:
    """Apply a named transform to signature arrays."""
    if transform is None or transform == "raw":
        return sigs
    if transform == "symlog":
        return np.sign(sigs) * np.log1p(np.abs(sigs))
    if transform == "level":
        if level_indices is None:
            _warn_level_fallback("level")
            return np.sign(sigs) * np.log1p(np.abs(sigs))
        scale = _level_scale(sigs.shape[1], level_indices)
        return np.sign(sigs) * (scale * np.abs(sigs))
    if transform == "level_symlog":
        if level_indices is None:
            _warn_level_fallback("level_symlog")
            return np.sign(sigs) * np.log1p(np.abs(sigs))
        scale = _level_scale(sigs.shape[1], level_indices)
        normed = np.sign(sigs) * (scale * np.abs(sigs))
        return np.sign(normed) * np.log1p(np.abs(normed))
    raise ValueError(f"Unknown sig_transform: {transform!r}")



def apply_path_transform(paths: np.ndarray, transform: str | None) -> np.ndarray:
    """Apply a named transform to path arrays."""
    if transform is None or transform == "raw":
        return paths
    if transform == "diff":
        transformed = np.empty_like(paths)
        transformed[:, 0] = paths[:, 0]
        transformed[:, 1:] = paths[:, 1:] - paths[:, :-1]
        return transformed
    raise ValueError(f"Unknown path_transform: {transform!r}")



def invert_path_transform(paths: np.ndarray, transform: str | None) -> np.ndarray:
    """Invert a named transform previously applied to path arrays."""
    if transform is None or transform == "raw":
        return paths
    if transform == "diff":
        return np.cumsum(paths, axis=-1)
    raise ValueError(f"Unknown path_transform: {transform!r}")


__all__ = [
    "apply_sig_transform",
    "apply_path_transform",
    "invert_path_transform",
]
