"""Path shape normalization helpers.

Artifact layout keeps time before path dimension: ``(N, T)`` for scalar
legacy paths and ``(N, T, D)`` for vector-valued paths. Model/process tensors
use channel-first layout: ``(N, D, T)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RealPathShape:
    n_paths: int
    seq_len: int
    path_dim: int


@dataclass(frozen=True)
class CandidatePathShape:
    n_paths: int
    n_samples: int
    seq_len: int
    path_dim: int


def real_path_shape(paths: np.ndarray, *, name: str = "paths") -> RealPathShape:
    """Return ``(N, T, D)`` metadata for scalar or vector real-path artifacts."""
    array = np.asarray(paths)
    if array.ndim == 2:
        return RealPathShape(int(array.shape[0]), int(array.shape[1]), 1)
    if array.ndim == 3:
        return RealPathShape(int(array.shape[0]), int(array.shape[1]), int(array.shape[2]))
    raise ValueError(f"{name} must have shape (N, T) or (N, T, D), got {array.shape}")


def as_model_paths(paths: np.ndarray, *, name: str = "paths") -> np.ndarray:
    """Convert artifact paths ``(N,T)`` / ``(N,T,D)`` to model layout ``(N,D,T)``."""
    array = np.asarray(paths)
    real_path_shape(array, name=name)
    if array.ndim == 2:
        return array[:, None, :]
    return np.moveaxis(array, -1, 1)


def from_model_paths(paths: np.ndarray, *, name: str = "paths") -> np.ndarray:
    """Convert model layout ``(N,D,T)`` back to scalar/vector artifact layout."""
    array = np.asarray(paths)
    if array.ndim == 2:
        return array
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (N, T) or (N, D, T), got {array.shape}")
    if array.shape[1] == 1:
        return array[:, 0, :]
    return np.moveaxis(array, 1, -1)


def candidate_path_shape(paths: np.ndarray, *, name: str = "paths") -> CandidatePathShape:
    """Return candidate metadata for scalar legacy or vector candidate artifacts."""
    array = np.asarray(paths)
    if array.ndim == 2:
        return CandidatePathShape(int(array.shape[0]), 1, int(array.shape[1]), 1)
    if array.ndim == 3:
        return CandidatePathShape(int(array.shape[0]), int(array.shape[1]), int(array.shape[2]), 1)
    if array.ndim == 4:
        return CandidatePathShape(
            int(array.shape[0]),
            int(array.shape[1]),
            int(array.shape[2]),
            int(array.shape[3]),
        )
    raise ValueError(f"{name} must have shape (N, T), (N, M, T), or (N, M, T, D), got {array.shape}")


def as_eval_real_paths(paths: np.ndarray, *, name: str = "real_paths") -> np.ndarray:
    """Convert real-path artifacts to explicit ``(N,T,D)`` evaluation layout."""
    array = np.asarray(paths)
    real_path_shape(array, name=name)
    if array.ndim == 2:
        return array[:, :, None]
    return array


def as_eval_candidate_paths(paths: np.ndarray, *, name: str = "gen_paths") -> np.ndarray:
    """Convert candidate artifacts to explicit ``(N,M,T,D)`` evaluation layout."""
    array = np.asarray(paths)
    candidate_path_shape(array, name=name)
    if array.ndim == 2:
        return array[:, None, :, None]
    if array.ndim == 3:
        return array[:, :, :, None]
    return array


def from_eval_candidate_paths(paths: np.ndarray, *, name: str = "paths") -> np.ndarray:
    """Convert explicit candidate layout back to public artifact layout."""
    array = np.asarray(paths)
    if array.ndim != 4:
        raise ValueError(f"{name} must have shape (N, M, T, D), got {array.shape}")
    if array.shape[-1] == 1:
        return array[..., 0]
    return array


def normalize_candidate_artifact(
    paths: np.ndarray,
    *,
    path_dim: int | None = None,
    name: str = "paths",
) -> np.ndarray:
    """Normalize candidate paths before writing an artifact.

    Scalar candidates keep the legacy public shapes: deterministic ``(N,T)`` is
    written as ``(N,1,T)`` and probabilistic ``(N,M,T)`` is left unchanged.
    Vector deterministic writer input ``(N,T,D)`` requires ``path_dim=D`` and is
    written as ``(N,1,T,D)``.
    """
    array = np.asarray(paths)
    if path_dim is not None and int(path_dim) < 1:
        raise ValueError(f"path_dim must be positive, got {path_dim}")
    expected_dim = None if path_dim is None else int(path_dim)

    if array.ndim == 2:
        if expected_dim not in (None, 1):
            raise ValueError(f"{name} has scalar shape {array.shape} but path_dim={expected_dim}")
        return array[:, None, :]

    if array.ndim == 3:
        if expected_dim is not None and expected_dim > 1:
            if array.shape[-1] != expected_dim:
                raise ValueError(
                    f"{name} deterministic vector input must have last dimension path_dim={expected_dim}, "
                    f"got {array.shape}"
                )
            return array[:, None, :, :]
        return array

    if array.ndim == 4:
        if expected_dim is not None and array.shape[-1] != expected_dim:
            raise ValueError(f"{name} path_dim mismatch: expected {expected_dim}, got {array.shape[-1]}")
        return from_eval_candidate_paths(array, name=name)

    raise ValueError(f"{name} must have shape (N,T), (N,M,T), (N,T,D), or (N,M,T,D), got {array.shape}")


__all__ = [
    "CandidatePathShape",
    "RealPathShape",
    "as_eval_candidate_paths",
    "as_eval_real_paths",
    "as_model_paths",
    "candidate_path_shape",
    "from_eval_candidate_paths",
    "from_model_paths",
    "normalize_candidate_artifact",
    "real_path_shape",
]
