"""Insertion inversion method.

Adapted from Signatory (https://github.com/patrick-kidger/signatory),
src/signatory/signature_inversion_module.py and signature_module.py.
Copyright 2019 Patrick Kidger. Licensed under the Apache License, Version 2.0;
see licenses/Signatory-LICENSE. Modified for this repository.
"""

from __future__ import annotations

import argparse
import time

import iisignature
import numpy as np
import torch

from probsiginv.inversion.contracts import InversionMethodContext, InversionResult


def signature_channels(channels: int, depth: int, scalar_term: bool = False) -> int:
    """Return ordinary signature channel count through ``depth``."""
    if channels < 1:
        raise ValueError("channels must be at least 1.")
    if depth < 0:
        raise ValueError("depth must be nonnegative.")
    if depth == 0:
        return int(scalar_term)
    return int(iisignature.siglength(int(channels), int(depth))) + int(scalar_term)


def extract_signature_term(
    sigtensor: torch.Tensor,
    channels: int,
    depth: int,
    scalar_term: bool = False,
) -> torch.Tensor:
    """Extract one homogeneous signature level from a flattened signature."""
    if channels < 1:
        raise ValueError("channels must be at least 1.")
    if depth < 1:
        raise ValueError("depth must be at least 1.")

    start = int(scalar_term) if depth == 1 else signature_channels(channels, depth - 1, scalar_term)
    length = int(channels) ** int(depth)
    if sigtensor.shape[-1] < start + length:
        raise ValueError("signature tensor is too short for requested channels/depth.")
    return sigtensor.narrow(dim=-1, start=start, length=length)


def get_insertion_matrix(
    signature: torch.Tensor,
    insertion_position: int,
    depth: int,
    channels: int,
) -> torch.Tensor:
    """Build the linear insertion map for a fixed insertion position."""
    batch = int(signature.shape[0])
    device = signature.device
    dtype = signature.dtype

    eye = torch.eye(channels, dtype=dtype, device=device).repeat(batch, 1)
    new_shape = (
        [batch, channels]
        + [1] * (insertion_position - 1)
        + [channels]
        + [1] * (depth + 1 - insertion_position)
    )
    repeat_points = (
        [1, 1]
        + [channels] * (insertion_position - 1)
        + [1]
        + [channels] * (depth + 1 - insertion_position)
    )
    basis_tensor = eye.view(new_shape).repeat(repeat_points)

    last_term = extract_signature_term(signature, channels, depth)
    last_term = last_term.view([batch] + [channels] * int(depth)).unsqueeze(insertion_position)
    repeat_points_sig = (
        [1, channels]
        + [1] * (insertion_position - 1)
        + [channels]
        + [1] * (depth + 1 - insertion_position)
    )
    signature_tensor = last_term.unsqueeze(1).repeat(repeat_points_sig)
    return (basis_tensor * signature_tensor).flatten(start_dim=2)


def solve_optimization_problem(
    signature: torch.Tensor,
    insertion_position: int,
    depth: int,
    channels: int,
) -> torch.Tensor:
    """Recover one insertion derivative from the top signature level."""
    lower_signature = signature[:, : -(channels**depth)]
    a_matrix = get_insertion_matrix(
        lower_signature,
        insertion_position,
        depth - 1,
        channels,
    )
    b_vector = depth * extract_signature_term(signature, channels, depth)
    numerator = torch.matmul(a_matrix, b_vector.unsqueeze(-1)).squeeze(-1)
    previous_level = extract_signature_term(signature, channels, depth - 1)
    denominator = torch.norm(previous_level, dim=1).unsqueeze(-1) ** 2
    return numerator / denominator


def invert_signature(
    signature: torch.Tensor,
    depth: int,
    channels: int,
    initial_position: torch.Tensor | None = None,
) -> torch.Tensor:
    """Invert ordinary signatures with the insertion algorithm."""
    if depth < 1:
        raise ValueError("depth must be at least 1.")
    if channels < 1:
        raise ValueError("channels must be at least 1.")
    if signature.ndim != 2:
        raise ValueError("signature must have shape (batch, signature_channels).")
    if signature_channels(channels, depth) != signature.shape[1]:
        raise ValueError("channels and depth do not correspond to signature shape.")

    batch = int(signature.shape[0])
    path = signature.new_zeros((batch, depth + 1, channels))

    if initial_position is not None:
        initial = initial_position.to(device=signature.device, dtype=signature.dtype)
        if initial.shape != (batch, channels):
            raise ValueError("initial_position must have shape (batch, channels).")
        path[:, 0, :] = initial

    if depth == 1:
        path[:, 1, :] = path[:, 0, :] + signature
        return path

    for insertion_position in range(1, depth + 1):
        derivative = solve_optimization_problem(
            signature,
            insertion_position,
            depth,
            channels,
        )
        path[:, insertion_position, :] = path[:, insertion_position - 1, :] + derivative / depth

    return path


def _normalize_path_batch(paths: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(paths, dtype=np.float32)
    squeeze = array.ndim == 2
    if squeeze:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError("paths must have shape (n_paths, seq_len) or (n_paths, seq_len, dim).")
    return array, squeeze


def _normalize_initial_values(
    initial_values: np.ndarray | None,
    *,
    n_paths: int,
    path_dim: int,
) -> torch.Tensor | None:
    if initial_values is None:
        return None

    array = np.asarray(initial_values, dtype=np.float32)
    if array.shape == (path_dim,):
        array = np.tile(array[None, :], (n_paths, 1))
    if array.shape != (n_paths, path_dim):
        raise ValueError(
            "initial_values must have shape (path_dim,) or (n_paths, path_dim)."
        )
    return torch.as_tensor(array, dtype=torch.float32)


def _resample_to_sequence_length(paths: np.ndarray, target_len: int) -> np.ndarray:
    """Resample depth-dependent insertion output to the source path length."""
    if paths.shape[1] == target_len:
        return paths

    src_grid = np.linspace(0.0, 1.0, paths.shape[1], dtype=np.float64)
    dst_grid = np.linspace(0.0, 1.0, target_len, dtype=np.float64)
    resampled = np.empty((paths.shape[0], target_len, paths.shape[2]), dtype=np.float64)
    for batch_idx in range(paths.shape[0]):
        for dim_idx in range(paths.shape[2]):
            resampled[batch_idx, :, dim_idx] = np.interp(
                dst_grid,
                src_grid,
                paths[batch_idx, :, dim_idx],
            )
    return resampled.astype(paths.dtype, copy=False)


def insertion_signatures_from_paths(paths: np.ndarray, sig_depth: int) -> np.ndarray:
    """Compute ordinary signatures for insertion inversion."""
    normalized, _ = _normalize_path_batch(paths)
    sigs = iisignature.sig(normalized, int(sig_depth))
    return np.asarray(sigs, dtype=np.float32)


def insertion_invert_signatures(
    sigs: np.ndarray,
    *,
    seq_len: int,
    path_dim: int,
    sig_depth: int,
    initial_values: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct paths from ordinary signatures via insertion inversion."""
    sig_tensor = torch.as_tensor(np.asarray(sigs), dtype=torch.float32)
    if sig_tensor.ndim == 1:
        sig_tensor = sig_tensor[None, :]

    initial_tensor = _normalize_initial_values(
        initial_values,
        n_paths=int(sig_tensor.shape[0]),
        path_dim=int(path_dim),
    )
    inverted = invert_signature(
        sig_tensor,
        int(sig_depth),
        int(path_dim),
        initial_position=initial_tensor,
    )
    reconstructed = inverted.detach().cpu().numpy().astype(np.float32, copy=False)
    return _resample_to_sequence_length(reconstructed, target_len=int(seq_len))


def insertion_invert_paths(paths: np.ndarray, sig_depth: int) -> np.ndarray:
    """Reconstruct a batch of paths via insertion inversion."""
    normalized, squeeze = _normalize_path_batch(paths)
    _, seq_len, path_dim = normalized.shape
    sigs = insertion_signatures_from_paths(normalized, sig_depth)
    reconstructed = insertion_invert_signatures(
        sigs,
        seq_len=seq_len,
        path_dim=path_dim,
        sig_depth=sig_depth,
        initial_values=normalized[:, 0, :],
    )
    return reconstructed[:, :, 0] if squeeze else reconstructed


def _augment_with_time(paths: np.ndarray, time_grid: np.ndarray | None = None) -> np.ndarray:
    """Add a leading time channel before insertion inversion."""
    n_paths, seq_len, _path_dim = paths.shape
    if time_grid is None:
        time = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
    else:
        time = np.asarray(time_grid, dtype=np.float32)
        if time.shape != (seq_len,):
            raise ValueError("time_grid must have shape (seq_len,).")
    time_channel = np.broadcast_to(time[None, :, None], (n_paths, seq_len, 1))
    return np.concatenate([time_channel, paths], axis=2).astype(np.float32, copy=False)


def insertion_invert_time_augmented_paths(
    paths: np.ndarray,
    sig_depth: int,
    *,
    time_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct paths from signatures of ``(time, path)`` augmented paths."""
    normalized, squeeze = _normalize_path_batch(paths)
    _, seq_len, path_dim = normalized.shape
    augmented = _augment_with_time(normalized, time_grid=time_grid)
    sigs = insertion_signatures_from_paths(augmented, sig_depth)
    reconstructed_augmented = insertion_invert_signatures(
        sigs,
        seq_len=seq_len,
        path_dim=path_dim + 1,
        sig_depth=sig_depth,
        initial_values=augmented[:, 0, :],
    )
    reconstructed = reconstructed_augmented[:, :, 1:]
    return reconstructed[:, :, 0] if squeeze else reconstructed


def insertion_invert_path(path: np.ndarray, sig_depth: int) -> np.ndarray:
    """Single-path convenience wrapper around ``insertion_invert_paths``."""
    result = insertion_invert_paths(path[None], sig_depth)
    return result[0]


def run_insertion_method(
    args: argparse.Namespace,
    method_context: InversionMethodContext,
    *,
    insertion_sig_depth: int,
) -> InversionResult:
    """Run insertion inversion and return the shared artifact payload."""
    del args
    paths = method_context.source.paths
    time_grid = method_context.source.time_grid
    n_paths = paths.shape[0]
    n_pieces = int(insertion_sig_depth)

    print(
        f"[insertion] inverting {n_paths} paths "
        f"(sig_depth={insertion_sig_depth}, n_pieces={n_pieces}) ..."
    )
    t0 = time.perf_counter()
    gen = insertion_invert_time_augmented_paths(
        paths,
        insertion_sig_depth,
        time_grid=time_grid,
    )
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s  ({elapsed / n_paths * 1000:.1f} ms/path)")

    return InversionResult(
        paths=np.asarray(gen),
        time_grid=np.asarray(time_grid),
        method="insertion",
        metadata={
            "sig_depth": int(insertion_sig_depth),
            "n_pieces": n_pieces,
            "augmentation": "time",
        },
    )


__all__ = [
    "extract_signature_term",
    "get_insertion_matrix",
    "insertion_invert_path",
    "insertion_invert_paths",
    "insertion_invert_signatures",
    "insertion_invert_time_augmented_paths",
    "insertion_signatures_from_paths",
    "invert_signature",
    "run_insertion_method",
    "signature_channels",
    "solve_optimization_problem",
]
