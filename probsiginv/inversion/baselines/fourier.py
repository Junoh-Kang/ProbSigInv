"""Fourier inversion method.

Adapted from SigDiffusions (https://github.com/Barb0ra/SigDiffusions),
signature_inversion_utils/fourier_inversion.py, by Barancikova et al.
Published here with permission from Barbora, with the requested note linking
to the original repository. This file is not covered by this repository's
MIT license; all rights to the adapted code remain with its authors.
"""

from __future__ import annotations

import argparse
import time
from math import comb

import iisignature
import numpy as np
from tqdm import tqdm

from probsiginv.inversion.artifacts import load_or_compute_deterministic_logsignatures
from probsiginv.inversion.contracts import InversionMethodContext, InversionResult
from probsiginv.inversion.baselines._free_lie_algebra import (
    Word,
    rightHalfShuffleProduct,
    shuffleProductMany,
    word2Elt,
)


_PI = np.pi
_ELT1 = word2Elt(Word("1"))
_ELT2 = word2Elt(Word("2"))
_ELT3 = word2Elt(Word("3"))
_ELTD = word2Elt(Word("4"))
_DEFAULT_N_COEFFS = 10


def _augment_for_fourier(paths: np.ndarray, mirror: bool = False) -> np.ndarray:
    """Apply (time, sin(t), cos(t)-1, path) augmentation over [0, 2pi]."""
    n_paths, seq_len, path_dim = paths.shape
    time_range = seq_len * 2 if mirror else seq_len

    t = np.linspace(0.0, 2.0 * np.pi, time_range)
    t_col = np.empty((n_paths, time_range, 1))
    t_col[:] = t[None, :, None]
    sin_col = np.sin(t_col)
    cos_col = np.cos(t_col) - 1.0

    data = paths
    if mirror:
        data = np.concatenate([data, data[:, ::-1, :]], axis=1)

    return np.concatenate([t_col, sin_col, cos_col, data], axis=2)


def _normalize_path_batch(paths: np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(paths, dtype=float)
    squeeze = array.ndim == 2
    if squeeze:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError("paths must have shape (n_paths, seq_len) or (n_paths, seq_len, dim).")
    return array, squeeze


def _sig_vector_to_tensors(sig: np.ndarray, path_dim: int, max_level: int) -> list[np.ndarray]:
    """Reshape a signature vector into one tensor per level."""
    tensors: list[np.ndarray] = []
    start = 0
    for level in range(1, max_level + 1):
        end = start + path_dim**level
        tensors.append(sig[start:end].reshape((path_dim,) * level))
        start = end
    return tensors


def _get_signature_at_index(
    signature_tensors: list[np.ndarray],
    level: int,
    index: list[int],
) -> float:
    level_tensor = signature_tensors[level - 1]
    return level_tensor[tuple(index)]


def _evaluate_linear_functional(
    signature_tensors: list[np.ndarray],
    elt,
    dim_index: int,
) -> float:
    total = 0.0
    for elt_dict in elt.data:
        for word, count in elt_dict.items():
            letters = [
                dim_index + 2 if int(letter) == 4 else int(letter) - 1
                for letter in word.letters
            ]
            element = _get_signature_at_index(
                signature_tensors,
                len(letters),
                letters,
            )
            total += count * element
    return total


def _get_bn(signature_tensors: list[np.ndarray], n: int, dim_index: int) -> float:
    result = 0.0
    for k in range(0, n + 1):
        for q in range(0, k + 1):
            big_shuffle = [_ELTD]
            if n - k != 0:
                big_shuffle.append(shuffleProductMany([_ELT2] * (n - k)))
            if q != 0:
                big_shuffle.append(shuffleProductMany([_ELT3] * q))
            linear_functional = rightHalfShuffleProduct(
                shuffleProductMany(big_shuffle),
                _ELT1,
            )
            coefficient = _evaluate_linear_functional(
                signature_tensors,
                linear_functional,
                dim_index,
            )
            coefficient *= comb(n, k) * comb(k, q) * np.sin((n - k) * _PI / 2)
            result += coefficient
    return result


def _get_an(signature_tensors: list[np.ndarray], n: int, dim_index: int) -> float:
    result = 0.0
    for k in range(0, n + 1):
        for q in range(0, k + 1):
            big_shuffle = [_ELTD]
            if n - k != 0:
                big_shuffle.append(shuffleProductMany([_ELT2] * (n - k)))
            if q != 0:
                big_shuffle.append(shuffleProductMany([_ELT3] * q))
            linear_functional = rightHalfShuffleProduct(
                shuffleProductMany(big_shuffle),
                _ELT1,
            )
            coefficient = _evaluate_linear_functional(
                signature_tensors,
                linear_functional,
                dim_index,
            )
            coefficient *= comb(n, k) * comb(k, q) * np.cos((n - k) * _PI / 2)
            result += coefficient
    return result


def _get_ans_bns(signature_tensors: list[np.ndarray], n_coeffs: int, dim_index: int):
    a_n = np.zeros(n_coeffs + 1)
    b_n = np.zeros(n_coeffs + 1)
    for i in range(0, n_coeffs + 1):
        a_n[i] = _get_an(signature_tensors, i, dim_index) / _PI
        b_n[i] = _get_bn(signature_tensors, i, dim_index) / _PI
    a_n[0] /= 2
    return a_n, b_n


def _get_fourier_coeffs_from_sig(
    sig: np.ndarray,
    dim: int,
    sig_depth: int,
    n_coeffs: int,
):
    signature_tensors = _sig_vector_to_tensors(sig, dim + 3, sig_depth)
    a_n_sig = []
    b_n_sig = []
    for dim_index in range(1, dim + 1):
        a_n_sig_d, b_n_sig_d = _get_ans_bns(signature_tensors, n_coeffs, dim_index)
        a_n_sig.append(a_n_sig_d)
        b_n_sig.append(b_n_sig_d)
    return a_n_sig, b_n_sig


def _reconstruct_from_fourier_coeffs(
    t: np.ndarray,
    a_n: list[np.ndarray],
    b_n: list[np.ndarray],
) -> np.ndarray:
    dim = len(a_n)
    n_coeffs = len(a_n[0]) - 1
    all_dims = []
    for dim_index in range(dim):
        result = np.full(t.shape[0], a_n[dim_index][0])
        for i in range(1, n_coeffs + 1):
            result += (
                a_n[dim_index][i] * np.cos(i * t)
                + b_n[dim_index][i] * np.sin(i * t)
            )
        all_dims.append(result)
    return np.stack(all_dims, axis=1)


def _resolve_fourier_sig_depth_and_n_coeffs(
    sig_depth: int | None,
    n_coeffs: int | None,
) -> tuple[int, int]:
    if n_coeffs is None:
        if sig_depth is None:
            n_coeffs = _DEFAULT_N_COEFFS
        else:
            n_coeffs = sig_depth - 2
    if n_coeffs < 0:
        raise ValueError("n_coeffs must be nonnegative; use sig_depth >= 2.")

    if sig_depth is None:
        sig_depth = n_coeffs + 2
    if n_coeffs > sig_depth - 2:
        raise ValueError("Fourier inversion requires n_coeffs <= sig_depth - 2.")

    return sig_depth, n_coeffs


def fourier_logsignatures_from_paths(
    paths: np.ndarray,
    sig_depth: int,
    *,
    mirror: bool = False,
) -> np.ndarray:
    """Compute true log-signatures of Fourier-augmented input paths."""
    normalized, _ = _normalize_path_batch(paths)
    _, _, path_dim = normalized.shape
    augmented = _augment_for_fourier(normalized, mirror=mirror)
    logsig_spec = iisignature.prepare(path_dim + 3, sig_depth)
    logsigs = iisignature.logsig(augmented, logsig_spec)
    return np.asarray(logsigs, dtype=np.float32)


def fourier_invert_tensor_signatures(
    sigs: np.ndarray,
    *,
    seq_len: int,
    path_dim: int,
    sig_depth: int,
    n_coeffs: int | None = None,
    mirror: bool = False,
    show_progress: bool = True,
) -> np.ndarray:
    """Reconstruct paths from ordinary Fourier tensor signatures."""
    sig_depth, n_coeffs = _resolve_fourier_sig_depth_and_n_coeffs(sig_depth, n_coeffs)
    sigs = np.asarray(sigs)
    if sigs.ndim == 1:
        sigs = sigs[None, :]

    time_range = seq_len * 2 if mirror else seq_len
    t = np.linspace(0.0, 2.0 * np.pi, time_range)
    reconstructed = []
    iterator = range(sigs.shape[0])
    if show_progress:
        iterator = tqdm(iterator, desc="Fourier inversion")

    for i in iterator:
        a_n, b_n = _get_fourier_coeffs_from_sig(sigs[i], path_dim, sig_depth, n_coeffs)
        rec = _reconstruct_from_fourier_coeffs(t, a_n, b_n)
        if mirror:
            rec = rec[:seq_len, :]
        reconstructed.append(rec)

    return np.stack(reconstructed, axis=0)


def fourier_invert_logsignatures(
    logsigs: np.ndarray,
    *,
    seq_len: int,
    path_dim: int,
    sig_depth: int,
    n_coeffs: int | None = None,
    mirror: bool = False,
    show_progress: bool = True,
) -> np.ndarray:
    """Expand and invert exactly one Fourier log-signature row at a time."""
    sig_depth, n_coeffs = _resolve_fourier_sig_depth_and_n_coeffs(sig_depth, n_coeffs)
    logsigs = np.asarray(logsigs)
    if logsigs.ndim == 1:
        logsigs = logsigs[None, :]
    expected_width = iisignature.logsiglength(path_dim + 3, sig_depth)
    if logsigs.ndim != 2 or logsigs.shape[1] != expected_width:
        raise ValueError(
            f"Fourier log-signatures must have shape (n_paths, {expected_width}); "
            f"got {logsigs.shape}."
        )
    if not np.isfinite(logsigs).all():
        raise ValueError("Fourier log-signatures must be finite.")

    sig_spec = iisignature.prepare(path_dim + 3, sig_depth, "COSX2")
    reconstructed = []
    iterator = range(logsigs.shape[0])
    if show_progress:
        iterator = tqdm(iterator, desc="Fourier inversion")
    for i in iterator:
        tensor_sig = iisignature.logsigtosig(logsigs[i], sig_spec)
        rec = fourier_invert_tensor_signatures(
            tensor_sig,
            seq_len=seq_len,
            path_dim=path_dim,
            sig_depth=sig_depth,
            n_coeffs=n_coeffs,
            mirror=mirror,
            show_progress=False,
        )[0]
        del tensor_sig
        reconstructed.append(rec)
    return np.stack(reconstructed, axis=0)


def fourier_invert_paths(
    paths: np.ndarray,
    sig_depth: int | None = None,
    *,
    n_coeffs: int | None = None,
    mirror: bool = False,
    show_progress: bool = True,
) -> np.ndarray:
    """Reconstruct paths from their signatures via Fourier inversion."""
    sig_depth, n_coeffs = _resolve_fourier_sig_depth_and_n_coeffs(sig_depth, n_coeffs)

    normalized, squeeze = _normalize_path_batch(paths)
    _, seq_len, path_dim = normalized.shape
    logsigs = fourier_logsignatures_from_paths(normalized, sig_depth, mirror=mirror)
    result = fourier_invert_logsignatures(
        logsigs,
        seq_len=seq_len,
        path_dim=path_dim,
        sig_depth=sig_depth,
        n_coeffs=n_coeffs,
        mirror=mirror,
        show_progress=show_progress,
    )
    return result[:, :, 0] if squeeze else result


def fourier_invert_path(
    path: np.ndarray,
    sig_depth: int | None = None,
    *,
    n_coeffs: int | None = None,
    mirror: bool = False,
) -> np.ndarray:
    """Single-path convenience wrapper around ``fourier_invert_paths``."""
    result = fourier_invert_paths(
        path[None],
        sig_depth,
        n_coeffs=n_coeffs,
        mirror=mirror,
        show_progress=False,
    )
    return result[0]


def fourier_invert_path_repeated(
    path: np.ndarray,
    n_repeats: int,
    sig_depth: int | None = None,
    *,
    n_coeffs: int | None = None,
    mirror: bool = False,
) -> np.ndarray:
    """Reconstruct a single path multiple times for API consistency."""
    tiled = np.tile(path[None], (n_repeats,) + (1,) * path.ndim)
    return fourier_invert_paths(
        tiled,
        sig_depth,
        n_coeffs=n_coeffs,
        mirror=mirror,
        show_progress=False,
    )


def run_fourier_method(
    args: argparse.Namespace,
    method_context: InversionMethodContext,
    *,
    fourier_sig_depth: int,
) -> InversionResult:
    """Run the Fourier method and return its canonical candidate payload."""
    paths = method_context.source.paths
    time_grid = method_context.source.time_grid
    n_paths = paths.shape[0]
    n_coeffs = getattr(args, "n_coeffs", None)
    mirror = bool(getattr(args, "mirror", False))
    effective_sig_depth = fourier_sig_depth
    if n_coeffs is not None:
        effective_sig_depth = max(fourier_sig_depth, n_coeffs + 2)
    effective_sig_depth, n_coeffs = _resolve_fourier_sig_depth_and_n_coeffs(
        effective_sig_depth,
        n_coeffs,
    )
    print(f"[fourier] inverting {n_paths} paths (sig_depth={effective_sig_depth}) ...")
    t0 = time.perf_counter()
    normalized, squeeze = _normalize_path_batch(paths)
    _, seq_len, path_dim = normalized.shape
    source_token = "fourier_mirror" if mirror else "fourier"
    logsigs = load_or_compute_deterministic_logsignatures(
        method_context.dataset_dir,
        source=source_token,
        sig_depth=effective_sig_depth,
        sig_channels=path_dim + 3,
        n_rows=n_paths,
        compute_logsignatures=lambda: fourier_logsignatures_from_paths(
            normalized,
            effective_sig_depth,
            mirror=mirror,
        ),
        metadata={"n_coeffs": n_coeffs, "mirror": mirror},
    )
    gen = fourier_invert_logsignatures(
        logsigs,
        seq_len=seq_len,
        path_dim=path_dim,
        sig_depth=effective_sig_depth,
        n_coeffs=n_coeffs,
        mirror=mirror,
    )
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s  ({elapsed / n_paths * 1000:.1f} ms/path)")
    gen = gen[:, :, 0] if squeeze else gen
    return InversionResult(
        paths=np.asarray(gen),
        time_grid=np.asarray(time_grid),
        method="fourier",
        metadata={
            "sig_depth": effective_sig_depth,
            "n_coeffs": n_coeffs,
        },
    )


__all__ = [
    "fourier_invert_path",
    "fourier_invert_path_repeated",
    "fourier_invert_logsignatures",
    "fourier_invert_paths",
    "fourier_invert_tensor_signatures",
    "fourier_logsignatures_from_paths",
    "run_fourier_method",
]
