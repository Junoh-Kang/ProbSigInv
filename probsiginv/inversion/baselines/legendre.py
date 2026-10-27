"""Legendre inversion method.

The signature helpers (_sig_vector_to_tensors, _get_signature_at_index,
_evaluate_linear_functional) are adapted from SigDiffusions
(https://github.com/Barb0ra/SigDiffusions),
signature_inversion_utils/fourier_inversion.py, by Barancikova et al.
Published here with permission from Barbora, with the requested note linking
to the original repository. These helpers are not covered by this repository's
MIT license; all rights to them remain with their authors.
"""

from __future__ import annotations

import argparse
import time

import iisignature
import numpy as np
from tqdm import tqdm

from probsiginv.inversion.artifacts import load_or_compute_deterministic_logsignatures
from probsiginv.inversion.contracts import InversionMethodContext, InversionResult
from probsiginv.inversion.baselines._free_lie_algebra import (
    Word,
    rightHalfShuffleProduct,
    word2Elt,
)


_ELT1 = word2Elt(Word("1"))
_ELT21 = word2Elt(Word("21"))
_DEFAULT_N_COEFFS = 10


def _augment_for_legendre(
    paths: np.ndarray,
    *,
    center_initial: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Legendre augmentation (time, path) over [0, 1]."""
    n_paths, seq_len, path_dim = paths.shape

    if center_initial:
        initial = paths[:, :1, :].copy()
        data = paths - initial
    else:
        initial = np.zeros((n_paths, 1, path_dim), dtype=paths.dtype)
        data = paths

    t = np.linspace(0.0, 1.0, seq_len, dtype=paths.dtype)
    t_col = np.empty((n_paths, seq_len, 1), dtype=paths.dtype)
    t_col[:] = t[None, :, None]

    return np.concatenate([t_col, data], axis=2), initial


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
    return float(level_tensor[tuple(index)])


def _evaluate_linear_functional(
    signature_tensors: list[np.ndarray],
    elt,
    dim_index: int,
) -> float:
    """Evaluate a free-tensor linear functional on a signature."""
    total = 0.0
    for elt_dict in elt.data:
        for word, count in elt_dict.items():
            letters = [
                dim_index if int(letter) == 2 else int(letter) - 1
                for letter in word.letters
            ]
            if len(letters) == 0:
                continue
            element = _get_signature_at_index(
                signature_tensors,
                len(letters),
                letters,
            )
            total += float(count) * element
    return float(total)


def _legendre_norm_sq(n: int) -> float:
    """Squared L2([0, 1]) norm of shifted Legendre P_n(2t-1)."""
    return 1.0 / (2 * n + 1)


def _legendre_recurrence_params(n: int) -> tuple[float, float, float | None]:
    """Return A_n, B_n, C_n for shifted Legendre recurrence."""
    if n < 1:
        raise ValueError("n must be at least 1.")
    if n == 1:
        return 2.0, -1.0, None

    a_n = (4 * n - 2) / n
    b_n = -(2 * n - 1) / n
    c_n = -(n - 1) / n
    return a_n, b_n, c_n


def _build_legendre_functionals(n_coeffs: int):
    """Build ell_0, ..., ell_N for shifted Legendre inversion."""
    if n_coeffs < 0:
        raise ValueError("n_coeffs must be nonnegative.")

    functionals = []

    ell_0 = (1.0 / _legendre_norm_sq(0)) * _ELT21
    functionals.append(ell_0)

    if n_coeffs == 0:
        return functionals

    a_1, b_1, _ = _legendre_recurrence_params(1)
    norm_1 = _legendre_norm_sq(1)
    ell_1 = (a_1 / norm_1) * rightHalfShuffleProduct(_ELT1, _ELT21)
    ell_1 = ell_1 + (b_1 / norm_1) * _ELT21
    functionals.append(ell_1)

    for n in range(2, n_coeffs + 1):
        a_n, b_n, c_n = _legendre_recurrence_params(n)

        norm_n = _legendre_norm_sq(n)
        norm_nm1 = _legendre_norm_sq(n - 1)
        norm_nm2 = _legendre_norm_sq(n - 2)

        term_1 = (
            a_n * norm_nm1 / norm_n
        ) * rightHalfShuffleProduct(_ELT1, functionals[n - 1])
        term_2 = (b_n * norm_nm1 / norm_n) * functionals[n - 1]
        term_3 = (c_n * norm_nm2 / norm_n) * functionals[n - 2]

        functionals.append(term_1 + term_2 + term_3)

    return functionals


def _get_legendre_coeffs_from_sig(
    sig: np.ndarray,
    dim: int,
    sig_depth: int,
    n_coeffs: int,
) -> list[np.ndarray]:
    """Extract shifted Legendre coefficients from a signature of (t, x)."""
    if n_coeffs > sig_depth - 2:
        raise ValueError("Legendre inversion requires n_coeffs <= sig_depth - 2.")

    signature_tensors = _sig_vector_to_tensors(sig, dim + 1, sig_depth)
    functionals = _build_legendre_functionals(n_coeffs)

    coeffs: list[np.ndarray] = []
    for dim_index in range(1, dim + 1):
        coeffs_d = np.zeros(n_coeffs + 1)
        for n, functional in enumerate(functionals):
            coeffs_d[n] = _evaluate_linear_functional(
                signature_tensors,
                functional,
                dim_index,
            )
        coeffs.append(coeffs_d)

    return coeffs


def _reconstruct_from_legendre_coeffs(
    t: np.ndarray,
    coeffs: list[np.ndarray],
) -> np.ndarray:
    """Reconstruct path values from shifted Legendre coefficients."""
    z = 2.0 * t - 1.0
    all_dims = [np.polynomial.legendre.legval(z, coeff_d) for coeff_d in coeffs]
    return np.stack(all_dims, axis=1)


def legendre_invert_tensor_signatures(
    sigs: np.ndarray,
    seq_len: int,
    dim: int,
    sig_depth: int,
    *,
    n_coeffs: int | None = None,
    initial_values: np.ndarray | None = None,
    show_progress: bool = True,
) -> np.ndarray:
    """Reconstruct paths from ordinary Legendre tensor signatures."""
    if n_coeffs is None:
        n_coeffs = sig_depth - 2
    if n_coeffs < 0:
        raise ValueError("n_coeffs must be nonnegative; use sig_depth >= 2.")
    if n_coeffs > sig_depth - 2:
        raise ValueError("Legendre inversion requires n_coeffs <= sig_depth - 2.")

    sigs = np.asarray(sigs)
    if sigs.ndim == 1:
        sigs = sigs[None, :]

    n_paths = sigs.shape[0]
    if initial_values is None:
        initial_values = np.zeros((n_paths, dim), dtype=float)
    else:
        initial_values = np.asarray(initial_values, dtype=float)
        if initial_values.shape == (dim,):
            initial_values = np.tile(initial_values[None, :], (n_paths, 1))
        if initial_values.shape != (n_paths, dim):
            raise ValueError(
                "initial_values must have shape (dim,) or (n_paths, dim)."
            )

    t = np.linspace(0.0, 1.0, seq_len)
    reconstructed = []
    iterator = range(n_paths)
    if show_progress:
        iterator = tqdm(iterator, desc="Legendre inversion")

    for i in iterator:
        coeffs = _get_legendre_coeffs_from_sig(sigs[i], dim, sig_depth, n_coeffs)
        rec = _reconstruct_from_legendre_coeffs(t, coeffs)
        rec = rec + initial_values[i][None, :]
        reconstructed.append(rec)

    return np.stack(reconstructed, axis=0)


def _resolve_legendre_sig_depth_and_n_coeffs(
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
        raise ValueError("Legendre inversion requires n_coeffs <= sig_depth - 2.")

    return sig_depth, n_coeffs


def legendre_logsignatures_from_paths(
    paths: np.ndarray,
    sig_depth: int,
    *,
    center_initial: bool = True,
) -> np.ndarray:
    """Compute true log-signatures of Legendre-augmented input paths."""
    normalized, _ = _normalize_path_batch(paths)
    _, _, path_dim = normalized.shape
    augmented, _initial = _augment_for_legendre(normalized, center_initial=center_initial)
    logsig_spec = iisignature.prepare(path_dim + 1, sig_depth)
    logsigs = iisignature.logsig(augmented, logsig_spec)
    return np.asarray(logsigs, dtype=np.float32)


def legendre_invert_logsignatures(
    logsigs: np.ndarray,
    seq_len: int,
    dim: int,
    sig_depth: int,
    *,
    n_coeffs: int | None = None,
    initial_values: np.ndarray | None = None,
    show_progress: bool = True,
) -> np.ndarray:
    """Expand and invert exactly one Legendre log-signature row at a time."""
    if n_coeffs is None:
        n_coeffs = sig_depth - 2
    logsigs = np.asarray(logsigs)
    if logsigs.ndim == 1:
        logsigs = logsigs[None, :]
    expected_width = iisignature.logsiglength(dim + 1, sig_depth)
    if logsigs.ndim != 2 or logsigs.shape[1] != expected_width:
        raise ValueError(
            f"Legendre log-signatures must have shape (n_paths, {expected_width}); "
            f"got {logsigs.shape}."
        )
    if not np.isfinite(logsigs).all():
        raise ValueError("Legendre log-signatures must be finite.")

    n_paths = logsigs.shape[0]
    if initial_values is None:
        normalized_initials = np.zeros((n_paths, dim), dtype=float)
    else:
        normalized_initials = np.asarray(initial_values, dtype=float)
        if normalized_initials.shape == (dim,):
            normalized_initials = np.tile(normalized_initials[None, :], (n_paths, 1))
        if normalized_initials.shape != (n_paths, dim):
            raise ValueError("initial_values must have shape (dim,) or (n_paths, dim).")

    sig_spec = iisignature.prepare(dim + 1, sig_depth, "COSX2")
    reconstructed = []
    iterator = range(n_paths)
    if show_progress:
        iterator = tqdm(iterator, desc="Legendre inversion")
    for i in iterator:
        tensor_sig = iisignature.logsigtosig(logsigs[i], sig_spec)
        rec = legendre_invert_tensor_signatures(
            tensor_sig,
            seq_len,
            dim,
            sig_depth,
            n_coeffs=n_coeffs,
            initial_values=normalized_initials[i],
            show_progress=False,
        )[0]
        del tensor_sig
        reconstructed.append(rec)
    return np.stack(reconstructed, axis=0)


def legendre_invert_paths(
    paths: np.ndarray,
    sig_depth: int | None = None,
    *,
    n_coeffs: int | None = None,
    center_initial: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """Reconstruct paths from their signatures via shifted Legendre inversion."""
    sig_depth, n_coeffs = _resolve_legendre_sig_depth_and_n_coeffs(sig_depth, n_coeffs)

    normalized, squeeze = _normalize_path_batch(paths)
    _, seq_len, path_dim = normalized.shape
    _augmented, initial = _augment_for_legendre(normalized, center_initial=center_initial)
    logsigs = legendre_logsignatures_from_paths(
        normalized,
        sig_depth,
        center_initial=center_initial,
    )

    reconstructed = legendre_invert_logsignatures(
        logsigs,
        seq_len,
        path_dim,
        sig_depth,
        n_coeffs=n_coeffs,
        initial_values=initial[:, 0, :] if center_initial else None,
        show_progress=show_progress,
    )

    return reconstructed[:, :, 0] if squeeze else reconstructed


def legendre_invert_path(
    path: np.ndarray,
    sig_depth: int | None = None,
    *,
    n_coeffs: int | None = None,
    center_initial: bool = True,
) -> np.ndarray:
    """Single-path convenience wrapper around ``legendre_invert_paths``."""
    result = legendre_invert_paths(
        path[None],
        sig_depth,
        n_coeffs=n_coeffs,
        center_initial=center_initial,
        show_progress=False,
    )
    return result[0]


def legendre_invert_path_repeated(
    path: np.ndarray,
    n_repeats: int,
    sig_depth: int | None = None,
    *,
    n_coeffs: int | None = None,
    center_initial: bool = True,
) -> np.ndarray:
    """Reconstruct a single path multiple times for API consistency."""
    tiled = np.tile(path[None], (n_repeats,) + (1,) * path.ndim)
    return legendre_invert_paths(
        tiled,
        sig_depth,
        n_coeffs=n_coeffs,
        center_initial=center_initial,
        show_progress=False,
    )


def run_legendre_method(
    args: argparse.Namespace,
    method_context: InversionMethodContext,
    *,
    legendre_sig_depth: int,
) -> InversionResult:
    """Run the Legendre method and return its canonical candidate payload."""
    paths = method_context.source.paths
    time_grid = method_context.source.time_grid
    n_paths = paths.shape[0]
    center_initial = getattr(args, "center_initial", True)
    n_coeffs = getattr(args, "n_coeffs", None)
    effective_sig_depth = legendre_sig_depth
    if n_coeffs is not None:
        effective_sig_depth = max(legendre_sig_depth, n_coeffs + 2)
    effective_sig_depth, n_coeffs = _resolve_legendre_sig_depth_and_n_coeffs(
        effective_sig_depth,
        n_coeffs,
    )

    print(f"[legendre] inverting {n_paths} paths (sig_depth={effective_sig_depth}) ...")
    t0 = time.perf_counter()
    normalized, squeeze = _normalize_path_batch(paths)
    _, seq_len, path_dim = normalized.shape
    _augmented, initial = _augment_for_legendre(normalized, center_initial=center_initial)
    logsigs = load_or_compute_deterministic_logsignatures(
        method_context.dataset_dir,
        source="legendre",
        sig_depth=effective_sig_depth,
        sig_channels=path_dim + 1,
        n_rows=n_paths,
        compute_logsignatures=lambda: legendre_logsignatures_from_paths(
            normalized,
            effective_sig_depth,
            center_initial=center_initial,
        ),
        metadata={"n_coeffs": n_coeffs, "center_initial": bool(center_initial)},
    )
    gen = legendre_invert_logsignatures(
        logsigs,
        seq_len,
        path_dim,
        effective_sig_depth,
        n_coeffs=n_coeffs,
        initial_values=initial[:, 0, :] if center_initial else None,
    )
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s  ({elapsed / n_paths * 1000:.1f} ms/path)")
    gen = gen[:, :, 0] if squeeze else gen
    return InversionResult(
        paths=np.asarray(gen),
        time_grid=np.asarray(time_grid),
        method="legendre",
        metadata={
            "sig_depth": effective_sig_depth,
            "n_coeffs": n_coeffs,
        },
    )


__all__ = [
    "legendre_invert_path",
    "legendre_invert_path_repeated",
    "legendre_invert_paths",
    "legendre_invert_logsignatures",
    "legendre_invert_tensor_signatures",
    "legendre_logsignatures_from_paths",
    "run_legendre_method",
]
