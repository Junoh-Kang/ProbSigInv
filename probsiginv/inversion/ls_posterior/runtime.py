"""Runtime assembly for leakage-free LS Posterior reference sampling."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from probsiginv.oracle.numerical import (
    build_process_gaussian_law,
    prepare_ls_observation_particles,
)
from probsiginv.inversion.artifacts import validate_existing_ls_posterior_artifact
from probsiginv.inversion.contracts import InversionResult
from probsiginv.inversion.ls_posterior.prior import resolve_declared_split_prior
from probsiginv.inversion.ls_posterior.sampling import sample_ls_posterior


_SUPPORTED_PROCESSES = {"gbm", "fbm", "ou"}
_OBSERVATION_DIGEST_DOMAIN = b"invertsig:ls-posterior-observation:v1\0"
_OBSERVATION_DIGEST_DTYPE = np.dtype("<f8")


def run_ls_posterior_method(
    *,
    n_rows: int,
    time_grid: np.ndarray,
    process_metadata: Mapping[str, Any],
    declared_bounds: Mapping[str, object | None],
    observations: np.ndarray,
    source_signature_artifact: str | None,
    sig_depth: int,
    n_particles_per_dim: int = 25,
    n_repeats: int = 30,
    seed: int = 2026,
    existing_artifact_path: str | os.PathLike[str] | None = None,
) -> InversionResult | None:
    """Sample an LS Posterior without accepting source path values.

    When ``existing_artifact_path`` exists, its complete reference contract is
    validated before any likelihood or posterior preparation. A compatible
    artifact returns ``None`` so the caller can report a safe cache hit.
    """
    process = str(process_metadata.get("process", "")).strip().lower()
    depth = _nonnegative_integer(sig_depth, "sig_depth")
    if process not in _SUPPORTED_PROCESSES:
        raise ValueError(
            f"unsupported LS posterior process={process!r}, depth={depth}; "
            "expected gbm, fbm, or ou"
        )
    path_dim = _positive_integer(
        process_metadata.get("path_dim"),
        "path_dim",
    )
    if path_dim != 1:
        raise ValueError(
            f"LS posterior requires scalar path_dim=1 for process={process}, "
            f"depth={depth}; got {path_dim}"
        )

    n_rows = _positive_integer(n_rows, "n_rows")
    n_repeats = _positive_integer(n_repeats, "n_repeats")
    n_particles_per_dim = _positive_integer(
        n_particles_per_dim,
        "n_particles_per_dim",
    )
    seed = _nonnegative_integer(seed, "seed")
    horizon = _positive_finite_float(
        process_metadata.get("T_horizon"),
        "T_horizon",
    )
    normalized_time_grid = _validate_normalized_time_grid(
        time_grid,
        process_metadata=process_metadata,
    )
    observation_batch = _validate_observations(
        observations,
        n_rows=n_rows,
        depth=depth,
        process=process,
    )
    observation_sha256 = _observation_sha256(observation_batch)

    source_depth = _nonnegative_integer(
        process_metadata.get("sig_depth", 0),
        "source sig_depth",
    )
    if depth > source_depth:
        raise ValueError(
            f"requested depth {depth} exceeds source depth {source_depth} "
            f"for process={process}"
        )
    source_artifact = "" if source_signature_artifact is None else str(
        source_signature_artifact
    )
    if depth > 0 and not source_artifact:
        raise ValueError(
            f"positive-depth LS posterior requires source signature artifact "
            f"for process={process}, depth={depth}"
        )

    try:
        prior = resolve_declared_split_prior(
            process_metadata,
            declared_bounds,
            n_particles_per_dim=n_particles_per_dim,
        )
    except ValueError as exc:
        raise ValueError(
            f"invalid declared prior for process={process}, depth={depth}: {exc}"
        ) from exc
    metadata = {
        "kind": "reference",
        "conditioning": f"LSd{depth}",
        "n_conditioning_rows": n_rows,
        "n_repeats": n_repeats,
        "seed": seed,
        "n_particles_per_dim": n_particles_per_dim,
        "process": process,
        "path_dim": path_dim,
        "T_horizon": horizon,
        "source_signature_artifact": source_artifact,
        "observation_sha256": observation_sha256,
        "continuous_ls": True,
        **{
            f"{name}_bounds": np.asarray(bound, dtype=float)
            for name, bound in zip(prior.parameter_names, prior.bounds)
        },
    }

    if existing_artifact_path is not None and Path(existing_artifact_path).exists():
        validate_existing_ls_posterior_artifact(
            existing_artifact_path,
            expected_metadata=metadata,
            expected_time_grid=normalized_time_grid,
        )
        return None

    law = build_process_gaussian_law(process)
    preparation = prepare_ls_observation_particles(
        particles=prior.particles,
        law=law,
        depth=depth,
        T=horizon,
    )
    physical_time_grid = horizon * normalized_time_grid
    samples = sample_ls_posterior(
        observation_batch,
        preparation,
        physical_time_grid=physical_time_grid,
        n_repeats=n_repeats,
        seed=seed,
    )
    return InversionResult(
        paths=samples.paths,
        time_grid=normalized_time_grid,
        method="ls_posterior",
        metadata=metadata,
        auxiliary_arrays={
            name: np.asarray(values)
            for name, values in samples.sampled_parameters.items()
        },
    )


def _validate_normalized_time_grid(
    time_grid: np.ndarray,
    *,
    process_metadata: Mapping[str, Any],
) -> np.ndarray:
    if np.iscomplexobj(time_grid):
        raise ValueError("time_grid must be real")
    grid = np.asarray(time_grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)):
        raise ValueError("time_grid must be a finite one-dimensional grid")
    if np.any(np.diff(grid) < 0.0):
        raise ValueError("time_grid must be non-decreasing")
    if not np.isclose(grid[0], 0.0, rtol=0.0, atol=1e-12) or not np.isclose(
        grid[-1],
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("LS posterior source time_grid must be normalized to [0,1]")
    if process_metadata.get("n_steps") is not None:
        n_steps = _nonnegative_integer(process_metadata["n_steps"], "n_steps")
        if grid.size != n_steps + 1:
            raise ValueError(
                f"time_grid length {grid.size} does not match n_steps={n_steps}"
            )
    return np.array(grid, dtype=float, copy=True)


def _validate_observations(
    observations: np.ndarray,
    *,
    n_rows: int,
    depth: int,
    process: str,
) -> np.ndarray:
    if np.iscomplexobj(observations):
        raise ValueError(
            f"LS observations must be real for process={process}, depth={depth}"
        )
    values = np.asarray(observations, dtype=float)
    expected_shape = (n_rows, depth)
    if values.shape != expected_shape or not np.all(np.isfinite(values)):
        raise ValueError(
            f"LS observations must be finite with shape {expected_shape} "
            f"for process={process}, depth={depth}; got {values.shape}"
    )
    return values


def _observation_sha256(observations: np.ndarray) -> str:
    """Hash canonical validated LS observations for reference cache identity."""
    canonical = np.ascontiguousarray(
        observations,
        dtype=_OBSERVATION_DIGEST_DTYPE,
    )
    digest = hashlib.sha256()
    digest.update(_OBSERVATION_DIGEST_DOMAIN)
    digest.update(b"dtype=<f8;order=C\0")
    digest.update(len(canonical.shape).to_bytes(4, byteorder="big"))
    for dimension in canonical.shape:
        digest.update(int(dimension).to_bytes(8, byteorder="big"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _positive_finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_integer(value: Any, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


__all__ = ["run_ls_posterior_method"]
