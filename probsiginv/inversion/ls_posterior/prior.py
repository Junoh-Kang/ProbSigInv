"""Resolve declared split priors without inspecting row parameter values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from probsiginv.oracle.numerical import build_midpoint_particles


_PROCESS_PARAMETER_NAMES = {
    "gbm": ("mu", "sigma"),
    "fbm": ("mu", "sigma", "hurst"),
    "ou": ("theta", "mu", "sigma"),
}
_MISSING = object()


@dataclass(frozen=True)
class DeclaredSplitPrior:
    """Canonical names, declared bounds, and their midpoint support."""

    parameter_names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
    particles: np.ndarray


def resolve_declared_split_prior(
    process_metadata: Mapping[str, object],
    explicit_bounds: Mapping[str, object | None],
    *,
    n_particles_per_dim: int = 25,
) -> DeclaredSplitPrior:
    """Resolve a box prior from declarations and split-wide scalar metadata.

    Explicit bounds fully determine row-varying parameters. Their values are
    never converted, iterated, indexed, or summarized.
    """
    n_particles_per_dim = _validate_particle_count(n_particles_per_dim)
    process = str(process_metadata.get("process", "")).lower()
    if process not in _PROCESS_PARAMETER_NAMES:
        raise ValueError(
            f"unsupported process {process!r}; expected gbm, fbm, or ou"
        )
    parameter_names = _PROCESS_PARAMETER_NAMES[process]

    supplied_names = {
        name for name, bound in explicit_bounds.items() if bound is not None
    }
    irrelevant = sorted(supplied_names - set(parameter_names))
    if irrelevant:
        raise ValueError(
            f"irrelevant explicit bounds for process {process}: {', '.join(irrelevant)}"
        )

    resolved_bounds = tuple(
        _resolve_parameter_bound(
            name=name,
            metadata_value=process_metadata.get(name, _MISSING),
            explicit_bound=explicit_bounds.get(name),
        )
        for name in parameter_names
    )
    particles = _readonly_float_array(
        build_midpoint_particles(
            bounds=resolved_bounds,
            n_points=n_particles_per_dim,
        )
    )
    return DeclaredSplitPrior(
        parameter_names=parameter_names,
        bounds=resolved_bounds,
        particles=particles,
    )


def _resolve_parameter_bound(
    *,
    name: str,
    metadata_value: object,
    explicit_bound: object | None,
) -> tuple[float, float]:
    declared = (
        None if explicit_bound is None else _validate_bound(name, explicit_bound)
    )
    if metadata_value is _MISSING:
        if declared is None:
            raise ValueError(f"{name} requires explicit bounds when metadata is missing")
        return declared

    if _metadata_ndim(metadata_value) != 0:
        if declared is None:
            raise ValueError(
                f"{name} row-varying metadata requires explicit bounds"
            )
        return declared

    scalar = _load_scalar_parameter(name, metadata_value)
    fixed_bound = (scalar, scalar)
    if declared is None:
        return fixed_bound
    if declared != fixed_bound:
        raise ValueError(
            f"{name} bounds {declared} must equal fixed split parameter {fixed_bound}"
        )
    return declared


def _metadata_ndim(value: object) -> int:
    ndim = getattr(value, "ndim", None)
    if ndim is not None:
        return int(ndim)
    return 0 if np.isscalar(value) else 1


def _load_scalar_parameter(name: str, value: object) -> float:
    array = np.asarray(value)
    if array.ndim != 0:
        raise RuntimeError(f"{name} scalar classification changed during conversion")
    try:
        scalar = float(array.item())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} fixed split parameter must be numeric") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} fixed split parameter must be finite")
    return scalar


def _validate_bound(name: str, bound: object) -> tuple[float, float]:
    try:
        lo, hi = bound
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} bounds must contain exactly two values") from exc
    try:
        lo = float(lo)
        hi = float(hi)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} bounds must be numeric") from exc
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError(f"{name} bounds must contain only finite values")
    if lo > hi:
        raise ValueError(f"{name} bounds must satisfy lower <= upper")
    return lo, hi


def _validate_particle_count(n_particles_per_dim: int) -> int:
    if isinstance(n_particles_per_dim, (bool, np.bool_)) or not isinstance(
        n_particles_per_dim, (int, np.integer)
    ):
        raise ValueError("n_particles_per_dim must be a positive integer")
    value = int(n_particles_per_dim)
    if value < 1:
        raise ValueError("n_particles_per_dim must be a positive integer")
    return value


def _readonly_float_array(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


__all__ = ["DeclaredSplitPrior", "resolve_declared_split_prior"]
