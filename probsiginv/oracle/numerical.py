"""Numerical LS oracle reconstruction error and the shared Gaussian conditioning engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve

__all__ = [
    "gbm_numerical_mse",
    "fbm_numerical_mse",
    "ou_numerical_mse",
]

LOG_2PI = float(np.log(2.0 * np.pi))

MeanFn = Callable[[np.ndarray, np.ndarray | float], np.ndarray | float]
KernelFn = Callable[
    [np.ndarray, np.ndarray | float, np.ndarray | float], np.ndarray | float
]


@dataclass(frozen=True)
class ProcessGaussianLaw:
    """Canonical parameter order, mean, and covariance for one process family."""

    process: str
    parameter_names: tuple[str, ...]
    covariance_parameter_indices: tuple[int, ...]
    mean_fn: MeanFn
    kernel_fn: KernelFn


@dataclass(frozen=True)
class LSObservationParticle:
    """Compact Gaussian law of one LS observation at a parameter particle.

    Dense path-grid terms are intentionally absent. They are built lazily only
    after this particle is selected for conditional path sampling.
    """

    theta: np.ndarray
    mean_vec: np.ndarray
    factor: np.ndarray
    log_det: float

    def log_likelihood(self, observation: np.ndarray) -> float:
        """Return the Gaussian log likelihood of one LS observation."""
        value = np.asarray(observation, dtype=float)
        if value.shape != self.mean_vec.shape:
            raise ValueError(
                "observation shape must match mean_vec shape "
                f"{self.mean_vec.shape}, got {value.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError("observation must contain only finite values")
        if value.size == 0:
            return 0.0

        delta = value - self.mean_vec
        alpha = cho_solve((self.factor, True), delta, check_finite=False)
        result = _log_gaussian_density(
            delta=delta,
            precision_times_delta=alpha,
            log_det=self.log_det,
        )
        if not np.isfinite(result):
            raise ValueError("observation log likelihood is non-finite")
        return result


@dataclass(frozen=True)
class LSObservationPreparation:
    """Compact particle table plus the exact LS quadrature provenance."""

    law: ProcessGaussianLaw
    particles: tuple[LSObservationParticle, ...]
    depth: int
    T: float
    n_quad_points: int
    jitter: float
    quadrature_grid: np.ndarray
    weighted_quadrature_basis: np.ndarray

    def __len__(self) -> int:
        return len(self.particles)

    def __iter__(self):
        return iter(self.particles)

    def __getitem__(self, index):
        return self.particles[index]


@dataclass(frozen=True)
class FixedThetaPathTerms:
    """Dense path-grid terms built lazily for one selected parameter."""

    theta: np.ndarray
    mean_path: np.ndarray
    K_XS: np.ndarray
    K_XX: np.ndarray
    regression_matrix: np.ndarray
    conditional_covariance: np.ndarray


def build_process_gaussian_law(process: str) -> ProcessGaussianLaw:
    """Return the canonical scalar Gaussian law for GBM, fBM, or OU."""
    normalized = str(process).lower()

    if normalized == "gbm":

        def mean_fn(theta: np.ndarray, t):
            mu_value, sigma_value = theta
            t = np.asarray(t, dtype=float)
            return (mu_value - 0.5 * sigma_value**2) * t

        def kernel_fn(theta: np.ndarray, s, t):
            _, sigma_value = theta
            s = np.asarray(s, dtype=float)
            t = np.asarray(t, dtype=float)
            return sigma_value**2 * np.minimum(s, t)

        return ProcessGaussianLaw(
            process=normalized,
            parameter_names=("mu", "sigma"),
            covariance_parameter_indices=(1,),
            mean_fn=mean_fn,
            kernel_fn=kernel_fn,
        )

    if normalized == "fbm":

        def mean_fn(theta: np.ndarray, t):
            mu_value, sigma_value, _ = theta
            t = np.asarray(t, dtype=float)
            return (mu_value - 0.5 * sigma_value**2) * t

        def kernel_fn(theta: np.ndarray, s, t):
            _, sigma_value, hurst_value = theta
            s = np.asarray(s, dtype=float)
            t = np.asarray(t, dtype=float)
            two_h = 2.0 * hurst_value
            return 0.5 * sigma_value**2 * (
                s**two_h + t**two_h - np.abs(t - s) ** two_h
            )

        return ProcessGaussianLaw(
            process=normalized,
            parameter_names=("mu", "sigma", "hurst"),
            covariance_parameter_indices=(1, 2),
            mean_fn=mean_fn,
            kernel_fn=kernel_fn,
        )

    if normalized == "ou":

        def mean_fn(theta: np.ndarray, t):
            theta_value, mu_value, _ = theta
            t = np.asarray(t, dtype=float)
            return mu_value * (1.0 - np.exp(-theta_value * t))

        def kernel_fn(theta: np.ndarray, s, t):
            theta_value, _, sigma_value = theta
            s = np.asarray(s, dtype=float)
            t = np.asarray(t, dtype=float)
            return (sigma_value**2 / (2.0 * theta_value)) * (
                np.exp(-theta_value * np.abs(t - s))
                - np.exp(-theta_value * (s + t))
            )

        return ProcessGaussianLaw(
            process=normalized,
            parameter_names=("theta", "mu", "sigma"),
            covariance_parameter_indices=(0, 2),
            mean_fn=mean_fn,
            kernel_fn=kernel_fn,
        )

    raise ValueError(f"unsupported process {process!r}; expected gbm, fbm, or ou")


@dataclass(frozen=True)
class PrecomputedParticle:
    """Store particle-specific paths and statistic covariances up to ``r_max``."""

    theta: np.ndarray
    mean_path: np.ndarray
    K_diag: np.ndarray
    mean_vec_max: np.ndarray | None
    cov_vector_max: np.ndarray | None
    cov_matrix_max: np.ndarray | None


@dataclass(frozen=True)
class PreparedDepthParticle:
    """Store depth-specific Gaussian regression terms for one particle."""

    theta: np.ndarray
    mean_path: np.ndarray
    mean_vec: np.ndarray
    cov_vector: np.ndarray
    conditional_var: np.ndarray
    factor: np.ndarray
    log_det: float


def gbm_numerical_mse(
    depths: list[int],
    mu: tuple[float, float],
    sigma: tuple[float, float],
    T_horizon: float = 1.0,
    n_particles_per_dim: int = 25,
    n_outer: int = 512,
    n_time_points: int = 200,
    n_quad_points: int = 200,
    distribution: str = "uniform",
    seed: int = 42,
    eps: float = 0.0,
    jitter: float = 1e-10,
) -> dict[int, dict[str, float]]:
    """Return numerical GBM pairwise path-MSE lower bounds by depth."""
    if distribution != "uniform":
        raise NotImplementedError(f"distribution={distribution!r} not supported")

    law = build_process_gaussian_law("gbm")
    particles = build_midpoint_particles(
        bounds=(mu, sigma),
        n_points=n_particles_per_dim,
    )

    return estimate_bayes_reconstruction_error(
        particles=particles,
        mean_fn=law.mean_fn,
        kernel_fn=law.kernel_fn,
        depths=depths,
        T=T_horizon,
        n_outer=n_outer,
        n_time_points=n_time_points,
        n_quad_points=n_quad_points,
        jitter=jitter,
        seed=seed,
        eps=eps,
    )


def fbm_numerical_mse(
    depths: list[int],
    mu: tuple[float, float],
    sigma: tuple[float, float],
    H: tuple[float, float],
    T_horizon: float = 1.0,
    n_particles_per_dim: int = 25,
    n_outer: int = 5120,
    n_time_points: int = 200,
    n_quad_points: int = 200,
    distribution: str = "uniform",
    seed: int = 42,
    eps: float = 0.0,
    jitter: float = 1e-10,
) -> dict[int, dict[str, float]]:
    """Return numerical fBM pairwise path-MSE lower bounds by depth."""
    if distribution != "uniform":
        raise NotImplementedError(f"distribution={distribution!r} not supported")

    law = build_process_gaussian_law("fbm")
    particles = build_midpoint_particles(
        bounds=(mu, sigma, H),
        n_points=n_particles_per_dim,
    )

    return estimate_bayes_reconstruction_error(
        particles=particles,
        mean_fn=law.mean_fn,
        kernel_fn=law.kernel_fn,
        depths=depths,
        T=T_horizon,
        n_outer=n_outer,
        n_time_points=n_time_points,
        n_quad_points=n_quad_points,
        jitter=jitter,
        seed=seed,
        eps=eps,
    )


def ou_numerical_mse(
    depths: list[int],
    theta: tuple[float, float],
    mu: tuple[float, float],
    sigma: tuple[float, float],
    T_horizon: float = 1.0,
    n_particles_per_dim: int = 25,
    n_outer: int = 512,
    n_time_points: int = 200,
    n_quad_points: int = 200,
    distribution: str = "uniform",
    seed: int = 42,
    eps: float = 0.0,
    jitter: float = 1e-10,
) -> dict[int, dict[str, float]]:
    """Return numerical OU pairwise path-MSE lower bounds by depth."""
    if distribution != "uniform":
        raise NotImplementedError(f"distribution={distribution!r} not supported")

    law = build_process_gaussian_law("ou")
    particles = build_midpoint_particles(
        bounds=(theta, mu, sigma),
        n_points=n_particles_per_dim,
    )

    return estimate_bayes_reconstruction_error(
        particles=particles,
        mean_fn=law.mean_fn,
        kernel_fn=law.kernel_fn,
        depths=depths,
        T=T_horizon,
        n_outer=n_outer,
        n_time_points=n_time_points,
        n_quad_points=n_quad_points,
        jitter=jitter,
        seed=seed,
        eps=eps,
    )


def estimate_bayes_reconstruction_error(
    particles,
    mean_fn: Callable[[np.ndarray, np.ndarray | float], np.ndarray | float],
    kernel_fn: Callable[
        [np.ndarray, np.ndarray | float, np.ndarray | float], np.ndarray | float
    ],
    depths: Sequence[int],
    T: float = 1.0,
    n_outer: int = 1000,
    n_time_points: int = 100,
    n_quad_points: int = 200,
    jitter: float = 1e-10,
    seed: int = 42,
    eps: float = 0.0,
) -> dict[int, dict[str, float]]:
    """Estimate pairwise Bayes reconstruction errors for the requested depths."""
    particles_array = np.asarray(particles, dtype=float)
    if particles_array.ndim != 2:
        raise ValueError("particles must have shape (n_particles, d_theta)")
    if particles_array.shape[0] == 0:
        raise ValueError("particles must contain at least one support point")

    requested_depths = [int(depth) for depth in depths]
    if len(requested_depths) == 0:
        raise ValueError("depths must contain at least one depth")
    requested_depths = list(dict.fromkeys(requested_depths))
    if any(depth < 0 for depth in requested_depths):
        raise ValueError("depths must be non-negative integers")

    T = float(T)
    eps = float(eps)
    if T <= 0.0:
        raise ValueError("T must be positive")
    if n_time_points < 2:
        raise ValueError("n_time_points must be at least 2")
    if n_quad_points < 2:
        raise ValueError("n_quad_points must be at least 2")
    if jitter < 0.0:
        raise ValueError("jitter must be non-negative")
    if eps < 0.0 or eps >= T:
        raise ValueError("eps must satisfy 0 <= eps < T")
    if any(depth > 0 for depth in requested_depths) and n_outer < 1:
        raise ValueError(
            "n_outer must be at least 1 when positive depths are requested"
        )

    t_grid, u_grid = _build_grids(
        T=T,
        n_time_points=n_time_points,
        n_quad_points=n_quad_points,
        eps=eps,
    )
    r_max = max(requested_depths)
    precomputed = _precompute_particles(
        particles=particles_array,
        mean_fn=mean_fn,
        kernel_fn=kernel_fn,
        r_max=r_max,
        t_grid=t_grid,
        u_grid=u_grid,
        jitter=jitter,
    )

    results: dict[int, dict[str, float]] = {}
    for depth in requested_depths:
        if depth == 0:
            results[depth] = _estimate_depth0(precomputed=precomputed, t_grid=t_grid)
        else:
            results[depth] = _estimate_positive_depth(
                precomputed=precomputed,
                r=depth,
                t_grid=t_grid,
                n_outer=int(n_outer),
                seed=int(seed),
            )
    return results


def build_midpoint_particles(
    bounds: Sequence[tuple[float, float]],
    n_points: int,
) -> np.ndarray:
    """Build an equal-weight Cartesian midpoint grid in declared bound order.

    A degenerate interval contributes one support point, so fixed parameters do
    not create duplicate particles.
    """
    if isinstance(n_points, (bool, np.bool_)) or not isinstance(
        n_points, (int, np.integer)
    ):
        raise ValueError("n_points must be an integer")
    n_points = int(n_points)
    if n_points < 1:
        raise ValueError("n_points must be at least 1")

    normalized_bounds: list[tuple[float, float]] = []
    for index, bound in enumerate(bounds):
        try:
            lo, hi = bound
        except (TypeError, ValueError) as exc:
            raise ValueError(f"bounds[{index}] must contain exactly two values") from exc
        lo = float(lo)
        hi = float(hi)
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"bounds[{index}] must contain only finite values")
        if lo > hi:
            raise ValueError(f"bounds[{index}] must satisfy lower <= upper")
        normalized_bounds.append((lo, hi))

    if not normalized_bounds:
        raise ValueError("bounds must contain at least one parameter interval")

    axes = [_midpoint_axis(lo, hi, n_points) for lo, hi in normalized_bounds]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([grid.ravel() for grid in mesh], axis=1)


def _readonly_array(values: np.ndarray) -> np.ndarray:
    """Return an owned float array that cannot mutate shared preparation state."""
    array = np.asarray(values, dtype=float).copy()
    array.setflags(write=False)
    return array


def prepare_ls_observation_particles(
    particles: np.ndarray,
    law: ProcessGaussianLaw,
    depth: int,
    T: float,
    n_quad_points: int = 200,
    jitter: float = 1e-10,
) -> LSObservationPreparation:
    """Prepare a compact LS table with exact numerical-policy provenance.

    Each particle retains only ``theta``, ``mean_vec``, ``factor``, and
    ``log_det``. The shared context owns the quadrature grid/basis needed later,
    so fixed-theta path terms cannot silently use a different policy. Covariance
    factors are reused within this call for equal covariance parameters.
    """
    particles_array = np.asarray(particles, dtype=float)
    if particles_array.ndim != 2:
        raise ValueError("particles must have shape (n_particles, d_theta)")
    if particles_array.shape[0] == 0:
        raise ValueError("particles must contain at least one support point")
    if particles_array.shape[1] != len(law.parameter_names):
        raise ValueError(
            f"{law.process} particles must have {len(law.parameter_names)} columns"
        )
    if not np.all(np.isfinite(particles_array)):
        raise ValueError("particles must contain only finite values")

    if isinstance(depth, (bool, np.bool_)) or not isinstance(
        depth, (int, np.integer)
    ):
        raise ValueError("depth must be a non-negative integer")
    depth = int(depth)
    if depth < 0:
        raise ValueError("depth must be a non-negative integer")

    T = float(T)
    jitter = float(jitter)
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T must be finite and positive")
    if not np.isfinite(jitter) or jitter < 0.0:
        raise ValueError("jitter must be finite and non-negative")
    if isinstance(n_quad_points, (bool, np.bool_)) or not isinstance(
        n_quad_points, (int, np.integer)
    ):
        raise ValueError("n_quad_points must be an integer")
    n_quad_points = int(n_quad_points)
    if n_quad_points < 2:
        raise ValueError("n_quad_points must be at least 2")

    u_grid = _readonly_array(np.linspace(0.0, T, n_quad_points, dtype=float))
    u_weights = _trapezoid_weights(u_grid)
    weighted_u_basis = _readonly_array(
        u_weights[:, None] * _power_basis(u_grid, max(depth - 1, 0))
    )

    if depth == 0:
        empty_mean = _readonly_array(np.empty(0, dtype=float))
        empty_factor = _readonly_array(np.empty((0, 0), dtype=float))
        prepared = tuple(
            LSObservationParticle(
                theta=_readonly_array(theta),
                mean_vec=empty_mean,
                factor=empty_factor,
                log_det=0.0,
            )
            for theta in particles_array
        )
        return LSObservationPreparation(
            law=law,
            particles=prepared,
            depth=depth,
            T=T,
            n_quad_points=n_quad_points,
            jitter=jitter,
            quadrature_grid=u_grid,
            weighted_quadrature_basis=weighted_u_basis,
        )

    prepared: list[LSObservationParticle] = []
    covariance_cache: dict[tuple[float, ...], tuple[np.ndarray, float]] = {}
    for particle_index, theta in enumerate(particles_array):
        theta_array = np.asarray(theta, dtype=float)
        mean_vec = _compute_mean_vec_max(
            theta=theta_array,
            mean_fn=law.mean_fn,
            r_max=depth,
            T=T,
            u_grid=u_grid,
            weighted_u_basis=weighted_u_basis,
        )
        if mean_vec is None:
            raise RuntimeError("positive-depth LS mean was not constructed")
        if not np.all(np.isfinite(mean_vec)):
            raise ValueError(
                f"non-finite LS statistic mean for process={law.process}, "
                f"depth={depth}, particle_index={particle_index}, "
                f"theta={theta_array.tolist()}"
            )

        covariance_key = tuple(
            float(theta_array[index])
            for index in law.covariance_parameter_indices
        )
        cached = covariance_cache.get(covariance_key)
        if cached is None:
            covariance = _compute_cov_matrix_max(
                theta=theta_array,
                kernel_fn=law.kernel_fn,
                r_max=depth,
                T=T,
                u_grid=u_grid,
                weighted_u_basis=weighted_u_basis,
                jitter=jitter,
            )
            if covariance is None:
                raise RuntimeError("positive-depth LS covariance was not constructed")
            if not np.all(np.isfinite(covariance)):
                raise ValueError(
                    f"non-finite LS statistic covariance for process={law.process}, "
                    f"depth={depth}, particle_index={particle_index}, "
                    f"theta={theta_array.tolist()}"
                )
            factor = _readonly_array(
                np.tril(
                    _factorize_covariance(
                        cov_matrix=covariance,
                        r=depth,
                        particle_index=particle_index,
                        theta=theta_array,
                    )
                )
            )
            log_det = float(2.0 * np.log(np.diag(factor)).sum())
            if not np.isfinite(log_det):
                raise ValueError(
                    f"non-finite LS statistic log determinant for "
                    f"process={law.process}, depth={depth}, "
                    f"particle_index={particle_index}, theta={theta_array.tolist()}"
                )
            cached = (factor, log_det)
            covariance_cache[covariance_key] = cached

        factor, log_det = cached
        prepared.append(
            LSObservationParticle(
                theta=_readonly_array(theta_array),
                mean_vec=_readonly_array(mean_vec),
                factor=factor,
                log_det=log_det,
            )
        )
    return LSObservationPreparation(
        law=law,
        particles=tuple(prepared),
        depth=depth,
        T=T,
        n_quad_points=n_quad_points,
        jitter=jitter,
        quadrature_grid=u_grid,
        weighted_quadrature_basis=weighted_u_basis,
    )


def prepare_fixed_theta_path_terms(
    preparation: LSObservationPreparation,
    particle_index: int,
    physical_time_grid: np.ndarray,
) -> FixedThetaPathTerms:
    """Build dense conditional-GP terms for one selected parameter only.

    The exact observation quadrature context is reused. Cholesky solves produce
    both the regression matrix and full Schur complement; no covariance inverse
    is formed.
    """
    if isinstance(particle_index, (bool, np.bool_)) or not isinstance(
        particle_index, (int, np.integer)
    ):
        raise ValueError("particle_index must be an integer")
    particle_index = int(particle_index)
    if particle_index < 0 or particle_index >= len(preparation):
        raise ValueError(
            f"particle_index must satisfy 0 <= index < {len(preparation)}"
        )

    law = preparation.law
    observation_particle = preparation[particle_index]
    theta_array = observation_particle.theta

    time_grid = np.asarray(physical_time_grid, dtype=float)
    if time_grid.ndim != 1 or time_grid.size == 0:
        raise ValueError("physical_time_grid must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(time_grid)):
        raise ValueError("physical_time_grid must contain only finite values")
    if np.any(np.diff(time_grid) < 0.0):
        raise ValueError("physical_time_grid must be non-decreasing")
    T = float(time_grid[-1])
    if T <= 0.0:
        raise ValueError("physical_time_grid must end at a positive horizon")
    horizon_tolerance = 1e-12 * max(1.0, abs(preparation.T))
    if not np.isclose(T, preparation.T, rtol=0.0, atol=horizon_tolerance):
        raise ValueError(
            f"physical_time_grid horizon {T} does not match prepared "
            f"horizon {preparation.T}"
        )

    depth = preparation.depth
    if observation_particle.mean_vec.shape != (depth,):
        raise ValueError("observation particle mean shape does not match prepared depth")
    if observation_particle.factor.shape != (depth, depth):
        raise ValueError("observation particle factor shape does not match its mean")

    mean_path = _as_broadcast_array(
        law.mean_fn(theta_array, time_grid),
        time_grid.shape,
        "mean_fn",
    )
    K_XX = _as_broadcast_array(
        law.kernel_fn(theta_array, time_grid[:, None], time_grid[None, :]),
        (time_grid.size, time_grid.size),
        "kernel_fn",
    )
    K_XX = 0.5 * (K_XX + K_XX.T)

    if depth == 0:
        K_XS = np.empty((time_grid.size, 0), dtype=float)
        regression_matrix = np.empty((time_grid.size, 0), dtype=float)
        conditional_covariance = K_XX.copy()
    else:
        K_XS = _compute_cov_vector_max(
            theta=theta_array,
            kernel_fn=law.kernel_fn,
            r_max=depth,
            T=preparation.T,
            t_grid=time_grid,
            u_grid=preparation.quadrature_grid,
            weighted_u_basis=preparation.weighted_quadrature_basis,
        )
        if K_XS is None:
            raise RuntimeError("positive-depth path/statistic covariance was not constructed")
        solved_K_SX = cho_solve(
            (observation_particle.factor, True),
            K_XS.T,
            check_finite=False,
        )
        regression_matrix = solved_K_SX.T
        conditional_covariance = K_XX - K_XS @ solved_K_SX
        conditional_covariance = 0.5 * (
            conditional_covariance + conditional_covariance.T
        )

    arrays = (mean_path, K_XS, K_XX, regression_matrix, conditional_covariance)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError(
            f"non-finite fixed-theta path terms for process={law.process}, "
            f"depth={depth}, theta={theta_array.tolist()}"
        )
    return FixedThetaPathTerms(
        theta=theta_array.copy(),
        mean_path=mean_path,
        K_XS=K_XS,
        K_XX=K_XX,
        regression_matrix=regression_matrix,
        conditional_covariance=conditional_covariance,
    )


def _cartesian_midpoint_particles(
    *bounds: tuple[float, float],
    n_points: int,
) -> np.ndarray:
    """Build an equal-weight Cartesian midpoint grid over the given bounds."""
    return build_midpoint_particles(bounds=bounds, n_points=n_points)


def _midpoint_axis(lo: float, hi: float, n_points: int) -> np.ndarray:
    """Return midpoint-rule support points on a uniform interval."""
    if n_points < 1:
        raise ValueError("n_points must be at least 1")
    if abs(hi - lo) < 1e-15:
        return np.array([float(lo)], dtype=float)
    step = (hi - lo) / n_points
    return lo + (np.arange(n_points, dtype=float) + 0.5) * step


def _build_grids(
    T: float,
    n_time_points: int,
    n_quad_points: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the time and quadrature grids on ``[eps, T]``."""
    t_grid = np.linspace(eps, T, n_time_points, dtype=float)
    u_grid = np.linspace(eps, T, n_quad_points, dtype=float)
    return t_grid, u_grid


def _trapezoid_weights(grid: np.ndarray) -> np.ndarray:
    """Return one-dimensional trapezoid weights for the given grid."""
    grid = np.asarray(grid, dtype=float)
    deltas = np.diff(grid)
    weights = np.empty_like(grid)
    weights[0] = 0.5 * deltas[0]
    weights[-1] = 0.5 * deltas[-1]
    if grid.size > 2:
        weights[1:-1] = 0.5 * (deltas[:-1] + deltas[1:])
    return weights


def _as_broadcast_array(values, shape: tuple[int, ...], fn_name: str) -> np.ndarray:
    """Broadcast a callable output to the requested shape."""
    array = np.asarray(values, dtype=float)
    squeezed = np.squeeze(array)

    if shape == ():
        if squeezed.shape == ():
            return np.asarray(squeezed, dtype=float)
        if squeezed.size == 1:
            return np.asarray(squeezed.reshape(()), dtype=float)
        raise ValueError(
            f"{fn_name} returned shape {array.shape}, "
            "which cannot broadcast to scalar output"
        )

    for candidate in (squeezed, array):
        try:
            return np.asarray(np.broadcast_to(candidate, shape), dtype=float)
        except ValueError:
            continue
    raise ValueError(
        f"{fn_name} returned shape {array.shape}, "
        f"which cannot broadcast to {shape}"
    )


def _power_basis(grid: np.ndarray, degree: int) -> np.ndarray:
    """Return columns ``grid**0`` through ``grid**(degree - 1)``."""
    if degree <= 0:
        return np.empty((grid.size, 0), dtype=float)
    powers = np.arange(degree, dtype=float)
    return grid[:, None] ** powers[None, :]


def _precompute_particles(
    particles: np.ndarray,
    mean_fn: Callable[[np.ndarray, np.ndarray | float], np.ndarray | float],
    kernel_fn: Callable[
        [np.ndarray, np.ndarray | float, np.ndarray | float], np.ndarray | float
    ],
    r_max: int,
    t_grid: np.ndarray,
    u_grid: np.ndarray,
    jitter: float,
) -> list[PrecomputedParticle]:
    """Precompute mean paths and Gaussian statistic terms for each particle."""
    T = float(u_grid[-1])
    u_weights = _trapezoid_weights(u_grid)
    weighted_u_basis = u_weights[:, None] * _power_basis(u_grid, max(r_max - 1, 0))

    precomputed: list[PrecomputedParticle] = []
    for theta in particles:
        theta_array = np.asarray(theta, dtype=float)
        mean_path = _as_broadcast_array(
            mean_fn(theta_array, t_grid), t_grid.shape, "mean_fn"
        )
        k_diag = _as_broadcast_array(
            kernel_fn(theta_array, t_grid, t_grid), t_grid.shape, "kernel_fn"
        )

        mean_vec_max = _compute_mean_vec_max(
            theta=theta_array,
            mean_fn=mean_fn,
            r_max=r_max,
            T=T,
            u_grid=u_grid,
            weighted_u_basis=weighted_u_basis,
        )
        cov_vector_max = _compute_cov_vector_max(
            theta=theta_array,
            kernel_fn=kernel_fn,
            r_max=r_max,
            T=T,
            t_grid=t_grid,
            u_grid=u_grid,
            weighted_u_basis=weighted_u_basis,
        )
        cov_matrix_max = _compute_cov_matrix_max(
            theta=theta_array,
            kernel_fn=kernel_fn,
            r_max=r_max,
            T=T,
            u_grid=u_grid,
            weighted_u_basis=weighted_u_basis,
            jitter=jitter,
        )

        precomputed.append(
            PrecomputedParticle(
                theta=theta_array.copy(),
                mean_path=mean_path,
                K_diag=k_diag,
                mean_vec_max=mean_vec_max,
                cov_vector_max=cov_vector_max,
                cov_matrix_max=cov_matrix_max,
            )
        )
    return precomputed


def _compute_mean_vec_max(
    theta: np.ndarray,
    mean_fn: Callable[[np.ndarray, np.ndarray | float], np.ndarray | float],
    r_max: int,
    T: float,
    u_grid: np.ndarray,
    weighted_u_basis: np.ndarray,
) -> np.ndarray | None:
    """Compute ``L^(r_max)(m_theta)`` for one particle."""
    if r_max == 0:
        return None

    mean_vec = np.empty(r_max, dtype=float)
    mean_vec[0] = float(_as_broadcast_array(mean_fn(theta, T), (), "mean_fn")[()])
    if r_max == 1:
        return mean_vec

    mean_u = _as_broadcast_array(mean_fn(theta, u_grid), u_grid.shape, "mean_fn")
    mean_vec[1:] = weighted_u_basis.T @ mean_u
    return mean_vec


def _compute_cov_vector_max(
    theta: np.ndarray,
    kernel_fn: Callable[
        [np.ndarray, np.ndarray | float, np.ndarray | float], np.ndarray | float
    ],
    r_max: int,
    T: float,
    t_grid: np.ndarray,
    u_grid: np.ndarray,
    weighted_u_basis: np.ndarray,
) -> np.ndarray | None:
    """Compute ``Cov_theta(Z_t, L^(r_max)(Z))`` on the time grid."""
    if r_max == 0:
        return None

    cov_vector = np.empty((t_grid.size, r_max), dtype=float)
    cov_vector[:, 0] = _as_broadcast_array(
        kernel_fn(theta, t_grid, T), t_grid.shape, "kernel_fn"
    )
    if r_max == 1:
        return cov_vector

    kernel_tu = _as_broadcast_array(
        kernel_fn(theta, t_grid[:, None], u_grid[None, :]),
        (t_grid.size, u_grid.size),
        "kernel_fn",
    )
    cov_vector[:, 1:] = kernel_tu @ weighted_u_basis
    return cov_vector


def _compute_cov_matrix_max(
    theta: np.ndarray,
    kernel_fn: Callable[
        [np.ndarray, np.ndarray | float, np.ndarray | float], np.ndarray | float
    ],
    r_max: int,
    T: float,
    u_grid: np.ndarray,
    weighted_u_basis: np.ndarray,
    jitter: float,
) -> np.ndarray | None:
    """Compute ``Cov_theta(L^(r_max)(Z), L^(r_max)(Z))`` for one particle."""
    if r_max == 0:
        return None

    cov_matrix = np.empty((r_max, r_max), dtype=float)
    cov_matrix[0, 0] = float(_as_broadcast_array(kernel_fn(theta, T, T), (), "kernel_fn")[()])
    if r_max == 1:
        cov_matrix += jitter * np.eye(1, dtype=float)
        return cov_matrix

    kernel_tu = _as_broadcast_array(kernel_fn(theta, T, u_grid), u_grid.shape, "kernel_fn")
    cov_matrix[0, 1:] = kernel_tu @ weighted_u_basis
    cov_matrix[1:, 0] = cov_matrix[0, 1:]

    kernel_uu = _as_broadcast_array(
        kernel_fn(theta, u_grid[:, None], u_grid[None, :]),
        (u_grid.size, u_grid.size),
        "kernel_fn",
    )
    cov_matrix[1:, 1:] = weighted_u_basis.T @ kernel_uu @ weighted_u_basis
    cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
    cov_matrix += jitter * np.eye(r_max, dtype=float)
    return cov_matrix


def _estimate_depth0(
    precomputed: Sequence[PrecomputedParticle],
    t_grid: np.ndarray,
) -> dict[str, float]:
    """Compute the depth-0 baseline with no posterior update."""
    mean_paths = np.stack([particle.mean_path for particle in precomputed], axis=0)
    kernel_diagonals = np.stack([particle.K_diag for particle in precomputed], axis=0)
    integrand = kernel_diagonals.mean(axis=0) + mean_paths.var(axis=0, ddof=0)
    return {
        "mean": float(2.0 * np.trapezoid(integrand, t_grid)),
        "stderr": 0.0,
    }


def _estimate_positive_depth(
    precomputed: Sequence[PrecomputedParticle],
    r: int,
    t_grid: np.ndarray,
    n_outer: int,
    seed: int,
) -> dict[str, float]:
    """Estimate the Bayes reconstruction error for one positive depth."""
    rng = _rng_for_depth(seed=seed, depth=r)
    prepared = _prepare_depth_particles(precomputed=precomputed, r=r)

    conditional_vars = np.stack(
        [particle.conditional_var for particle in prepared], axis=0
    )
    replications = np.empty(n_outer, dtype=float)

    for draw_idx in range(n_outer):
        true_index = int(rng.integers(len(prepared)))
        sample = _sample_statistic(prepared[true_index], rng)

        log_weights = np.empty(len(prepared), dtype=float)
        conditional_means = np.empty((len(prepared), t_grid.size), dtype=float)

        for particle_index, particle in enumerate(prepared):
            delta = sample - particle.mean_vec
            alpha = cho_solve((particle.factor, True), delta, check_finite=False)
            log_weights[particle_index] = _log_gaussian_density(
                delta=delta,
                precision_times_delta=alpha,
                log_det=particle.log_det,
            )
            conditional_means[particle_index] = (
                particle.mean_path + particle.cov_vector @ alpha
            )

        weights = np.exp(log_weights - _logsumexp(log_weights))
        posterior_second_moment = weights @ (conditional_means * conditional_means)
        posterior_mean = weights @ conditional_means
        posterior_var = (
            weights @ conditional_vars
            + posterior_second_moment
            - posterior_mean**2
        )
        replications[draw_idx] = float(
            2.0 * np.trapezoid(np.maximum(posterior_var, 0.0), t_grid)
        )

    stderr = 0.0
    if n_outer > 1:
        stderr = float(replications.std(ddof=1) / np.sqrt(n_outer))
    return {
        "mean": float(replications.mean()),
        "stderr": stderr,
    }


def _prepare_depth_particles(
    precomputed: Sequence[PrecomputedParticle],
    r: int,
) -> list[PreparedDepthParticle]:
    """Slice particle precomputes to depth ``r`` and pre-factorize covariances."""
    prepared: list[PreparedDepthParticle] = []
    for particle_index, particle in enumerate(precomputed):
        if (
            particle.mean_vec_max is None
            or particle.cov_vector_max is None
            or particle.cov_matrix_max is None
        ):
            raise ValueError(
                f"particle precompute is missing positive-depth terms for depth {r}"
            )

        mean_vec = np.asarray(particle.mean_vec_max[:r], dtype=float)
        cov_vector = np.asarray(particle.cov_vector_max[:, :r], dtype=float)
        cov_matrix = np.asarray(particle.cov_matrix_max[:r, :r], dtype=float)

        factor = _factorize_covariance(
            cov_matrix=cov_matrix,
            r=r,
            particle_index=particle_index,
            theta=particle.theta,
        )
        solved_cov_vector = cho_solve((factor, True), cov_vector.T, check_finite=False)
        quadratic = np.sum(cov_vector * solved_cov_vector.T, axis=1)
        conditional_var = np.maximum(particle.K_diag - quadratic, 0.0)
        log_det = float(2.0 * np.log(np.diag(factor)).sum())

        prepared.append(
            PreparedDepthParticle(
                theta=particle.theta,
                mean_path=particle.mean_path,
                mean_vec=mean_vec,
                cov_vector=cov_vector,
                conditional_var=conditional_var,
                factor=factor,
                log_det=log_det,
            )
        )
    return prepared


def _factorize_covariance(
    cov_matrix: np.ndarray,
    r: int,
    particle_index: int,
    theta: np.ndarray,
) -> np.ndarray:
    """Compute a Cholesky factor for one particle-depth covariance matrix."""
    try:
        factor, _ = cho_factor(cov_matrix, lower=True, check_finite=False)
    except LinAlgError as exc:
        raise ValueError(
            f"Cholesky factorization failed for depth={r}, "
            f"particle_index={particle_index}, theta={theta.tolist()}"
        ) from exc
    return factor


def _sample_statistic(
    particle: PreparedDepthParticle,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one observed statistic vector from a particle's Gaussian law."""
    chol = np.tril(particle.factor)
    return particle.mean_vec + chol @ rng.standard_normal(particle.mean_vec.shape[0])


def _log_gaussian_density(
    delta: np.ndarray,
    precision_times_delta: np.ndarray,
    log_det: float,
) -> float:
    """Compute a Gaussian log density from centered data and a solved system."""
    dim = delta.shape[0]
    return float(-0.5 * (dim * LOG_2PI + log_det + delta @ precision_times_delta))


def _logsumexp(logw: np.ndarray) -> float:
    """Compute ``log(sum(exp(logw)))`` stably for one-dimensional inputs."""
    max_logw = float(np.max(logw))
    return float(max_logw + np.log(np.exp(logw - max_logw).sum()))


def _rng_for_depth(seed: int, depth: int) -> np.random.Generator:
    """Create a depth-local RNG so one depth does not depend on others."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(depth)]))
