"""Hierarchical parameter and conditional-Gaussian sampling for LS observations."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from probsiginv.oracle.numerical import (
    LSObservationPreparation,
    prepare_fixed_theta_path_terms,
)


_EIGEN_TOLERANCE_SCALE = 1e-10
_EIGEN_POLICY_IDENTITY = (
    "symmetrize-eigh-clip-strict-positive-v1",
    _EIGEN_TOLERANCE_SCALE,
)
_DEFAULT_FACTOR_CACHE_CAPACITY = 128


@dataclass(frozen=True)
class ConditionalCovarianceFactor:
    """The clipped PSD covariance and its retained positive-eigenvalue factor."""

    covariance: np.ndarray
    factor: np.ndarray
    eigenvalues: np.ndarray
    tolerance: float


@dataclass(frozen=True)
class FixedThetaConditional:
    """Conditional grid law for one observation and one parameter particle."""

    mean: np.ndarray
    covariance: np.ndarray
    factor: np.ndarray


@dataclass(frozen=True)
class LSPosteriorSamples:
    """Sampled paths and their aligned named process parameters."""

    paths: np.ndarray
    sampled_parameters: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class _DeterministicConditionalTerms:
    """Cacheable fixed-particle terms; sampled paths and noise never enter here."""

    mean_path: np.ndarray
    regression_matrix: np.ndarray
    covariance: np.ndarray
    factor: np.ndarray


class _ConditionalFactorCache:
    """Small LRU for deterministic conditional factors within a sampler run."""

    def __init__(self, capacity: int = _DEFAULT_FACTOR_CACHE_CAPACITY):
        if isinstance(capacity, (bool, np.bool_)) or not isinstance(
            capacity, (int, np.integer)
        ):
            raise ValueError("cache capacity must be a positive integer")
        self.capacity = int(capacity)
        if self.capacity < 1:
            raise ValueError("cache capacity must be a positive integer")
        self._entries: OrderedDict[tuple[object, ...], _DeterministicConditionalTerms] = (
            OrderedDict()
        )

    def __len__(self) -> int:
        return len(self._entries)

    def get_or_build(
        self,
        key: tuple[object, ...],
        builder: Callable[[], _DeterministicConditionalTerms],
    ) -> _DeterministicConditionalTerms:
        cached = self._entries.pop(key, None)
        if cached is not None:
            self._entries[key] = cached
            return cached
        built = builder()
        self._entries[key] = built
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
        return built


def compute_posterior_weights(
    observation: np.ndarray,
    preparation: LSObservationPreparation,
    *,
    row_index: int,
) -> np.ndarray:
    """Return equal-prior midpoint-particle weights for one LS observation."""
    row_index = _validate_nonnegative_integer(row_index, "row_index")
    try:
        value = _validate_observation(observation, preparation)
    except ValueError as exc:
        raise ValueError(
            "invalid observation for "
            f"process={preparation.law.process}, depth={preparation.depth}, "
            f"row_index={row_index}: {exc}"
        ) from exc
    n_particles = len(preparation)
    if n_particles == 0:
        raise ValueError("LS observation preparation must contain particles")
    if preparation.depth == 0:
        return _readonly_array(np.full(n_particles, 1.0 / n_particles))

    log_weights = np.empty(n_particles, dtype=float)
    for particle_index, particle in enumerate(preparation):
        context = _context(preparation, row_index, particle_index)
        try:
            log_weight = float(particle.log_likelihood(value))
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"likelihood evaluation failed for {context}: {exc}") from exc
        if not np.isfinite(log_weight):
            raise ValueError(
                f"non-finite likelihood normalization input for {context}"
            )
        log_weights[particle_index] = log_weight

    max_log_weight = float(np.max(log_weights))
    shifted_mass = float(np.exp(log_weights - max_log_weight).sum())
    log_normalizer = max_log_weight + float(np.log(shifted_mass))
    if not np.isfinite(log_normalizer):
        raise ValueError(
            "non-finite likelihood normalization for "
            f"process={preparation.law.process}, depth={preparation.depth}, "
            f"row_index={row_index}, particle_index=all"
        )
    weights = np.exp(log_weights - log_normalizer)
    total = float(weights.sum())
    if not np.all(np.isfinite(weights)) or not np.isfinite(total) or total <= 0.0:
        raise ValueError(
            "non-finite posterior weights for "
            f"process={preparation.law.process}, depth={preparation.depth}, "
            f"row_index={row_index}, particle_index=all"
        )
    weights /= total
    return _readonly_array(weights)


def build_fixed_theta_conditional(
    observation: np.ndarray,
    preparation: LSObservationPreparation,
    *,
    particle_index: int,
    physical_time_grid: np.ndarray,
    row_index: int,
) -> FixedThetaConditional:
    """Build one fixed-particle conditional mean and full covariance factor."""
    row_index = _validate_nonnegative_integer(row_index, "row_index")
    particle_index = _validate_particle_index(particle_index, preparation)
    context = _context(preparation, row_index, particle_index)
    try:
        value = _validate_observation(observation, preparation)
    except ValueError as exc:
        raise ValueError(f"invalid observation for {context}: {exc}") from exc
    deterministic = _build_deterministic_conditional_terms(
        preparation,
        particle_index,
        physical_time_grid,
        context=context,
    )

    mean = _conditional_mean(
        value,
        preparation,
        particle_index,
        deterministic,
        context=context,
    )
    return FixedThetaConditional(
        mean=mean,
        covariance=deterministic.covariance,
        factor=deterministic.factor,
    )


def sample_ls_posterior(
    observations: np.ndarray,
    preparation: LSObservationPreparation,
    *,
    physical_time_grid: np.ndarray,
    n_repeats: int,
    seed: int = 2026,
    row_indices: Sequence[int] | None = None,
    _factor_cache: _ConditionalFactorCache | None = None,
) -> LSPosteriorSamples:
    """Draw one parameter and one conditional GP path per row and repeat."""
    n_repeats = _validate_positive_integer(n_repeats, "n_repeats")
    seed = _validate_nonnegative_integer(seed, "seed")
    observation_batch = _validate_observation_batch(observations, preparation)
    global_rows = _validate_row_indices(row_indices, observation_batch.shape[0])
    time_grid = _validate_physical_time_grid(physical_time_grid, preparation)
    cache = (
        _ConditionalFactorCache()
        if _factor_cache is None
        else _validate_factor_cache(_factor_cache)
    )

    n_rows = observation_batch.shape[0]
    n_time_points = time_grid.size
    paths = np.empty((n_rows, n_repeats, n_time_points), dtype=float)
    parameter_arrays = {
        f"sampled_{name}": np.empty((n_rows, n_repeats), dtype=float)
        for name in preparation.law.parameter_names
    }

    for local_row, global_row in enumerate(global_rows):
        observation = observation_batch[local_row]
        weights = compute_posterior_weights(
            observation,
            preparation,
            row_index=global_row,
        )
        parameter_sequence, gaussian_sequence = np.random.SeedSequence(
            [seed, global_row]
        ).spawn(2)
        parameter_rng = np.random.default_rng(parameter_sequence)
        gaussian_rng = np.random.default_rng(gaussian_sequence)
        selected_particles = _draw_particle_indices(
            parameter_rng,
            weights,
            n_repeats,
        )
        selected_particles = _validate_selected_particles(
            selected_particles,
            n_repeats=n_repeats,
            n_particles=len(preparation),
            row_index=global_row,
        )
        gaussian_noise = _draw_gaussian_noise(
            gaussian_rng,
            n_repeats,
            n_time_points,
        )
        gaussian_noise = _validate_gaussian_noise(
            gaussian_noise,
            n_repeats=n_repeats,
            n_time_points=n_time_points,
            row_index=global_row,
        )

        for repeat_index, particle_index in enumerate(selected_particles):
            particle_index = int(particle_index)
            context = _context(preparation, global_row, particle_index)
            cache_key = _conditional_cache_key(
                preparation,
                particle_index,
                time_grid,
            )
            deterministic = cache.get_or_build(
                cache_key,
                lambda particle_index=particle_index, context=context: (
                    _build_deterministic_conditional_terms(
                        preparation,
                        particle_index,
                        time_grid,
                        context=context,
                    )
                ),
            )
            mean = _conditional_mean(
                observation,
                preparation,
                particle_index,
                deterministic,
                context=context,
            )
            rank = deterministic.factor.shape[1]
            path = mean + deterministic.factor @ gaussian_noise[repeat_index, :rank]
            if not np.all(np.isfinite(path)):
                raise ValueError(
                    f"non-finite sampled path for {context}, repeat_index={repeat_index}"
                )
            paths[local_row, repeat_index] = path
            theta = preparation[particle_index].theta
            for parameter_index, name in enumerate(preparation.law.parameter_names):
                parameter_arrays[f"sampled_{name}"][
                    local_row, repeat_index
                ] = theta[parameter_index]

    _validate_sampled_outputs(
        paths,
        parameter_arrays,
        n_rows=n_rows,
        n_repeats=n_repeats,
        n_time_points=n_time_points,
    )
    readonly_parameters = MappingProxyType(
        {name: _readonly_array(values) for name, values in parameter_arrays.items()}
    )
    return LSPosteriorSamples(
        paths=_readonly_array(paths),
        sampled_parameters=readonly_parameters,
    )


def _build_deterministic_conditional_terms(
    preparation: LSObservationPreparation,
    particle_index: int,
    physical_time_grid: np.ndarray,
    *,
    context: str,
) -> _DeterministicConditionalTerms:
    try:
        terms = prepare_fixed_theta_path_terms(
            preparation=preparation,
            particle_index=particle_index,
            physical_time_grid=physical_time_grid,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ValueError(f"fixed-theta conditioning failed for {context}: {exc}") from exc
    covariance_factor = _factorize_psd_covariance(
        terms.conditional_covariance,
        context=context,
    )
    return _DeterministicConditionalTerms(
        mean_path=_readonly_array(terms.mean_path),
        regression_matrix=_readonly_array(terms.regression_matrix),
        covariance=covariance_factor.covariance,
        factor=covariance_factor.factor,
    )


def _conditional_mean(
    observation: np.ndarray,
    preparation: LSObservationPreparation,
    particle_index: int,
    deterministic: _DeterministicConditionalTerms,
    *,
    context: str,
) -> np.ndarray:
    delta = observation - preparation[particle_index].mean_vec
    mean = deterministic.mean_path + deterministic.regression_matrix @ delta
    if not np.all(np.isfinite(mean)):
        raise ValueError(f"non-finite conditional mean for {context}")
    return _readonly_array(mean)


def _factorize_psd_covariance(
    covariance: np.ndarray,
    *,
    context: str,
) -> ConditionalCovarianceFactor:
    """Apply the exact no-jitter eigen policy for a conditional covariance."""
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"conditional covariance must be square for {context}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"conditional covariance must be finite for {context}")
    symmetric = 0.5 * (matrix + matrix.T)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"conditional covariance eigh failed for {context}") from exc

    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    tolerance = _EIGEN_TOLERANCE_SCALE * scale
    if np.any(eigenvalues < -tolerance):
        minimum = float(np.min(eigenvalues))
        raise ValueError(
            "conditional covariance has a materially negative eigenvalue "
            f"{minimum} below {-tolerance} for {context}"
        )

    clipped = np.where(eigenvalues < 0.0, 0.0, eigenvalues)
    positive = clipped > 0.0
    factor = eigenvectors[:, positive] * np.sqrt(clipped[positive])[None, :]
    clipped_covariance = factor @ factor.T
    clipped_covariance = 0.5 * (clipped_covariance + clipped_covariance.T)
    return ConditionalCovarianceFactor(
        covariance=_readonly_array(clipped_covariance),
        factor=_readonly_array(factor),
        eigenvalues=_readonly_array(clipped),
        tolerance=tolerance,
    )


def _validate_observation(
    observation: np.ndarray,
    preparation: LSObservationPreparation,
) -> np.ndarray:
    if np.iscomplexobj(observation):
        raise ValueError("observation must be real")
    value = np.asarray(observation, dtype=float)
    expected_shape = (preparation.depth,)
    if value.shape != expected_shape:
        raise ValueError(
            f"observation shape must be {expected_shape}, got {value.shape}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError("observation must contain only finite values")
    return value


def _validate_particle_index(
    particle_index: int,
    preparation: LSObservationPreparation,
) -> int:
    value = _validate_nonnegative_integer(particle_index, "particle_index")
    if value >= len(preparation):
        raise ValueError(
            f"particle_index must satisfy 0 <= index < {len(preparation)}"
        )
    return value


def _validate_positive_integer(value: int, name: str) -> int:
    result = _validate_nonnegative_integer(value, name)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _validate_nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _validate_observation_batch(
    observations: np.ndarray,
    preparation: LSObservationPreparation,
) -> np.ndarray:
    if np.iscomplexobj(observations):
        raise ValueError("observations must be real")
    values = np.asarray(observations, dtype=float)
    expected_width = preparation.depth
    if values.ndim != 2 or values.shape[1] != expected_width:
        raise ValueError(
            "observations must have shape "
            f"(n_rows, {expected_width}), got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("observations must contain only finite values")
    return values


def _validate_row_indices(
    row_indices: Sequence[int] | None,
    n_rows: int,
) -> tuple[int, ...]:
    if row_indices is None:
        return tuple(range(n_rows))
    if isinstance(row_indices, np.ndarray) and row_indices.ndim != 1:
        raise ValueError("row_indices must be a one-dimensional integer sequence")
    try:
        values = tuple(row_indices)
    except TypeError as exc:
        raise ValueError("row_indices must be a one-dimensional integer sequence") from exc
    if len(values) != n_rows:
        raise ValueError(
            f"row_indices length must match observations row count {n_rows}"
        )
    result: list[int] = []
    for value in values:
        try:
            result.append(_validate_nonnegative_integer(value, "row_indices"))
        except ValueError as exc:
            raise ValueError(
                "row_indices must contain only non-negative integers"
            ) from exc
    return tuple(result)


def _validate_physical_time_grid(
    physical_time_grid: np.ndarray,
    preparation: LSObservationPreparation,
) -> np.ndarray:
    if np.iscomplexobj(physical_time_grid):
        raise ValueError("physical_time_grid must be real")
    grid = np.asarray(physical_time_grid, dtype=float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError(
            "physical_time_grid must be a non-empty one-dimensional array"
        )
    if not np.all(np.isfinite(grid)):
        raise ValueError("physical_time_grid must contain only finite values")
    if np.any(np.diff(grid) < 0.0):
        raise ValueError("physical_time_grid must be non-decreasing")
    tolerance = 1e-12 * max(1.0, abs(preparation.T))
    if not np.isclose(
        float(grid[-1]), preparation.T, rtol=0.0, atol=tolerance
    ):
        raise ValueError(
            f"physical_time_grid horizon must match prepared horizon {preparation.T}"
        )
    return np.array(grid, dtype=float, copy=True)


def _validate_factor_cache(cache: object) -> _ConditionalFactorCache:
    if not isinstance(cache, _ConditionalFactorCache):
        raise ValueError("_factor_cache must be a _ConditionalFactorCache")
    return cache


def _draw_particle_indices(
    rng: np.random.Generator,
    weights: np.ndarray,
    n_repeats: int,
) -> np.ndarray:
    return np.asarray(
        rng.choice(weights.size, size=n_repeats, replace=True, p=weights),
        dtype=int,
    )


def _draw_gaussian_noise(
    rng: np.random.Generator,
    n_repeats: int,
    n_time_points: int,
) -> np.ndarray:
    return rng.standard_normal((n_repeats, n_time_points))


def _validate_selected_particles(
    selected_particles: np.ndarray,
    *,
    n_repeats: int,
    n_particles: int,
    row_index: int,
) -> np.ndarray:
    selected = np.asarray(selected_particles)
    if selected.shape != (n_repeats,) or selected.dtype.kind not in "iu":
        raise ValueError(
            f"particle draws must be an integer array of shape ({n_repeats},) "
            f"for row_index={row_index}"
        )
    selected = np.asarray(selected, dtype=int)
    if np.any(selected < 0) or np.any(selected >= n_particles):
        raise ValueError(
            f"particle draws are out of range for row_index={row_index}"
        )
    return selected


def _validate_gaussian_noise(
    gaussian_noise: np.ndarray,
    *,
    n_repeats: int,
    n_time_points: int,
    row_index: int,
) -> np.ndarray:
    noise = np.asarray(gaussian_noise, dtype=float)
    expected_shape = (n_repeats, n_time_points)
    if noise.shape != expected_shape or not np.all(np.isfinite(noise)):
        raise ValueError(
            f"Gaussian noise must be finite with shape {expected_shape} "
            f"for row_index={row_index}"
        )
    return noise


def _conditional_cache_key(
    preparation: LSObservationPreparation,
    particle_index: int,
    physical_time_grid: np.ndarray,
) -> tuple[object, ...]:
    """Return an exact identity for every deterministic numerical input."""
    particle_index = _validate_particle_index(particle_index, preparation)
    grid = np.asarray(physical_time_grid, dtype=float)
    particle = preparation[particle_index]
    numerical_policy = (
        float(preparation.T),
        int(preparation.n_quad_points),
        float(preparation.jitter),
        _array_identity(preparation.quadrature_grid),
        _array_identity(preparation.weighted_quadrature_basis),
        _EIGEN_POLICY_IDENTITY,
    )
    return (
        preparation.law.process,
        particle_index,
        tuple(float(value) for value in particle.theta),
        int(preparation.depth),
        _array_identity(grid),
        numerical_policy,
    )


def _array_identity(values: np.ndarray) -> tuple[object, ...]:
    array = np.ascontiguousarray(values)
    return (array.dtype.str, array.shape, array.tobytes(order="C"))


def _validate_sampled_outputs(
    paths: np.ndarray,
    parameter_arrays: Mapping[str, np.ndarray],
    *,
    n_rows: int,
    n_repeats: int,
    n_time_points: int,
) -> None:
    expected_path_shape = (n_rows, n_repeats, n_time_points)
    if paths.shape != expected_path_shape or not np.all(np.isfinite(paths)):
        raise ValueError(
            f"sampled paths must be finite with shape {expected_path_shape}"
        )
    expected_parameter_shape = (n_rows, n_repeats)
    for name, values in parameter_arrays.items():
        if values.shape != expected_parameter_shape or not np.all(np.isfinite(values)):
            raise ValueError(
                f"{name} must be finite with shape {expected_parameter_shape}"
            )


def _context(
    preparation: LSObservationPreparation,
    row_index: int,
    particle_index: int,
) -> str:
    theta = preparation[particle_index].theta.tolist()
    return (
        f"process={preparation.law.process}, depth={preparation.depth}, "
        f"row_index={row_index}, particle_index={particle_index}, theta={theta}"
    )


def _readonly_array(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


__all__ = [
    "FixedThetaConditional",
    "LSPosteriorSamples",
    "build_fixed_theta_conditional",
    "compute_posterior_weights",
    "sample_ls_posterior",
]
