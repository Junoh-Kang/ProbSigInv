from __future__ import annotations

from typing import Any

import numpy as np

from probsiginv.evaluation.metrics.summary import sample_std
from probsiginv.evaluation.types import EvalBatch

try:
    from scipy.linalg import solve_toeplitz as _scipy_solve_toeplitz
except Exception:  # pragma: no cover
    _scipy_solve_toeplitz = None


_EPS = 1e-12


def _summarize_parameter(
    real_values: np.ndarray,
    gen_values: np.ndarray,
    *,
    target_values: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    real = np.asarray(real_values, dtype=float).reshape(-1)
    gen = np.asarray(gen_values, dtype=float)
    if gen.ndim == 1:
        gen = gen[:, None]

    target = real if target_values is None else np.asarray(target_values, dtype=float).reshape(-1)
    if target.shape != real.shape:
        raise ValueError("target_values must have the same flattened shape as real_values")
    real_abs_error = np.abs(real - target)
    gen_mean = np.mean(gen, axis=1)
    gen_abs_error = np.abs(gen - target[:, None])
    return {
        "real": {
            "mean": float(np.mean(real)),
            "median": float(np.median(real)),
            "std": sample_std(real),
        },
        "gen": {
            "mean": float(np.mean(gen_mean)),
            "median": float(np.median(gen_mean)),
            "std": sample_std(gen_mean),
        },
        "real_abs_error": {
            "mean": float(np.mean(real_abs_error)),
            "median": float(np.median(real_abs_error)),
            "std": sample_std(real_abs_error),
        },
        "gen_abs_error": {
            "mean": float(np.mean(gen_abs_error)),
            "median": float(np.median(gen_abs_error)),
            "std": sample_std(gen_abs_error.reshape(-1)),
        },
    }


def _broadcast_known_parameter(value: float | np.ndarray | None, n_real: int, name: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} metadata is required for parameter evaluation")
    return np.broadcast_to(np.asarray(value, dtype=float), (n_real,))


def _broadcast_known_parameter_by_dim(
    value: float | np.ndarray | None,
    n_real: int,
    path_dim: int,
    name: str,
) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} metadata is required for parameter evaluation")
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full((n_real, path_dim), float(array), dtype=float)
    if array.shape == (n_real, path_dim):
        return array.astype(float)
    flat = array.reshape(-1)
    if flat.size == 1:
        return np.full((n_real, path_dim), float(flat[0]), dtype=float)
    if flat.size == n_real:
        return np.broadcast_to(flat[:, None], (n_real, path_dim)).astype(float)
    raise ValueError(
        f"{name} metadata for vector paths must be scalar, length n_real, "
        f"or shape (n_real, path_dim); got {array.shape}"
    )


def _detect_process(batch: EvalBatch) -> str:
    if batch.process is None:
        raise ValueError("process metadata is required for parameter evaluation")
    process = str(batch.process).strip().lower()
    if process not in {"gbm", "fbm", "ou"}:
        raise ValueError(f"Unsupported process for parameter evaluation: {process!r}")
    return process


def _as_2d_paths(paths: np.ndarray) -> np.ndarray:
    paths = np.asarray(paths, dtype=float)
    if paths.ndim == 3 and paths.shape[-1] == 1:
        paths = paths[..., 0]
    if paths.ndim != 2:
        raise ValueError("paths must have shape (N, T)")
    if paths.shape[1] < 2:
        raise ValueError("paths must contain at least 2 time points")
    return paths


def _as_3d_paths(paths: np.ndarray) -> np.ndarray:
    paths = np.asarray(paths, dtype=float)
    if paths.ndim == 2:
        paths = paths[:, :, None]
    if paths.ndim != 3:
        raise ValueError("paths must have shape (N, T) or (N, T, D)")
    if paths.shape[1] < 2:
        raise ValueError("paths must contain at least 2 time points")
    return paths


def _estimate_gbm_sigma(paths: np.ndarray, dt: float) -> np.ndarray:
    paths = _as_2d_paths(paths)
    increments = np.diff(paths, axis=-1)
    mean_increment = np.mean(increments, axis=-1, keepdims=True)
    residual = increments - mean_increment
    sigma2_hat = np.mean(residual**2, axis=-1) / dt
    return np.sqrt(np.maximum(sigma2_hat, 0.0))


def _estimate_gbm_sigma_by_dim(paths: np.ndarray, dt: float) -> np.ndarray:
    array = _as_3d_paths(paths)
    values = [
        _estimate_gbm_sigma(array[..., dim], dt)
        for dim in range(array.shape[-1])
    ]
    return np.stack(values, axis=1)


def _summarize_coordinate_parameter(
    real_values: np.ndarray,
    gen_values: np.ndarray,
    *,
    target_values: np.ndarray,
) -> dict[str, dict[str, float]]:
    real = np.asarray(real_values, dtype=float)
    gen = np.asarray(gen_values, dtype=float)
    target = np.asarray(target_values, dtype=float)
    if real.shape != target.shape:
        raise ValueError("target_values must have the same shape as real_values")
    if gen.shape != target.shape:
        raise ValueError("gen_values must have the same shape as target_values")

    real_abs_error = np.mean(np.abs(real - target), axis=1)
    gen_abs_error = np.mean(np.abs(gen - target), axis=1)
    return {
        "real": {
            "mean": float(np.mean(real)),
            "median": float(np.median(real)),
            "std": sample_std(real),
        },
        "gen": {
            "mean": float(np.mean(gen)),
            "median": float(np.median(gen)),
            "std": sample_std(gen),
        },
        "real_abs_error": {
            "mean": float(np.mean(real_abs_error)),
            "median": float(np.median(real_abs_error)),
            "std": sample_std(real_abs_error),
        },
        "gen_abs_error": {
            "mean": float(np.mean(gen_abs_error)),
            "median": float(np.median(gen_abs_error)),
            "std": sample_std(gen_abs_error),
        },
    }


def _estimate_increment_rho_dim01(paths: np.ndarray) -> np.ndarray:
    array = _as_3d_paths(paths)
    if array.shape[-1] < 2:
        raise ValueError("rho requires at least two path dimensions")
    increments = np.diff(array[..., :2], axis=1)
    left = increments[..., 0]
    right = increments[..., 1]
    left_centered = left - np.mean(left, axis=1, keepdims=True)
    right_centered = right - np.mean(right, axis=1, keepdims=True)
    numerator = np.mean(left_centered * right_centered, axis=1)
    denominator = np.sqrt(np.mean(left_centered**2, axis=1) * np.mean(right_centered**2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = numerator / denominator
    return np.asarray(rho, dtype=float)


def _summarize_rho_parameter(
    real_values: np.ndarray,
    gen_values: np.ndarray,
    *,
    target_values: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    real = np.asarray(real_values, dtype=float).reshape(-1)
    gen = np.asarray(gen_values, dtype=float).reshape(-1)
    if real.shape != gen.shape:
        raise ValueError("real and generated rho estimates must have the same shape")

    if target_values is None:
        keep = np.isfinite(real) & np.isfinite(gen)
        if not np.any(keep):
            real = np.zeros(1, dtype=float)
            gen = np.zeros(1, dtype=float)
            return _summarize_parameter(real, gen)
        return _summarize_parameter(real[keep], gen[keep])

    target = np.asarray(target_values, dtype=float).reshape(-1)
    if target.shape != real.shape:
        raise ValueError("target_values must have the same flattened shape as real_values")
    keep = np.isfinite(real) & np.isfinite(gen) & np.isfinite(target)
    if not np.any(keep):
        real = np.zeros(1, dtype=float)
        gen = np.zeros(1, dtype=float)
        target = np.zeros(1, dtype=float)
    else:
        real = real[keep]
        gen = gen[keep]
        target = target[keep]
    return _summarize_parameter(real, gen, target_values=target)


def _fgn_cov_first_col(n: int, dt: float, hurst: float) -> np.ndarray:
    if not (0.0 < hurst < 1.0):
        raise ValueError(f"hurst must be in (0, 1), got {hurst}")

    k = np.arange(n, dtype=float)
    col = 0.5 * (dt ** (2.0 * hurst)) * (
        np.abs(k + 1.0) ** (2.0 * hurst)
        - 2.0 * np.abs(k) ** (2.0 * hurst)
        + np.abs(k - 1.0) ** (2.0 * hurst)
    )
    col[0] = dt ** (2.0 * hurst)
    return col


def _solve_symmetric_toeplitz(first_col: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    first_col = np.asarray(first_col, dtype=float)
    rhs = np.asarray(rhs, dtype=float)

    if _scipy_solve_toeplitz is not None:
        return _scipy_solve_toeplitz((first_col, first_col), rhs, check_finite=False)

    n = first_col.shape[0]
    idx = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    mat = first_col[idx] + 1e-10 * np.eye(n)
    return np.linalg.solve(mat, rhs)


def _estimate_fbm_sigma_given_hurst(
    paths: np.ndarray,
    dt: float,
    hurst: float | np.ndarray,
) -> np.ndarray:
    paths = _as_2d_paths(paths)
    increments = np.diff(paths, axis=-1)

    n_paths, n_increments = increments.shape
    hurst_arr = np.broadcast_to(np.asarray(hurst, dtype=float), (n_paths,))
    sigma_hat = np.empty(n_paths, dtype=float)
    design = np.full(n_increments, dt, dtype=float)

    hurst_keys = np.round(hurst_arr, 10)
    for hurst_key in np.unique(hurst_keys):
        idx = np.where(hurst_keys == hurst_key)[0]
        hurst_value = float(hurst_arr[idx[0]])
        cov_col = _fgn_cov_first_col(n_increments, dt, hurst_value)

        gamma_inv_design = _solve_symmetric_toeplitz(cov_col, design)
        denom = max(float(design @ gamma_inv_design), _EPS)

        y = increments[idx].T
        gamma_inv_y = _solve_symmetric_toeplitz(cov_col, y)
        nu_hat = (design @ gamma_inv_y) / denom
        residual = y - design[:, None] * nu_hat[None, :]

        gamma_inv_residual = _solve_symmetric_toeplitz(cov_col, residual)
        sigma2 = np.sum(residual * gamma_inv_residual, axis=0) / n_increments
        sigma_hat[idx] = np.sqrt(np.maximum(sigma2, 0.0))

    return sigma_hat


def _estimate_fbm_hurst_given_sigma(
    paths: np.ndarray,
    dt: float,
    sigma: float | np.ndarray,
) -> np.ndarray:
    paths = _as_2d_paths(paths)
    n_paths, n_steps = paths.shape
    sigma_arr = np.broadcast_to(np.asarray(sigma, dtype=float), (n_paths,))
    max_m = max(1, n_steps // 8)
    ms = [1]
    while ms[-1] * 2 <= max_m:
        ms.append(ms[-1] * 2)
    ms_array = np.asarray(ms, dtype=int)

    empirical = np.empty((n_paths, len(ms_array)), dtype=float)
    for j, m in enumerate(ms_array):
        diff2 = paths[:, 2 * m :] - 2.0 * paths[:, m : n_steps - m] + paths[:, : n_steps - 2 * m]
        empirical[:, j] = np.mean(diff2**2, axis=-1)

    log_empirical = np.log(np.maximum(empirical, 1e-30))
    hurst_grid = np.linspace(0.05, 0.95, 181)
    losses = np.empty((n_paths, len(hurst_grid)), dtype=float)
    h_scales = ms_array.astype(float) * dt

    for j, hurst_value in enumerate(hurst_grid):
        second_var_const = 4.0 - 2.0 ** (2.0 * hurst_value)
        theory = (
            sigma_arr[:, None] ** 2
            * second_var_const
            * h_scales[None, :] ** (2.0 * hurst_value)
        )
        log_theory = np.log(np.maximum(theory, 1e-30))
        losses[:, j] = np.mean((log_empirical - log_theory) ** 2, axis=1)

    return hurst_grid[np.argmin(losses, axis=1)]


def _estimate_ou_sigma_given_theta_mu(
    paths: np.ndarray,
    dt: float,
    theta: float | np.ndarray,
    mu: float | np.ndarray,
) -> np.ndarray:
    paths = _as_2d_paths(paths)
    n_paths = paths.shape[0]
    theta_arr = np.broadcast_to(np.asarray(theta, dtype=float), (n_paths,))
    mu_arr = np.broadcast_to(np.asarray(mu, dtype=float), (n_paths,))
    conditional_mean = paths[:, :-1] + theta_arr[:, None] * (mu_arr[:, None] - paths[:, :-1]) * dt
    residual = paths[:, 1:] - conditional_mean
    sigma2_hat = np.mean(residual**2, axis=-1) / dt
    return np.sqrt(np.maximum(sigma2_hat, 0.0))


def _compute_gbm_parameter_metrics(batch: EvalBatch) -> dict[str, Any]:
    representative_gen_paths = batch.gen_paths[:, 0, :]

    if batch.path_dim == 1:
        known_sigma = _broadcast_known_parameter(batch.sigma, batch.n_real, "sigma")
        real_sigma = _estimate_gbm_sigma(batch.real_paths, batch.dt)
        gen_sigma = _estimate_gbm_sigma(representative_gen_paths, batch.dt)
        sigma_payload = _summarize_parameter(real_sigma, gen_sigma, target_values=known_sigma)
        rho_payload = None
    elif batch.path_dim == 2:
        known_sigma = _broadcast_known_parameter_by_dim(
            batch.sigma,
            batch.n_real,
            batch.path_dim,
            "sigma",
        )
        real_sigma = _estimate_gbm_sigma_by_dim(batch.real_paths, batch.dt)
        gen_sigma = _estimate_gbm_sigma_by_dim(representative_gen_paths, batch.dt)
        sigma_payload = _summarize_coordinate_parameter(
            real_sigma,
            gen_sigma,
            target_values=known_sigma,
        )

        real_rho = _estimate_increment_rho_dim01(batch.real_paths)
        gen_rho = _estimate_increment_rho_dim01(representative_gen_paths)
        if batch.metadata.get("gbm_rho") is not None:
            reference = np.asarray(batch.metadata["gbm_rho"], dtype=float)
            known_rho = _broadcast_known_parameter(reference, batch.n_real, "gbm_rho")
            rho_payload = _summarize_rho_parameter(real_rho, gen_rho, target_values=known_rho)
            flat_reference = reference.reshape(-1)
            if flat_reference.size == 1:
                rho_payload["reference"] = {"gbm_rho": float(flat_reference[0])}
            else:
                rho_payload["reference"] = {
                    "gbm_rho_min": float(flat_reference.min()),
                    "gbm_rho_max": float(flat_reference.max()),
                }
        else:
            rho_payload = _summarize_rho_parameter(real_rho, gen_rho)
    else:
        raise ValueError(f"GBM parameter evaluation is unsupported for path_dim={batch.path_dim}")

    payload = {
        "sigma": sigma_payload,
        "hurst": None,
        "sigma_known_hurst": None,
        "hurst_known_sigma": None,
    }
    if rho_payload is not None:
        payload["rho"] = rho_payload
    return payload


def _compute_fbm_parameter_metrics(batch: EvalBatch) -> dict[str, Any]:
    known_hurst = _broadcast_known_parameter(batch.hurst, batch.n_real, "hurst")
    known_sigma = _broadcast_known_parameter(batch.sigma, batch.n_real, "sigma")

    real_sigma_known_hurst = _estimate_fbm_sigma_given_hurst(batch.real_paths, batch.dt, known_hurst)
    real_hurst_known_sigma = _estimate_fbm_hurst_given_sigma(batch.real_paths, batch.dt, known_sigma)

    representative_gen_paths = batch.gen_paths[:, 0, :]
    gen_sigma_known_hurst = _estimate_fbm_sigma_given_hurst(representative_gen_paths, batch.dt, known_hurst)
    gen_hurst_known_sigma = _estimate_fbm_hurst_given_sigma(representative_gen_paths, batch.dt, known_sigma)

    return {
        "sigma": None,
        "hurst": None,
        "sigma_known_hurst": _summarize_parameter(
            real_sigma_known_hurst,
            gen_sigma_known_hurst,
            target_values=known_sigma,
        ),
        "hurst_known_sigma": _summarize_parameter(
            real_hurst_known_sigma,
            gen_hurst_known_sigma,
            target_values=known_hurst,
        ),
    }


def _compute_ou_parameter_metrics(batch: EvalBatch) -> dict[str, Any]:
    if batch.theta is None or batch.mu is None:
        raise ValueError("OU evaluation requires theta and mu metadata in real.npz")
    known_sigma = _broadcast_known_parameter(batch.sigma, batch.n_real, "sigma")

    real_sigma = _estimate_ou_sigma_given_theta_mu(batch.real_paths, batch.dt, theta=batch.theta, mu=batch.mu)
    representative_gen_paths = batch.gen_paths[:, 0, :]
    gen_sigma = _estimate_ou_sigma_given_theta_mu(
        representative_gen_paths,
        batch.dt,
        theta=batch.theta,
        mu=batch.mu,
    )
    return {
        "sigma": _summarize_parameter(real_sigma, gen_sigma, target_values=known_sigma),
        "hurst": None,
        "sigma_known_hurst": None,
        "hurst_known_sigma": None,
    }


def compute_parameter_metrics(batch: EvalBatch) -> dict[str, Any]:
    process = _detect_process(batch)
    if process == "gbm":
        return _compute_gbm_parameter_metrics(batch)
    if process == "fbm":
        return _compute_fbm_parameter_metrics(batch)
    if process == "ou":
        return _compute_ou_parameter_metrics(batch)
    raise ValueError(f"Unsupported process for parameter evaluation: {process!r}")
