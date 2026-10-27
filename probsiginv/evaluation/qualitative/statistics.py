from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from probsiginv.utils.path_shapes import as_eval_candidate_paths, as_eval_real_paths


@dataclass(frozen=True)
class QQCurve:
    quantiles: np.ndarray
    real: np.ndarray
    candidate: np.ndarray


@dataclass(frozen=True)
class ACFCurve:
    tau: np.ndarray
    rho: np.ndarray
    target: str
    center: str
    companions: tuple["ACFCurve", ...] = ()


def normalize_path_array(paths: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(paths, dtype=float)
    if array.ndim == 2:
        return array[:, None, :]
    if array.ndim == 3:
        return array
    if array.ndim == 4:
        return array
    raise ValueError(f"{name} paths must have shape (N, T) or (N, M, T)")


def _public_path_array(array: np.ndarray) -> np.ndarray:
    if array.shape[-1] == 1:
        return array[..., 0]
    return array


def _as_explicit_path_array(paths: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(paths, dtype=float)
    if array.ndim == 2:
        return array[:, None, :, None]
    if array.ndim == 3:
        return array[:, :, :, None]
    if array.ndim == 4:
        return array
    raise ValueError(f"{name} paths must have shape (N, T), (N, M, T), or (N, M, T, D)")


def _as_explicit_real_paths(paths: np.ndarray, *, name: str, time_steps: int | None = None) -> np.ndarray:
    array = np.asarray(paths, dtype=float)
    if array.ndim == 4:
        if array.shape[1] != 1:
            raise ValueError(f"{name} paths must have one real-path repeat axis")
        if time_steps is not None and array.shape[2] != time_steps:
            raise ValueError(f"{name} paths and time_grid must share T")
        return array
    if array.ndim == 3 and array.shape[1] == 1 and (time_steps is None or array.shape[2] == time_steps):
        return array[:, :, :, None]
    real = as_eval_real_paths(array, name=name)
    if time_steps is not None and real.shape[1] != time_steps:
        raise ValueError(f"{name} paths and time_grid must share T")
    return real[:, None, :, :]


def _as_explicit_paths_for_acf(paths: np.ndarray, *, time_steps: int) -> np.ndarray:
    array = np.asarray(paths, dtype=float)
    if array.ndim == 3 and array.shape[1] == int(time_steps) and array.shape[2] != int(time_steps):
        return _as_explicit_real_paths(array, name="paths", time_steps=time_steps)
    if array.ndim == 2 and array.shape[1] == int(time_steps):
        return _as_explicit_real_paths(array, name="paths", time_steps=time_steps)
    return _as_explicit_path_array(array, name="paths")


def _as_explicit_candidate_paths(
    paths: np.ndarray,
    *,
    name: str,
    time_steps: int,
    path_dim: int,
) -> np.ndarray:
    array = np.asarray(paths, dtype=float)
    if path_dim > 1 and array.ndim == 3 and array.shape[1] == time_steps and array.shape[2] == path_dim:
        return array[:, None, :, :]
    return as_eval_candidate_paths(array, name=name)


def _as_real_eval_paths(paths: np.ndarray, *, time_steps: int) -> np.ndarray:
    array = np.asarray(paths, dtype=float)
    if array.ndim == 3 and array.shape[1] == 1 and array.shape[2] == time_steps:
        return array[:, 0, :, None]
    return as_eval_real_paths(array, name="real")


def validate_path_arrays(
    *,
    time_grid: np.ndarray,
    real_paths: np.ndarray,
    named_paths: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    time_values = np.asarray(time_grid, dtype=float)
    if time_values.ndim != 1:
        raise ValueError("time_grid must have shape (T,)")

    real_eval = _as_real_eval_paths(np.asarray(real_paths, dtype=float), time_steps=time_values.shape[0])
    if real_eval.shape[1] != time_values.shape[0]:
        raise ValueError("real paths and time_grid must share T")

    normalized: dict[str, np.ndarray] = {}
    for name, paths in named_paths.items():
        array = _as_explicit_candidate_paths(
            paths,
            name=name,
            time_steps=real_eval.shape[1],
            path_dim=real_eval.shape[2],
        )
        if array.shape[0] != real_eval.shape[0]:
            raise ValueError(f"{name} paths and real paths must share N")
        if array.shape[2] != real_eval.shape[1]:
            raise ValueError(f"{name} paths and real paths must share T")
        if array.shape[3] != real_eval.shape[2]:
            raise ValueError(f"{name} paths and real paths must share D")
        normalized[name] = _public_path_array(array)

    return time_values, _public_path_array(real_eval[:, None, :, :]), normalized


def pooled_increment_qq(
    *,
    real_paths: np.ndarray,
    candidate_paths: np.ndarray,
    n_quantiles: int = 101,
) -> QQCurve:
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2")

    real = _as_explicit_real_paths(real_paths, name="real")
    candidate = _as_explicit_candidate_paths(
        candidate_paths,
        name="candidate",
        time_steps=real.shape[2],
        path_dim=real.shape[3],
    )
    if real.shape[2] < 2:
        raise ValueError("paths must contain at least 2 time points")
    if real.shape[3] != candidate.shape[3]:
        raise ValueError("real and candidate paths must share D")

    if real.shape[3] > 1:
        curves = [
            pooled_increment_qq(
                real_paths=real[..., dim],
                candidate_paths=candidate[..., dim],
                n_quantiles=n_quantiles,
            )
            for dim in range(real.shape[3])
        ]
        return QQCurve(
            quantiles=curves[0].quantiles,
            real=np.stack([curve.real for curve in curves], axis=0),
            candidate=np.stack([curve.candidate for curve in curves], axis=0),
        )

    quantiles = np.linspace(0.0, 1.0, int(n_quantiles))
    real_samples = np.diff(real[..., 0], axis=-1).reshape(-1)
    candidate_samples = np.diff(candidate[..., 0], axis=-1).reshape(-1)
    return QQCurve(
        quantiles=quantiles,
        real=np.quantile(real_samples, quantiles),
        candidate=np.quantile(candidate_samples, quantiles),
    )


def _broadcast_meta(value: Any, n_paths: int, n_repeats: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(n_paths * n_repeats, float(array), dtype=float)
    flat = array.reshape(-1)
    if flat.size == 1:
        return np.full(n_paths * n_repeats, float(flat[0]), dtype=float)
    if flat.size != n_paths:
        return None
    return np.repeat(flat, n_repeats)


def _flatten_ensemble(paths: np.ndarray) -> np.ndarray:
    array = _as_explicit_path_array(paths, name="paths")
    if array.shape[-1] != 1:
        raise ValueError("_flatten_ensemble expects scalar paths")
    scalar = array[..., 0]
    return scalar.reshape(scalar.shape[0] * scalar.shape[1], scalar.shape[2])


def _theory_mean(
    *,
    paths: np.ndarray,
    time_grid: np.ndarray,
    process: str,
    metadata: Mapping[str, Any],
) -> np.ndarray | None:
    explicit_paths = _as_explicit_path_array(paths, name="paths")
    if explicit_paths.shape[-1] != 1:
        raise ValueError("_theory_mean expects scalar paths")
    scalar_paths = explicit_paths[..., 0]
    n_paths, n_repeats, _ = scalar_paths.shape
    t = np.asarray(time_grid, dtype=float)
    x0 = _flatten_ensemble(scalar_paths)[:, 0]
    mu = _broadcast_meta(metadata.get("mu"), n_paths, n_repeats)
    sigma = _broadcast_meta(metadata.get("sigma"), n_paths, n_repeats)

    if process == "ou":
        theta = _broadcast_meta(metadata.get("theta"), n_paths, n_repeats)
        if theta is None:
            theta = _broadcast_meta(metadata.get("kappa"), n_paths, n_repeats)
        if theta is None or mu is None:
            return None
        return mu[:, None] + (x0[:, None] - mu[:, None]) * np.exp(-theta[:, None] * (t[None, :] - t[0]))

    if process in {"gbm", "fbm"} and mu is not None:
        if sigma is not None:
            drift = mu - 0.5 * sigma**2
        else:
            drift = mu
        return x0[:, None] + drift[:, None] * (t[None, :] - t[0])

    return None


def _centered_values(
    *,
    paths: np.ndarray,
    time_grid: np.ndarray,
    process: str,
    metadata: Mapping[str, Any],
    target: str,
) -> tuple[np.ndarray, str]:
    ensemble = _flatten_ensemble(paths)
    mean = _theory_mean(
        paths=paths,
        time_grid=time_grid,
        process=process,
        metadata=metadata,
    )

    if target == "level":
        if mean is not None:
            return ensemble - mean, "theory"
        return ensemble - np.nanmean(ensemble, axis=0, keepdims=True), "ensemble"

    if target == "increment":
        values = np.diff(ensemble, axis=1)
        if mean is not None:
            return values - np.diff(mean, axis=1), "theory"
        return values - np.nanmean(values, axis=0, keepdims=True), "ensemble"

    raise ValueError("target must be 'level' or 'increment'")


def _level_anchor_acf(
    values: np.ndarray,
    time_grid: np.ndarray,
    *,
    anchor: float | None = None,
    max_tau: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time_grid, dtype=float)
    if t.size < 2:
        raise ValueError("paths must contain at least 2 time points")

    if anchor is None:
        anchor = 0.5 * (float(t[0]) + float(t[-1]))
    anchor_index = int(np.argmin(np.abs(t - float(anchor))))
    if anchor_index == 0:
        anchor_index = 1

    indices = np.arange(anchor_index, t.size)
    tau = t[indices] - t[anchor_index]
    if max_tau is not None:
        keep = tau <= float(max_tau) + 1e-12
        indices = indices[keep]
        tau = tau[keep]

    anchor_values = values[:, anchor_index][:, None]
    lag_values = values[:, indices]
    numerator = np.nanmean(anchor_values * lag_values, axis=0)
    denominator = np.sqrt(
        np.nanmean(anchor_values * anchor_values) * np.nanmean(lag_values * lag_values, axis=0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = numerator / denominator
    return tau, np.asarray(rho, dtype=float)


def _pooled_acf(
    values: np.ndarray,
    time_grid: np.ndarray,
    *,
    max_lag: int | None = None,
    default_max_lag: int | None = 20,
) -> tuple[np.ndarray, np.ndarray]:
    if values.shape[1] < 1:
        raise ValueError("ACF requires at least one time value")

    n_steps = values.shape[1]
    if max_lag is None:
        resolved_default = n_steps - 1 if default_max_lag is None else int(default_max_lag)
        if resolved_default < 0:
            raise ValueError("default_max_lag must be non-negative")
        resolved_max_lag = min(resolved_default, n_steps - 1)
    else:
        if max_lag < 0:
            raise ValueError("max_lag must be non-negative")
        resolved_max_lag = min(int(max_lag), n_steps - 1)

    rho = np.empty(resolved_max_lag + 1, dtype=float)
    for lag in range(resolved_max_lag + 1):
        left = values[:, : n_steps - lag]
        right = values[:, lag:n_steps]
        numerator = np.nanmean(left * right)
        denominator = np.sqrt(np.nanmean(left * left) * np.nanmean(right * right))
        with np.errstate(divide="ignore", invalid="ignore"):
            rho[lag] = numerator / denominator

    dt = float(np.median(np.diff(np.asarray(time_grid, dtype=float))))
    return np.arange(resolved_max_lag + 1, dtype=float) * dt, rho


def _finance_volatility_values(paths: np.ndarray) -> tuple[np.ndarray, str]:
    ensemble = _flatten_ensemble(paths)
    returns = np.diff(ensemble, axis=1)
    values = np.abs(returns)
    return values - np.nanmean(values, axis=0, keepdims=True), "ensemble"


def process_acf(
    *,
    time_grid: np.ndarray,
    paths: np.ndarray,
    process: str,
    metadata: Mapping[str, Any] | None = None,
    max_lag: int | None = None,
    anchor: float | None = None,
) -> ACFCurve:
    time_values = np.asarray(time_grid, dtype=float)
    explicit_paths = _as_explicit_paths_for_acf(paths, time_steps=time_values.shape[0])
    if explicit_paths.shape[-1] > 1:
        curves = [
            process_acf(
                time_grid=time_grid,
                paths=explicit_paths[..., dim],
                process=process,
                metadata=metadata,
                max_lag=max_lag,
                anchor=anchor,
            )
            for dim in range(explicit_paths.shape[-1])
        ]
        return ACFCurve(
            tau=curves[0].tau,
            rho=np.stack([curve.rho for curve in curves], axis=0),
            target=curves[0].target,
            center=curves[0].center,
        )

    normalized_process = str(process).strip().lower()
    supported_processes = {"ou", "gbm", "fbm", "finance"}
    if normalized_process not in supported_processes:
        raise ValueError(
            f"unsupported qualitative ACF process {process!r}; "
            f"expected one of {sorted(supported_processes)}"
        )

    target = "level" if normalized_process == "ou" else "increment"
    resolved_max_lag = 30 if normalized_process == "finance" and max_lag is None else max_lag
    centered, center = _centered_values(
        paths=explicit_paths,
        time_grid=np.asarray(time_grid, dtype=float),
        process=normalized_process,
        metadata=metadata or {},
        target=target,
    )

    if target == "level":
        time_values = np.asarray(time_grid, dtype=float)
        if anchor is None:
            # OU level ACF needs enough samples for a stable real curve. Pooling
            # all time pairs at each lag avoids the noisy single-anchor estimate.
            tau, rho = _pooled_acf(
                centered,
                time_values,
                max_lag=max_lag,
                default_max_lag=(centered.shape[1] - 1) // 2,
            )
        else:
            max_tau = None
            if max_lag is not None:
                if max_lag < 0:
                    raise ValueError("max_lag must be non-negative")
                dt = float(np.median(np.diff(time_values)))
                max_tau = int(max_lag) * dt
            tau, rho = _level_anchor_acf(centered, time_values, anchor=anchor, max_tau=max_tau)
    else:
        tau, rho = _pooled_acf(centered, np.asarray(time_grid, dtype=float), max_lag=resolved_max_lag)

    companions: tuple[ACFCurve, ...] = ()
    if normalized_process == "finance":
        volatility_values, volatility_center = _finance_volatility_values(
            explicit_paths
        )
        volatility_tau, volatility_rho = _pooled_acf(
            volatility_values,
            np.asarray(time_grid, dtype=float),
            max_lag=resolved_max_lag,
        )
        companions = (
            ACFCurve(
                tau=volatility_tau,
                rho=volatility_rho,
                target="volatility",
                center=volatility_center,
            ),
        )

    return ACFCurve(tau=tau, rho=rho, target=target, center=center, companions=companions)


__all__ = [
    "ACFCurve",
    "QQCurve",
    "normalize_path_array",
    "pooled_increment_qq",
    "process_acf",
    "validate_path_arrays",
]
