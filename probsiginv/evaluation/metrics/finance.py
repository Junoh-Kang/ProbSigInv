from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy import stats

from probsiginv.evaluation.metrics.summary import sample_std
from probsiginv.evaluation.types import EvalBatch


ACF_ERROR_LAGS = tuple(range(1, 31))
ACF_ERROR_REDUCTION = "mean_absolute_difference_over_lags"


def _summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float(np.nan), "median": float(np.nan), "std": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": sample_std(arr),
    }


def _summarize_real_vs_generated(real_values: np.ndarray, gen_values: np.ndarray) -> dict[str, dict[str, float]]:
    real = np.asarray(real_values, dtype=float).reshape(-1)
    gen = np.asarray(gen_values, dtype=float).reshape(-1)
    if gen.shape != real.shape:
        raise ValueError("real_values and gen_values must have the same flattened shape")

    gen_abs_error = np.full(real.shape, np.nan, dtype=float)
    valid = np.isfinite(real) & np.isfinite(gen)
    gen_abs_error[valid] = np.abs(gen[valid] - real[valid])
    return {
        "real": _summary(real),
        "gen": _summary(gen),
        "gen_abs_error": _summary(gen_abs_error),
    }


def _summarize_finite_real_vs_generated(
    real_values: np.ndarray,
    gen_values: np.ndarray,
    *,
    expected_size: int,
    label: str,
) -> dict[str, dict[str, float]]:
    real = np.asarray(real_values, dtype=float).reshape(-1)
    gen = np.asarray(gen_values, dtype=float).reshape(-1)
    if real.size != expected_size or gen.size != expected_size:
        raise ValueError(
            f"{label} must contain exactly {expected_size} aligned window values"
        )
    if not np.isfinite(real).all() or not np.isfinite(gen).all():
        raise ValueError(f"{label} contains a non-finite or undefined statistic")
    return _summarize_real_vs_generated(real, gen)


def _returns_from_paths(paths: np.ndarray) -> np.ndarray:
    arr = np.asarray(paths, dtype=float)
    if arr.ndim == 3 and arr.shape[-1] == 1 and arr.shape[1] >= 2:
        arr = arr[..., 0]
    if arr.ndim not in {2, 3}:
        raise ValueError("paths must have shape (N, T) or (N, M, T)")
    if arr.shape[-1] < 2:
        raise ValueError("paths must contain at least 2 time points")
    return np.diff(arr, axis=-1)


def _pathwise_realized_volatility(paths: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(_returns_from_paths(paths) ** 2, axis=-1))

def _pathwise_skewness(paths: np.ndarray) -> np.ndarray:
    """Per-path Fisher-Pearson skewness from raw paths."""
    returns = _returns_from_paths(paths)
    flat_returns = returns.reshape(-1, returns.shape[-1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        skewness = np.asarray(stats.skew(flat_returns, axis=1, bias=False), dtype=float)
    return skewness.reshape(returns.shape[:-1])

def _pathwise_excess_kurtosis(paths: np.ndarray) -> np.ndarray:
    """Per-path Fisher's adjusted (bias-corrected) excess kurtosis from raw paths.

    Matches CoFinDiff (arXiv:2503.04164) eq. (3) via
    scipy.stats.kurtosis(., fisher=True, bias=False). Returns NaN for
    degenerate paths (zero variance, n < 4); these are excluded by _summary's
    finite-value filter rather than silently contributing 0.
    """
    returns = _returns_from_paths(paths)
    flat_returns = returns.reshape(-1, returns.shape[-1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        kurtosis = np.asarray(
            stats.kurtosis(flat_returns, axis=1, fisher=True, bias=False),
            dtype=float,
        )
    return kurtosis.reshape(returns.shape[:-1])


def _pathwise_acf_curves_from_increments(
    increments: np.ndarray,
    max_lag: int,
    *,
    label: str,
) -> np.ndarray:
    values = np.asarray(increments, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{label} must have shape (n_windows, n_returns)")
    if max_lag < 1:
        raise ValueError(f"{label} max_lag must be positive")
    if values.shape[1] <= max_lag:
        raise ValueError(
            f"{label} requires at least {max_lag + 1} returns "
            f"({max_lag + 2} raw time points)"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values")

    centered = values - np.mean(values, axis=1, keepdims=True)
    denominator = np.sum(centered ** 2, axis=1)
    if not np.isfinite(denominator).all() or np.any(denominator <= 0.0):
        raise ValueError(
            f"{label} autocorrelation is undefined for a zero-variance window"
        )

    curves = np.empty((values.shape[0], max_lag), dtype=float)
    for lag in range(1, max_lag + 1):
        numerator = np.sum(centered[:, :-lag] * centered[:, lag:], axis=1)
        curves[:, lag - 1] = numerator / denominator
    if not np.isfinite(curves).all():
        raise ValueError(f"{label} produced non-finite autocorrelation values")
    return curves


def _pathwise_acf_error(
    real_paths: np.ndarray,
    gen_paths: np.ndarray,
    *,
    absolute: bool,
    label: str,
) -> np.ndarray:
    kind = "absolute_return_acf_error" if absolute else "return_acf_error"
    metric_label = f"{label}.{kind}"
    real_returns = _returns_from_paths(real_paths)
    gen_returns = _returns_from_paths(gen_paths)
    if real_returns.ndim != 2 or gen_returns.ndim != 2:
        raise ValueError(
            f"{metric_label} paths must resolve to scalar return matrices"
        )
    if real_returns.shape != gen_returns.shape:
        raise ValueError(
            f"{metric_label} real/generated returns must have identical aligned shape; "
            f"got {real_returns.shape} and {gen_returns.shape}"
        )
    if absolute:
        real_returns = np.abs(real_returns)
        gen_returns = np.abs(gen_returns)
    max_lag = max(ACF_ERROR_LAGS)
    real_curves = _pathwise_acf_curves_from_increments(
        real_returns,
        max_lag,
        label=f"{metric_label}.real",
    )
    gen_curves = _pathwise_acf_curves_from_increments(
        gen_returns,
        max_lag,
        label=f"{metric_label}.gen",
    )
    errors = np.mean(np.abs(gen_curves - real_curves), axis=1)
    if errors.shape != (real_returns.shape[0],) or not np.isfinite(errors).all():
        raise ValueError(f"{metric_label} did not produce one finite value per window")
    return errors


def _compute_scalar_stylized_payload(
    real_paths: np.ndarray,
    gen_paths: np.ndarray,
    *,
    strict: bool,
    label: str,
) -> dict[str, Any]:
    expected_size = int(np.asarray(real_paths).shape[0])

    def summarize(real_values: np.ndarray, gen_values: np.ndarray, metric: str) -> dict[str, Any]:
        if strict:
            return _summarize_finite_real_vs_generated(
                real_values,
                gen_values,
                expected_size=expected_size,
                label=f"{label}.{metric}",
            )
        return _summarize_real_vs_generated(real_values, gen_values)

    realized_volatility = summarize(
        _pathwise_realized_volatility(real_paths),
        _pathwise_realized_volatility(gen_paths),
        "realized_volatility",
    )
    skewness = summarize(
        _pathwise_skewness(real_paths),
        _pathwise_skewness(gen_paths),
        "skewness",
    )
    kurtosis = summarize(
        _pathwise_excess_kurtosis(real_paths),
        _pathwise_excess_kurtosis(gen_paths),
        "kurtosis",
    )

    return {
        "realized_volatility": realized_volatility,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "return_acf_error": _pathwise_acf_error(
            real_paths,
            gen_paths,
            absolute=False,
            label=label,
        ).tolist(),
        "absolute_return_acf_error": _pathwise_acf_error(
            real_paths,
            gen_paths,
            absolute=True,
            label=label,
        ).tolist(),
    }


def _pathwise_cross_channel_correlation(paths: np.ndarray, *, absolute: bool) -> np.ndarray:
    arr = np.asarray(paths, dtype=float)
    if arr.ndim != 3 or arr.shape[-1] != 2 or arr.shape[1] < 2:
        raise ValueError("dim-2 correlation paths must have shape (N, T, 2) with T >= 2")
    returns = np.diff(arr, axis=1)
    if absolute:
        returns = np.abs(returns)
    centered = returns - np.mean(returns, axis=1, keepdims=True)
    numerator = np.sum(centered[:, :, 0] * centered[:, :, 1], axis=1)
    denominator = np.sqrt(
        np.sum(centered[:, :, 0] ** 2, axis=1)
        * np.sum(centered[:, :, 1] ** 2, axis=1)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return numerator / denominator


def _dim2_coordinate_names(batch: EvalBatch) -> tuple[str, str]:
    raw_names = batch.metadata.get("coordinate_names")
    if not isinstance(raw_names, (list, tuple)) or len(raw_names) != batch.path_dim:
        raise ValueError("dim-2 finance coordinate names must match path width exactly")
    names = tuple(raw_names)
    if any(
        not isinstance(name, str) or not name or name != name.strip()
        for name in names
    ):
        raise ValueError("dim-2 finance coordinate names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("dim-2 finance coordinate names must be unique")
    return names[0], names[1]


def compute_stylized_metrics(batch: EvalBatch) -> dict[str, Any]:
    representative_gen_paths = batch.gen_paths[:, 0, :]
    if batch.path_dim == 1:
        return _compute_scalar_stylized_payload(
            batch.real_paths,
            representative_gen_paths,
            strict=True,
            label="finance",
        )
    if batch.path_dim != 2:
        raise ValueError(f"finance stylized metrics do not support path_dim={batch.path_dim}")

    coordinate_names = _dim2_coordinate_names(batch)
    channels = {
        name: _compute_scalar_stylized_payload(
            batch.real_paths[:, :, index],
            representative_gen_paths[:, :, index],
            strict=True,
            label=f"channels.{name}",
        )
        for index, name in enumerate(coordinate_names)
    }
    cross_channel: dict[str, Any] = {}
    for key, absolute in (
        ("return_correlation", False),
        ("absolute_return_correlation", True),
    ):
        cross_channel[key] = _summarize_finite_real_vs_generated(
            _pathwise_cross_channel_correlation(batch.real_paths, absolute=absolute),
            _pathwise_cross_channel_correlation(representative_gen_paths, absolute=absolute),
            expected_size=batch.n_real,
            label=f"cross_channel.{key}",
        )
    return {"channels": channels, "cross_channel": cross_channel}


__all__ = ["ACF_ERROR_LAGS", "ACF_ERROR_REDUCTION", "compute_stylized_metrics"]
