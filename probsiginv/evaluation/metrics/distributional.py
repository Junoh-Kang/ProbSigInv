from __future__ import annotations

import numpy as np
from scipy import stats

from probsiginv.evaluation.types import EvalBatch


METRIC_KEYS = (
    "wasserstein",
    "ks_statistic",
    "js_divergence",
    "mean_diff",
    "std_diff",
    "skew_diff",
    "kurtosis_diff",
)


def _json_float(value: float) -> float:
    return float(value)


def _empty_metrics() -> dict[str, float]:
    return {key: float("nan") for key in METRIC_KEYS}


def _flat_finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _histogram_bins(real: np.ndarray, synthetic: np.ndarray) -> np.ndarray:
    lower = float(min(real.min(), synthetic.min()))
    upper = float(max(real.max(), synthetic.max()))
    if lower == upper:
        half_width = max(abs(lower), 1.0) * 0.5
        lower -= half_width
        upper += half_width
    return np.linspace(lower, upper, 100)


def _js_divergence(real: np.ndarray, synthetic: np.ndarray) -> float:
    bins = _histogram_bins(real, synthetic)
    hist_real, _ = np.histogram(real, bins=bins, density=True)
    hist_syn, _ = np.histogram(synthetic, bins=bins, density=True)

    eps = 1e-10
    hist_real = hist_real + eps
    hist_syn = hist_syn + eps
    hist_real /= hist_real.sum()
    hist_syn /= hist_syn.sum()

    midpoint = 0.5 * (hist_real + hist_syn)
    return float(0.5 * (stats.entropy(hist_real, midpoint) + stats.entropy(hist_syn, midpoint)))


def distribution_metrics(real: np.ndarray, synthetic: np.ndarray) -> dict[str, float]:
    real_flat = _flat_finite(real)
    synthetic_flat = _flat_finite(synthetic)
    if real_flat.size == 0 or synthetic_flat.size == 0:
        return _empty_metrics()

    ks_stat, _ = stats.ks_2samp(real_flat, synthetic_flat)
    return {
        "wasserstein": _json_float(stats.wasserstein_distance(real_flat, synthetic_flat)),
        "ks_statistic": _json_float(ks_stat),
        "js_divergence": _json_float(_js_divergence(real_flat, synthetic_flat)),
        "mean_diff": _json_float(abs(np.mean(synthetic_flat) - np.mean(real_flat))),
        "std_diff": _json_float(abs(np.std(synthetic_flat) - np.std(real_flat))),
        "skew_diff": _json_float(abs(stats.skew(synthetic_flat) - stats.skew(real_flat))),
        "kurtosis_diff": _json_float(abs(stats.kurtosis(synthetic_flat) - stats.kurtosis(real_flat))),
    }


def compute_distributional_metrics(batch: EvalBatch) -> dict[str, float]:
    real_returns = np.diff(batch.real_paths, axis=1)
    representative_gen_returns = np.diff(batch.gen_paths[:, 0, :, :], axis=1)
    return distribution_metrics(real_returns, representative_gen_returns)


__all__ = ["METRIC_KEYS", "compute_distributional_metrics", "distribution_metrics"]
