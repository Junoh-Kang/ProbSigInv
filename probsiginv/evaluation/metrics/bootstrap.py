from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from probsiginv.evaluation.metrics.summary import sample_std


def build_mean_bootstrap_summary(
    values: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    ci_level: float,
    resample_axis: str,
    statistic: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("bootstrap requires at least one statistic")
    if not np.all(np.isfinite(values)):
        raise ValueError("bootstrap statistics must be finite")
    if int(n_resamples) < 1:
        raise ValueError("bootstrap requires n_resamples >= 1")
    if not 0.0 < float(ci_level) < 1.0:
        raise ValueError("bootstrap requires 0 < ci_level < 1")

    n_resamples = int(n_resamples)
    seed = int(seed)
    ci_level = float(ci_level)
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(n_resamples, dtype=float)

    chunk_size = max(1, min(n_resamples, max(1, 1_000_000 // values.size)))
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        indices = rng.integers(0, values.size, size=(stop - start, values.size), dtype=np.int64)
        bootstrap_means[start:stop] = values[indices].mean(axis=1)

    alpha = (1.0 - ci_level) / 2.0
    ci_low, ci_high = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    payload = {
        "n_resamples": n_resamples,
        "seed": seed,
        "resample_axis": str(resample_axis),
        "statistic": str(statistic),
        "n_statistics": int(values.size),
        "mean": float(np.mean(bootstrap_means)),
        "standard_error": sample_std(bootstrap_means),
        "ci_method": "percentile",
        "ci_level": ci_level,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }
    if metadata is not None:
        payload.update(dict(metadata))
    return payload
