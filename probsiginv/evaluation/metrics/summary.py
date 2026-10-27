from __future__ import annotations

import numpy as np


def sample_std(values: np.ndarray) -> float:
    """Sample standard deviation (ddof=1); 0.0 for fewer than two values."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def mean_std(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    return {"mean": float(np.mean(values)), "std": sample_std(values)}
