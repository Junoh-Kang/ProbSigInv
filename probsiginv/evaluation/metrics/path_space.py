from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from probsiginv.evaluation.metrics.summary import mean_std, sample_std
from probsiginv.evaluation.types import EvalBatch

# Path errors are squared distances summed over path dimensions and averaged over time.


def compute_path_mse_per_real(batch: EvalBatch) -> np.ndarray:
    """Per reference path, the path MSE averaged over generated samples."""
    diff = batch.real_paths[:, None, :] - batch.gen_paths
    return np.sum(diff**2, axis=-1).mean(axis=-1).mean(axis=1)


def compute_path_mse(batch: EvalBatch) -> dict[str, float]:
    return mean_std(compute_path_mse_per_real(batch))


def compute_cond_mean_mse(batch: EvalBatch) -> dict[str, float]:
    diff = batch.real_paths - batch.gen_paths.mean(axis=1)
    return mean_std(np.sum(diff**2, axis=-1).mean(axis=-1))


def compute_spread_mse(batch: EvalBatch) -> dict[str, float] | None:
    """Mean pairwise path MSE between generated samples of the same reference.

    Uses sum_{j<k} |x_j - x_k|^2 = M * sum_j |x_j - mean|^2, so the pairwise mean is
    twice the unbiased sample variance.
    """
    if batch.n_samples < 2:
        return None
    variance = np.var(batch.gen_paths, axis=1, ddof=1)
    return mean_std(2.0 * np.sum(variance, axis=-1).mean(axis=-1))


def compute_energy_distance(
    reference_paths: np.ndarray,
    model_paths: np.ndarray,
) -> dict[str, float | int | np.ndarray]:
    """Signed unbiased squared Energy Distance per conditioning row.

    Inputs have shape (S, N, K) or (S, N, K, D): conditions, samples, time, path dim.
    Distances are Euclidean over the flattened path, scaled by 1/sqrt(K).
    """
    reference = np.asarray(reference_paths, dtype=np.float64)
    model = np.asarray(model_paths, dtype=np.float64)
    if reference.ndim == 3:
        reference = reference[..., None]
    if model.ndim == 3:
        model = model[..., None]
    for name, paths in (("reference_paths", reference), ("model_paths", model)):
        if paths.ndim != 4:
            raise ValueError(f"{name} must have shape (S,N,K) or (S,N,K,D), got {paths.shape}")
        if paths.shape[1] < 2:
            raise ValueError(f"{name} must contain at least two samples per condition")
        if not np.all(np.isfinite(paths)):
            raise ValueError(f"{name} contains non-finite values")
    if reference.shape[0] != model.shape[0] or reference.shape[2:] != model.shape[2:]:
        raise ValueError(
            "reference_paths and model_paths must share condition, time, and path-dimension axes: "
            f"got {reference.shape} and {model.shape}"
        )

    n_conditions, n_reference, n_steps, path_dim = reference.shape
    n_model = model.shape[1]
    scale = np.sqrt(float(n_steps))
    per_condition = np.empty(n_conditions, dtype=float)
    for i in range(n_conditions):
        ref = reference[i].reshape(n_reference, n_steps * path_dim)
        gen = model[i].reshape(n_model, n_steps * path_dim)
        cross = float(np.mean(cdist(ref, gen))) / scale
        within_ref = float(np.sum(cdist(ref, ref))) / float(n_reference * (n_reference - 1)) / scale
        within_gen = float(np.sum(cdist(gen, gen))) / float(n_model * (n_model - 1)) / scale
        per_condition[i] = 2.0 * cross - within_ref - within_gen

    mean = float(np.mean(per_condition))
    std = sample_std(per_condition)
    standard_error = float(std / np.sqrt(n_conditions))
    return {
        "per_condition": per_condition,
        "mean": mean,
        "std": std,
        "standard_error": standard_error,
        "ci95_low": mean - 1.959963984540054 * standard_error,
        "ci95_high": mean + 1.959963984540054 * standard_error,
        "n_conditions": int(n_conditions),
        "n_reference_samples": int(n_reference),
        "n_model_samples": int(n_model),
    }
