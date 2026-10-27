from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from probsiginv.utils.path_shapes import as_eval_candidate_paths, as_eval_real_paths


@dataclass
class EvalBatch:
    real_paths: np.ndarray
    gen_paths: np.ndarray
    time_grid: np.ndarray
    process: str | None = None
    theta: float | np.ndarray | None = None
    mu: float | np.ndarray | None = None
    sigma: float | np.ndarray | None = None
    hurst: float | np.ndarray | None = None
    dataset_name: str | None = None
    model_name: str | None = None
    aug_type: str = "TA"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.real_paths = as_eval_real_paths(np.asarray(self.real_paths, dtype=float), name="real_paths")
        self.gen_paths = as_eval_candidate_paths(np.asarray(self.gen_paths, dtype=float), name="gen_paths")
        self.time_grid = np.asarray(self.time_grid, dtype=float)

        if self.real_paths.ndim != 3:
            raise ValueError("real_paths must have shape (N, T, D)")
        if self.gen_paths.ndim != 4:
            raise ValueError("gen_paths must have shape (N, M, T, D)")
        if self.real_paths.shape[0] != self.gen_paths.shape[0]:
            raise ValueError("real_paths and gen_paths must share N")
        if self.real_paths.shape[1] != self.gen_paths.shape[2]:
            raise ValueError("real_paths and gen_paths must share T")
        if self.real_paths.shape[2] != self.gen_paths.shape[3]:
            raise ValueError("real_paths and gen_paths must share D")
        if self.time_grid.ndim != 1 or self.time_grid.shape[0] != self.real_paths.shape[1]:
            raise ValueError("time_grid must have shape (T,)")

    @property
    def n_real(self) -> int:
        return int(self.real_paths.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.gen_paths.shape[1])

    @property
    def n_steps(self) -> int:
        return int(self.real_paths.shape[1])

    @property
    def path_dim(self) -> int:
        return int(self.real_paths.shape[2])

    @property
    def dt(self) -> float:
        return float(self.time_grid[1] - self.time_grid[0])


@dataclass
class PairwiseResult:
    name: str
    values: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerRealResult:
    name: str
    values: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributionResult:
    name: str
    values: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationBundle:
    batch: EvalBatch
    pairwise: dict[str, PairwiseResult] = field(default_factory=dict)
    per_real: dict[str, PerRealResult] = field(default_factory=dict)
    distribution: dict[str, DistributionResult] = field(default_factory=dict)
    estimated: dict[str, Any] = field(default_factory=dict)
    coverage: Any = None
    summaries: dict[str, Any] = field(default_factory=dict)
