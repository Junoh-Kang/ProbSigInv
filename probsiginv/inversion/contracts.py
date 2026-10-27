"""Shared contracts between inversion runtime and method modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InversionSourceContext:
    """Shared source data resolved once by the top-level inversion runtime."""

    paths: np.ndarray
    source_sigs: np.ndarray | None
    time_grid: np.ndarray
    meta: dict[str, Any]
    dataset_name: str

    @property
    def n_paths(self) -> int:
        return int(self.paths.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.paths.shape[1])

    @property
    def path_dim(self) -> int:
        if "path_dim" in self.meta:
            return int(self.meta["path_dim"])
        return int(self.paths.shape[-1]) if self.paths.ndim == 3 else 1


@dataclass(frozen=True)
class InversionMethodContext:
    """Shared runtime state handed off to a single inversion method."""

    source: InversionSourceContext
    dataset_dir: str
    output_dir: str


@dataclass(frozen=True)
class InversionResult:
    """Canonical method output returned before runtime-level artifact writing."""

    paths: np.ndarray
    time_grid: np.ndarray
    method: str
    metadata: dict[str, Any]
    auxiliary_arrays: dict[str, np.ndarray] = field(default_factory=dict)
