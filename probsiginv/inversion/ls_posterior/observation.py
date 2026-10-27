"""Load raw scalar Linear Signature observations from split artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import iisignature
import numpy as np

from probsiginv.utils.signature_config import linear_time_level_indices


def build_raw_ls_to_moments(depth: int) -> np.ndarray:
    """Build the triangular raw-logsignature to LS-moment transform."""
    if isinstance(depth, (bool, np.bool_)) or not isinstance(
        depth, (int, np.integer)
    ):
        raise ValueError("depth must be an integer")
    depth = int(depth)
    if depth < 0:
        raise ValueError("depth must be non-negative")

    transform = np.zeros((depth, depth), dtype=float)
    if depth == 0:
        return transform
    transform[0, 0] = 1.0
    for row in range(1, depth):
        transform[row, 0] = 1.0 / (row + 1)
        transform[row, 1] = -1.0 / row
        coefficient = -1.0
        for column in range(2, row + 1):
            if column > 2:
                coefficient *= row - column + 2
            transform[row, column] = coefficient
    return transform


RAW_LS_TO_MOMENTS = build_raw_ls_to_moments(6)
_REQUIRED_KEYS = ("sigs", "sig_depth", "sig_channels", "sig_type", "aug_type")


@dataclass(frozen=True)
class LSObservationBatch:
    """Physical-time LS observations and their raw artifact provenance."""

    observations: np.ndarray
    source_artifact: str | None
    source_depth: int | None


def load_ls_observations(
    split_or_artifact: str | Path,
    *,
    n_rows: int,
    depth: int,
    T: float,
) -> LSObservationBatch:
    """Return raw physical LS observations without consulting source paths.

    Positive depths read only ``TA_logsigs.npz`` and select the canonical
    scalar LS coordinates. Depth zero is artifact-free.
    """
    n_rows = _validate_n_rows(n_rows)
    depth = _validate_depth(depth)
    T = _validate_horizon(T)

    if depth == 0:
        return LSObservationBatch(
            observations=_readonly_float_array(np.empty((n_rows, 0), dtype=float)),
            source_artifact=None,
            source_depth=None,
        )

    artifact_path = _resolve_artifact_path(split_or_artifact)
    with np.load(artifact_path, allow_pickle=False) as artifact:
        missing = [key for key in _REQUIRED_KEYS if key not in artifact.files]
        if missing:
            raise ValueError(
                f"{artifact_path}: missing required key(s): {', '.join(missing)}"
            )

        aug_type = _load_scalar(artifact, "aug_type", artifact_path)
        sig_type = _load_scalar(artifact, "sig_type", artifact_path)
        sig_channels = _load_integer_scalar(
            artifact, "sig_channels", artifact_path
        )
        source_depth = _load_integer_scalar(artifact, "sig_depth", artifact_path)
        raw_sigs = np.asarray(artifact["sigs"])

    if str(aug_type) != "TA":
        raise ValueError(f"{artifact_path}: aug_type must be 'TA', got {aug_type!r}")
    if str(sig_type) != "logsig":
        raise ValueError(
            f"{artifact_path}: sig_type must be 'logsig', got {sig_type!r}"
        )
    if sig_channels != 2:
        raise ValueError(
            f"{artifact_path}: sig_channels must be 2 for scalar LS, got {sig_channels}"
        )
    if source_depth < 1:
        raise ValueError(
            f"{artifact_path}: source sig_depth must be positive, got {source_depth}"
        )
    if depth > source_depth:
        raise ValueError(
            f"requested depth {depth} exceeds source depth {source_depth} in {artifact_path}"
        )
    _validate_raw_sigs(
        raw_sigs,
        n_rows=n_rows,
        source_depth=source_depth,
        artifact_path=artifact_path,
    )

    level_indices = linear_time_level_indices(2, depth)
    selected_indices = tuple(level_indices[level][0] for level in range(1, depth + 1))
    with np.errstate(over="ignore", invalid="ignore"):
        raw_coordinates = np.asarray(raw_sigs[:, selected_indices], dtype=float)
        transform = build_raw_ls_to_moments(depth)
        normalized_moments = raw_coordinates @ transform.T
        physical_moments = normalized_moments * T ** np.arange(depth, dtype=float)
    if not np.all(np.isfinite(physical_moments)):
        raise ValueError(
            f"{artifact_path}: LS transform produced non-finite observations "
            f"for depth={depth}, T={T:g}"
        )
    return LSObservationBatch(
        observations=_readonly_float_array(physical_moments),
        source_artifact=str(artifact_path),
        source_depth=source_depth,
    )


def _resolve_artifact_path(split_or_artifact: str | Path) -> Path:
    path = Path(split_or_artifact)
    if path.is_dir() or path.suffix != ".npz":
        return path / "TA_logsigs.npz"
    if path.name != "TA_logsigs.npz":
        raise ValueError(
            f"LS observations require split-local TA_logsigs.npz, got {path.name!r}"
        )
    return path


def _validate_n_rows(n_rows: int) -> int:
    if isinstance(n_rows, (bool, np.bool_)) or not isinstance(
        n_rows, (int, np.integer)
    ):
        raise ValueError("n_rows must be a non-negative integer")
    value = int(n_rows)
    if value < 0:
        raise ValueError("n_rows must be a non-negative integer")
    return value


def _validate_depth(depth: int) -> int:
    if isinstance(depth, (bool, np.bool_)) or not isinstance(
        depth, (int, np.integer)
    ):
        raise ValueError("depth must be an integer")
    value = int(depth)
    if value < 0:
        raise ValueError("depth must be non-negative")
    return value


def _validate_horizon(T: float) -> float:
    value = float(T)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("T must be finite and positive")
    return value


def _readonly_float_array(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _load_scalar(artifact, key: str, artifact_path: Path):
    value = np.asarray(artifact[key])
    if value.ndim != 0:
        raise ValueError(f"{artifact_path}: {key} must be scalar")
    return value.item()


def _load_integer_scalar(artifact, key: str, artifact_path: Path) -> int:
    value = _load_scalar(artifact, key, artifact_path)
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{artifact_path}: {key} must be an integer")
    return int(value)


def _validate_raw_sigs(
    raw_sigs: np.ndarray,
    *,
    n_rows: int,
    source_depth: int,
    artifact_path: Path,
) -> None:
    if raw_sigs.ndim != 2:
        raise ValueError(
            f"{artifact_path}: sigs must be two-dimensional, got shape {raw_sigs.shape}"
        )
    if raw_sigs.shape[0] != n_rows:
        raise ValueError(
            f"{artifact_path}: sigs row count {raw_sigs.shape[0]} does not match {n_rows}"
        )
    expected_width = iisignature.logsiglength(2, source_depth)
    if raw_sigs.shape[1] != expected_width:
        raise ValueError(
            f"{artifact_path}: sigs width {raw_sigs.shape[1]} does not match "
            f"source depth {source_depth} width {expected_width}"
        )
    if not np.issubdtype(raw_sigs.dtype, np.number) or np.issubdtype(
        raw_sigs.dtype, np.complexfloating
    ):
        raise ValueError(f"{artifact_path}: sigs must contain real numeric values")
    if not np.all(np.isfinite(raw_sigs)):
        raise ValueError(f"{artifact_path}: sigs must contain only finite values")


__all__ = [
    "LSObservationBatch",
    "build_raw_ls_to_moments",
    "load_ls_observations",
]
