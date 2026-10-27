"""Dataset-level shared artifact preparation for inversion."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Callable

import iisignature
import numpy as np

from probsiginv.inversion.contracts import InversionResult
from probsiginv.inversion.outputs import save_npz_artifact
from probsiginv.inversion.provenance import file_sha256
from probsiginv.utils.contract import OUTPUT_ARTIFACT_NAME, parse_conditioning_label
from probsiginv.utils.path_shapes import normalize_candidate_artifact


@dataclass
class PreparedSharedArtifacts:
    """Dataset-level artifacts shared across inversion jobs."""

    real_path: str
    real_sigs_path: str | None


def _normalize_single_sample_paths(paths: np.ndarray) -> np.ndarray:
    """Normalize deterministic candidate paths to the canonical (N, 1, T) shape."""
    return normalize_candidate_artifact(paths)


def save_candidate_artifact(output_dir: str, result: InversionResult) -> str:
    """Persist a method result under the canonical candidate artifact contract."""
    if result.method == "ls_posterior":
        return save_ls_posterior_artifact(output_dir, result)
    path_dim = result.metadata.get("path_dim")
    result_paths = np.asarray(result.paths)
    if path_dim is None and result_paths.ndim == 3 and result_paths.shape[1] == np.asarray(result.time_grid).shape[0]:
        path_dim = int(result_paths.shape[-1])
    arrays = {
        "paths": normalize_candidate_artifact(result_paths, path_dim=path_dim),
        "time_grid": np.asarray(result.time_grid),
        "method": result.method,
    }
    for key, value in result.metadata.items():
        if value is not None:
            arrays[key] = value
    reserved = set(arrays).intersection(result.auxiliary_arrays)
    if reserved:
        raise ValueError(
            "auxiliary arrays conflict with canonical artifact field(s): "
            + ", ".join(sorted(reserved))
        )
    arrays.update(result.auxiliary_arrays)
    return save_npz_artifact(output_dir, OUTPUT_ARTIFACT_NAME, **arrays)


_LS_POSTERIOR_PARAMETERS = {
    "gbm": ("mu", "sigma"),
    "fbm": ("mu", "sigma", "hurst"),
    "ou": ("theta", "mu", "sigma"),
}
_LS_POSTERIOR_METADATA = {
    "kind",
    "conditioning",
    "n_conditioning_rows",
    "n_repeats",
    "seed",
    "n_particles_per_dim",
    "process",
    "path_dim",
    "T_horizon",
    "source_signature_artifact",
    "observation_sha256",
    "continuous_ls",
}


def save_ls_posterior_artifact(output_dir: str, result: InversionResult) -> str:
    """Validate and write one collision-safe LS Posterior reference artifact."""
    arrays = _validate_ls_posterior_result(result)
    path = os.path.join(output_dir, OUTPUT_ARTIFACT_NAME)
    if os.path.exists(path):
        raise FileExistsError(
            f"refusing to overwrite existing LS posterior artifact: {path}"
        )
    os.makedirs(output_dir, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{OUTPUT_ARTIFACT_NAME}.",
        suffix=".tmp",
        dir=output_dir,
    )
    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            np.savez(temporary_file, **arrays)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            validate_existing_ls_posterior_artifact(
                path,
                expected_metadata=result.metadata,
                expected_time_grid=result.time_grid,
            )
        return path
    finally:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass


def validate_existing_ls_posterior_artifact(
    artifact_path: str | os.PathLike[str],
    *,
    expected_metadata: dict[str, Any],
    expected_time_grid: np.ndarray,
) -> bool:
    """Return true only for an exact compatible, finite reference artifact."""
    path = os.fspath(artifact_path)
    try:
        process = str(expected_metadata["process"]).lower()
        parameter_names = _LS_POSTERIOR_PARAMETERS[process]
        expected_fields = {
            "paths",
            "time_grid",
            "method",
            *expected_metadata,
            *(f"sampled_{name}" for name in parameter_names),
        }
        n_rows = int(expected_metadata["n_conditioning_rows"])
        n_repeats = int(expected_metadata["n_repeats"])
        grid = _validate_time_grid(expected_time_grid)
        _validate_observation_sha256(expected_metadata["observation_sha256"])
        parameter_bounds = {
            name: _validate_bounds(
                expected_metadata[f"{name}_bounds"],
                name=f"{name}_bounds",
            )
            for name in parameter_names
        }

        with np.load(path, allow_pickle=False) as artifact:
            actual_fields = set(artifact.files)
            if actual_fields != expected_fields:
                missing = sorted(expected_fields - actual_fields)
                extra = sorted(actual_fields - expected_fields)
                raise ValueError(f"field mismatch; missing={missing}, extra={extra}")

            paths = _validate_reference_paths(
                artifact["paths"],
                expected_shape=(n_rows, n_repeats, grid.size),
            )
            del paths
            actual_grid = _validate_time_grid(artifact["time_grid"])
            if not np.array_equal(actual_grid, grid):
                raise ValueError("time_grid mismatch")
            if _scalar_value(artifact["method"]) != "ls_posterior":
                raise ValueError("method mismatch")
            for key, expected in expected_metadata.items():
                actual = artifact[key]
                if not _artifact_value_equal(actual, expected):
                    raise ValueError(f"metadata mismatch for {key}")
            for name in parameter_names:
                _validate_sampled_parameter(
                    artifact[f"sampled_{name}"],
                    name=f"sampled_{name}",
                    expected_shape=(n_rows, n_repeats),
                    bounds=parameter_bounds[name],
                )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ValueError(
            f"incompatible LS posterior artifact at {path}: {exc}"
        ) from exc
    return True


def _validate_ls_posterior_result(result: InversionResult) -> dict[str, Any]:
    if result.method != "ls_posterior":
        raise ValueError("LS posterior reference method must be 'ls_posterior'")
    metadata = dict(result.metadata)
    process = str(metadata.get("process", "")).lower()
    if process not in _LS_POSTERIOR_PARAMETERS:
        raise ValueError(f"unsupported LS posterior process {process!r}")
    required_metadata = _LS_POSTERIOR_METADATA | {
        f"{name}_bounds" for name in _LS_POSTERIOR_PARAMETERS[process]
    }
    if set(metadata) != required_metadata:
        missing = sorted(required_metadata - set(metadata))
        extra = sorted(set(metadata) - required_metadata)
        raise ValueError(
            f"LS posterior metadata field mismatch; missing={missing}, extra={extra}"
        )
    if metadata["kind"] != "reference":
        raise ValueError("LS posterior artifact kind must be 'reference'")
    aug_type, depth = parse_conditioning_label(str(metadata["conditioning"]))
    if aug_type != "LS":
        raise ValueError("LS posterior conditioning must use LS")
    if int(metadata["path_dim"]) != 1:
        raise ValueError("LS posterior path_dim must be 1")
    if not bool(metadata["continuous_ls"]):
        raise ValueError("LS posterior artifact must declare continuous_ls")

    n_rows = _positive_int(metadata["n_conditioning_rows"], "n_conditioning_rows")
    n_repeats = _positive_int(metadata["n_repeats"], "n_repeats")
    _positive_int(metadata["n_particles_per_dim"], "n_particles_per_dim")
    _nonnegative_int(metadata["seed"], "seed")
    horizon = float(metadata["T_horizon"])
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("T_horizon must be finite and positive")
    source_artifact = metadata["source_signature_artifact"]
    if not isinstance(source_artifact, str):
        raise ValueError("source_signature_artifact must be a string")
    if depth > 0 and not source_artifact:
        raise ValueError("positive-depth LS posterior requires a source signature artifact")
    _validate_observation_sha256(metadata["observation_sha256"])
    parameter_bounds = {
        name: _validate_bounds(
            metadata[f"{name}_bounds"],
            name=f"{name}_bounds",
        )
        for name in _LS_POSTERIOR_PARAMETERS[process]
    }

    time_grid = _validate_time_grid(result.time_grid)
    paths = _validate_reference_paths(
        result.paths,
        expected_shape=(n_rows, n_repeats, time_grid.size),
    )
    expected_auxiliary = {
        f"sampled_{name}" for name in _LS_POSTERIOR_PARAMETERS[process]
    }
    if set(result.auxiliary_arrays) != expected_auxiliary:
        missing = sorted(expected_auxiliary - set(result.auxiliary_arrays))
        extra = sorted(set(result.auxiliary_arrays) - expected_auxiliary)
        raise ValueError(
            f"LS posterior sampled parameter field mismatch; missing={missing}, extra={extra}"
        )
    sampled = {
        f"sampled_{name}": _validate_sampled_parameter(
            result.auxiliary_arrays[f"sampled_{name}"],
            name=f"sampled_{name}",
            expected_shape=(n_rows, n_repeats),
            bounds=parameter_bounds[name],
        )
        for name in _LS_POSTERIOR_PARAMETERS[process]
    }
    return {
        "paths": paths,
        "time_grid": time_grid,
        "method": result.method,
        **metadata,
        **sampled,
    }


def _validate_reference_paths(
    values: np.ndarray,
    *,
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    if np.iscomplexobj(values):
        raise ValueError("paths must be real")
    array = np.asarray(values, dtype=float)
    if array.shape != expected_shape or not np.all(np.isfinite(array)):
        raise ValueError(f"paths must be finite with shape {expected_shape}")
    return array


def _validate_time_grid(values: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(values):
        raise ValueError("time_grid must be real")
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("time_grid must be a non-empty finite vector")
    if np.any(np.diff(array) < 0.0):
        raise ValueError("time_grid must be non-decreasing")
    return array


def _validate_sampled_parameter(
    values: np.ndarray,
    *,
    name: str,
    expected_shape: tuple[int, int],
    bounds: np.ndarray,
) -> np.ndarray:
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real")
    array = np.asarray(values, dtype=float)
    if array.shape != expected_shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {expected_shape}")
    if np.any(array < bounds[0]) or np.any(array > bounds[1]):
        raise ValueError(
            f"{name} must lie within inclusive bounds "
            f"[{bounds[0]}, {bounds[1]}]"
        )
    return array


def _validate_bounds(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain two finite values")
    if array[0] > array[1]:
        raise ValueError(f"{name} must satisfy lower <= upper")
    return array


def _validate_observation_sha256(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError("observation_sha256 must be a lowercase SHA-256 hex digest")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _scalar_value(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim != 0:
        raise ValueError("expected scalar metadata")
    return array.item()


def _artifact_value_equal(actual: Any, expected: Any) -> bool:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if actual_array.shape != expected_array.shape:
        return False
    return bool(np.array_equal(actual_array, expected_array))


def save_real_paths(
    dataset_dir: str,
    paths: np.ndarray,
    time_grid: np.ndarray,
    meta: dict,
    *,
    source_dataset_path: str | None = None,
) -> str:
    """Save dataset-level real paths if they do not already exist."""
    os.makedirs(dataset_dir, exist_ok=True)
    real_out = os.path.join(dataset_dir, "real.npz")
    if os.path.exists(real_out):
        if source_dataset_path is None or file_sha256(real_out) != file_sha256(source_dataset_path):
            raise ValueError(f"Existing reference paths differ from the source dataset: {real_out}")
        print(f"  real.npz already exists, skipping -> {real_out}")
        return real_out

    if source_dataset_path is None:
        raise ValueError("dataset-backed inversion requires source_dataset_path to materialize real.npz")

    shutil.copy2(source_dataset_path, real_out)
    print(f"  copied -> {real_out}")
    return real_out


def save_fourier_paths(
    output_dir: str,
    paths: np.ndarray,
    time_grid: np.ndarray,
    *,
    sig_depth: int,
    n_coeffs: int | None = None,
) -> str:
    """Save Fourier inversion output under the canonical baseline artifact contract."""
    return save_candidate_artifact(
        output_dir,
        InversionResult(
            paths=np.asarray(paths),
            time_grid=np.asarray(time_grid),
            method="fourier",
            metadata={
                "sig_depth": sig_depth,
                "n_coeffs": n_coeffs,
            },
        ),
    )


def save_legendre_paths(
    output_dir: str,
    paths: np.ndarray,
    time_grid: np.ndarray,
    *,
    sig_depth: int,
    n_coeffs: int | None = None,
) -> str:
    """Save Legendre inversion output under the canonical baseline artifact contract."""
    return save_candidate_artifact(
        output_dir,
        InversionResult(
            paths=np.asarray(paths),
            time_grid=np.asarray(time_grid),
            method="legendre",
            metadata={
                "sig_depth": sig_depth,
                "n_coeffs": n_coeffs,
            },
        ),
    )


def save_real_signatures(
    dataset_dir: str,
    sigs: np.ndarray,
    *,
    aug_type: str,
    sig_type: str,
    sig_depth: int,
) -> str:
    """Save the dataset-backed conditioning signatures alongside real.npz."""
    out = os.path.join(dataset_dir, f"real_sigs_{aug_type}_{sig_type}.npz")
    if os.path.exists(out):
        print(f"  real_sigs already exists, skipping -> {out}")
        return out
    out = save_npz_artifact(
        dataset_dir,
        f"real_sigs_{aug_type}_{sig_type}.npz",
        sigs=sigs,
        aug_type=aug_type,
        sig_depth=sig_depth,
        sig_type=sig_type,
    )
    print(f"  saved -> {out}")
    return out


def build_deterministic_signature_artifact_name(
    source: str,
    *,
    sig_type: str,
    sig_depth: int,
) -> str:
    """Build the canonical deterministic signature cache artifact name."""
    return f"real_sigs_{str(source).lower()}_{str(sig_type).lower()}_d{int(sig_depth)}.npz"


def _npz_scalar(npz, key: str) -> Any:
    value = npz.get(key)
    if value is None:
        return None
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array


def _validate_logsignature_cache_array(
    logsigs: np.ndarray,
    *,
    expected_shape: tuple[int, int],
    context: str,
) -> np.ndarray:
    array = np.asarray(logsigs)
    if array.shape != expected_shape:
        raise ValueError(f"{context}: expected shape {expected_shape}, got {array.shape}")
    if array.dtype != np.dtype(np.float32):
        raise ValueError(f"{context}: expected dtype float32, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{context}: log-signatures must be finite")
    return array


def load_or_compute_deterministic_logsignatures(
    dataset_dir: str,
    *,
    source: str,
    sig_depth: int,
    sig_channels: int,
    n_rows: int,
    compute_logsignatures: Callable[[], np.ndarray],
    metadata: dict[str, Any] | None = None,
) -> np.ndarray:
    """Load or create one strictly validated float32 log-signature cache."""
    metadata = dict(metadata or {})
    artifact_name = build_deterministic_signature_artifact_name(
        source,
        sig_type="logsig",
        sig_depth=sig_depth,
    )
    artifact_path = os.path.join(dataset_dir, artifact_name)
    expected_shape = (
        int(n_rows),
        int(iisignature.logsiglength(int(sig_channels), int(sig_depth))),
    )
    expected = {
        "source": str(source).lower(),
        "sig_type": "logsig",
        "representation": "logsignature",
        "sig_channels": int(sig_channels),
        "sig_depth": int(sig_depth),
        "dtype": "float32",
        **metadata,
    }

    if os.path.exists(artifact_path):
        try:
            with np.load(artifact_path, allow_pickle=False) as npz:
                if "logsigs" not in npz.files or "sigs" in npz.files:
                    raise ValueError("required payload is exactly 'logsigs', not 'sigs'")
                allowed_fields = {"logsigs", *expected}
                actual_fields = set(npz.files)
                unexpected_fields = sorted(actual_fields - allowed_fields)
                missing_fields = sorted(allowed_fields - actual_fields)
                if unexpected_fields:
                    raise ValueError(f"unexpected fields: {', '.join(unexpected_fields)}")
                if missing_fields:
                    raise ValueError(f"missing fields: {', '.join(missing_fields)}")
                mismatches = [
                    key
                    for key, value in expected.items()
                    if _npz_scalar(npz, key) != value
                ]
                if mismatches:
                    raise ValueError(f"metadata mismatch for {', '.join(mismatches)}")
                logsigs = _validate_logsignature_cache_array(
                    npz["logsigs"],
                    expected_shape=expected_shape,
                    context="cached log-signatures",
                )
        except (KeyError, OSError, ValueError) as exc:
            raise ValueError(
                f"incompatible deterministic log-signature cache at {artifact_path}: {exc}"
            ) from exc
        print(f"  deterministic log-signature cache hit -> {artifact_path}")
        return logsigs

    logsigs = np.asarray(compute_logsignatures(), dtype=np.float32)
    logsigs = _validate_logsignature_cache_array(
        logsigs,
        expected_shape=expected_shape,
        context="computed log-signatures",
    )
    out = save_npz_artifact(dataset_dir, artifact_name, logsigs=logsigs, **expected)
    print(f"  saved -> {out}")
    return logsigs


def copy_real_signature_artifact(
    dataset_dir: str,
    *,
    source_sig_path: str,
    output_name: str,
) -> str:
    """Copy a canonical split-local signature artifact into the output dataset root."""
    os.makedirs(dataset_dir, exist_ok=True)
    out = os.path.join(dataset_dir, output_name)
    if os.path.exists(out):
        if file_sha256(out) != file_sha256(source_sig_path):
            raise ValueError(f"Existing reference signatures differ from the source dataset: {out}")
        print(f"  real_sigs already exists, skipping -> {out}")
        return out
    shutil.copy2(source_sig_path, out)
    print(f"  copied -> {out}")
    return out


def prepare_shared_dataset_artifacts(
    args: argparse.Namespace | None,
    paths: np.ndarray,
    source_sigs: np.ndarray | None,
    time_grid: np.ndarray,
    meta: dict,
    dataset_dir: str,
    *,
    ckpt_args: dict | None = None,
    include_source_signatures: bool | None = None,
    source_dataset_path: str | None = None,
) -> PreparedSharedArtifacts:
    """Prepare dataset-level artifacts shared by all inversion jobs."""
    source_dataset_path = source_dataset_path or (
        str(getattr(args, "dataset")) if args is not None and getattr(args, "dataset", None) is not None else None
    )
    real_path = save_real_paths(
        dataset_dir,
        paths,
        time_grid,
        meta,
        source_dataset_path=source_dataset_path,
    )
    real_sigs_path: str | None = None
    source_split_dir = os.path.dirname(source_dataset_path) if source_dataset_path is not None else None

    if source_split_dir is not None:
        ta_sig_path = os.path.join(source_split_dir, "TA_logsigs.npz")
        tll3_sig_path = os.path.join(source_split_dir, "TLL3_logsigs.npz")
        if os.path.exists(ta_sig_path):
            real_sigs_path = copy_real_signature_artifact(
                dataset_dir,
                source_sig_path=ta_sig_path,
                output_name="real_sigs_TA_logsig.npz",
            )
        if os.path.exists(tll3_sig_path):
            copy_real_signature_artifact(
                dataset_dir,
                source_sig_path=tll3_sig_path,
                output_name="real_sigs_TLL3_logsig.npz",
            )

    if include_source_signatures is None:
        include_source_signatures = args is not None and args.method == "generative"

    if include_source_signatures and real_sigs_path is None:
        if source_sigs is None:
            raise ValueError("dataset-backed inversion requires source signatures")

        train_aug_type = str(meta.get("aug_type", "TA"))
        sig_type = str(meta.get("sig_type", "logsig"))
        sig_depth = int(meta.get("sig_depth", 0))
        real_sigs_path = save_real_signatures(
            dataset_dir,
            source_sigs,
            aug_type=train_aug_type,
            sig_type=sig_type,
            sig_depth=sig_depth,
        )

    return PreparedSharedArtifacts(real_path=real_path, real_sigs_path=real_sigs_path)
