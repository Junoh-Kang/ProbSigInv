"""Build multivariate FX sliding-window datasets from a declared NPY schema.

Usage
-----
    python -m probsiginv.cli.build_dataset configs/dataset/fx_daily.yaml
    python -m probsiginv.cli.build_dataset configs/dataset/fx_daily.yaml --sigs-only
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from probsiginv.repo import resolve_repo_path
from probsiginv.utils.contract import build_split_paths_path, build_split_sigs_path
from probsiginv.utils.signatures import augment_path, compute_logsignatures


DEFAULT_SIG_AUG_TYPES = ("TA",)
SUPPORTED_SIG_AUG_TYPES = ("TA", "TLL3")


def _require_positive_int(value: Any, name: str) -> int:
    resolved = int(value)
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if resolved <= 0:
        raise ValueError(f"{name} must be positive, got {resolved}")
    return resolved


def _require_positive_float(value: Any, name: str) -> float:
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return resolved


def _require_string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    resolved = [str(item) for item in value]
    if any(not item for item in resolved):
        raise ValueError(f"{name} must not contain empty values")
    return resolved


def resolve_source_columns(source_labels: list[str], currencies: list[str]) -> list[int]:
    """Resolve a duplicate-free currency selection against the ordered source schema."""
    if len(set(source_labels)) != len(source_labels):
        raise ValueError("source_labels must not contain duplicate labels")
    if len(set(currencies)) != len(currencies):
        raise ValueError("currencies must not contain duplicate values")

    label_to_column = {label: idx for idx, label in enumerate(source_labels)}
    unknown = [currency for currency in currencies if currency not in label_to_column]
    if unknown:
        raise ValueError(f"Unknown currencies: {unknown!r}")
    return [label_to_column[currency] for currency in currencies]


def build_windows(rates: np.ndarray, n_steps: int, stride: int) -> np.ndarray:
    """Return coordinate-wise log-relative windows with ``n_steps + 1`` points."""
    window_len = n_steps + 1
    if rates.ndim != 2:
        raise ValueError(f"rates must have shape (rows, columns), got {rates.shape}")
    if rates.shape[0] < window_len:
        raise ValueError(
            f"split has {rates.shape[0]} rows but needs at least {window_len} rows"
        )
    log_rates = np.log(rates)
    starts = range(0, rates.shape[0] - window_len + 1, stride)
    return np.stack(
        [log_rates[start : start + window_len] - log_rates[start] for start in starts]
    )


def estimate_params(paths: np.ndarray, T_horizon: float) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-window, per-coordinate annualized drift and volatility."""
    n_steps = paths.shape[1] - 1
    dt = T_horizon / n_steps
    log_returns = np.diff(paths, axis=1)
    sigma = np.std(log_returns, axis=1, ddof=1) / np.sqrt(dt)
    mu = np.mean(log_returns, axis=1) / dt + 0.5 * sigma**2
    return mu, sigma


def _resolve_sig_aug_types(config: dict[str, Any]) -> tuple[str, ...]:
    values = config.get("aug_type", DEFAULT_SIG_AUG_TYPES)
    if isinstance(values, str):
        resolved = (values.upper(),)
    elif isinstance(values, list) and values:
        resolved = tuple(str(value).upper() for value in values)
    else:
        raise ValueError("aug_type must be a string or a non-empty list")

    invalid = [value for value in resolved if value not in SUPPORTED_SIG_AUG_TYPES]
    if invalid:
        raise ValueError(f"Unsupported aug_type values: {invalid!r}")
    return tuple(dict.fromkeys(resolved))


FX_TEST_TAGS = ("in-distribution", "out-distribution")


def _resolve_test_strides(
    config: dict[str, Any], *, window_stride: int
) -> tuple[int, int | None]:
    """Return (out-distribution stride, in-distribution stride or None) from ``test`` entries."""
    legacy = [key for key in ("test_window_stride", "in_distribution_window_stride") if key in config]
    if legacy:
        raise ValueError(
            f"{legacy!r} are no longer supported; declare test splits as "
            "test: [{tag: in-distribution|out-distribution, window_stride: N}]"
        )
    entries = config.get("test")
    if entries is None:
        return window_stride, None
    if not isinstance(entries, list) or not entries:
        raise ValueError("test must be a non-empty list of split entries")

    strides: dict[str, int] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"test entries must be mappings, got {entry!r}")
        tag = entry.get("tag")
        if tag not in FX_TEST_TAGS:
            raise ValueError(f"test entry tag must be one of {FX_TEST_TAGS!r}, got {tag!r}")
        if tag in strides:
            raise ValueError(f"Duplicate test dataset tag: {tag}")
        unknown = sorted(set(entry) - {"tag", "window_stride"})
        if unknown:
            raise ValueError(f"test entry {tag!r} has unsupported keys {unknown!r}")
        stride = entry.get("window_stride", window_stride)
        if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
            raise ValueError(
                f"test entry {tag!r} window_stride must be a positive integer, got {stride!r}"
            )
        strides[tag] = stride
    if "out-distribution" not in strides:
        raise ValueError("test must include an out-distribution entry")
    return strides["out-distribution"], strides.get("in-distribution")


def _format_number(value: float) -> str:
    return f"{value:g}"


def _dataset_stem(config: dict[str, Any]) -> str:
    currency_tag = "-".join(currency.lower() for currency in config["currencies"])
    train_percent = _format_number(100.0 * config["train_ratio"])
    return (
        f"fx_{currency_tag}_tr{train_percent}_n{config['n_steps']}"
        f"_s{config['window_stride']}_ps{_format_number(config['path_scale'])}"
    )


def _load_config(
    config_path: str,
    *,
    aug_type_override: str | None,
    sig_depth_override: int | None,
    sig_type_override: str | None,
) -> dict[str, Any]:
    loaded = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(loaded, dict):
        raise ValueError("FX config must be a YAML mapping")

    source_labels = _require_string_list(loaded.get("source_labels"), "source_labels")
    currencies = _require_string_list(loaded.get("currencies"), "currencies")
    source_columns = resolve_source_columns(source_labels, currencies)
    n_steps = _require_positive_int(loaded.get("n_steps"), "n_steps")
    if n_steps < 2:
        raise ValueError(f"n_steps must be at least 2, got {n_steps}")
    window_stride = _require_positive_int(
        loaded.get("window_stride", 1), "window_stride"
    )
    test_window_stride, in_distribution_window_stride = _resolve_test_strides(
        loaded, window_stride=window_stride
    )
    train_ratio = float(loaded.get("train_ratio", 0.8))
    if not np.isfinite(train_ratio) or not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be strictly between 0 and 1, got {train_ratio!r}")

    path_scale = _require_positive_float(loaded.get("path_scale", 1.0), "path_scale")
    T_horizon = _require_positive_float(loaded.get("T_horizon", 1.0), "T_horizon")
    S0 = _require_positive_float(loaded.get("S0", 1.0), "S0")
    sig_depth_source = sig_depth_override if sig_depth_override is not None else loaded.get("sig_depth", 6)
    sig_depth = _require_positive_int(sig_depth_source, "sig_depth")
    sig_type = str(sig_type_override or loaded.get("sig_type", "logsig"))
    if sig_type != "logsig":
        raise ValueError(f'Only sig_type="logsig" is supported, got {sig_type!r}')

    if "source_path" not in loaded:
        raise ValueError("source_path is required")
    source_path = resolve_repo_path(str(loaded["source_path"])).resolve()
    output_dir = resolve_repo_path(str(loaded.get("output_root", loaded.get("output_dir", "_data"))))
    sig_aug_types = (
        (str(aug_type_override).upper(),)
        if aug_type_override is not None
        else _resolve_sig_aug_types(loaded)
    )
    invalid_aug_types = [value for value in sig_aug_types if value not in SUPPORTED_SIG_AUG_TYPES]
    if invalid_aug_types:
        raise ValueError(f"Unsupported aug_type values: {invalid_aug_types!r}")

    return {
        **loaded,
        "process": str(loaded.get("process", "finance")),
        "source_path": source_path,
        "source_labels": source_labels,
        "currencies": currencies,
        "source_columns": source_columns,
        "n_steps": n_steps,
        "train_ratio": train_ratio,
        "window_stride": window_stride,
        "test_window_stride": test_window_stride,
        "in_distribution_window_stride": in_distribution_window_stride,
        "T_horizon": T_horizon,
        "S0": S0,
        "path_scale": path_scale,
        "sig_depth": sig_depth,
        "sig_type": sig_type,
        "sig_aug_types": sig_aug_types,
        "seed": int(loaded.get("seed", 0)),
        "output_dir": output_dir,
    }


def _load_selected_rates(config: dict[str, Any]) -> np.ndarray:
    source_path = config["source_path"]
    source = np.load(source_path, allow_pickle=False)
    if not isinstance(source, np.ndarray) or source.ndim != 2:
        raise ValueError(f"FX source must have shape (rows, columns), got {getattr(source, 'shape', None)}")
    expected_width = len(config["source_labels"])
    if source.shape[1] != expected_width:
        raise ValueError(
            f"FX source width {source.shape[1]} does not match source_labels width {expected_width}"
        )

    selected = np.asarray(source[:, config["source_columns"]], dtype=np.float64)
    if not np.isfinite(selected).all():
        raise ValueError("Selected FX source values must all be finite")
    if not (selected > 0.0).all():
        raise ValueError("Selected FX source values must all be positive")
    return selected


def _identity(config: dict[str, Any], *, n_rows: int) -> dict[str, Any]:
    split_at = int(n_rows * config["train_ratio"])
    return {
        "source_path": str(config["source_path"]),
        "source_labels": config["source_labels"],
        "currencies": config["currencies"],
        "source_columns": config["source_columns"],
        "source_n_rows": n_rows,
        "n_steps": config["n_steps"],
        "train_ratio": config["train_ratio"],
        "train_window_stride": config["window_stride"],
        "test_window_stride": config["test_window_stride"],
        "path_scale": config["path_scale"],
        "split_at": split_at,
    }


def _expected_split_metadata(
    identity: dict[str, Any],
    split: str,
    *,
    in_distribution_window_stride: int | None = None,
) -> dict[str, Any]:
    if split == "train":
        start, end = 0, identity["split_at"]
        stride = identity["train_window_stride"]
    elif split == "test/out-distribution":
        start, end = identity["split_at"], identity["source_n_rows"]
        stride = identity["test_window_stride"]
    elif split == "test/in-distribution":
        if in_distribution_window_stride is None:
            raise ValueError(
                "test/in-distribution requires in_distribution_window_stride"
            )
        start, end = 0, identity["split_at"]
        stride = in_distribution_window_stride
    else:
        raise ValueError(f"unsupported FX split {split!r}")
    metadata = {
        **identity,
        "split": split,
        "raw_interval_start": start,
        "raw_interval_end": end,
        "raw_row_count": end - start,
        "window_stride": stride,
    }
    if split == "test/in-distribution":
        metadata["in_distribution_window_stride"] = stride
    return metadata


def _npz_value(data: np.lib.npyio.NpzFile, key: str) -> Any:
    if key not in data.files:
        raise ValueError(f"missing identity field {key!r}")
    value = np.asarray(data[key])
    return value.item() if value.ndim == 0 else value.tolist()


def _normalize_legacy_split_identity(
    key: str,
    actual_value: Any,
    expected_value: Any,
) -> Any:
    if (
        key == "split"
        and expected_value == "test/out-distribution"
        and actual_value == "test/chronological"
    ):
        return expected_value
    return actual_value


def _assert_matching_identity(
    artifact_path: Path,
    expected: dict[str, Any],
) -> None:
    try:
        with np.load(artifact_path, allow_pickle=False) as stored:
            for key, expected_value in expected.items():
                actual_value = _normalize_legacy_split_identity(
                    key,
                    _npz_value(stored, key),
                    expected_value,
                )
                if actual_value != expected_value:
                    raise ValueError(
                        f"{artifact_path}: dataset identity mismatch for {key}: "
                        f"stored={actual_value!r}, requested={expected_value!r}"
                    )
    except FileNotFoundError as exc:
        raise ValueError(
            f"{artifact_path}: existing dataset root is missing identity artifact"
        ) from exc
    except (KeyError, OSError) as exc:
        raise ValueError(f"{artifact_path}: could not read dataset identity") from exc


def _validate_existing_root(
    dataset_root: Path,
    identity: dict[str, Any],
    *,
    require_existing: bool,
) -> None:
    if not dataset_root.exists():
        if require_existing:
            raise ValueError(f"{dataset_root}: --sigs-only requires an existing dataset identity")
        return
    if not dataset_root.is_dir():
        raise ValueError(f"{dataset_root}: existing dataset root is not a directory")

    for split in ("train", "test/out-distribution"):
        artifact_path = build_split_paths_path(dataset_root, split)
        _assert_matching_identity(
            artifact_path,
            _expected_split_metadata(identity, split),
        )


def _inspect_targeted_id_state(
    dataset_root: Path,
    *,
    sigs_only: bool,
    sig_aug_types: tuple[str, ...],
) -> None:
    split_dir = dataset_root / "test/in-distribution"
    allowed = {"paths.npz", "TA_logsigs.npz", "TLL3_logsigs.npz"}
    if split_dir.is_symlink():
        raise ValueError(
            f"{split_dir}: targeted ID split must not be a symbolic link"
        )
    if split_dir.exists():
        if not split_dir.is_dir():
            raise ValueError(
                f"{split_dir}: targeted ID split is not a directory"
            )
        entries = list(split_dir.iterdir())
    else:
        entries = []
    for path in entries:
        if path.name not in allowed:
            continue
        if path.is_symlink():
            raise ValueError(
                f"{path}: targeted ID artifact must not be a symbolic link"
            )
        if not path.is_file():
            raise ValueError(
                f"{path}: targeted ID artifact must be a regular file"
            )
    existing = {path.name for path in entries}
    unexpected = sorted(existing - allowed)
    if unexpected:
        raise ValueError(
            f"{split_dir}: unexpected targeted ID artifacts {unexpected}"
        )

    if not sigs_only:
        expected = {"paths.npz"} | {
            f"{aug_type}_logsigs.npz" for aug_type in sig_aug_types
        }
        if existing:
            if existing != expected:
                raise ValueError(
                    f"{split_dir}: partial targeted ID state: "
                    f"expected absent {sorted(expected)}, found {sorted(existing)}"
                )
            raise ValueError(f"{split_dir}: targeted ID base artifacts already exist")
        return

    base = {"paths.npz", "TA_logsigs.npz"}
    if not base.issubset(existing):
        raise ValueError(
            f"{split_dir}: partial targeted ID base state: found {sorted(existing)}"
        )
    requested = {f"{aug_type}_logsigs.npz" for aug_type in sig_aug_types}
    if existing & requested:
        raise ValueError(
            f"{split_dir}: targeted ID signature artifacts already exist: "
            f"{sorted(existing & requested)}"
        )
    if existing != base:
        raise ValueError(f"{split_dir}: unexpected targeted ID state {sorted(existing)}")


def _assert_id_path_subset(
    dataset_root: Path,
    *,
    paths: np.ndarray,
    raw_paths: np.ndarray,
    time_grid: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    config: dict[str, Any],
    identity: dict[str, Any],
) -> None:
    id_stride = config["in_distribution_window_stride"]
    train_stride = config["window_stride"]
    if id_stride % train_stride != 0:
        raise ValueError(
            "in_distribution_window_stride must align with train window_stride "
            "for exact subset identity"
        )
    subset_step = id_stride // train_stride
    train_path = build_split_paths_path(dataset_root, "train")
    _assert_matching_identity(
        train_path,
        _expected_split_metadata(identity, "train"),
    )
    try:
        with np.load(train_path, allow_pickle=False) as train:
            expected = {
                field: np.asarray(train[field])[::subset_step]
                for field in ("paths", "raw_paths", "mu", "sigma")
            }
            train_time_grid = np.asarray(train["time_grid"])
    except (KeyError, OSError) as exc:
        raise ValueError(f"{train_path}: could not load train subset fields") from exc
    actual = {
        "paths": paths,
        "raw_paths": raw_paths,
        "mu": mu,
        "sigma": sigma,
    }
    for field, expected_value in expected.items():
        if not np.isfinite(expected_value).all() or not np.array_equal(
            actual[field], expected_value
        ):
            raise ValueError(
                f"test/in-distribution {field} is not the exact train subset"
            )
    if not np.array_equal(time_grid, train_time_grid):
        raise ValueError("test/in-distribution time_grid differs from train")


def _assert_id_signature_subset(
    dataset_root: Path,
    *,
    sigs: np.ndarray,
    raw_sigs: np.ndarray,
    sig_channels: int,
    aug_type: str,
    config: dict[str, Any],
    identity: dict[str, Any],
) -> None:
    subset_step = (
        config["in_distribution_window_stride"] // config["window_stride"]
    )
    train_path = build_split_sigs_path(
        dataset_root,
        "train",
        aug_type,
        config["sig_type"],
    )
    expected_identity = _expected_split_metadata(identity, "train")
    expected_metadata = {
        **{
            key: value
            for key, value in expected_identity.items()
            if key not in {"process", "seed"}
        },
        "sig_depth": config["sig_depth"],
        "sig_channels": sig_channels,
        "sig_type": config["sig_type"],
        "aug_type": aug_type,
        "path_dim": len(config["currencies"]),
        "n_steps": config["n_steps"],
    }
    _assert_matching_identity(train_path, expected_metadata)
    try:
        with np.load(train_path, allow_pickle=False) as train:
            expected_sigs = np.asarray(train["sigs"])[::subset_step]
            expected_raw_sigs = np.asarray(train["raw_sigs"])[::subset_step]
    except (KeyError, OSError) as exc:
        raise ValueError(f"{train_path}: could not load train signatures") from exc
    if not np.isfinite(expected_sigs).all() or not np.array_equal(sigs, expected_sigs):
        raise ValueError(
            f"test/in-distribution {aug_type} sigs are not the exact train subset"
        )
    if not np.isfinite(expected_raw_sigs).all() or not np.array_equal(
        raw_sigs,
        expected_raw_sigs,
    ):
        raise ValueError(
            f"test/in-distribution {aug_type} raw_sigs are not the exact train subset"
        )


def _preflight_split_arrays(
    artifact_path: Path,
    *,
    paths: np.ndarray,
    raw_paths: np.ndarray,
    time_grid: np.ndarray,
    config: dict[str, Any],
    split_metadata: dict[str, Any],
) -> None:
    window_len = config["n_steps"] + 1
    expected_rows = (
        (split_metadata["raw_row_count"] - window_len)
        // split_metadata["window_stride"]
        + 1
    )
    expected_shape = (expected_rows, window_len, len(config["currencies"]))
    if paths.shape != expected_shape or raw_paths.shape != expected_shape:
        raise ValueError(
            f"{artifact_path}: preflight expected paths/raw_paths shape "
            f"{expected_shape}, got {paths.shape}/{raw_paths.shape}"
        )
    if not np.isfinite(paths).all() or not np.isfinite(raw_paths).all():
        raise ValueError(f"{artifact_path}: preflight paths/raw_paths must be finite")
    if not np.array_equal(raw_paths[:, 0, :], np.zeros((expected_rows, expected_shape[-1]))):
        raise ValueError(f"{artifact_path}: preflight raw_paths must start at zero")

    # Stored paths are produced by this exact float64 multiplication, so bitwise
    # equality is part of the strict rebuild identity rather than a tolerance check.
    if not np.array_equal(paths, config["path_scale"] * raw_paths):
        raise ValueError(
            f"{artifact_path}: preflight paths must equal path_scale * raw_paths exactly"
        )

    expected_time_grid = np.linspace(0.0, 1.0, window_len)
    if time_grid.shape != expected_time_grid.shape or not np.array_equal(
        time_grid, expected_time_grid
    ):
        raise ValueError(
            f"{artifact_path}: preflight time_grid must exactly match "
            f"the {window_len}-point normalized grid"
        )


def _atomic_savez(out_path: Path, **payload: Any) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=out_path.parent,
        prefix=f".{out_path.name}.",
        suffix=".tmp.npz",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    try:
        np.savez(temp_path, **payload)
        temp_path.replace(out_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _save_timeseries(
    out_path: Path,
    *,
    paths: np.ndarray,
    raw_paths: np.ndarray,
    time_grid: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    config: dict[str, Any],
    split_metadata: dict[str, Any],
) -> None:
    _atomic_savez(
        out_path,
        paths=paths,
        raw_paths=raw_paths,
        time_grid=time_grid,
        process=np.array(config["process"]),
        path_dim=np.int64(len(config["currencies"])),
        path_scale=np.float64(config["path_scale"]),
        mu=mu,
        sigma=sigma,
        S0=np.float64(config["S0"]),
        T_horizon=np.float64(config["T_horizon"]),
        n_steps=np.int64(config["n_steps"]),
        seed=np.int64(config["seed"]),
        split=np.array(split_metadata["split"]),
        source_path=np.array(str(config["source_path"])),
        source_labels=np.asarray(config["source_labels"]),
        currencies=np.asarray(config["currencies"]),
        source_columns=np.asarray(config["source_columns"], dtype=np.int64),
        source_n_rows=np.int64(split_metadata["source_n_rows"]),
        train_ratio=np.float64(config["train_ratio"]),
        train_window_stride=np.int64(config["window_stride"]),
        test_window_stride=np.int64(config["test_window_stride"]),
        split_at=np.int64(split_metadata["split_at"]),
        raw_interval_start=np.int64(split_metadata["raw_interval_start"]),
        raw_interval_end=np.int64(split_metadata["raw_interval_end"]),
        raw_row_count=np.int64(split_metadata["raw_row_count"]),
        window_stride=np.int64(split_metadata["window_stride"]),
        **(
            {
                "in_distribution_window_stride": np.int64(
                    split_metadata["in_distribution_window_stride"]
                )
            }
            if "in_distribution_window_stride" in split_metadata
            else {}
        ),
    )
    print(f"Saved timeseries -> {out_path}  {paths.shape}")


def _save_sigs(
    out_path: Path,
    *,
    sigs: np.ndarray,
    raw_sigs: np.ndarray,
    sig_channels: int,
    config: dict[str, Any],
    split_metadata: dict[str, Any],
    aug_type: str,
) -> None:
    _atomic_savez(
        out_path,
        sigs=sigs,
        raw_sigs=raw_sigs,
        sig_depth=np.int64(config["sig_depth"]),
        sig_channels=np.int64(sig_channels),
        sig_type=np.array(config["sig_type"]),
        aug_type=np.array(aug_type),
        path_scale=np.float64(config["path_scale"]),
        split=np.array(split_metadata["split"]),
        path_dim=np.int64(len(config["currencies"])),
        n_steps=np.int64(config["n_steps"]),
        source_path=np.array(str(config["source_path"])),
        source_labels=np.asarray(config["source_labels"]),
        currencies=np.asarray(config["currencies"]),
        source_columns=np.asarray(config["source_columns"], dtype=np.int64),
        source_n_rows=np.int64(split_metadata["source_n_rows"]),
        train_ratio=np.float64(config["train_ratio"]),
        train_window_stride=np.int64(config["window_stride"]),
        test_window_stride=np.int64(config["test_window_stride"]),
        split_at=np.int64(split_metadata["split_at"]),
        raw_interval_start=np.int64(split_metadata["raw_interval_start"]),
        raw_interval_end=np.int64(split_metadata["raw_interval_end"]),
        raw_row_count=np.int64(split_metadata["raw_row_count"]),
        window_stride=np.int64(split_metadata["window_stride"]),
        **(
            {
                "in_distribution_window_stride": np.int64(
                    split_metadata["in_distribution_window_stride"]
                )
            }
            if "in_distribution_window_stride" in split_metadata
            else {}
        ),
    )
    print(f"Saved sigs       -> {out_path}  {sigs.shape}")


def build_fx_dataset(
    config_path: str,
    sigs_only: bool = False,
    aug_type_override: str | None = None,
    sig_depth_override: int | None = None,
    sig_type_override: str | None = None,
    split_override: str | None = None,
    output_root_override: str | None = None,
) -> Path:
    """Build a split-before-windowing FX dataset from a YAML config."""
    config = _load_config(
        config_path,
        aug_type_override=aug_type_override,
        sig_depth_override=sig_depth_override,
        sig_type_override=sig_type_override,
    )
    if output_root_override is not None:
        config["output_dir"] = resolve_repo_path(output_root_override)
    selected_rates = _load_selected_rates(config)
    identity = _identity(config, n_rows=selected_rates.shape[0])
    split_at = identity["split_at"]
    window_len = config["n_steps"] + 1
    if split_at < window_len:
        raise ValueError(
            f"train split has {split_at} rows but needs at least {window_len} rows"
        )
    test_rows = selected_rates.shape[0] - split_at
    if test_rows < window_len:
        raise ValueError(
            f"test split has {test_rows} rows but needs at least {window_len} rows"
        )

    dataset_root = config["output_dir"] / _dataset_stem(config)
    if split_override not in {None, "test/in-distribution"}:
        raise ValueError(f"unsupported targeted split {split_override!r}")
    if (
        split_override == "test/in-distribution"
        and config["in_distribution_window_stride"] is None
    ):
        raise ValueError(
            "--split test/in-distribution requires in_distribution_window_stride"
        )
    targeted_id = split_override == "test/in-distribution"
    _validate_existing_root(
        dataset_root,
        identity,
        require_existing=sigs_only or targeted_id,
    )
    if targeted_id:
        _inspect_targeted_id_state(
            dataset_root,
            sigs_only=sigs_only,
            sig_aug_types=config["sig_aug_types"],
        )

    split_inputs: tuple[tuple[str, np.ndarray], ...] = (
        ("train", selected_rates[:split_at]),
        ("test/out-distribution", selected_rates[split_at:]),
    )
    if config["in_distribution_window_stride"] is not None:
        split_inputs += (("test/in-distribution", selected_rates[:split_at]),)
    if targeted_id:
        split_inputs = tuple(
            item for item in split_inputs if item[0] == "test/in-distribution"
        )
    time_grid = np.linspace(0.0, 1.0, window_len)

    # Phase 1: load/build and preflight every split before signature work or writes.
    prepared_splits: list[dict[str, Any]] = []
    for split, split_rates in split_inputs:
        split_metadata = _expected_split_metadata(
            identity,
            split,
            in_distribution_window_stride=config[
                "in_distribution_window_stride"
            ],
        )
        paths_path = build_split_paths_path(dataset_root, split)
        if sigs_only:
            _assert_matching_identity(paths_path, split_metadata)
            try:
                with np.load(paths_path, allow_pickle=False) as stored:
                    paths = np.asarray(stored["paths"], dtype=np.float64)
                    raw_paths = np.asarray(stored["raw_paths"], dtype=np.float64)
                    split_time_grid = np.asarray(stored["time_grid"], dtype=np.float64)
            except (KeyError, OSError) as exc:
                raise ValueError(
                    f"{paths_path}: preflight could not load stored path arrays"
                ) from exc
            mu = None
            sigma = None
        else:
            raw_paths = build_windows(
                split_rates,
                config["n_steps"],
                split_metadata["window_stride"],
            )
            paths = raw_paths * config["path_scale"]
            mu, sigma = estimate_params(raw_paths, config["T_horizon"])
            split_time_grid = time_grid

        _preflight_split_arrays(
            paths_path,
            paths=paths,
            raw_paths=raw_paths,
            time_grid=split_time_grid,
            config=config,
            split_metadata=split_metadata,
        )
        if split == "test/in-distribution" and targeted_id:
            if sigs_only:
                try:
                    with np.load(paths_path, allow_pickle=False) as stored:
                        mu = np.asarray(stored["mu"], dtype=np.float64)
                        sigma = np.asarray(stored["sigma"], dtype=np.float64)
                except (KeyError, OSError) as exc:
                    raise ValueError(
                        f"{paths_path}: preflight could not load stored parameter arrays"
                    ) from exc
            _assert_id_path_subset(
                dataset_root,
                paths=paths,
                raw_paths=raw_paths,
                time_grid=split_time_grid,
                mu=mu,
                sigma=sigma,
                config=config,
                identity=identity,
            )
        prepared_splits.append(
            {
                "split": split,
                "split_metadata": split_metadata,
                "paths_path": paths_path,
                "paths": paths,
                "raw_paths": raw_paths,
                "time_grid": split_time_grid,
                "mu": mu,
                "sigma": sigma,
            }
        )

    # Phase 2: compute every requested signature before writing either split.
    for prepared in prepared_splits:
        signature_artifacts = []
        for aug_type in config["sig_aug_types"]:
            augmented = augment_path(
                prepared["paths"],
                prepared["time_grid"],
                aug_type=aug_type,
            )
            raw_augmented = augment_path(
                prepared["raw_paths"],
                prepared["time_grid"],
                aug_type=aug_type,
            )
            sigs = compute_logsignatures(augmented, depth=config["sig_depth"])
            raw_sigs = compute_logsignatures(raw_augmented, depth=config["sig_depth"])
            if prepared["split"] == "test/in-distribution" and targeted_id:
                _assert_id_signature_subset(
                    dataset_root,
                    sigs=sigs,
                    raw_sigs=raw_sigs,
                    sig_channels=augmented.shape[-1],
                    aug_type=aug_type,
                    config=config,
                    identity=identity,
                )
            sig_path = build_split_sigs_path(
                dataset_root,
                prepared["split"],
                aug_type,
                config["sig_type"],
            )
            signature_artifacts.append(
                {
                    "path": sig_path,
                    "sigs": sigs,
                    "raw_sigs": raw_sigs,
                    "sig_channels": augmented.shape[-1],
                    "aug_type": aug_type,
                }
            )
        prepared["signature_artifacts"] = signature_artifacts

    # Phase 3: replace individual artifacts atomically, preserving optional extras.
    for prepared in prepared_splits:
        if not sigs_only:
            _save_timeseries(
                prepared["paths_path"],
                paths=prepared["paths"],
                raw_paths=prepared["raw_paths"],
                time_grid=prepared["time_grid"],
                mu=prepared["mu"],
                sigma=prepared["sigma"],
                config=config,
                split_metadata=prepared["split_metadata"],
            )
        for signature in prepared["signature_artifacts"]:
            _save_sigs(
                signature["path"],
                sigs=signature["sigs"],
                raw_sigs=signature["raw_sigs"],
                sig_channels=signature["sig_channels"],
                config=config,
                split_metadata=prepared["split_metadata"],
                aug_type=signature["aug_type"],
            )

    print(f"Dataset root      -> {dataset_root}")
    return dataset_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a multivariate FX dataset from NPY data.")
    parser.add_argument("config", type=str, help="Path to the required YAML config")
    parser.add_argument("--output-root", "--output-dir", dest="output_root", type=str,
                        help="Dataset root (default: _data); --output-dir is a legacy alias")
    parser.add_argument(
        "--sigs-only",
        action="store_true",
        help="Load identity-matched paths and recompute signatures only.",
    )
    parser.add_argument("--aug-type", choices=["TA", "TLL3"])
    parser.add_argument("--sig-depth", type=int)
    parser.add_argument("--sig-type", choices=["logsig"])
    parser.add_argument("--split", choices=["test/in-distribution"])
    args = parser.parse_args()
    build_fx_dataset(
        args.config,
        sigs_only=args.sigs_only,
        aug_type_override=args.aug_type,
        sig_depth_override=args.sig_depth,
        sig_type_override=args.sig_type,
        split_override=args.split,
        output_root_override=args.output_root,
    )
