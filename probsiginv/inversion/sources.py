"""Real-path source loading/generation for inversion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from probsiginv.dataset.dataset import PathSigDataset
from probsiginv.utils.contract import resolve_split_artifact_paths_for_aug
from probsiginv.utils.layout import build_dataset_name_from_dataset_path


_DEFAULT_INSERTION_SIG_DEPTH = 12


def load_from_dataset(
    dataset_path: str,
    *,
    preferred_aug_type: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load real paths, pre-computed signatures, and metadata from a PathSigDataset .npz."""
    ds = PathSigDataset(dataset_path, preferred_aug_type=preferred_aug_type)
    meta = {
        "process": getattr(ds, "process", None),
        "theta": getattr(ds, "theta", None),
        "mu": ds.mu,
        "sigma": ds.sigma,
        "hurst": ds.hurst,
        "sig_depth": int(ds.sig_depth),
        "sig_channels": int(ds.sig_channels),
        "sig_type": ds.sig_type,
        "aug_type": getattr(ds, "aug_type", "TA"),
        "source_kind": "dataset",
        "fourier_sig_depth_default": int(ds.sig_depth),
        "n_steps": int(ds.n_steps),
        "T_horizon": float(ds.T_horizon),
        "path_dim": int(getattr(ds, "path_dim", 1)),
    }
    return ds.paths, ds.sigs, ds.time_grid, meta


def _load_split_signature_metadata(split_dir: Path) -> dict:
    sig_depths: list[int] = []
    sig_channels: list[int] = []
    for candidate in sorted(split_dir.glob("*_logsigs.npz")):
        with np.load(candidate, allow_pickle=True) as sig_npz:
            if "sig_depth" in sig_npz:
                sig_depths.append(int(np.asarray(sig_npz["sig_depth"]).item()))
            if "sig_channels" in sig_npz:
                sig_channels.append(int(np.asarray(sig_npz["sig_channels"]).item()))
    return {
        "sig_depth": max(sig_depths) if sig_depths else 0,
        "sig_channels": max(sig_channels) if sig_channels else 0,
    }


def load_paths_only(
    dataset_path: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, dict, str]:
    """Load the canonical path artifact without constructing a dataset object."""
    resolved = Path(dataset_path)
    ts_path = resolved if resolved.name == "paths.npz" else resolved / "paths.npz"
    split_dir = ts_path.parent
    with np.load(ts_path, allow_pickle=True) as ts_data:
        paths = np.asarray(ts_data["paths"])
        raw_time_grid = np.asarray(ts_data["time_grid"])
        if paths.ndim not in (2, 3):
            raise ValueError(
                f"{ts_path}: paths must have shape (N,T) or (N,T,D), got {paths.shape}"
            )
        actual_path_dim = 1 if paths.ndim == 2 else int(paths.shape[-1])
        if actual_path_dim < 1:
            raise ValueError(
                f"{ts_path}: paths shape must have a positive path dimension, got {paths.shape}"
            )
        if "path_dim" in ts_data:
            stored_path_dim = np.asarray(ts_data["path_dim"])
            stored_value = (
                stored_path_dim.item() if stored_path_dim.ndim == 0 else None
            )
            if stored_path_dim.ndim != 0 or isinstance(
                stored_value, (bool, np.bool_)
            ) or not isinstance(stored_value, (int, np.integer)):
                raise ValueError(f"{ts_path}: stored path_dim must be an integer scalar")
            stored_path_dim = int(stored_value)
            if stored_path_dim != actual_path_dim:
                raise ValueError(
                    f"{ts_path}: stored path_dim={stored_path_dim} is inconsistent "
                    f"with paths shape {paths.shape} (actual path_dim={actual_path_dim})"
                )
        if np.iscomplexobj(raw_time_grid):
            raise ValueError(f"{ts_path}: time_grid must be real")
        try:
            time_grid = np.asarray(raw_time_grid, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{ts_path}: time_grid must be real") from exc
        if time_grid.ndim != 1:
            raise ValueError(f"{ts_path}: time_grid must be one-dimensional")
        if time_grid.size == 0:
            raise ValueError(f"{ts_path}: time_grid must be non-empty")
        if not np.all(np.isfinite(time_grid)):
            raise ValueError(f"{ts_path}: time_grid must be finite")
        if paths.shape[1] != time_grid.size:
            raise ValueError(
                f"{ts_path}: time_grid length {time_grid.size} does not match "
                f"paths time dimension {paths.shape[1]}"
            )

        if "n_steps" not in ts_data:
            raise ValueError(f"{ts_path}: n_steps must be an integer scalar")
        raw_n_steps = np.asarray(ts_data["n_steps"])
        n_steps_value = raw_n_steps.item() if raw_n_steps.ndim == 0 else None
        if raw_n_steps.ndim != 0 or isinstance(
            n_steps_value, (bool, np.bool_)
        ) or not isinstance(n_steps_value, (int, np.integer)):
            raise ValueError(f"{ts_path}: n_steps must be an integer scalar")
        n_steps = int(n_steps_value)
        expected_n_steps = time_grid.size - 1
        if n_steps != expected_n_steps:
            raise ValueError(
                f"{ts_path}: n_steps={n_steps} does not match "
                f"time_grid.size - 1={expected_n_steps}"
            )

        def _optional_scalar_or_array(key: str):
            if key not in ts_data:
                return None
            raw = ts_data[key]
            return float(raw) if raw.ndim == 0 else raw

        sig_meta = _load_split_signature_metadata(split_dir)
        meta = {
            "process": str(ts_data["process"]) if "process" in ts_data else None,
            "theta": _optional_scalar_or_array("theta"),
            "mu": _optional_scalar_or_array("mu"),
            "sigma": _optional_scalar_or_array("sigma"),
            "hurst": _optional_scalar_or_array("hurst"),
            "sig_depth": int(sig_meta["sig_depth"]),
            "sig_channels": int(sig_meta["sig_channels"]),
            "sig_type": "logsig",
            "aug_type": None,
            "source_kind": "dataset",
            "fourier_sig_depth_default": int(sig_meta["sig_depth"]),
            "n_steps": n_steps,
            "T_horizon": float(ts_data["T_horizon"]),
            "path_dim": actual_path_dim,
        }
    dataset_name = build_dataset_name_from_dataset_path(split_dir)
    return paths, None, time_grid, meta, dataset_name


def resolve_fourier_sig_depth(args: argparse.Namespace, meta: dict) -> int:
    """Resolve the Fourier inversion depth for the current source."""
    if args.fourier_sig_depth is not None:
        return int(args.fourier_sig_depth)
    default_depth = meta.get("fourier_sig_depth_default")
    if default_depth is not None:
        return int(default_depth)
    return int(meta["sig_depth"])


def resolve_legendre_sig_depth(args: argparse.Namespace, meta: dict) -> int:
    """Resolve the Legendre inversion depth for the current source."""
    if args.legendre_sig_depth is not None:
        return int(args.legendre_sig_depth)
    return int(meta["sig_depth"])


def resolve_insertion_sig_depth(args: argparse.Namespace, meta: dict) -> int:
    """Resolve the insertion inversion depth for the current source."""
    del meta
    insertion_sig_depth = getattr(args, "insertion_sig_depth", None)
    if insertion_sig_depth is not None:
        depth = int(insertion_sig_depth)
    else:
        depth = _DEFAULT_INSERTION_SIG_DEPTH
    if depth < 1:
        raise ValueError(
            "Insertion inversion requires a positive signature depth; provide args.insertion_sig_depth."
        )
    return depth


def load_real_path_source(
    args: argparse.Namespace,
    *,
    preferred_source_aug_type: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, str, str]:
    """Load the real-path source from a pre-built dataset."""
    print(f"Loading dataset: {args.dataset}")
    if getattr(args, "method", None) in {
        "fourier",
        "legendre",
        "insertion",
        "ls_posterior",
    }:
        resolved = Path(args.dataset)
        ts_path = resolved if resolved.name == "paths.npz" else resolved / "paths.npz"
        paths, sigs, time_grid, meta, dataset_name = load_paths_only(str(ts_path))
        return paths, sigs, time_grid, meta, dataset_name, str(ts_path)

    split_dir, ts_path, _sig_path = resolve_split_artifact_paths_for_aug(
        args.dataset,
        preferred_aug_type=preferred_source_aug_type,
    )
    paths, sigs, time_grid, meta = load_from_dataset(
        args.dataset,
        preferred_aug_type=preferred_source_aug_type,
    )
    dataset_name = build_dataset_name_from_dataset_path(split_dir)
    return paths, sigs, time_grid, meta, dataset_name, str(ts_path)
