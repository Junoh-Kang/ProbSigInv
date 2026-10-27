from __future__ import annotations

from pathlib import Path

import numpy as np

from probsiginv.utils.contract import OUTPUT_ARTIFACT_NAME


def _unwrap_optional(value):
    if value is None:
        return None
    arr = np.asarray(value)
    return arr.item() if arr.ndim == 0 else arr


def _resolve_path_scale(real_npz) -> float:
    value = real_npz.get("path_scale", None)
    if value is None:
        return 1.0
    arr = np.asarray(value, dtype=float)
    return float(arr.item()) if arr.ndim == 0 else float(arr.reshape(-1)[0])


def _resolve_real_paths(real_npz) -> np.ndarray:
    if "raw_paths" in real_npz:
        return np.asarray(real_npz["raw_paths"], dtype=float)
    return np.asarray(real_npz["paths"], dtype=float)


def _resolve_path_dim(real_npz) -> int:
    if "path_dim" in real_npz:
        return int(np.asarray(real_npz["path_dim"]).item())
    paths = _resolve_real_paths(real_npz)
    return int(paths.shape[-1]) if paths.ndim == 3 else 1


def discover_dataset_dirs(root_dir: str | Path) -> list[Path]:
    root = Path(root_dir)
    if (root / "real.npz").exists():
        return [root]

    data_dirs = sorted(p.parent for p in root.glob("*/real.npz"))
    if not data_dirs:
        data_dirs = sorted(p.parent for p in root.glob("**/real.npz"))
    return list(dict.fromkeys(data_dirs))


def load_dataset_artifacts(data_dir: str | Path) -> dict:
    data_dir = Path(data_dir)
    real_path = data_dir / "real.npz"
    if not real_path.exists():
        raise FileNotFoundError(f"real.npz not found in {data_dir}")

    real_npz = np.load(real_path)
    model_paths = sorted(data_dir.glob(f"**/{OUTPUT_ARTIFACT_NAME}"))

    return {
        "data_dir": data_dir,
        "real_paths": _resolve_real_paths(real_npz),
        "time_grid": real_npz["time_grid"],
        "path_dim": _resolve_path_dim(real_npz),
        "path_scale": _resolve_path_scale(real_npz),
        "process": _unwrap_optional(real_npz.get("process", None)),
        "ticker": _unwrap_optional(real_npz.get("ticker", None)),
        "theta": _unwrap_optional(real_npz.get("theta", None)),
        "mu": _unwrap_optional(real_npz.get("mu", None)),
        "sigma": _unwrap_optional(real_npz.get("sigma", None)),
        "gbm_rho": _unwrap_optional(real_npz.get("gbm_rho", None)),
        "hurst": _unwrap_optional(real_npz.get("hurst", None)),
        "T_horizon": _unwrap_optional(real_npz.get("T_horizon", None)),
        "model_paths": model_paths,
    }


def filter_model_paths(model_paths, include=None, exclude=None):
    result = list(model_paths)
    if include:
        result = [p for p in result if all(s in str(p) for s in include)]
    if exclude:
        result = [p for p in result if not any(s in str(p) for s in exclude)]
    return result
