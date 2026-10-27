"""Orchestrator: generate paths → compute log-signatures → save .npz.

Supports GBM, fractional Brownian motion (fBM), and Ornstein-Uhlenbeck (OU) via pluggable generators.

Directory structure produced
----------------------------
  <output_dir>/<process_stem>/train/paths.npz
  <output_dir>/<process_stem>/train/TA_logsigs.npz
  <output_dir>/<process_stem>/train/TLL3_logsigs.npz
  <output_dir>/<process_stem>/val/paths.npz
  <output_dir>/<process_stem>/val/TA_logsigs.npz
  <output_dir>/<process_stem>/val/TLL3_logsigs.npz
  <output_dir>/<process_stem>/test/<tag>/paths.npz
  <output_dir>/<process_stem>/test/<tag>/TA_logsigs.npz
  <output_dir>/<process_stem>/test/<tag>/TLL3_logsigs.npz

CLI flags
---------
  python dataset/build_dataset.py config.yaml            # full build
  python dataset/build_dataset.py config.yaml --sigs-only  # reuse existing timeseries
"""

import argparse
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import yaml

import numpy as np

from probsiginv.repo import resolve_repo_path
from probsiginv.utils.contract import build_split_dir, build_split_paths_path, build_split_sigs_path
from probsiginv.utils.signatures import augment_path, compute_logsignatures


DEFAULT_SIG_AUG_TYPES = ("TA", "TLL3")
SUPPORTED_SIG_AUG_TYPES = ("TA", "TLL3", "TLL")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _fmt_param(val) -> str:
    """Format a fixed float or (low, high) tuple for use in a filename."""
    if isinstance(val, tuple):
        return f"{val[0]:g}-{val[1]:g}"
    return f"{val:g}"


def _sample_param(spec, size, rng: np.random.Generator) -> np.ndarray:
    """Convert fixed float or (low, high) tuple to an array with ``size``."""
    if isinstance(spec, tuple):
        return rng.uniform(spec[0], spec[1], size=size)
    return np.full(size, spec)


# ---------------------------------------------------------------------------
# Config hierarchy
# ---------------------------------------------------------------------------

@dataclass
class PathConfig:
    """Base config — process-agnostic params only."""
    process: str = ""
    n_steps: int = 1000
    T_horizon: float = 1.0
    S0: float = 1.0
    n_paths: int = 10000
    path_dim: int = 1
    sig_depth: int = 8
    sig_type: str = "logsig"
    aug_type: str = "TA"      # "TA" | "TLL3" | "TLL"
    seed: int = 42

    def __post_init__(self) -> None:
        self.path_dim = int(self.path_dim)
        if self.path_dim < 1:
            raise ValueError(f"path_dim must be positive, got {self.path_dim}")


def _append_dim_suffix(stem: str, path_dim: int) -> str:
    return stem if int(path_dim) == 1 else f"{stem}_d{int(path_dim)}"


@dataclass
class GBMConfig(PathConfig):
    mu: Union[float, tuple] = 0.05
    sigma: Union[float, tuple] = 0.2
    gbm_rho: Union[float, list, None] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.gbm_rho is not None:
            if self.path_dim == 1:
                warnings.warn(
                    "gbm_rho is set but path_dim=1; gbm_rho will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )
            elif self.path_dim > 2:
                raise ValueError(
                    f"gbm_rho requires path_dim=2, got path_dim={self.path_dim}. "
                    "Only path_dim=2 supports correlated GBM via Cholesky."
                )
            if isinstance(self.gbm_rho, list):
                if len(self.gbm_rho) != 2:
                    raise ValueError(
                        f"gbm_rho list must have exactly 2 elements [lo, hi], got {self.gbm_rho}"
                    )
                lo, hi = float(self.gbm_rho[0]), float(self.gbm_rho[1])
                if not (-1.0 <= lo <= hi <= 1.0):
                    raise ValueError(
                        f"gbm_rho [lo, hi] must satisfy -1 <= lo <= hi <= 1, got [{lo}, {hi}]"
                    )
            else:
                rho_val = float(self.gbm_rho)
                if not (-1.0 <= rho_val <= 1.0):
                    raise ValueError(
                        f"gbm_rho must be in [-1, 1], got {self.gbm_rho}"
                    )

    @property
    def process_stem(self) -> str:
        stem = f"gbm_mu{_fmt_param(self.mu)}_sig{_fmt_param(self.sigma)}"
        if self.gbm_rho is not None:
            rho_value = tuple(self.gbm_rho) if isinstance(self.gbm_rho, list) else self.gbm_rho
            stem += f"_rho{_fmt_param(rho_value)}"
        if self.n_steps != 1000:
            stem += f"_n{self.n_steps}"
        return _append_dim_suffix(stem, self.path_dim)

    @property
    def sig_tag(self) -> str:
        return f"{self.aug_type}_{self.sig_type}"


@dataclass
class FBMConfig(PathConfig):
    mu: Union[float, tuple] = 0.05
    sigma: Union[float, tuple] = 0.2
    hurst: Union[float, tuple] = 0.7
    method: str = "daviesharte"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.path_dim > 1:
            raise ValueError(
                f"{self.__class__.__name__} does not support path_dim>1 (got path_dim={self.path_dim}). "
                "Only GBM supports multi-dimensional paths."
            )

    @property
    def process_stem(self) -> str:
        stem = f"fbm_mu{_fmt_param(self.mu)}_sig{_fmt_param(self.sigma)}_H{_fmt_param(self.hurst)}"
        if self.n_steps != 1000:
            stem += f"_n{self.n_steps}"
        return _append_dim_suffix(stem, self.path_dim)

    @property
    def sig_tag(self) -> str:
        return f"{self.aug_type}_{self.sig_type}"


@dataclass
class OUConfig(PathConfig):
    theta: Union[float, tuple] = 1.0
    mu: Union[float, tuple] = 0.0
    sigma: Union[float, tuple] = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.path_dim > 1:
            raise ValueError(
                f"{self.__class__.__name__} does not support path_dim>1 (got path_dim={self.path_dim}). "
                "Only GBM supports multi-dimensional paths."
            )

    @property
    def process_stem(self) -> str:
        stem = f"ou_theta{_fmt_param(self.theta)}_mu{_fmt_param(self.mu)}_sig{_fmt_param(self.sigma)}"
        if self.n_steps != 1000:
            stem += f"_n{self.n_steps}"
        return _append_dim_suffix(stem, self.path_dim)

    @property
    def sig_tag(self) -> str:
        return f"{self.aug_type}_{self.sig_type}"


# ---------------------------------------------------------------------------
# Generator ABC + implementations
# ---------------------------------------------------------------------------

class PathGenerator(ABC):
    @staticmethod
    @abstractmethod
    def generate(config: PathConfig) -> dict:
        """Returns dict with 'paths', 'time_grid', 'metadata' (dict of arrays)."""


class GBMGenerator(PathGenerator):
    @staticmethod
    def generate(config: GBMConfig) -> dict:
        dt = config.T_horizon / config.n_steps
        rng = np.random.default_rng(config.seed)

        param_size = (
            (config.n_paths, config.path_dim)
            if config.path_dim > 1
            else config.n_paths
        )
        mu_arr = _sample_param(config.mu, param_size, rng)
        sigma_arr = _sample_param(config.sigma, param_size, rng)

        # Resolve gbm_rho and apply Cholesky if applicable
        realized_rho = 0.0
        if config.gbm_rho is not None and config.path_dim == 2 and isinstance(config.gbm_rho, list):
            # Range: draw an independent rho for every path.
            lo, hi = config.gbm_rho[0], config.gbm_rho[1]
            realized_rho = rng.uniform(lo, hi, size=config.n_paths)
            Z = rng.normal(0.0, np.sqrt(dt), size=(config.n_paths, config.n_steps, 2))
            rho_col = realized_rho[:, None]
            dW = np.stack(
                [Z[..., 0], rho_col * Z[..., 0] + np.sqrt(1.0 - rho_col**2) * Z[..., 1]],
                axis=-1,
            )
        elif config.gbm_rho is not None and config.path_dim == 2:
            realized_rho = float(config.gbm_rho)
            corr = np.array([[1.0, realized_rho], [realized_rho, 1.0]])
            L = np.linalg.cholesky(corr)
            Z = rng.normal(0.0, np.sqrt(dt), size=(config.n_paths, config.n_steps, 2))
            dW = Z @ L.T
        else:
            dW = rng.normal(0.0, np.sqrt(dt), size=(config.n_paths, config.n_steps, config.path_dim))

        mu = mu_arr[:, None, :] if config.path_dim > 1 else mu_arr[:, None, None]
        sigma = sigma_arr[:, None, :] if config.path_dim > 1 else sigma_arr[:, None, None]
        log_increments = ((mu - 0.5 * sigma**2) * dt) + sigma * dW

        paths = np.concatenate(
            [
                np.zeros((config.n_paths, 1, config.path_dim)),
                np.cumsum(log_increments, axis=1),
            ],
            axis=1,
        )
        if config.path_dim == 1:
            paths = paths[:, :, 0]

        time_grid = np.linspace(0.0, 1.0, config.n_steps + 1)

        mu_out = config.mu if not isinstance(config.mu, tuple) else mu_arr
        sigma_out = config.sigma if not isinstance(config.sigma, tuple) else sigma_arr

        return {
            "paths": paths,
            "time_grid": time_grid,
            "metadata": {"mu": mu_out, "sigma": sigma_out, "gbm_rho": realized_rho},
        }


class FBMGenerator(PathGenerator):
    @staticmethod
    def generate(config: FBMConfig) -> dict:
        from fbm import FBM

        rng = np.random.default_rng(config.seed)

        mu_arr = _sample_param(config.mu, config.n_paths, rng)
        sigma_arr = _sample_param(config.sigma, config.n_paths, rng)
        hurst_arr = _sample_param(config.hurst, config.n_paths, rng)

        time_grid = np.linspace(0.0, 1.0, config.n_steps + 1)
        paths = np.empty((config.n_paths, config.n_steps + 1, config.path_dim))
        paths[:, 0] = 0.0

        # Set legacy global seed for fbm library reproducibility
        np.random.seed(config.seed)

        # Group by unique hurst to maximise FBM covariance caching
        unique_hursts = np.unique(hurst_arr)
        for h in unique_hursts:
            mask = hurst_arr == h
            f = FBM(n=config.n_steps, hurst=float(h),
                    length=config.T_horizon, method=config.method)
            for idx in np.where(mask)[0]:
                drift = (mu_arr[idx] - 0.5 * sigma_arr[idx] ** 2) * (time_grid * config.T_horizon)
                for dim_idx in range(config.path_dim):
                    fbm_path = f.fbm()  # (n_steps+1,) starting at 0
                    paths[idx, :, dim_idx] = drift + sigma_arr[idx] * fbm_path
        if config.path_dim == 1:
            paths = paths[:, :, 0]

        mu_out = config.mu if not isinstance(config.mu, tuple) else mu_arr
        sigma_out = config.sigma if not isinstance(config.sigma, tuple) else sigma_arr
        hurst_out = config.hurst if not isinstance(config.hurst, tuple) else hurst_arr

        return {
            "paths": paths,
            "time_grid": time_grid,
            "metadata": {"mu": mu_out, "sigma": sigma_out, "hurst": hurst_out},
        }


class OUGenerator(PathGenerator):
    @staticmethod
    def generate(config: OUConfig) -> dict:
        dt = config.T_horizon / config.n_steps
        rng = np.random.default_rng(config.seed)

        theta_arr = _sample_param(config.theta, config.n_paths, rng)
        mu_arr = _sample_param(config.mu, config.n_paths, rng)
        sigma_arr = _sample_param(config.sigma, config.n_paths, rng)

        dW = rng.normal(0.0, np.sqrt(dt), size=(config.n_paths, config.n_steps, config.path_dim))

        paths = np.empty((config.n_paths, config.n_steps + 1, config.path_dim))
        paths[:, 0] = 0.0
        for t in range(config.n_steps):
            paths[:, t + 1, :] = (
                paths[:, t, :]
                + theta_arr[:, None] * (mu_arr[:, None] - paths[:, t, :]) * dt
                + sigma_arr[:, None] * dW[:, t, :]
            )
        if config.path_dim == 1:
            paths = paths[:, :, 0]

        time_grid = np.linspace(0.0, 1.0, config.n_steps + 1)

        theta_out = config.theta if not isinstance(config.theta, tuple) else theta_arr
        mu_out = config.mu if not isinstance(config.mu, tuple) else mu_arr
        sigma_out = config.sigma if not isinstance(config.sigma, tuple) else sigma_arr

        return {
            "paths": paths,
            "time_grid": time_grid,
            "metadata": {"theta": theta_out, "mu": mu_out, "sigma": sigma_out},
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS = {
    "gbm": (GBMConfig, GBMGenerator),
    "fbm": (FBMConfig, FBMGenerator),
    "ou": (OUConfig, OUGenerator),
}


# ---------------------------------------------------------------------------
# BuildDataset
# ---------------------------------------------------------------------------

class BuildDataset:
    def __init__(
        self,
        train_config: PathConfig,
        val_config: PathConfig | None = None,
        test_configs: dict[str, PathConfig] | None = None,
        output_dir: str | None = None,
        generator: type[PathGenerator] | None = None,
        sig_aug_types: tuple[str, ...] = DEFAULT_SIG_AUG_TYPES,
    ):
        self.train_config = train_config
        self.val_config = val_config
        self.test_configs = dict(test_configs or {})
        self.output_dir = resolve_repo_path(output_dir)
        self.generator = generator or GBMGenerator
        self.sig_aug_types = tuple(sig_aug_types)

    def _compute_sigs(self, paths: np.ndarray, time_grid: np.ndarray, *, aug_type: str, config: PathConfig):
        if str(config.sig_type) != "logsig":
            raise ValueError(f"Only sig_type=\"logsig\" is supported, got {config.sig_type!r}")
        augmented = augment_path(paths, time_grid, aug_type=aug_type)
        sig_channels = augmented.shape[-1]
        return compute_logsignatures(augmented, depth=config.sig_depth), sig_channels

    def _save_timeseries(
        self,
        out_path: Path,
        paths: np.ndarray,
        time_grid: np.ndarray,
        config: PathConfig,
        metadata: dict,
        split: str,
    ) -> None:
        save_kwargs = dict(
            paths=paths,
            time_grid=time_grid,
            process=np.array(getattr(config, "process", "")),
            S0=np.float64(config.S0),
            T_horizon=np.float64(config.T_horizon),
            n_steps=np.int64(config.n_steps),
            path_dim=np.int64(config.path_dim),
            seed=np.int64(config.seed),
            split=np.array(split),
        )
        for key, val in metadata.items():
            save_kwargs[key] = np.asarray(val)
        np.savez(out_path, **save_kwargs)
        print(f"Saved timeseries to {out_path}")

    def _save_sigs(
        self,
        out_path: Path,
        sigs: np.ndarray,
        sig_channels: int,
        *,
        aug_type: str,
        sig_type: str,
        sig_depth: int,
        split: str,
    ) -> None:
        np.savez(
            out_path,
            sigs=sigs,
            sig_depth=np.int64(sig_depth),
            sig_channels=np.int64(sig_channels),
            sig_type=np.array(sig_type),
            aug_type=np.array(aug_type),
            split=np.array(split),
        )
        print(f"Saved sigs      to {out_path}")

    def _build_one(
        self,
        config: PathConfig,
        split: str,
        dataset_root: Path,
        sigs_only: bool = False,
    ) -> None:
        split_dir = build_split_dir(dataset_root, split)
        split_dir.mkdir(parents=True, exist_ok=True)
        ts_path = build_split_paths_path(dataset_root, split)

        if sigs_only:
            ts_data = np.load(ts_path, allow_pickle=True)
            paths     = ts_data["paths"]
            time_grid = ts_data["time_grid"]
            print(f"Loaded {paths.shape[0]} paths from {ts_path}")
        else:
            result    = self.generator.generate(config)
            paths     = result["paths"]
            time_grid = result["time_grid"]
            metadata  = result["metadata"]
            print(f"Generated {config.n_paths} paths: {paths.shape}")
            self._save_timeseries(ts_path, paths, time_grid, config, metadata, split)

        for aug_type in self.sig_aug_types:
            sig_path = build_split_sigs_path(dataset_root, split, aug_type, config.sig_type)
            sigs, sig_channels = self._compute_sigs(paths, time_grid, aug_type=aug_type, config=config)
            print(
                f"Computed {config.sig_type} [{aug_type}]: "
                f"sigs {sigs.shape} (depth={config.sig_depth}, channels={sig_channels})"
            )
            self._save_sigs(
                sig_path,
                sigs,
                sig_channels,
                aug_type=aug_type,
                sig_type=config.sig_type,
                sig_depth=config.sig_depth,
                split=split,
            )

    def build(self, sigs_only: bool = False) -> None:
        """Build and save dataset(s)."""
        stem    = self.train_config.process_stem
        dataset_root = self.output_dir / stem

        dataset_root.mkdir(parents=True, exist_ok=True)

        split_configs: list[tuple[PathConfig, str]] = [(self.train_config, "train")]
        if self.val_config is not None:
            split_configs.append((self.val_config, "val"))
        split_configs.extend(
            (cfg, str(Path("test") / tag))
            for tag, cfg in sorted(self.test_configs.items(), key=lambda item: item[0].casefold())
        )

        for cfg, split in split_configs:
            self._build_one(cfg, split, dataset_root, sigs_only=sigs_only)

        print(f"\nDone.")
        print(f"  dataset    : {dataset_root}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_BASE_KEYS = [
    "process",
    "n_steps",
    "T_horizon",
    "S0",
    "n_paths",
    "path_dim",
    "sig_depth",
    "sig_type",
    "aug_type",
    "seed",
]


def _parse_param(value):
    """YAML/CLI value → scalar float or (float, float) range tuple."""
    if isinstance(value, list):
        if len(value) == 1:
            return float(value[0])
        if len(value) == 2:
            return (float(value[0]), float(value[1]))
        raise ValueError(f"Expected 1 or 2 elements, got: {value}")
    return value


def _parse_gbm_rho(value):
    """YAML/CLI gbm_rho -> scalar float or [lo, hi] range list."""
    if value is None:
        return None
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if len(value) == 1:
            return float(value[0])
        if len(value) == 2:
            return [float(value[0]), float(value[1])]
        raise ValueError(f"gbm_rho expects a scalar or [lo, hi], got: {value}")
    return float(value)


def _resolve_sig_aug_types(cfg: dict) -> tuple[str, ...]:
    values = cfg.get("aug_type", None)
    if values is None:
        values = cfg.get("sig_aug_types", None)
    if values is None:
        return DEFAULT_SIG_AUG_TYPES

    if isinstance(values, str):
        resolved = (str(values).upper(),)
    elif isinstance(values, list) and values:
        resolved = tuple(str(value).upper() for value in values)
    else:
        raise ValueError("aug_type must be a string or a non-empty list")

    invalid = [value for value in resolved if value not in SUPPORTED_SIG_AUG_TYPES]
    if invalid:
        raise ValueError(f"Unsupported aug_type values: {invalid!r}")
    return tuple(dict.fromkeys(resolved))


def _build_config_kwargs(process: str, split_cfg: dict, cfg: dict, base_kwargs: dict) -> dict:
    kwargs = {
        **base_kwargs,
        "n_paths": split_cfg.get("n_paths", base_kwargs["n_paths"]),
        "path_dim": split_cfg.get("path_dim", base_kwargs.get("path_dim", 1)),
        "seed": split_cfg.get("seed", base_kwargs["seed"]),
    }
    if process == "ou":
        kwargs["theta"] = _parse_param(split_cfg.get("theta", cfg["theta"]))
        kwargs["mu"] = _parse_param(split_cfg.get("mu", cfg["mu"]))
        kwargs["sigma"] = _parse_param(split_cfg.get("sigma", cfg["sigma"]))
    else:
        kwargs["mu"] = _parse_param(split_cfg.get("mu", cfg["mu"]))
        kwargs["sigma"] = _parse_param(split_cfg.get("sigma", cfg["sigma"]))
    if process == "gbm":
        kwargs["gbm_rho"] = _parse_gbm_rho(split_cfg.get("gbm_rho", cfg.get("gbm_rho")))
    if process == "fbm":
        kwargs["hurst"] = _parse_param(split_cfg.get("hurst", cfg["hurst"]))
        kwargs["method"] = split_cfg.get("method", cfg.get("method", "daviesharte"))
    return kwargs


def _is_single_split_config(raw_cfg: dict | None) -> bool:
    return bool(raw_cfg) and "n_paths" in raw_cfg


def _normalize_test_split_specs(raw_cfg) -> list[tuple[str | None, dict]]:
    if not raw_cfg:
        return []
    if not isinstance(raw_cfg, list):
        raise ValueError("test must be a list of mappings")
    specs: list[tuple[str | None, dict]] = []
    for item in raw_cfg:
        if not isinstance(item, dict):
            raise ValueError("Each test split entry must be a mapping")
        payload = dict(item)
        tag = payload.pop("tag", None)
        specs.append((str(tag) if tag is not None else None, payload))
    return specs


def main():
    parser = argparse.ArgumentParser(
        description="Build path + signature dataset. CLI args override YAML values.")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    # Overridable args (all optional; None = use YAML value)
    parser.add_argument("--process", type=str, choices=list(GENERATORS))
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--n-paths", type=int)
    parser.add_argument("--path-dim", type=int)
    parser.add_argument("--sig-depth", type=int)
    parser.add_argument("--sig-type", type=str, choices=["logsig"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--T-horizon", type=float)
    parser.add_argument("--S0", type=float)
    parser.add_argument("--mu", type=float, nargs="+")
    parser.add_argument("--sigma", type=float, nargs="+")
    parser.add_argument("--gbm-rho", type=float, nargs="+", help="gbm only; scalar rho or lo hi range")
    parser.add_argument("--theta", type=float, nargs="+", help="ou only")
    parser.add_argument("--hurst", type=float, nargs="+", help="fbm only")
    parser.add_argument("--method", type=str, help="fbm only")
    parser.add_argument("--aug-type", type=str, nargs="+", choices=list(SUPPORTED_SIG_AUG_TYPES), default=None)
    parser.add_argument("--output-root", "--output-dir", dest="output_root", type=str,
                        help="Dataset root (default: _data); --output-dir is a legacy alias")
    parser.add_argument("--val-n-paths", type=int)
    parser.add_argument("--val-seed", type=int)
    parser.add_argument("--monitor-n-paths", type=int)
    parser.add_argument("--monitor-seed", type=int)
    parser.add_argument(
        "--sigs-only", action="store_true",
        help="Skip path generation; load existing timeseries/ files and only compute sigs.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    # CLI overrides YAML
    def _override(key, arg_val):
        if arg_val is not None:
            cfg[key] = arg_val

    _override("process", args.process)
    _override("n_steps", args.n_steps)
    _override("n_paths", args.n_paths)
    _override("path_dim", args.path_dim)
    _override("sig_depth", args.sig_depth)
    _override("sig_type", args.sig_type)

    if str(cfg.get("sig_type", "logsig")) != "logsig":
        raise ValueError(f"Only sig_type=\"logsig\" is supported, got {cfg.get('sig_type')!r}")
    _override("seed", args.seed)
    _override("T_horizon", args.T_horizon)
    _override("S0", args.S0)
    _override("mu", args.mu)           # list or None
    _override("sigma", args.sigma)
    _override("gbm_rho", args.gbm_rho)
    _override("theta", args.theta)
    _override("hurst", args.hurst)
    _override("method", args.method)
    _override("output_root", args.output_root)
    if args.aug_type is not None:
        cfg["aug_type"] = list(args.aug_type)
    if args.val_n_paths is not None:
        cfg.setdefault("val", {})["n_paths"] = args.val_n_paths
    if args.val_seed is not None:
        cfg.setdefault("val", {})["seed"] = args.val_seed
    if args.monitor_n_paths is not None:
        cfg.setdefault("val", {})["n_paths"] = args.monitor_n_paths
    if args.monitor_seed is not None:
        cfg.setdefault("val", {})["seed"] = args.monitor_seed

    process = cfg["process"]
    if process not in GENERATORS:
        raise ValueError(f"Unknown process '{process}'. Choose from: {list(GENERATORS)}")
    config_cls, generator_cls = GENERATORS[process]

    resolved_sig_aug_types = _resolve_sig_aug_types(cfg)
    base_kwargs = {k: cfg[k] for k in _BASE_KEYS if k in cfg}
    base_kwargs["aug_type"] = resolved_sig_aug_types[0]

    train_config = config_cls(**_build_config_kwargs(process, cfg, cfg, base_kwargs))

    raw_val_cfg = cfg.get("val")
    legacy_monitor_cfg = cfg.get("monitor")
    raw_test_cfgs = cfg.get("test")

    canonical_val_cfg: dict | None = None
    if raw_val_cfg is not None:
        canonical_val_cfg = dict(raw_val_cfg)
    elif legacy_monitor_cfg is not None:
        canonical_val_cfg = dict(legacy_monitor_cfg)

    val_config = None
    if canonical_val_cfg is not None:
        val_config = config_cls(**_build_config_kwargs(process, canonical_val_cfg, cfg, base_kwargs))

    test_configs: dict[str, PathConfig] = {}
    for explicit_tag, split_cfg in _normalize_test_split_specs(raw_test_cfgs):
        test_config = config_cls(**_build_config_kwargs(process, split_cfg, cfg, base_kwargs))
        tag = explicit_tag or test_config.process_stem
        if tag in test_configs:
            raise ValueError(f"Duplicate test dataset tag: {tag}")
        test_configs[tag] = test_config

    output_dir = cfg.get("output_root", cfg.get("output_dir", "_data"))
    builder = BuildDataset(
        train_config,
        val_config=val_config,
        test_configs=test_configs,
        output_dir=output_dir,
        generator=generator_cls,
        sig_aug_types=resolved_sig_aug_types,
    )
    builder.build(sigs_only=args.sigs_only)
