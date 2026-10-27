"""Download S&P 500 daily data via yfinance, build sliding-window dataset, compute log-signatures.

Produces the same directory layout as build_dataset.py:
    <output_dir>/<process_stem>/train/paths.npz
    <output_dir>/<process_stem>/train/TA_logsigs.npz
    <output_dir>/<process_stem>/train/TLL3_logsigs.npz
    <output_dir>/<process_stem>/test/<tag>/paths.npz
    <output_dir>/<process_stem>/test/<tag>/TA_logsigs.npz
    <output_dir>/<process_stem>/test/<tag>/TLL3_logsigs.npz

Usage
-----
    python dataset/sp500_data.py configs/dataset/sp500_daily.yaml
    python dataset/sp500_data.py configs/dataset/sp500_daily.yaml --sigs-only
"""

import argparse
import sys
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probsiginv.repo import resolve_repo_path
from probsiginv.utils.contract import build_split_paths_path, build_split_sigs_path
from probsiginv.utils.signatures import augment_path, compute_logsignatures


DEFAULT_SIG_AUG_TYPES = ("TA", "TLL3")
SUPPORTED_SIG_AUG_TYPES = ("TA", "TLL3", "TLL")


# ---------------------------------------------------------------------------
# Download + cache
# ---------------------------------------------------------------------------

def download_sp500(
    ticker: str = "^GSPC",
    start: str = "2000-01-01",
    end: str = "2025-12-31",
    cache_dir: str = "_data/sp500",
) -> pd.DataFrame:
    """Download daily adjusted close via yfinance and cache locally.

    Returns
    -------
    df : DataFrame with DatetimeIndex and 'Close' column.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    safe_ticker = ticker.replace("^", "").replace("/", "_")
    csv_file = cache_path / f"{safe_ticker}_{start}_{end}.csv"

    if csv_file.exists():
        print(f"Loading cached data from {csv_file}")
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    else:
        import yfinance as yf
        print(f"Downloading {ticker} from {start} to {end} ...")
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            raise RuntimeError(f"No data returned for {ticker} ({start} → {end})")
        # Keep only Close
        if isinstance(df.columns, pd.MultiIndex):
            df = df["Close"]
            if isinstance(df, pd.DataFrame):
                df = df.iloc[:, 0].to_frame("Close")
            else:
                df = df.to_frame("Close")
        elif "Close" in df.columns:
            df = df[["Close"]]
        else:
            df = df.iloc[:, :1]
            df.columns = ["Close"]
        df.to_csv(csv_file)
        print(f"Cached to {csv_file}  ({len(df)} rows)")

    return df


# ---------------------------------------------------------------------------
# Sliding-window construction
# ---------------------------------------------------------------------------

def build_windows(prices: np.ndarray, n_steps: int, stride: int) -> np.ndarray:
    """Create sliding windows of log-normalized paths.

    Parameters
    ----------
    prices  : 1-D array of raw prices (daily close)
    n_steps : window length (number of steps, so each window has n_steps+1 points)
    stride  : step between window starts

    Returns
    -------
    windows : (P, n_steps+1)  log(S_t/S_0) paths starting at 0
    """
    log_prices = np.log(prices)
    window_len = n_steps + 1

    windows = []
    for i in range(0, len(log_prices) - window_len + 1, stride):
        w = log_prices[i : i + window_len]
        windows.append(w - w[0])

    return np.stack(windows)


def estimate_params(paths: np.ndarray, T_horizon: float) -> tuple:
    """Estimate annualized mu and sigma for each window.

    Parameters
    ----------
    paths     : (P, n_steps+1)  log-price paths (starting at 0)
    T_horizon : time horizon of each window

    Returns
    -------
    mu_arr    : (P,) annualized drift
    sigma_arr : (P,) annualized volatility
    """
    n_steps = paths.shape[1] - 1
    dt = T_horizon / n_steps
    log_returns = np.diff(paths, axis=1)  # (P, n_steps)

    sigma_arr = np.std(log_returns, axis=1, ddof=1) / np.sqrt(dt)
    mu_arr = np.mean(log_returns, axis=1) / dt + 0.5 * sigma_arr ** 2

    return mu_arr, sigma_arr


# ---------------------------------------------------------------------------
# Save helpers (matching build_dataset.py format)
# ---------------------------------------------------------------------------

def _save_timeseries(
    out_path: Path,
    paths: np.ndarray,
    raw_paths: np.ndarray,
    time_grid: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    config: dict,
    split: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        paths=paths,
        raw_paths=raw_paths,
        time_grid=time_grid,
        process=np.array(str(config.get("process", "finance"))),
        mu=mu,
        sigma=sigma,
        S0=np.float64(config.get("S0", 1.0)),
        T_horizon=np.float64(config.get("T_horizon", 1.0)),
        n_steps=np.int64(config["n_steps"]),
        path_scale=np.float64(config.get("path_scale", 1.0)),
        seed=np.int64(config.get("seed", 0)),
        split=np.array(split),
        ticker=np.array(config.get("ticker", "^GSPC")),
    )
    print(f"Saved timeseries → {out_path}  ({paths.shape[0]} windows)")


def _save_sigs(
    out_path: Path,
    sigs: np.ndarray,
    raw_sigs: np.ndarray,
    sig_channels: int,
    config: dict,
    split: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        sigs=sigs,
        raw_sigs=raw_sigs,
        sig_depth=np.int64(config["sig_depth"]),
        sig_channels=np.int64(sig_channels),
        sig_type=np.array(config.get("sig_type", "logsig")),
        aug_type=np.array(config.get("aug_type", "TA")),
        path_scale=np.float64(config.get("path_scale", 1.0)),
        split=np.array(split),
    )
    print(f"Saved sigs       → {out_path}  {sigs.shape}")


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def _month_tag(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y%m")


def _process_stem(cfg: dict, *, train_start: str, train_end: str) -> str:
    n = cfg["n_steps"]
    s = cfg.get("window_stride", 5)
    stem = f"sp500_{_month_tag(train_start)}-{_month_tag(train_end)}_n{n}_s{s}"
    path_scale = float(cfg.get("path_scale", 1.0))
    if not np.isclose(path_scale, 1.0):
        stem += f"_ps{path_scale:g}"
    return stem


def _resolve_date(cfg: dict, key: str, legacy_key: str, default: str | None = None) -> str | None:
    value = cfg.get(key)
    if value is not None:
        return str(value)
    legacy_value = cfg.get(legacy_key)
    if legacy_value is not None:
        return str(legacy_value)
    return default


def _normalize_test_split_specs(
    raw_test_cfg,
    *,
    legacy_val_cfg: dict | None,
) -> list[tuple[str | None, dict]]:
    if raw_test_cfg is None:
        if legacy_val_cfg is None:
            return []
        payload = dict(legacy_val_cfg)
        tag = payload.pop("tag", "default")
        return [(str(tag), payload)]

    if not isinstance(raw_test_cfg, list):
        raise ValueError("test must be a list of mappings")

    specs: list[tuple[str | None, dict]] = []
    for item in raw_test_cfg:
        if not isinstance(item, dict):
            raise ValueError("Each test split entry must be a mapping")
        payload = dict(item)
        tag = payload.pop("tag", None)
        specs.append((str(tag) if tag is not None else None, payload))
    return specs


def _default_test_tag(split_cfg: dict, *, default_stride: int) -> str:
    start = _resolve_date(split_cfg, "start", "start_date")
    end = _resolve_date(split_cfg, "end", "end_date")
    if start is None or end is None:
        raise ValueError("S&P500 test split entries require start/end (or start_date/end_date)")
    stride = int(split_cfg.get("window_stride", default_stride))
    safe_start = start.replace("-", "")
    safe_end = end.replace("-", "")
    return f"{safe_start}_{safe_end}_s{stride}"


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


def build_sp500_dataset(
    config_path: str,
    sigs_only: bool = False,
    aug_type_override: str | None = None,
    sig_depth_override: int | None = None,
    sig_type_override: str | None = None,
    output_root_override: str | None = None,
) -> None:
    """Build S&P 500 sliding-window dataset from a YAML config."""
    cfg = yaml.safe_load(Path(config_path).read_text())

    ticker = cfg.get("ticker", "^GSPC")
    n_steps = cfg["n_steps"]
    stride = cfg.get("window_stride", 5)
    T_horizon = cfg.get("T_horizon", 1.0)
    path_scale = float(cfg.get("path_scale", 1.0))
    sig_depth = sig_depth_override or cfg.get("sig_depth", 8)
    sig_type = str(sig_type_override or cfg.get("sig_type", "logsig"))
    if sig_type != "logsig":
        raise ValueError(f"Only sig_type=\"logsig\" is supported, got {sig_type!r}")
    output_dir = resolve_repo_path(output_root_override or cfg.get("output_root", cfg.get("output_dir", "_data")))
    cache_dir = cfg.get("cache_dir", "_data/sp500")

    train_start = _resolve_date(cfg, "start", "start_date", "2000-01-01")
    train_end = _resolve_date(cfg, "end", "end_date", "2022-12-31")
    stem = _process_stem(cfg, train_start=train_start, train_end=train_end)
    dataset_root = output_dir / stem
    dataset_root.mkdir(parents=True, exist_ok=True)
    sig_aug_types = (str(aug_type_override).upper(),) if aug_type_override is not None else _resolve_sig_aug_types(cfg)
    raw_test_cfgs = cfg.get("test")
    legacy_val_cfg = cfg.get("val")
    test_specs = _normalize_test_split_specs(raw_test_cfgs, legacy_val_cfg=legacy_val_cfg)

    download_starts = [pd.Timestamp(train_start)]
    download_ends = [pd.Timestamp(train_end)]
    for _tag, split_cfg in test_specs:
        split_start = _resolve_date(
            split_cfg,
            "start",
            "start_date",
            train_end if raw_test_cfgs is None else None,
        )
        split_end = _resolve_date(
            split_cfg,
            "end",
            "end_date",
            train_end if raw_test_cfgs is None else None,
        )
        if split_start is None or split_end is None:
            raise ValueError("S&P500 test split entries require start/end (or start_date/end_date)")
        download_starts.append(pd.Timestamp(split_start))
        download_ends.append(pd.Timestamp(split_end))

    # --- Download full price series (once, covering train + all tests) ---
    all_start = min(download_starts).date().isoformat()
    all_end = max(download_ends).date().isoformat()

    df = download_sp500(ticker=ticker, start=all_start, end=all_end, cache_dir=cache_dir)
    prices_full = df["Close"].values.astype(np.float64)
    dates_full = df.index

    # --- Define splits ---
    splits = []

    # Train split
    train_mask = (dates_full >= pd.Timestamp(train_start)) & (dates_full <= pd.Timestamp(train_end))
    splits.append(("train", prices_full[train_mask], cfg))

    # Held-out test splits
    seen_test_tags: set[str] = set()
    for idx, (explicit_tag, split_cfg) in enumerate(test_specs):
        split_start = _resolve_date(
            split_cfg,
            "start",
            "start_date",
            train_end if raw_test_cfgs is None else None,
        )
        split_end = _resolve_date(
            split_cfg,
            "end",
            "end_date",
            all_end if raw_test_cfgs is None else None,
        )
        if split_start is None or split_end is None:
            raise ValueError("S&P500 test split entries require start/end (or start_date/end_date)")
        test_stride = int(split_cfg.get("window_stride", stride))
        test_seed = int(split_cfg.get("seed", cfg.get("seed", 0) + idx + 1))
        test_tag = explicit_tag or _default_test_tag(split_cfg, default_stride=stride)
        if test_tag in seen_test_tags:
            raise ValueError(f"Duplicate test dataset tag: {test_tag}")
        seen_test_tags.add(test_tag)
        test_mask = (dates_full >= pd.Timestamp(split_start)) & (dates_full <= pd.Timestamp(split_end))
        test_config = {
            **cfg,
            **split_cfg,
            "start": split_start,
            "end": split_end,
            "window_stride": test_stride,
            "seed": test_seed,
        }
        splits.append((str(Path("test") / test_tag), prices_full[test_mask], test_config))

    for split_name, prices, split_cfg in splits:
        print(f"\n--- {split_name} split: {len(prices)} trading days ---")

        if len(prices) < n_steps + 1:
            print(f"  WARNING: only {len(prices)} days, need {n_steps + 1}. Skipping.")
            continue

        ts_path = build_split_paths_path(dataset_root, split_name)
        s = split_cfg.get("window_stride", stride)

        if sigs_only:
            ts_data = np.load(ts_path, allow_pickle=True)
            paths = ts_data["paths"]
            stored_path_scale = float(np.asarray(ts_data.get("path_scale", 1.0)).item())
            if not np.isclose(stored_path_scale, path_scale):
                raise ValueError(
                    f"{ts_path}: stored path_scale={stored_path_scale:g} does not match requested path_scale={path_scale:g}"
                )
            if "raw_paths" in ts_data:
                raw_paths = np.asarray(ts_data["raw_paths"], dtype=np.float64)
            elif np.isclose(path_scale, 1.0):
                raw_paths = np.asarray(paths, dtype=np.float64)
            else:
                raw_paths = np.asarray(paths, dtype=np.float64) / path_scale
            print(f"  Loaded {paths.shape[0]} windows from {ts_path}")
        else:
            raw_paths = build_windows(prices, n_steps, s)
            paths = raw_paths * path_scale
            mu_arr, sigma_arr = estimate_params(raw_paths, T_horizon)
            time_grid = np.linspace(0.0, 1.0, n_steps + 1)
            _save_timeseries(ts_path, paths, raw_paths, time_grid, mu_arr, sigma_arr, split_cfg, split_name)
            print(f"  Windows: {paths.shape}, stride={s}")

        # Signatures
        time_grid = np.linspace(0.0, 1.0, n_steps + 1)
        for aug_type in sig_aug_types:
            sig_path = build_split_sigs_path(dataset_root, split_name, aug_type, sig_type)
            augmented = augment_path(paths, time_grid, aug_type=aug_type)
            raw_augmented = augment_path(raw_paths, time_grid, aug_type=aug_type)
            sig_channels = augmented.shape[-1]
            sigs = compute_logsignatures(augmented, depth=sig_depth)
            raw_sigs = compute_logsignatures(raw_augmented, depth=sig_depth)
            print(f"  Computed {sig_type} [{aug_type}]: {sigs.shape} (depth={sig_depth}, channels={sig_channels})")
            _save_sigs(sig_path, sigs, raw_sigs, sig_channels, {**split_cfg, "aug_type": aug_type}, split_name)

    print(f"\nDone.")
    print(f"  dataset    : {dataset_root}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build S&P 500 sliding-window dataset.")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--output-root", "--output-dir", dest="output_root", type=str,
                        help="Dataset root (default: _data); --output-dir is a legacy alias")
    parser.add_argument("--sigs-only", action="store_true",
                        help="Skip windowing; load existing timeseries and recompute sigs only.")
    parser.add_argument("--aug-type", type=str, choices=["TA", "TLL3"],
                        help="Build only one stored signature artifact instead of the default TA/TLL3 pair.")
    parser.add_argument("--sig-depth", type=int, help="Override sig_depth from config.")
    parser.add_argument("--sig-type", type=str, choices=["logsig"],
                        help="Override sig_type from config (logsig only).")
    args = parser.parse_args()
    build_sp500_dataset(args.config, sigs_only=args.sigs_only,
                        aug_type_override=args.aug_type,
                        sig_depth_override=args.sig_depth,
                        sig_type_override=args.sig_type,
                        output_root_override=args.output_root)
