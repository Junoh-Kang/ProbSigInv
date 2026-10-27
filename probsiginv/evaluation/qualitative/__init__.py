"""Qualitative evaluation panel assembly and storage."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from probsiginv.evaluation.candidate_selection import CandidateArtifact, discover_candidates
from probsiginv.evaluation.data import load_dataset_artifacts
from probsiginv.evaluation.qualitative.panels import (
    build_acf_plot_spec as _panel_build_acf_plot_spec,
    build_candidate_baseline_routes as _panel_build_candidate_baseline_routes,
    build_overlay_plot_spec as _panel_build_overlay_plot_spec,
    build_qq_plot_spec as _panel_build_qq_plot_spec,
    resolve_panel_targets as _panel_resolve_targets,
    sample_candidate_indices as _panel_sample_candidate_indices,
    select_path_indices as _panel_select_path_indices,
)
from probsiginv.evaluation.qualitative.render import render_plot_spec_png as _render_plot_spec_png
from probsiginv.evaluation.qualitative.statistics import (
    ACFCurve as _ACFCurve,
    QQCurve as _QQCurve,
    pooled_increment_qq as _pooled_increment_qq,
    process_acf as _process_acf,
    validate_path_arrays as _validate_path_arrays,
)
from probsiginv.evaluation.quantitative import detect_process
from probsiginv.repo import repo_relative_path


SUPPORTED_QUALITATIVE_METRICS = ("qq", "acf", "overlay")
DEFAULT_N_SHOW = 5
DEFAULT_N_GEN = 5
QUAL_CACHE_SCHEMA_VERSION = 1
DEFAULT_BASELINE_ORDER = ("insertion", "fourier", "legendre", "reg")
DEFAULT_OVERLAY_RANDOM_SEED = 42


def _split_keys(data_dir: str | Path) -> tuple[str, str]:
    path = Path(data_dir)
    if path.parent == path:
        raise ValueError(f"Could not derive train_key/test_key from {path}")
    return path.parent.name, path.name


def _qual_root(data_dir: Path) -> Path:
    return data_dir / "results" / "qual"


def _replace_plot_dir(*, final_root: Path, staging_root: Path) -> None:
    if final_root.exists():
        shutil.rmtree(final_root)
    final_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root.rename(final_root)


def _load_candidate_paths(candidate: CandidateArtifact, *, path_scale: float = 1.0) -> np.ndarray:
    with np.load(candidate.artifact_path) as npz:
        paths = np.asarray(npz["paths"], dtype=float)
    if not np.isclose(path_scale, 1.0):
        paths = paths / path_scale
    return paths


def _metric_dir_name(metric: str) -> str:
    return "vis" if metric == "overlay" else metric


def _plot_root(qual_root: Path, metric: str) -> Path:
    return qual_root / _metric_dir_name(metric)


def _plot_staging_root(qual_root: Path, metric: str) -> Path:
    return qual_root / f".{_metric_dir_name(metric)}_tmp"


def _cleanup_obsolete_outputs(root: Path) -> None:
    for name in ("qqplot.json", "acf.json"):
        path = root / name
        if path.exists():
            path.unlink()
    for name in ("panels",):
        path = root / name
        if path.exists():
            shutil.rmtree(path)


def _metadata_from_artifacts(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: artifacts.get(key)
        for key in ("theta", "kappa", "mu", "sigma", "hurst")
        if artifacts.get(key) is not None
    }


def _resolve_overlay_targets(
    data_dir: str | Path,
    candidates: list[CandidateArtifact],
) -> tuple[list[CandidateArtifact], list[CandidateArtifact]]:
    return _panel_resolve_targets(candidates, fallback_candidates=discover_candidates(data_dir))


def _first_sample_paths(paths: np.ndarray) -> np.ndarray:
    array = np.asarray(paths, dtype=float)
    if array.ndim == 4:
        if array.shape[1] < 1:
            raise ValueError("QQ/ACF candidate paths must contain at least one sample")
        return array[:, :1, :, :]
    if array.ndim == 3:
        if array.shape[1] < 1:
            raise ValueError("QQ/ACF candidate paths must contain at least one sample")
        return array[:, :1, :]
    return array


def _statistics_for_lines(
    *,
    time_grid: np.ndarray,
    real_paths: np.ndarray,
    named_paths: Mapping[str, np.ndarray],
    process: str,
    metadata: Mapping[str, Any],
) -> tuple[dict[str, _QQCurve], dict[str, _ACFCurve], _ACFCurve]:
    line_paths = {
        key: _first_sample_paths(paths)
        for key, paths in named_paths.items()
    }
    qq_curves = {
        key: _pooled_increment_qq(real_paths=real_paths, candidate_paths=paths)
        for key, paths in line_paths.items()
    }
    acf_curves = {
        key: _process_acf(
            time_grid=time_grid,
            paths=paths,
            process=process,
            metadata=metadata,
        )
        for key, paths in line_paths.items()
    }
    real_acf = _process_acf(
        time_grid=time_grid,
        paths=real_paths,
        process=process,
        metadata=metadata,
    )
    return qq_curves, acf_curves, real_acf


def _normalize_metric_names(metric_names: tuple[str, ...]) -> set[str]:
    selected = set(metric_names)
    if "panels" in selected:
        selected.remove("panels")
        selected.update(SUPPORTED_QUALITATIVE_METRICS)
    unknown = selected.difference(SUPPORTED_QUALITATIVE_METRICS)
    if unknown:
        raise ValueError(f"unsupported qualitative metric(s): {sorted(unknown)}")
    return selected


def _plain_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(repo_relative_path(value))
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _plain_value(value.item())
        return [_plain_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value


def _cache_output_dir(data_dir: Path) -> Path:
    return data_dir / "results" / "qual_cache"


def _cache_plot_style(*, baseline_order: tuple[str, ...]) -> dict[str, Any]:
    return {
        "baseline_order": tuple(baseline_order),
        "baseline_colors": {},
        "baseline_fallback_colors": ("tab:purple", "tab:brown", "tab:pink", "tab:gray"),
    }


_RESERVED_CACHE_PREFIXES = frozenset({"real", "time_grid", "path_indices", "quantiles"})


def _method_prefixes(candidates: list[CandidateArtifact]) -> dict[str, str]:
    prefixes: dict[str, str] = {}
    seen: set[str] = set()
    for candidate in candidates:
        prefix = candidate.key
        if not prefix:
            raise ValueError("Qualitative cache candidate key must be non-empty")
        if prefix in _RESERVED_CACHE_PREFIXES:
            raise ValueError(f"Qualitative cache candidate key collides with reserved prefix: {prefix}")
        if prefix in seen:
            raise ValueError(f"Duplicate qualitative cache candidate key: {prefix}")
        seen.add(prefix)
        prefixes[candidate.key] = prefix
    return prefixes


def _candidate_cache_metadata(candidate: CandidateArtifact, *, prefix: str, role: str) -> dict[str, Any]:
    return {
        "key": candidate.key,
        "role": role,
        "npz_prefix": prefix,
        "kind": candidate.kind,
        "method": candidate.method,
        "conditioning": candidate.conditioning,
        "source_artifact": candidate.source_artifact,
        "aug_type": candidate.aug_type,
        "depth": candidate.depth,
        "artifact_path": candidate.artifact_path,
    }


def _curve_manifest(prefix: str, curve: _ACFCurve) -> dict[str, Any]:
    companion_items = []
    for index, companion in enumerate(curve.companions):
        companion_prefix = f"{prefix}_companion{index:02d}"
        companion_items.append(
            {
                "target": companion.target,
                "center": companion.center,
                "tau": f"{companion_prefix}_tau",
                "rho": f"{companion_prefix}_rho",
            }
        )
    return {
        "target": curve.target,
        "center": curve.center,
        "tau": f"{prefix}_tau",
        "rho": f"{prefix}_rho",
        "companions": companion_items,
    }


def _append_curve_arrays(arrays: dict[str, np.ndarray], *, prefix: str, curve: _ACFCurve) -> None:
    arrays[f"{prefix}_tau"] = np.asarray(curve.tau, dtype=float)
    arrays[f"{prefix}_rho"] = np.asarray(curve.rho, dtype=float)
    for index, companion in enumerate(curve.companions):
        companion_prefix = f"{prefix}_companion{index:02d}"
        arrays[f"{companion_prefix}_tau"] = np.asarray(companion.tau, dtype=float)
        arrays[f"{companion_prefix}_rho"] = np.asarray(companion.rho, dtype=float)


def _selected_candidate_paths(
    *,
    paths: np.ndarray,
    path_indices: list[int],
    candidate_key: str,
    is_target: bool,
    n_gen: int,
    overlay_seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if is_target:
        sample_indices = [
            _panel_sample_candidate_indices(
                paths.shape[1],
                n_gen,
                path_idx=path_idx,
                candidate_key=candidate_key,
                random_seed=overlay_seed,
            )
            for path_idx in path_indices
        ]
        selected = np.stack(
            [paths[path_idx, sample_indices[index], :] for index, path_idx in enumerate(path_indices)],
            axis=0,
        )
        return selected, np.asarray(sample_indices, dtype=int)

    selected = _first_sample_paths(paths)[path_indices, :, :]
    return selected, None


def write_qualitative_cache_for_dataset(
    data_dir: str | Path,
    *,
    candidates: list[CandidateArtifact],
    out_dir: str | Path | None = None,
    n_show: int = DEFAULT_N_SHOW,
    n_gen: int = DEFAULT_N_GEN,
    overlay_seed: int = DEFAULT_OVERLAY_RANDOM_SEED,
    baseline_order: tuple[str, ...] = DEFAULT_BASELINE_ORDER,
) -> Path:
    """Write reusable qualitative numeric artifacts for paper figure notebooks."""

    data_dir = Path(data_dir)
    train_key, test_key = _split_keys(data_dir)
    artifacts = load_dataset_artifacts(data_dir)
    process = detect_process(artifacts)
    metadata = _metadata_from_artifacts(artifacts)
    baselines, targets = _resolve_overlay_targets(data_dir, candidates)
    by_key = {candidate.key: candidate for candidate in [*targets, *baselines]}
    panel_candidates = list(by_key.values())
    prefixes = _method_prefixes(panel_candidates)
    target_keys = {candidate.key for candidate in targets}
    path_scale = float(artifacts.get("path_scale", 1.0))

    loaded_paths = {
        candidate.key: _load_candidate_paths(candidate, path_scale=path_scale)
        for candidate in panel_candidates
    }
    time_grid, real_paths, normalized_paths = _validate_path_arrays(
        time_grid=np.asarray(artifacts["time_grid"], dtype=float),
        real_paths=np.asarray(artifacts["real_paths"], dtype=float),
        named_paths=loaded_paths,
    )

    output_dir = Path(out_dir) if out_dir is not None else _cache_output_dir(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path_indices = _panel_select_path_indices(real_paths.shape[0], n_show)
    qq_curves, acf_curves, real_acf = _statistics_for_lines(
        time_grid=time_grid,
        real_paths=real_paths,
        named_paths=normalized_paths,
        process=process,
        metadata=metadata,
    )

    path_arrays: dict[str, np.ndarray] = {
        "time_grid": time_grid,
        "path_indices": np.asarray(path_indices, dtype=int),
        "real_paths": real_paths[path_indices, 0, :],
    }
    path_manifest_methods: dict[str, Any] = {}
    for candidate in panel_candidates:
        prefix = prefixes[candidate.key]
        selected_paths, sample_indices = _selected_candidate_paths(
            paths=normalized_paths[candidate.key],
            path_indices=path_indices,
            candidate_key=candidate.key,
            is_target=candidate.key in target_keys,
            n_gen=n_gen,
            overlay_seed=overlay_seed,
        )
        path_arrays[f"{prefix}_paths"] = selected_paths
        method_entry: dict[str, Any] = {"paths": f"{prefix}_paths"}
        if sample_indices is not None:
            path_arrays[f"{prefix}_sample_indices"] = sample_indices
            method_entry["sample_indices"] = f"{prefix}_sample_indices"
        path_manifest_methods[candidate.key] = method_entry

    first_curve = next(iter(qq_curves.values()))
    qq_arrays: dict[str, np.ndarray] = {
        "quantiles": np.asarray(first_curve.quantiles, dtype=float),
        "real": np.asarray(first_curve.real, dtype=float),
    }
    qq_manifest_methods: dict[str, Any] = {}
    for candidate in panel_candidates:
        prefix = prefixes[candidate.key]
        qq_arrays[f"{prefix}_candidate"] = np.asarray(qq_curves[candidate.key].candidate, dtype=float)
        qq_manifest_methods[candidate.key] = {"candidate": f"{prefix}_candidate"}

    acf_arrays: dict[str, np.ndarray] = {}
    _append_curve_arrays(acf_arrays, prefix="real", curve=real_acf)
    acf_manifest_methods: dict[str, Any] = {}
    for candidate in panel_candidates:
        prefix = prefixes[candidate.key]
        _append_curve_arrays(acf_arrays, prefix=prefix, curve=acf_curves[candidate.key])
        acf_manifest_methods[candidate.key] = _curve_manifest(prefix, acf_curves[candidate.key])

    plot_style = _cache_plot_style(baseline_order=baseline_order)
    baseline_routes = _panel_build_candidate_baseline_routes(
        baselines,
        targets,
        plot_style=plot_style,
    )
    manifest = {
        "schema_version": QUAL_CACHE_SCHEMA_VERSION,
        "train_key": train_key,
        "test_key": test_key,
        "process": process,
        "source": {
            "data_dir": data_dir,
            "real_artifact": data_dir / "real.npz",
        },
        "metadata": metadata,
        "artifacts": {
            "path": "path.npz",
            "qq": "qq.npz",
            "acf": "acf.npz",
        },
        "selection": {
            "n_show": int(n_show),
            "n_gen": int(n_gen),
            "path_indices": path_indices,
            "overlay_seed": int(overlay_seed),
        },
        "sample_policy": {
            "qq_acf": "candidate_sample_index_0",
            "qq_acf_sample_index": 0,
            "path_overlay": "deterministic_path_indices_and_seeded_candidate_samples",
        },
        "targets": [candidate.key for candidate in targets],
        "baseline_routes": baseline_routes,
        "methods": [
            _candidate_cache_metadata(
                candidate,
                prefix=prefixes[candidate.key],
                role="target" if candidate.key in target_keys else "baseline",
            )
            for candidate in panel_candidates
        ],
        "path": {
            "time_grid": "time_grid",
            "path_indices": "path_indices",
            "real_paths": "real_paths",
            "methods": path_manifest_methods,
        },
        "qq": {
            "quantiles": "quantiles",
            "real": "real",
            "methods": qq_manifest_methods,
        },
        "acf": {
            "real": _curve_manifest("real", real_acf),
            "methods": acf_manifest_methods,
        },
    }

    np.savez(output_dir / "path.npz", **path_arrays)
    np.savez(output_dir / "qq.npz", **qq_arrays)
    np.savez(output_dir / "acf.npz", **acf_arrays)
    (output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(_plain_value(manifest), sort_keys=False),
        encoding="utf-8",
    )
    return output_dir


def _write_png_artifacts(
    *,
    qual_root: Path,
    time_grid: np.ndarray,
    real_paths: np.ndarray,
    candidate_paths: Mapping[str, np.ndarray],
    candidates: list[CandidateArtifact],
    baselines: list[CandidateArtifact],
    targets: list[CandidateArtifact],
    process: str,
    metadata: Mapping[str, Any],
    selected_metrics: set[str],
    n_show: int,
    n_gen: int,
    plot_style: Mapping[str, Any],
) -> None:
    del candidates
    staging_roots = {
        metric: _plot_staging_root(qual_root, metric)
        for metric in selected_metrics
    }
    for staging_root in staging_roots.values():
        if staging_root.exists():
            shutil.rmtree(staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)

    path_indices = (
        _panel_select_path_indices(real_paths.shape[0], n_show)
        if "overlay" in selected_metrics
        else []
    )
    baseline_routes = _panel_build_candidate_baseline_routes(
        baselines,
        targets,
        plot_style=plot_style,
    )
    qq_curves: dict[str, _QQCurve] = {}
    acf_curves: dict[str, _ACFCurve] = {}
    real_acf: _ACFCurve | None = None
    if "qq" in selected_metrics or "acf" in selected_metrics:
        qq_curves, acf_curves, real_acf = _statistics_for_lines(
            time_grid=time_grid,
            real_paths=real_paths,
            named_paths=candidate_paths,
            process=process,
            metadata=metadata,
        )

    try:
        for target in targets:
            target_paths = candidate_paths[target.key]
            routed_baseline_keys = [
                key
                for key in baseline_routes[target.key]
                if key in candidate_paths and key in qq_curves and key in acf_curves
            ]
            if "qq" in selected_metrics:
                baseline_qq = {key: qq_curves[key] for key in routed_baseline_keys}
                qq_spec = _panel_build_qq_plot_spec(
                    candidate_key=target.key,
                    candidate_kind=target.kind,
                    qq_curve=qq_curves[target.key],
                    baseline_curves=baseline_qq,
                    plot_style=plot_style,
                )
                _render_plot_spec_png(
                    plot_spec=qq_spec,
                    output_path=staging_roots["qq"] / f"{target.key}.png",
                    figsize=(4.0, 4.0),
                )
            if "acf" in selected_metrics:
                if real_acf is None:
                    raise RuntimeError("real ACF curve was not computed")
                baseline_acf = {key: acf_curves[key] for key in routed_baseline_keys}
                acf_spec = _panel_build_acf_plot_spec(
                    candidate_key=target.key,
                    candidate_kind=target.kind,
                    real_curve=real_acf,
                    candidate_curve=acf_curves[target.key],
                    baseline_curves=baseline_acf,
                    plot_style=plot_style,
                )
                _render_plot_spec_png(
                    plot_spec=acf_spec,
                    output_path=staging_roots["acf"] / f"{target.key}.png",
                    figsize=(4.0, 4.0),
                )
            if "overlay" in selected_metrics:
                for path_idx in path_indices:
                    sample_indices = _panel_sample_candidate_indices(
                        target_paths.shape[1],
                        n_gen,
                        path_idx=path_idx,
                        candidate_key=target.key,
                        random_seed=int(plot_style["overlay_random_seed"]),
                    )
                    deterministic_paths = {
                        key: candidate_paths[key][path_idx, 0]
                        for key in baseline_routes[target.key]
                        if key in candidate_paths
                    }
                    overlay_spec = _panel_build_overlay_plot_spec(
                        time_grid=time_grid,
                        real_path=real_paths[path_idx, 0],
                        deterministic_paths=deterministic_paths,
                        candidate_paths=target_paths[path_idx, sample_indices, :],
                        candidate_key=target.key,
                        path_idx=path_idx,
                        plot_style=plot_style,
                    )
                    _render_plot_spec_png(
                        plot_spec=overlay_spec,
                        output_path=staging_roots["overlay"] / f"path_{path_idx:03d}" / f"{target.key}.png",
                        figsize=(6.0, 4.0),
                    )

        for metric, staging_root in staging_roots.items():
            _replace_plot_dir(
                final_root=_plot_root(qual_root, metric),
                staging_root=staging_root,
            )
    finally:
        for staging_root in staging_roots.values():
            if staging_root.exists():
                shutil.rmtree(staging_root)


def write_qualitative_results_for_dataset(
    data_dir: str | Path,
    *,
    candidates: list[CandidateArtifact],
    plot_style: Mapping[str, Any],
    metric_names: tuple[str, ...] = SUPPORTED_QUALITATIVE_METRICS,
    n_show: int = DEFAULT_N_SHOW,
    n_gen: int = DEFAULT_N_GEN,
) -> Path:
    data_dir = Path(data_dir)
    artifacts = load_dataset_artifacts(data_dir)
    process = detect_process(artifacts)
    metadata = _metadata_from_artifacts(artifacts)
    selected_metrics = _normalize_metric_names(metric_names)
    baselines, targets = _resolve_overlay_targets(data_dir, candidates)
    by_key = {candidate.key: candidate for candidate in [*targets, *baselines]}
    panel_candidates = list(by_key.values())
    path_scale = float(artifacts.get("path_scale", 1.0))

    loaded_paths = {
        candidate.key: _load_candidate_paths(candidate, path_scale=path_scale)
        for candidate in panel_candidates
    }
    time_grid, real_paths, normalized_paths = _validate_path_arrays(
        time_grid=np.asarray(artifacts["time_grid"], dtype=float),
        real_paths=np.asarray(artifacts["real_paths"], dtype=float),
        named_paths=loaded_paths,
    )

    qual_root = _qual_root(data_dir)
    qual_root.mkdir(parents=True, exist_ok=True)
    _cleanup_obsolete_outputs(qual_root)
    _write_png_artifacts(
        qual_root=qual_root,
        time_grid=time_grid,
        real_paths=real_paths,
        candidate_paths=normalized_paths,
        candidates=panel_candidates,
        baselines=baselines,
        targets=targets,
        process=process,
        metadata=metadata,
        selected_metrics=selected_metrics,
        n_show=n_show,
        n_gen=n_gen,
        plot_style=plot_style,
    )
    return qual_root


__all__ = [
    "DEFAULT_BASELINE_ORDER",
    "DEFAULT_N_GEN",
    "DEFAULT_N_SHOW",
    "DEFAULT_OVERLAY_RANDOM_SEED",
    "QUAL_CACHE_SCHEMA_VERSION",
    "SUPPORTED_QUALITATIVE_METRICS",
    "_resolve_overlay_targets",
    "write_qualitative_cache_for_dataset",
    "write_qualitative_results_for_dataset",
]
