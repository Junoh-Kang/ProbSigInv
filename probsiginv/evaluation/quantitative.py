"""Quantitative evaluation artifact assembly and storage."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from probsiginv.evaluation.candidate_selection import CandidateArtifact
from probsiginv.evaluation.data import load_dataset_artifacts
from probsiginv.evaluation.metrics.finance import (
    ACF_ERROR_LAGS,
    ACF_ERROR_REDUCTION,
    compute_stylized_metrics,
)
from probsiginv.evaluation.metrics.bootstrap import build_mean_bootstrap_summary
from probsiginv.evaluation.metrics.parameters import compute_parameter_metrics
from probsiginv.evaluation.metrics.path_space import (
    compute_cond_mean_mse,
    compute_energy_distance,
    compute_path_mse,
    compute_path_mse_per_real,
    compute_spread_mse,
)
from probsiginv.evaluation.metrics.signature_space import (
    compute_cond_consistency,
    compute_sig_mse,
    parse_signature_space_key,
)
from probsiginv.evaluation.metrics.summary import mean_std
from probsiginv.evaluation.types import EvalBatch
from probsiginv.utils.contract import load_candidate_contract
from probsiginv.inversion.provenance import file_sha256


SUPPORTED_METRICS = (
    "mse",
    "cond-mean-mse",
    "spread-mse",
    "energy-distance",
    "params",
    "cond-consistency",
    "sig-mse",
    "stylized",
    "oracle",
)
DEFAULT_METRICS = (
    "mse",
    "cond-mean-mse",
    "spread-mse",
    "params",
    "cond-consistency",
    "sig-mse",
)
FINANCE_DEFAULT_METRICS = (
    "mse",
    "cond-mean-mse",
    "spread-mse",
    "cond-consistency",
    "sig-mse",
    "stylized",
)
CANDIDATE_BATCH_METRICS = (
    "mse",
    "cond-mean-mse",
    "spread-mse",
    "params",
    "cond-consistency",
    "sig-mse",
    "stylized",
)
REFERENCE_FREE_METRICS = tuple(metric for metric in SUPPORTED_METRICS if metric != "energy-distance")
DEFAULT_SIG_SPACE_KEYS = ("TAd6",)
DEFAULT_PATH_MSE_BOOTSTRAP_RESAMPLES = 10000
DEFAULT_PATH_MSE_BOOTSTRAP_SEED = 20260504
DEFAULT_PATH_MSE_BOOTSTRAP_CI_LEVEL = 0.95
SCHEMA_VERSION = 1
ENERGY_DISTANCE_ESTIMATOR = "unbiased_u"
# Seeds of the numerical fBM / OU oracle averaged in the paper.
DEFAULT_ORACLE_SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1001)
ORACLE_PRIOR_KEYS = {
    "gbm": ("mu", "sigma"),
    "fbm": ("mu", "sigma", "hurst"),
    "ou": ("theta", "mu", "sigma"),
}


def _normalize_process(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip().lower()


def detect_process(artifacts: dict[str, Any]) -> str:
    process = _normalize_process(artifacts.get("process"))
    if process:
        return process
    if artifacts.get("ticker") is not None:
        return "finance"
    if artifacts.get("theta") is not None:
        return "ou"
    if artifacts.get("hurst") is not None:
        return "fbm"
    return "gbm"


def default_metrics_for_process(process: str) -> tuple[str, ...]:
    return FINANCE_DEFAULT_METRICS if process == "finance" else DEFAULT_METRICS


def _quant_root(data_dir: Path) -> Path:
    return data_dir / "results" / "quant"


def _validate_finite_json(value: Any, *, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_json(item, label=f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json(item, label=f"{label}[{index}]")
        return
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError(f"{label} contains a non-finite value")


def _validate_stylized_summary(value: Any, *, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {"real", "gen", "gen_abs_error"}:
        raise ValueError(f"{label} must contain real, gen, and gen_abs_error summaries")
    for key, summary in value.items():
        if not isinstance(summary, dict) or set(summary) != {"mean", "median", "std"}:
            raise ValueError(f"{label}.{key} must contain mean, median, and std")
        for statistic, item in summary.items():
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float, np.integer, np.floating))
                or not np.isfinite(item)
            ):
                raise ValueError(
                    f"{label}.{key}.{statistic} contains a non-finite or non-numeric value"
                )


def _validate_stylized_vector(value: Any, *, n_windows: int, label: str) -> None:
    if not isinstance(value, list) or len(value) != n_windows:
        raise ValueError(f"{label} must contain exactly {n_windows} window values")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float, np.integer, np.floating))
        for item in value
    ):
        raise ValueError(f"{label} must be a finite non-negative vector")
    array = np.asarray(value, dtype=float)
    if array.shape != (n_windows,) or not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError(f"{label} must be a finite non-negative vector")


def _validate_scalar_stylized_candidate(
    value: Any,
    *,
    n_windows: int,
    label: str,
) -> None:
    expected = {
        "realized_volatility",
        "skewness",
        "kurtosis",
        "return_acf_error",
        "absolute_return_acf_error",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} has an incompatible stylized schema")
    for metric in ("realized_volatility", "skewness", "kurtosis"):
        _validate_stylized_summary(value[metric], label=f"{label}.{metric}")
    for metric in ("return_acf_error", "absolute_return_acf_error"):
        _validate_stylized_vector(
            value[metric],
            n_windows=n_windows,
            label=f"{label}.{metric}",
        )


def _validate_stylized_candidates(
    candidates: Any,
    *,
    coordinate_names: list[str] | None,
    expected_n_windows: int | None = None,
) -> int:
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError("stylized candidates must be a nonempty mapping")
    if coordinate_names is not None and (
        not isinstance(coordinate_names, list)
        or len(coordinate_names) != 2
        or any(
            not isinstance(name, str) or not name or name != name.strip()
            for name in coordinate_names
        )
        or len(set(coordinate_names)) != len(coordinate_names)
    ):
        raise ValueError("stylized coordinate_names must contain two unique names")

    n_windows = expected_n_windows
    if n_windows is not None and n_windows < 1:
        raise ValueError("stylized n_windows must be positive")
    for candidate_key, candidate in candidates.items():
        if not isinstance(candidate_key, str) or not candidate_key:
            raise ValueError("stylized candidate keys must be nonempty strings")
        if coordinate_names is None:
            scalar_payloads = [(candidate_key, candidate)]
        else:
            if not isinstance(candidate, dict) or set(candidate) != {"channels", "cross_channel"}:
                raise ValueError(
                    f"candidates.{candidate_key} has an incompatible dim-2 schema"
                )
            channels = candidate["channels"]
            if not isinstance(channels, dict) or list(channels) != coordinate_names:
                raise ValueError(
                    f"candidates.{candidate_key}.channels has incompatible order"
                )
            scalar_payloads = [
                (f"{candidate_key}.channels.{name}", channels[name])
                for name in coordinate_names
            ]
            cross_channel = candidate["cross_channel"]
            if not isinstance(cross_channel, dict) or set(cross_channel) != {
                "return_correlation",
                "absolute_return_correlation",
            }:
                raise ValueError(
                    f"candidates.{candidate_key}.cross_channel is incompatible"
                )
            for metric, summary in cross_channel.items():
                _validate_stylized_summary(
                    summary,
                    label=f"candidates.{candidate_key}.cross_channel.{metric}",
                )

        for label, scalar in scalar_payloads:
            vector = scalar.get("return_acf_error") if isinstance(scalar, dict) else None
            if n_windows is None:
                if not isinstance(vector, list) or not vector:
                    raise ValueError(
                        f"candidates.{label}.return_acf_error must be nonempty"
                    )
                n_windows = len(vector)
            _validate_scalar_stylized_candidate(
                scalar,
                n_windows=n_windows,
                label=f"candidates.{label}",
            )

    if n_windows is None:
        raise ValueError("stylized candidates did not define n_windows")
    return n_windows


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(
        payload,
        indent=2,
        allow_nan=False,
    ) + "\n"
    _write_text_atomic(path, serialized)


def _write_text_atomic(path: Path, serialized: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _flatten_summary(value: dict[str, Any], prefix: str, *, include_lists: bool = False) -> dict[str, Any]:
    row = {}
    for key, item in value.items():
        name = f"{prefix}.{key}"
        if isinstance(item, dict):
            row.update(_flatten_summary(item, name, include_lists=include_lists))
        elif isinstance(item, list):
            if include_lists:
                row[name] = json.dumps(item, separators=(",", ":"))
        else:
            row[name] = item
    return row


def _write_candidate_results(data_dir: Path, updates: dict[Path, dict[str, Any]]) -> None:
    """Replace selected results and rebuild the CSV from all saved candidates."""
    saved = {
        path: json.loads(path.read_text())
        for path in sorted(data_dir.glob("**/quant.json"))
        if path not in updates and "results" not in path.relative_to(data_dir).parts
        and (path.parent / "paths.npz").is_file()
    }
    saved.update(updates)
    rows = []
    for path, payload in sorted(saved.items()):
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{path}: unsupported candidate quant schema")
        rows.append({
            "candidate": payload["candidate"]["key"],
            "folder": path.parent.relative_to(data_dir).as_posix(),
            "process": payload["process"],
            **_flatten_summary({key: value for key, value in payload["candidate"].items() if key != "key"}, "candidate"),
            **_flatten_summary(payload["inputs"], "inputs", include_lists=True),
            **_flatten_summary(payload["evaluation"], "evaluation", include_lists=True),
            **_flatten_summary(payload["metrics"], "metrics"),
        })
    columns = ["candidate", "folder", "process"]
    extra_columns = {key for row in rows for key in row} - set(columns)
    columns += sorted(extra_columns, key=lambda key: (not key.startswith("metrics."), key))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    for path, payload in updates.items():
        _write_json_atomic(path, payload)
    _write_text_atomic(_quant_root(data_dir) / "summary.csv", buffer.getvalue())


def _validate_stylized_metadata_values(
    metadata: dict[str, Any],
    *,
    label: str,
) -> None:
    lags = metadata.get("acf_error_lags")
    if (
        type(lags) is not list
        or any(type(lag) is not int for lag in lags)
        or lags != list(ACF_ERROR_LAGS)
    ):
        raise ValueError(
            f"{label} metadata mismatch for acf_error_lags: "
            f"expected {list(ACF_ERROR_LAGS)!r}, got {lags!r}"
        )
    reduction = metadata.get("acf_error_reduction")
    if type(reduction) is not str or reduction != ACF_ERROR_REDUCTION:
        raise ValueError(
            f"{label} metadata mismatch for acf_error_reduction: "
            f"expected {ACF_ERROR_REDUCTION!r}, got {reduction!r}"
        )
    sample_index = metadata.get("representative_generated_sample_index")
    if type(sample_index) is not int or sample_index != 0:
        raise ValueError(
            f"{label} metadata mismatch for representative_generated_sample_index: "
            f"expected 0, got {sample_index!r}"
        )


def _oracle_key(depth: int) -> str:
    return f"LSd{depth}"


def _build_oracle_path_mse(
    process: str,
    depths: list[int],
    *,
    priors: dict[str, tuple[float, float]],
    seeds: tuple[int, ...],
    T_horizon: float,
) -> dict[str, Any] | None:
    if process not in ORACLE_PRIOR_KEYS or not depths:
        return None
    missing = [key for key in ORACLE_PRIOR_KEYS[process] if priors.get(key) is None]
    if missing:
        flags = ", ".join(f"--{key}-bounds" for key in missing)
        raise ValueError(f"oracle for process={process} requires the prior bounds {flags}")

    from probsiginv.oracle import (
        fbm_numerical_mse,
        gbm_analytic_mse,
        ou_numerical_mse,
    )

    if process == "gbm":
        result = gbm_analytic_mse(depths, priors["mu"], priors["sigma"], T_horizon=T_horizon)
        return {_oracle_key(depth): float(result[depth]) for depth in depths}

    if not seeds:
        raise ValueError("oracle for fbm/ou requires at least one seed")
    per_seed = []
    for seed in seeds:
        if process == "fbm":
            result = fbm_numerical_mse(
                depths,
                priors["mu"],
                priors["sigma"],
                priors["hurst"],
                T_horizon=T_horizon,
                seed=int(seed),
            )
        else:
            result = ou_numerical_mse(
                depths,
                priors["theta"],
                priors["mu"],
                priors["sigma"],
                T_horizon=T_horizon,
                seed=int(seed),
            )
        per_seed.append(result)
    return {
        _oracle_key(depth): {
            **mean_std([result[depth]["mean"] for result in per_seed]),
            "n_seeds": len(per_seed),
        }
        for depth in depths
    }


def _build_batch(artifacts: dict[str, Any], process: str, candidate: CandidateArtifact) -> EvalBatch:
    with np.load(candidate.artifact_path) as npz:
        gen_paths = np.asarray(npz["paths"], dtype=float)
        if gen_paths.ndim not in (2, 3, 4):
            raise ValueError(
                f"{candidate.artifact_path}: candidate paths must have shape "
                f"(N,T), (N,M,T), or (N,M,T,D), got {gen_paths.shape}"
            )
    path_dim = int(artifacts.get("path_dim", 1))
    sample_valued = gen_paths.ndim >= 3
    if gen_paths.ndim == 3 and path_dim > 1:
        sample_valued = False
        gen_paths = gen_paths[:, None, :, :]
    path_scale = float(artifacts.get("path_scale", 1.0))
    if not np.isclose(path_scale, 1.0):
        gen_paths = gen_paths / path_scale
    return EvalBatch(
        real_paths=artifacts["real_paths"],
        gen_paths=gen_paths,
        time_grid=artifacts["time_grid"],
        process=process,
        theta=artifacts["theta"],
        mu=artifacts["mu"],
        sigma=artifacts["sigma"],
        hurst=artifacts["hurst"],
        dataset_name=Path(artifacts["data_dir"]).name,
        model_name=candidate.artifact_path.parent.name,
        aug_type=candidate.aug_type or "TA",
        metadata={
            **{
                key: artifacts[key]
                for key in ("gbm_rho", "coordinate_names")
                if artifacts.get(key) is not None
            },
            "candidate_sample_valued": sample_valued,
        },
    )


def _resolve_dim2_finance_coordinate_names(data_dir: str | Path, path_dim: int) -> tuple[str, str]:
    real_path = Path(data_dir) / "real.npz"
    with np.load(real_path, allow_pickle=False) as real_npz:
        if "currencies" not in real_npz:
            raise ValueError(f"{real_path}: missing dim-2 finance coordinate names in currencies")
        raw_names = np.asarray(real_npz["currencies"])

    if raw_names.ndim != 1 or raw_names.size != path_dim:
        raise ValueError(
            f"{real_path}: coordinate name width must match path_dim={path_dim}, "
            f"got shape {raw_names.shape}"
        )
    names = tuple(raw_names.tolist())
    if any(
        not isinstance(name, str) or not name or name != name.strip()
        for name in names
    ):
        raise ValueError(f"{real_path}: coordinate names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{real_path}: coordinate names must be unique")
    return names[0], names[1]


def _load_energy_sample_artifact(
    artifact_path: str | Path,
    *,
    path_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Energy Distance sample artifact does not exist: {artifact_path}")
    with np.load(artifact_path) as npz:
        missing = [key for key in ("paths", "time_grid") if key not in npz]
        if missing:
            raise ValueError(f"{artifact_path}: missing required arrays: {', '.join(missing)}")
        paths = np.asarray(npz["paths"])
        time_grid = np.asarray(npz["time_grid"], dtype=float)

    if not np.isclose(path_scale, 1.0):
        paths = np.asarray(paths, dtype=float) / path_scale
    return paths, time_grid


def _compute_energy_distance_payload(
    *,
    reference_paths: np.ndarray,
    reference_time_grid: np.ndarray,
    model_artifact: str | Path,
    path_scale: float,
) -> dict[str, Any]:
    model_paths, model_time_grid = _load_energy_sample_artifact(
        model_artifact,
        path_scale=path_scale,
    )
    if reference_time_grid.shape != model_time_grid.shape or not np.array_equal(
        reference_time_grid,
        model_time_grid,
    ):
        raise ValueError(
            "Energy Distance reference and model artifacts must have identical time_grid arrays: "
            f"{reference_time_grid.shape} vs {model_time_grid.shape}"
        )

    payload = compute_energy_distance(reference_paths, model_paths)
    payload["per_condition"] = payload["per_condition"].tolist()
    return payload


def _compute_path_mse_payload(
    batch: EvalBatch,
    *,
    bootstrap_mse: bool,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    bootstrap_ci_level: float,
) -> dict[str, Any]:
    if not bootstrap_mse:
        return compute_path_mse(batch)

    per_real = compute_path_mse_per_real(batch)
    payload: dict[str, Any] = mean_std(per_real)
    payload["bootstrap"] = build_mean_bootstrap_summary(
        per_real,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        ci_level=bootstrap_ci_level,
        resample_axis="real_path",
        statistic="mean_path_mse_after_repeat_average",
        metadata={"n_repeats": batch.n_samples},
    )
    return payload


def _compute_candidate_artifacts(
    data_dir: str | Path,
    *,
    candidates: list[CandidateArtifact],
    metric_names: tuple[str, ...],
    sig_space_keys: tuple[str, ...] = (),
    bootstrap_mse: bool = False,
    bootstrap_resamples: int = DEFAULT_PATH_MSE_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_PATH_MSE_BOOTSTRAP_SEED,
    bootstrap_ci_level: float = DEFAULT_PATH_MSE_BOOTSTRAP_CI_LEVEL,
    energy_reference: str | Path | None = None,
    oracle_priors: dict[str, tuple[float, float]] | None = None,
    oracle_seeds: tuple[int, ...] = DEFAULT_ORACLE_SEEDS,
) -> tuple[
    str,
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, Any],
]:
    artifacts = load_dataset_artifacts(data_dir)
    process = detect_process(artifacts)
    selected = set(metric_names)
    stylized_metadata: dict[str, Any] = {}
    if "stylized" in selected and process == "finance":
        stylized_metadata = {
            "acf_error_lags": list(ACF_ERROR_LAGS),
            "acf_error_reduction": ACF_ERROR_REDUCTION,
            "representative_generated_sample_index": 0,
        }
        if int(artifacts.get("path_dim", 1)) == 2:
            coordinate_names = _resolve_dim2_finance_coordinate_names(
                data_dir,
                int(artifacts["path_dim"]),
            )
            artifacts["coordinate_names"] = coordinate_names
            stylized_metadata["coordinate_names"] = list(coordinate_names)
    needs_candidate_batches = any(metric in selected for metric in CANDIDATE_BATCH_METRICS)
    parsed_sig_spaces = {
        f"{signature_augment}d{signature_depth}": (signature_augment, signature_depth)
        for signature_augment, signature_depth in (parse_signature_space_key(space_key) for space_key in sig_space_keys)
    }
    parsed_sig_spaces = dict(sorted(parsed_sig_spaces.items(), key=lambda item: item[0].casefold()))

    reference_paths = None
    reference_time_grid = None
    reference_conditioning = None
    if "energy-distance" in selected:
        if energy_reference is None:
            raise ValueError("energy-distance requires energy_reference")
        reference_contract = load_candidate_contract(energy_reference)
        if reference_contract.kind != "reference":
            raise ValueError(
                f"Energy Distance reference must have kind='reference', got {reference_contract.kind!r}"
            )
        reference_conditioning = reference_contract.conditioning
        reference_paths, reference_time_grid = _load_energy_sample_artifact(energy_reference)

    metric_payloads: dict[str, dict[str, Any]] = {
        "mse": {},
        "cond_mean_mse": {},
        "spread_mse": {},
        "energy_distance": {},
        "params": {},
        "cond_consistency": {},
        "stylized": {},
    }
    consistency_metadata: dict[str, Any] = {}
    space_metric_payloads: dict[str, Any] = {}
    depths: set[int] = set()

    for candidate in candidates:
        if needs_candidate_batches:
            batch = _build_batch(artifacts, process, candidate)
            if "mse" in selected:
                if batch.metadata["candidate_sample_valued"]:
                    metric_payloads["mse"][candidate.key] = _compute_path_mse_payload(
                        batch,
                        bootstrap_mse=bootstrap_mse,
                        bootstrap_resamples=bootstrap_resamples,
                        bootstrap_seed=bootstrap_seed,
                        bootstrap_ci_level=bootstrap_ci_level,
                    )
                else:
                    metric_payloads["mse"][candidate.key] = None
            if "cond-mean-mse" in selected:
                metric_payloads["cond_mean_mse"][candidate.key] = compute_cond_mean_mse(batch)
            if "spread-mse" in selected:
                metric_payloads["spread_mse"][candidate.key] = (
                    compute_spread_mse(batch)
                    if batch.metadata["candidate_sample_valued"]
                    else None
                )
            if "params" in selected and process != "finance":
                metric_payloads["params"][candidate.key] = (
                    compute_parameter_metrics(batch)
                    if batch.path_dim == 1 or (process == "gbm" and batch.path_dim == 2)
                    else {"unsupported_for_path_dim": int(batch.path_dim)}
                )
            if "cond-consistency" in selected:
                consistency_payload, signature_metadata = compute_cond_consistency(
                    batch,
                    data_dir=data_dir,
                    candidate=candidate,
                )
                metric_payloads["cond_consistency"][candidate.key] = consistency_payload
                if signature_metadata is not None:
                    consistency_metadata[candidate.key] = signature_metadata
            if "sig-mse" in selected:
                for canonical_space_key, (signature_augment, signature_depth) in parsed_sig_spaces.items():
                    space_payload = space_metric_payloads.setdefault(
                        canonical_space_key,
                        {
                            "signature_augment": signature_augment,
                            "signature_depth": signature_depth,
                            "candidates": {},
                        },
                    )
                    space_payload["candidates"][candidate.key] = compute_sig_mse(
                        batch,
                        data_dir=data_dir,
                        candidate=candidate,
                        signature_augment=signature_augment,
                        signature_depth=signature_depth,
                    )
            if "stylized" in selected:
                try:
                    metric_payloads["stylized"][candidate.key] = (
                        compute_stylized_metrics(batch)
                        if batch.path_dim in {1, 2}
                        else {"unsupported_for_path_dim": int(batch.path_dim)}
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"stylized candidate {candidate.key}: {exc}"
                    ) from exc
        if "energy-distance" in selected:
            if candidate.conditioning != reference_conditioning:
                raise ValueError(
                    "Energy Distance candidate and reference must have matching conditioning: "
                    f"{candidate.conditioning!r} vs {reference_conditioning!r}"
                )
            metric_payloads["energy_distance"][candidate.key] = (
                _compute_energy_distance_payload(
                    reference_paths=reference_paths,
                    reference_time_grid=reference_time_grid,
                    model_artifact=candidate.artifact_path,
                    path_scale=float(artifacts.get("path_scale", 1.0)),
                )
                if candidate.kind == "generative"
                else None
            )
        if "oracle" in selected and candidate.kind == "generative" and candidate.depth is not None:
            depths.add(candidate.depth)

    if "params" in selected and process == "finance":
        metric_payloads["params"] = {candidate.key: None for candidate in candidates}

    T_horizon = artifacts["T_horizon"]
    if T_horizon is None:
        T_horizon = float(artifacts["time_grid"][-1] - artifacts["time_grid"][0])
    oracle_payload = None
    if "oracle" in selected:
        if int(artifacts.get("path_dim", 1)) > 1:
            oracle_payload = {"unsupported_for_path_dim": int(artifacts["path_dim"])}
        else:
            oracle_payload = _build_oracle_path_mse(
                process,
                sorted(depths),
                priors=oracle_priors or {},
                seeds=oracle_seeds,
                T_horizon=float(T_horizon),
            )

    filtered_payloads = {
        metric: payload
        for metric, payload in metric_payloads.items()
        if payload
    }
    return (
        process,
        filtered_payloads,
        consistency_metadata,
        space_metric_payloads,
        oracle_payload,
        stylized_metadata,
    )


def write_quantitative_results_for_dataset(
    data_dir: str | Path,
    *,
    candidates: list[CandidateArtifact],
    metric_names: tuple[str, ...] = REFERENCE_FREE_METRICS,
    sig_space_keys: tuple[str, ...] = (),
    bootstrap_mse: bool = False,
    bootstrap_resamples: int = DEFAULT_PATH_MSE_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_PATH_MSE_BOOTSTRAP_SEED,
    bootstrap_ci_level: float = DEFAULT_PATH_MSE_BOOTSTRAP_CI_LEVEL,
    energy_reference: str | Path | None = None,
    oracle_priors: dict[str, tuple[float, float]] | None = None,
    oracle_seeds: tuple[int, ...] = DEFAULT_ORACLE_SEEDS,
) -> Path:
    data_dir = Path(data_dir)
    train_key = data_dir.name
    quant_dir = _quant_root(data_dir)
    selected = set(metric_names)
    if "sig-mse" in selected and not sig_space_keys:
        sig_space_keys = DEFAULT_SIG_SPACE_KEYS
    if "sig-mse" not in selected and sig_space_keys:
        raise ValueError("--sig-space requires sig-mse")
    if "energy-distance" in selected and energy_reference is None:
        raise ValueError("energy-distance requires --energy-reference")
    if "energy-distance" not in selected and energy_reference is not None:
        raise ValueError("--energy-reference requires energy-distance")

    (
        process,
        metric_payloads,
        consistency_metadata,
        space_metric_payloads,
        oracle_payload,
        stylized_metadata,
    ) = _compute_candidate_artifacts(
        data_dir,
        candidates=candidates,
        metric_names=metric_names,
        sig_space_keys=sig_space_keys,
        bootstrap_mse=bootstrap_mse,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_ci_level=bootstrap_ci_level,
        energy_reference=energy_reference,
        oracle_priors=oracle_priors,
        oracle_seeds=oracle_seeds,
    )

    inputs = {"real_sha256": file_sha256(data_dir / "real.npz")}
    if selected - {"oracle"}:
        if selected & {"cond-consistency", "sig-mse"}:
            inputs["signature_sha256"] = {
                path.name: file_sha256(path)
                for path in sorted(data_dir.glob("real_*logsigs.npz"))
            }
        protocol: dict[str, Any] = {"metrics": sorted(selected - {"oracle"})}
        if "sig-mse" in selected:
            protocol["sig_spaces"] = list(space_metric_payloads)
        if "mse" in selected:
            protocol["bootstrap_mse"] = bootstrap_mse
            if bootstrap_mse:
                protocol.update(bootstrap_resamples=bootstrap_resamples,
                                bootstrap_seed=bootstrap_seed,
                                bootstrap_ci_level=bootstrap_ci_level)
        if "energy-distance" in selected:
            protocol["energy_distance_estimator"] = ENERGY_DISTANCE_ESTIMATOR
            protocol["energy_reference"] = str(Path(energy_reference).resolve())
            inputs["energy_reference_sha256"] = file_sha256(energy_reference)
        if stylized_metadata:
            _validate_stylized_metadata_values(stylized_metadata, label="stylized")
            _validate_stylized_candidates(
                metric_payloads["stylized"],
                coordinate_names=stylized_metadata.get("coordinate_names"),
            )
            protocol["stylized"] = stylized_metadata

        updates = {}
        for candidate in candidates:
            metrics = {
                name: values[candidate.key]
                for name, values in metric_payloads.items()
                if candidate.key in values
            }
            if "sig-mse" in selected:
                metrics["sig_mse"] = {
                    space: {**{key: value for key, value in values.items() if key != "candidates"},
                            "result": values["candidates"][candidate.key]}
                    for space, values in space_metric_payloads.items()
                }
            candidate_protocol = dict(protocol)
            if candidate.key in consistency_metadata:
                candidate_protocol["conditioning_signature"] = consistency_metadata[candidate.key]
            payload = {
                "schema_version": SCHEMA_VERSION,
                "split": train_key,
                "process": process,
                "candidate": {"key": candidate.key, "kind": candidate.kind,
                              "method": candidate.method, "conditioning": candidate.conditioning},
                "inputs": {**inputs, "paths_sha256": file_sha256(candidate.artifact_path)},
                "evaluation": candidate_protocol,
                "metrics": metrics,
            }
            _validate_finite_json(payload, label=candidate.key)
            updates[candidate.artifact_path.parent / "quant.json"] = payload
        _write_candidate_results(data_dir, updates)

    if "oracle" in selected:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "train_key": train_key,
            "process": process,
            "metric": "oracle",
            "path_mse": oracle_payload or {},
            "inputs": {"real_sha256": inputs["real_sha256"]},
            "evaluation": {"priors": oracle_priors or {}, "seeds": list(oracle_seeds),
                           "depths": sorted({candidate.depth for candidate in candidates
                                             if candidate.kind == "generative" and candidate.depth is not None})},
        }
        if process == "finance":
            payload["unsupported_for_process"] = "finance"
        _write_json_atomic(quant_dir / "oracle.json", payload)

    return quant_dir
