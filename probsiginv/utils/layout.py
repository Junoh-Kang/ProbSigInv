"""Helpers for checkpoint/output naming conventions."""

from __future__ import annotations

from pathlib import Path

from probsiginv.repo import resolve_repo_path
from probsiginv.utils.contract import (
    build_checkpoint_candidate_dir,
    build_conditioning_label,
    checkpoint_filename,
)


def _format_param_label(value) -> str:
    if isinstance(value, tuple):
        return f"{value[0]}-{value[1]}"
    return f"{value}"


def canonical_arch_token(arch: str) -> str:
    token = str(arch).strip().lower()
    if token in {"transformer", "dit"}:
        return "dit"
    return token


def legacy_arch_token(arch: str) -> str:
    token = canonical_arch_token(arch)
    if token == "dit":
        return "transformer"
    return token


def normalize_training_method_token(training_method: str | None) -> str:
    if training_method is None:
        return "ddpm"

    method = str(training_method).strip().lower()
    if method in {"ddpm", "diffusion"}:
        return "ddpm"
    if method in {"flow", "flowmatching", "flow_matching", "fm"}:
        return "flow"
    raise ValueError(f"Unknown training_method: {training_method!r}")


def build_dataset_tag(data_path: str | Path, path_transform: str | None = None) -> str:
    base = Path(data_path).name
    if path_transform in (None, "", "raw"):
        return base
    return f"{base}_path-{path_transform}"


def build_model_tag(
    arch: str,
    aug_type: str,
    sig_depth: int,
    sig_transform: str | None = None,
    sig_full: bool = False,
    training_method: str | None = None,
) -> str:
    method = normalize_training_method_token(training_method)
    arch_token = canonical_arch_token(arch)
    transform_tag = f"_{sig_transform}" if sig_transform else ""
    full_tag = "_full" if sig_full else ""
    return f"{method}_{arch_token}_{aug_type}_d{sig_depth}{transform_tag}{full_tag}"


def build_legacy_model_tags(
    arch: str,
    aug_type: str,
    sig_depth: int,
    sig_transform: str | None = None,
    sig_full: bool = False,
    training_method: str | None = None,
    sig_type: str = "logsig",
) -> list[str]:
    method = normalize_training_method_token(training_method)
    arch_token = legacy_arch_token(arch)
    transform_tag = f"_{sig_transform}" if sig_transform else ""
    full_tag = "_full" if sig_full else ""
    base = f"{arch_token}_{aug_type}_{sig_type}_d{sig_depth}{transform_tag}{full_tag}"
    tags = [f"{base}_{method}"]
    if method == "ddpm":
        tags.append(base)
    return list(dict.fromkeys(tags))


def build_checkpoint_exp_path(save_path: str | Path, dataset_tag: str, candidate_label: str) -> Path:
    return build_checkpoint_candidate_dir(save_path, dataset_tag, candidate_label)


def build_checkpoint_dir(save_path: str | Path, dataset_tag: str, candidate_label: str) -> Path:
    return build_checkpoint_exp_path(save_path, dataset_tag, candidate_label)


def build_checkpoint_candidates(
    save_path: str | Path,
    dataset_tag: str,
    candidate_label: str,
    legacy_candidate_labels: list[str] | None = None,
    *,
    epoch: int = 1000,
) -> list[Path]:
    candidates = [build_checkpoint_dir(save_path, dataset_tag, candidate_label) / checkpoint_filename(epoch)]
    for legacy_candidate_label in legacy_candidate_labels or []:
        candidates.append(
            build_checkpoint_dir(save_path, dataset_tag, legacy_candidate_label) / checkpoint_filename(epoch)
        )
    return list(dict.fromkeys(candidates))


def build_output_dataset_dir(
    dataset_name: str,
    *,
    project: str | None = None,
    base_dir: str | Path = "_outputs",
) -> Path:
    parts = [resolve_repo_path(base_dir)]
    if project:
        parts.append(Path(project))
    parts.append(Path(dataset_name))
    out = parts[0]
    for part in parts[1:]:
        out /= part
    return out


def build_generated_output_dir(
    dataset_name: str,
    exp_name: str,
    *,
    project: str | None = None,
    base_dir: str | Path = "_outputs",
) -> Path:
    return build_output_dataset_dir(
        dataset_name,
        project=project,
        base_dir=base_dir,
    ) / exp_name


def build_sampling_exp_name(model_tag: str, sampling_steps: int) -> str:
    return f"{model_tag}_s{int(sampling_steps)}"


def build_simulation_family_name(mu, sigma, hurst=None) -> str:
    if hurst is None:
        return f"gbm_mu{_format_param_label(mu)}_sig{_format_param_label(sigma)}"
    return (
        f"fbm_mu{_format_param_label(mu)}_sig{_format_param_label(sigma)}"
        f"_H{_format_param_label(hurst)}"
    )


def build_simulation_dataset_name(mu, sigma, n_paths: int, hurst=None) -> str:
    family = build_simulation_family_name(mu, sigma, hurst=hurst)
    return f"{family}_N{int(n_paths)}"


def build_dataset_name_from_dataset_path(dataset_path: str | Path) -> str:
    split_dir = Path(dataset_path)
    if split_dir.name == "paths.npz":
        split_dir = split_dir.parent
    if split_dir.parent.name == "test":
        return str(Path(split_dir.parent.parent.name) / "test" / split_dir.name)
    if split_dir.name in {"train", "val"}:
        return str(Path(split_dir.parent.name) / split_dir.name)
    # Explicit split directories outside the builder's layout remain supported.
    return split_dir.name


def default_project_name(config: dict) -> str:
    method = str(config.get("training_method", "ddpm")).lower()
    method = "reg" if method == "regression" else normalize_training_method_token(method)
    arch = canonical_arch_token(config.get("arch", "dit"))
    transform = config.get("sig_transform") or "raw"
    return f"{method}_{arch}_{transform}"


def validate_project_name(value: str) -> str:
    value = str(value).strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("project must be a single non-empty directory name")
    return value


def build_dataset_split_name(
    dataset_root: str | Path,
    split: str,
    dataset_name_tag: str | None = None,
) -> str:
    if dataset_name_tag is not None:
        return str(dataset_name_tag)
    return f"{Path(dataset_root).name}_{split}"


__all__ = [
    "build_conditioning_label",
    "build_checkpoint_candidates",
    "build_checkpoint_dir",
    "build_checkpoint_exp_path",
    "build_dataset_name_from_dataset_path",
    "build_dataset_split_name",
    "build_dataset_tag",
    "build_generated_output_dir",
    "build_legacy_model_tags",
    "build_model_tag",
    "build_output_dataset_dir",
    "build_sampling_exp_name",
    "build_simulation_dataset_name",
    "build_simulation_family_name",
    "canonical_arch_token",
    "checkpoint_filename",
    "legacy_arch_token",
    "normalize_training_method_token",
]
