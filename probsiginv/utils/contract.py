"""Canonical save/load contract helpers for checkpoints and inversion artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np

from probsiginv.repo import resolve_repo_path


PATHS_ARTIFACT_NAME = "paths.npz"
OUTPUT_ARTIFACT_NAME = PATHS_ARTIFACT_NAME
DEFAULT_GENERATIVE_SAMPLING_STEPS = 1000
_SIG_ARTIFACT_RE = re.compile(r"^(?P<aug>[A-Za-z0-9]+)_(?P<sig_type>[A-Za-z0-9]+)s\.npz$")
_CONDITIONING_RE = re.compile(r"^(?P<aug>[A-Za-z0-9]+)d(?P<depth>\d+)$")


@dataclass(frozen=True)
class CandidateContract:
    path: Path
    method: str
    kind: str
    key: str
    conditioning: str | None = None
    aug_type: str | None = None
    depth: int | None = None
    source_artifact: str | None = None
    metadata: dict[str, Any] | None = None


def build_conditioning_label(aug_type: str, sig_depth: int) -> str:
    return f"{str(aug_type).upper()}d{int(sig_depth)}"


def parse_conditioning_label(conditioning: str) -> tuple[str, int]:
    label = str(conditioning).strip()
    match = _CONDITIONING_RE.fullmatch(label)
    if match is None:
        raise ValueError(f"Invalid conditioning label: {conditioning!r}")
    return str(match.group("aug")).upper(), int(match.group("depth"))


def build_output_candidate_label(
    method: str,
    *,
    conditioning: str | None = None,
    sampling_steps: int | None = None,
    n_coeffs: int | None = None,
) -> str:
    normalized = str(method).strip().lower()
    if normalized == "generative":
        if not conditioning:
            raise ValueError("generative outputs require conditioning")
        if sampling_steps is None or int(sampling_steps) == DEFAULT_GENERATIVE_SAMPLING_STEPS:
            return str(conditioning)
        return f"{conditioning}_s{int(sampling_steps)}"
    if normalized == "regression":
        if not conditioning:
            raise ValueError("regression outputs require conditioning")
        return f"reg_{conditioning}"
    if normalized == "ls_posterior":
        if not conditioning:
            raise ValueError("ls_posterior outputs require conditioning")
        return f"ls_posterior_{conditioning}"
    if normalized in {"fourier", "legendre"} and n_coeffs is not None:
        return f"{normalized}_n{int(n_coeffs)}"
    return normalized


def build_output_candidate_dir(
    dataset_name: str | Path,
    candidate_label: str,
    *,
    project: str | None = None,
    base_dir: str | Path = "_outputs",
) -> Path:
    root = resolve_repo_path(base_dir)
    if project:
        root /= str(project)
    return root / Path(dataset_name) / candidate_label


def build_output_artifact_path(
    dataset_name: str | Path,
    candidate_label: str,
    *,
    project: str | None = None,
    base_dir: str | Path = "_outputs",
) -> Path:
    return build_output_candidate_dir(
        dataset_name,
        candidate_label,
        project=project,
        base_dir=base_dir,
    ) / PATHS_ARTIFACT_NAME


def build_split_dir(
    dataset_root: str | Path,
    split: str,
) -> Path:
    return resolve_repo_path(dataset_root) / split


def build_split_paths_path(
    dataset_root: str | Path,
    split: str,
) -> Path:
    return build_split_dir(dataset_root, split) / PATHS_ARTIFACT_NAME


def build_split_sig_dir(
    dataset_root: str | Path,
    split: str,
    sig_tag: str,
) -> Path:
    return build_split_dir(dataset_root, split) / sig_tag


def build_split_sig_artifact_name(
    aug_type: str,
    sig_type: str = "logsig",
) -> str:
    return f"{str(aug_type).upper()}_{str(sig_type).lower()}s.npz"


def build_split_sigs_path(
    dataset_root: str | Path,
    split: str,
    aug_type: str,
    sig_type: str = "logsig",
) -> Path:
    return build_split_dir(dataset_root, split) / build_split_sig_artifact_name(aug_type, sig_type)


def resolve_split_artifact_paths(path: str | Path) -> tuple[Path, Path, Path]:
    return resolve_split_artifact_paths_for_aug(path)


def resolve_split_artifact_paths_for_aug(
    path: str | Path,
    preferred_aug_type: str | None = None,
) -> tuple[Path, Path, Path]:
    resolved = resolve_repo_path(path)
    if _SIG_ARTIFACT_RE.fullmatch(resolved.name):
        sig_path = resolved
        split_dir = resolved.parent
    elif resolved.is_dir() and (resolved / PATHS_ARTIFACT_NAME).exists():
        split_dir = resolved
        sig_paths = sorted(
            candidate
            for candidate in split_dir.iterdir()
            if candidate.is_file() and _SIG_ARTIFACT_RE.fullmatch(candidate.name)
        )
        if len(sig_paths) == 1:
            sig_path = sig_paths[0]
        elif len(sig_paths) > 1:
            preferred_candidates = []
            if preferred_aug_type is not None:
                preferred_candidates.append(
                    split_dir / build_split_sig_artifact_name(preferred_aug_type, "logsig")
                )
            preferred_candidates.append(split_dir / build_split_sig_artifact_name("TA", "logsig"))
            for candidate in preferred_candidates:
                if candidate in sig_paths:
                    sig_path = candidate
                    break
            else:
                raise ValueError(
                    f"{split_dir}: expected exactly one signature artifact, found {len(sig_paths)}"
                )
        else:
            raise ValueError(f"{split_dir}: expected exactly one signature artifact, found {len(sig_paths)}")
    else:
        raise FileNotFoundError(f"Could not resolve split artifacts from {resolved}")
    return split_dir, split_dir / PATHS_ARTIFACT_NAME, sig_path


def build_checkpoint_candidate_dir(
    save_path: str | Path,
    dataset_tag: str,
    candidate_label: str,
) -> Path:
    return resolve_repo_path(save_path) / dataset_tag / candidate_label


def checkpoint_filename(epoch: int) -> str:
    return f"ckpt_epoch{int(epoch):04d}.pt"


def build_checkpoint_path(
    save_path: str | Path,
    dataset_tag: str,
    candidate_label: str,
    *,
    epoch: int,
) -> Path:
    return build_checkpoint_candidate_dir(save_path, dataset_tag, candidate_label) / checkpoint_filename(epoch)


def _unwrap_npz_value(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array


def load_candidate_metadata(artifact_path: str | Path) -> dict[str, Any]:
    path = Path(artifact_path)
    with np.load(path) as npz:
        metadata = {
            key: _unwrap_npz_value(npz[key])
            for key in npz.files
            if key not in {"paths", "time_grid"}
        }

    if "method" not in metadata:
        raise ValueError(f"{path}: candidate artifact is missing required metadata field 'method'")

    metadata["method"] = str(metadata["method"]).strip().lower()
    if "conditioning" in metadata:
        metadata["conditioning"] = str(metadata["conditioning"]).strip()
    if "source_artifact" in metadata:
        metadata["source_artifact"] = str(metadata["source_artifact"]).strip()
    return metadata


def load_candidate_contract(artifact_path: str | Path) -> CandidateContract:
    path = Path(artifact_path)
    metadata = load_candidate_metadata(path)
    method = str(metadata["method"])

    if method == "generative":
        conditioning = metadata.get("conditioning")
        if not conditioning:
            raise ValueError(f"{path}: generative artifact is missing required metadata field 'conditioning'")
        aug_type, depth = parse_conditioning_label(str(conditioning))
        source_artifact = metadata.get("source_artifact")
        return CandidateContract(
            path=path,
            method=method,
            kind="generative",
            key=str(conditioning),
            conditioning=str(conditioning),
            aug_type=aug_type,
            depth=depth,
            source_artifact=str(source_artifact) if source_artifact else None,
            metadata=metadata,
        )

    if method == "regression":
        conditioning = metadata.get("conditioning")
        if not conditioning:
            raise ValueError(f"{path}: regression artifact is missing required metadata field 'conditioning'")
        aug_type, depth = parse_conditioning_label(str(conditioning))
        source_artifact = metadata.get("source_artifact")
        return CandidateContract(
            path=path,
            method=method,
            kind="baseline",
            key=build_output_candidate_label(method, conditioning=str(conditioning)),
            conditioning=str(conditioning),
            aug_type=aug_type,
            depth=depth,
            source_artifact=str(source_artifact) if source_artifact else None,
            metadata=metadata,
        )

    if method in {"fourier", "legendre", "insertion"}:
        n_coeffs = metadata.get("n_coeffs")
        return CandidateContract(
            path=path,
            method=method,
            kind="baseline",
            key=build_output_candidate_label(method, n_coeffs=int(n_coeffs)) if n_coeffs is not None else method,
            depth=int(metadata["sig_depth"]) if "sig_depth" in metadata else None,
            metadata=metadata,
        )

    if method == "ls_posterior":
        if metadata.get("kind") != "reference":
            raise ValueError(
                f"{path}: ls_posterior artifact requires kind='reference'"
            )
        conditioning = metadata.get("conditioning")
        if not conditioning:
            raise ValueError(
                f"{path}: ls_posterior reference is missing required metadata field 'conditioning'"
            )
        aug_type, depth = parse_conditioning_label(str(conditioning))
        if aug_type != "LS":
            raise ValueError(
                f"{path}: ls_posterior reference conditioning must use LS"
            )
        source_artifact = metadata.get(
            "source_signature_artifact",
            metadata.get("source_artifact"),
        )
        return CandidateContract(
            path=path,
            method=method,
            kind="reference",
            key=build_output_candidate_label(
                method,
                conditioning=str(conditioning),
            ),
            conditioning=str(conditioning),
            aug_type=aug_type,
            depth=depth,
            source_artifact=str(source_artifact) if source_artifact else None,
            metadata=metadata,
        )

    raise ValueError(f"{path}: unsupported candidate method {method!r}")


__all__ = [
    "CandidateContract",
    "DEFAULT_GENERATIVE_SAMPLING_STEPS",
    "OUTPUT_ARTIFACT_NAME",
    "PATHS_ARTIFACT_NAME",
    "build_checkpoint_candidate_dir",
    "build_checkpoint_path",
    "build_conditioning_label",
    "build_output_artifact_path",
    "build_output_candidate_dir",
    "build_output_candidate_label",
    "build_split_dir",
    "build_split_paths_path",
    "build_split_sig_dir",
    "build_split_sig_artifact_name",
    "build_split_sigs_path",
    "checkpoint_filename",
    "load_candidate_contract",
    "load_candidate_metadata",
    "parse_conditioning_label",
    "resolve_split_artifact_paths",
    "resolve_split_artifact_paths_for_aug",
]
