from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from probsiginv.evaluation.candidate_selection import CandidateArtifact
from probsiginv.evaluation.metrics.summary import sample_std
from probsiginv.evaluation.types import EvalBatch
from probsiginv.repo import resolve_repo_path
from probsiginv.utils.conditioning import build_conditioning_from_signatures, compute_raw_signatures_from_paths
from probsiginv.utils.contract import load_candidate_metadata
from probsiginv.utils.signatures import augment_path, compute_logsignatures_dim


_REAL_SIG_RE = re.compile(r"^real_sigs_(?P<aug>LS|TA|TLL3|TLL)_(?P<sig_type>[a-z]+)(?:_d(?P<depth>\d+))?\.npz$")
_SPACE_KEY_RE = re.compile(r"^(?P<aug>LS|TA|TLL3|TLL)d(?P<depth>\d+)$", re.IGNORECASE)


def parse_signature_space_key(space_key: str) -> tuple[str, int]:
    match = _SPACE_KEY_RE.match(space_key.strip())
    if match is None:
        raise ValueError(f"Invalid --sig-space value: {space_key!r}")
    return str(match.group("aug")).upper(), int(match.group("depth"))


def _load_raw_signature_cache(
    path: Path,
    *,
    aug_type: str,
    sig_depth: int,
    n_paths: int,
) -> tuple[np.ndarray, int] | None:
    if not path.exists():
        return None
    with np.load(path) as npz:
        candidate_aug = str(np.asarray(npz.get("aug_type", aug_type)).item())
        candidate_sig_type = str(np.asarray(npz.get("sig_type", "logsig")).item())
        candidate_depth = int(np.asarray(npz.get("sig_depth", 0)).item())
        sigs = np.asarray(npz["sigs"], dtype=np.float32)
    if candidate_aug != aug_type or candidate_sig_type != "logsig":
        return None
    if candidate_depth < sig_depth:
        return None
    if sigs.shape[0] != n_paths:
        return None
    return sigs, candidate_depth


def _write_raw_signature_cache(
    path: Path,
    sigs: np.ndarray,
    *,
    aug_type: str,
    sig_depth: int,
    sig_type: str = "logsig",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sigs=np.asarray(sigs, dtype=np.float32),
        aug_type=np.array(str(aug_type).upper()),
        sig_type=np.array(str(sig_type).lower()),
        sig_depth=np.array(int(sig_depth), dtype=np.int32),
    )


def _build_raw_logsig_level_indices(
    batch: EvalBatch,
    *,
    aug_type: str,
    sig_depth: int,
) -> dict[int, list[int]]:
    # Evaluation-space sig-mse lives in raw logsignature space, so level
    # boundaries come from the augmentation dimension directly.
    probe_paths = batch.real_paths[:1]
    if probe_paths.shape[0] == 0:
        probe_paths = np.zeros((1, batch.n_steps), dtype=float)
    channels = int(augment_path(probe_paths, batch.time_grid, aug_type=aug_type).shape[-1])
    level_ends = compute_logsignatures_dim(channels, sig_depth)
    level_starts = np.concatenate(([0], level_ends[:-1]))
    return {
        level: list(range(int(start), int(stop)))
        for level, start, stop in zip(range(1, sig_depth + 1), level_starts, level_ends)
    }


def _build_signature_mse_summary(
    real_sigs: np.ndarray,
    gen_sigs: np.ndarray,
) -> dict[str, Any]:
    diff = real_sigs[:, None, :] - gen_sigs
    absolute = np.mean(np.mean(diff ** 2, axis=-1), axis=1)
    denom = np.mean(real_sigs ** 2, axis=-1)
    relative = np.divide(
        absolute,
        denom,
        out=np.zeros_like(absolute),
        where=denom > 0.0,
    )
    return {
        "absolute": {
            "mean": float(np.mean(absolute)),
            "median": float(np.median(absolute)),
            "std": sample_std(absolute),
        },
        "relative": {
            "mean": float(np.mean(relative)),
            "median": float(np.median(relative)),
            "std": sample_std(relative),
        }
    }


def build_sig_mse_summary(
    real_sigs: np.ndarray,
    gen_sigs: np.ndarray,
    level_indices: dict[int, list[int]],
) -> dict[str, Any]:
    levels: dict[str, dict[str, Any]] = {}
    full_indices: list[int] = []
    for level, indices in sorted(level_indices.items()):
        level_index_list = list(indices)
        full_indices.extend(level_index_list)
        levels[str(level)] = _build_signature_mse_summary(
            real_sigs[:, level_index_list],
            gen_sigs[:, :, level_index_list],
        )

    return {
        "full": _build_signature_mse_summary(
            real_sigs[:, full_indices],
            gen_sigs[:, :, full_indices],
        ),
        "levels": levels,
    }


def build_cond_consistency_summary(
    real_sigs: np.ndarray,
    gen_sigs: np.ndarray,
) -> dict[str, Any]:
    diff = real_sigs[:, None, :] - gen_sigs
    absolute = np.mean(np.mean(diff ** 2, axis=-1), axis=1)
    denom = np.mean(real_sigs ** 2, axis=-1)
    relative = np.divide(
        absolute,
        denom,
        out=np.zeros_like(absolute),
        where=denom > 0.0,
    )
    return {
        "absolute": {
            "mean": float(np.mean(absolute)),
            "median": float(np.median(absolute)),
            "std": sample_std(absolute),
        },
        "relative": {
            "mean": float(np.mean(relative)),
            "median": float(np.median(relative)),
            "std": sample_std(relative),
        },
    }


def _load_real_signature_artifact(
    data_dir: str | Path,
    *,
    aug_type: str,
) -> tuple[np.ndarray, int] | None:
    data_dir = Path(data_dir)
    best_match: tuple[np.ndarray, int] | None = None
    for candidate in sorted(data_dir.glob("real_sigs_*.npz")):
        match = _REAL_SIG_RE.match(candidate.name)
        if match is None:
            continue
        with np.load(candidate) as npz:
            candidate_aug = str(np.asarray(npz.get("aug_type", match.group("aug"))).item())
            candidate_sig_type = str(np.asarray(npz.get("sig_type", match.group("sig_type"))).item())
            if candidate_aug != aug_type or candidate_sig_type != "logsig":
                continue
            filename_depth = match.group("depth")
            candidate_depth = int(np.asarray(npz.get("sig_depth", filename_depth or 0)).item())
            sig_key = "raw_sigs" if "raw_sigs" in npz else "sigs"
            candidate_sigs = np.asarray(npz[sig_key], dtype=np.float32)
        if best_match is None or candidate_depth > best_match[1]:
            best_match = (candidate_sigs, candidate_depth)
    return best_match


def resolve_signature_metadata(
    data_dir: str | Path,
    candidate: CandidateArtifact,
) -> dict[str, Any]:
    if candidate.kind != "generative":
        raise ValueError("signature metadata is only defined for generative candidates")

    metadata = load_candidate_metadata(candidate.artifact_path)
    source_artifact = metadata.get("source_artifact")
    if not source_artifact:
        raise ValueError(f"{candidate.artifact_path}: generative artifact is missing source_artifact")

    checkpoint = torch.load(resolve_repo_path(source_artifact), map_location="cpu", weights_only=False)
    ckpt_args = dict(checkpoint.get("args", {}))
    aug_type = ckpt_args.get("aug_type")
    sig_depth = ckpt_args.get("sig_depth")
    sig_transform = ckpt_args.get("sig_transform")
    sig_full = bool(ckpt_args.get("sig_full", False))

    if aug_type is None or sig_depth is None:
        raise ValueError(
            "Could not resolve training signature metadata from generative source artifact "
            f"for {candidate.artifact_path.parent.name}"
        )
    aug_type = str(aug_type)
    sig_depth = int(sig_depth)

    real_sig_artifact = _load_real_signature_artifact(data_dir, aug_type=aug_type)
    if real_sig_artifact is not None:
        _, dataset_sig_depth = real_sig_artifact
        if sig_depth > dataset_sig_depth:
            raise ValueError(
                f"model sig_depth {sig_depth} exceeds available dataset signature depth {dataset_sig_depth} "
                f"for aug_type={aug_type}"
            )

    return {
        "aug_type": aug_type,
        "sig_type": "logsig",
        "sig_depth": sig_depth,
        "sig_transform": sig_transform,
        "sig_full": sig_full,
        "sig_channels": ckpt_args.get("sig_channels"),
        "real_sig_artifact": real_sig_artifact,
    }


def _load_real_raw_signatures(
    batch: EvalBatch,
    *,
    data_dir: str | Path,
    aug_type: str,
    sig_depth: int,
) -> np.ndarray:
    real_sig_artifact = _load_real_signature_artifact(data_dir, aug_type=aug_type)
    if real_sig_artifact is not None:
        _, dataset_sig_depth = real_sig_artifact
        if sig_depth <= dataset_sig_depth:
            return np.asarray(real_sig_artifact[0], dtype=np.float32)

    return compute_raw_signatures_from_paths(
        batch.real_paths,
        batch.time_grid,
        aug_type=aug_type,
        sig_depth=sig_depth,
        sig_type="logsig",
    )


def _load_or_compute_sig_mse_real_signatures(
    batch: EvalBatch,
    *,
    data_dir: str | Path,
    aug_type: str,
    sig_depth: int,
) -> np.ndarray:
    data_dir = Path(data_dir)
    cache_path = data_dir / f"real_{str(aug_type).upper()}d{int(sig_depth)}_logsigs.npz"
    cached = _load_raw_signature_cache(
        cache_path,
        aug_type=aug_type,
        sig_depth=sig_depth,
        n_paths=batch.n_real,
    )
    if cached is not None:
        return cached[0]

    sigs = _load_real_raw_signatures(
        batch,
        data_dir=data_dir,
        aug_type=aug_type,
        sig_depth=sig_depth,
    )
    _write_raw_signature_cache(
        cache_path,
        sigs,
        aug_type=aug_type,
        sig_depth=sig_depth,
    )
    return sigs


def _load_or_compute_sig_mse_candidate_signatures(
    batch: EvalBatch,
    *,
    candidate: CandidateArtifact,
    signature_augment: str,
    signature_depth: int,
) -> np.ndarray:
    cache_path = (
        candidate.artifact_path.parent
        / f"{str(signature_augment).upper()}d{int(signature_depth)}_logsigs.npz"
    )
    cached = _load_raw_signature_cache(
        cache_path,
        aug_type=signature_augment,
        sig_depth=signature_depth,
        n_paths=batch.n_real,
    )
    if cached is not None:
        return cached[0][:, None, :]

    representative_paths = batch.gen_paths[:, 0, :]
    sigs = compute_raw_signatures_from_paths(
        representative_paths,
        batch.time_grid,
        aug_type=signature_augment,
        sig_depth=signature_depth,
        sig_type="logsig",
    )
    _write_raw_signature_cache(
        cache_path,
        sigs,
        aug_type=signature_augment,
        sig_depth=signature_depth,
    )
    return sigs[:, None, :]


def compute_cond_consistency(
    batch: EvalBatch,
    *,
    data_dir: str | Path,
    candidate: CandidateArtifact,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if candidate.kind != "generative":
        return None, None

    spec = resolve_signature_metadata(data_dir, candidate)
    sig_depth = int(spec["sig_depth"])
    if sig_depth == 0:
        return None, None

    aug_type = str(spec["aug_type"])
    sig_type = str(spec["sig_type"])
    sig_transform = spec["sig_transform"]
    sig_full = bool(spec["sig_full"])
    sig_channels = spec.get("sig_channels")
    if sig_channels is None:
        sig_channels = int(augment_path(batch.real_paths[:1], batch.time_grid, aug_type=aug_type).shape[-1])

    real_raw_sigs = _load_real_raw_signatures(
        batch,
        data_dir=data_dir,
        aug_type=aug_type,
        sig_depth=sig_depth,
    )
    real_cond_sigs, _, level_indices = build_conditioning_from_signatures(
        real_raw_sigs,
        aug_type=aug_type,
        sig_type=sig_type,
        sig_depth=sig_depth,
        sig_transform=sig_transform,
        sig_full=sig_full,
        sig_channels=int(sig_channels),
    )

    if batch.path_dim == 1:
        flat_gen_paths = batch.gen_paths[..., 0].reshape(batch.n_real * batch.n_samples, batch.n_steps)
    else:
        flat_gen_paths = batch.gen_paths.reshape(batch.n_real * batch.n_samples, batch.n_steps, batch.path_dim)
    gen_raw_sigs = compute_raw_signatures_from_paths(
        flat_gen_paths,
        batch.time_grid,
        aug_type=aug_type,
        sig_depth=sig_depth,
        sig_type=sig_type,
    )
    gen_cond_sigs, _, _ = build_conditioning_from_signatures(
        gen_raw_sigs,
        aug_type=aug_type,
        sig_type=sig_type,
        sig_depth=sig_depth,
        sig_transform=sig_transform,
        sig_full=sig_full,
        level_indices=level_indices,
        sig_channels=int(sig_channels),
    )
    gen_cond_sigs = gen_cond_sigs.reshape(batch.n_real, batch.n_samples, -1)

    metadata = {
        "signature_augment": aug_type,
        "signature_depth": sig_depth,
        "signature_transform": sig_transform,
        "signature_full": sig_full,
    }
    return build_cond_consistency_summary(real_cond_sigs, gen_cond_sigs), metadata


def compute_sig_mse(
    batch: EvalBatch,
    *,
    data_dir: str | Path,
    candidate: CandidateArtifact,
    signature_augment: str,
    signature_depth: int,
) -> dict[str, Any] | None:
    if signature_depth == 0:
        return None

    real_raw_sigs = _load_or_compute_sig_mse_real_signatures(
        batch,
        data_dir=data_dir,
        aug_type=signature_augment,
        sig_depth=signature_depth,
    )

    gen_raw_sigs = _load_or_compute_sig_mse_candidate_signatures(
        batch,
        candidate=candidate,
        signature_augment=signature_augment,
        signature_depth=signature_depth,
    )

    level_indices = _build_raw_logsig_level_indices(
        batch,
        aug_type=signature_augment,
        sig_depth=signature_depth,
    )
    return build_sig_mse_summary(real_raw_sigs, gen_raw_sigs, level_indices)
