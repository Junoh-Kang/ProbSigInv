"""Thin generative inversion glue over shared conditioning and loading utilities."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from probsiginv.inversion.contracts import InversionMethodContext, InversionResult, InversionSourceContext
from probsiginv.inversion.outputs import resolve_output_layout
from probsiginv.repo import resolve_repo_path
from probsiginv.training.processes import normalize_training_method
from probsiginv.utils.contract import build_conditioning_label, build_output_candidate_label
from probsiginv.utils.conditioning import (
    build_conditioning_from_signatures,
    resolve_data_aug_type,
    resolve_effective_sig_depth,
)
from probsiginv.utils.path_shapes import from_model_paths
from probsiginv.utils.transforms import invert_path_transform
from probsiginv.utils.utils import load_model_and_process


@dataclass(frozen=True)
class CheckpointInspection:
    """Checkpoint metadata needed before loading the full backend."""

    ckpt_args: dict[str, Any] = field(default_factory=dict)
    expected_cond_dim: int | None = None
    checkpoint_payload: dict[str, Any] | None = None


@dataclass
class PreparedGenerativeInversion:
    """Resolved conditioning tensors and backend objects for generative inversion."""

    ckpt_args: dict
    cond_sigs: np.ndarray
    cond_sig_depth: int
    path_transform: str
    training_method: str
    model: object | None
    process: object | None


def inspect_generative_checkpoint(args: argparse.Namespace) -> CheckpointInspection:
    """Read checkpoint args without constructing the model backend."""
    if args.method != "generative":
        return CheckpointInspection()
    if args.ckpt is None:
        raise ValueError("--ckpt is required when --method=generative")

    peek = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ckpt_args = dict(peek.get("args", {}))
    if normalize_training_method(ckpt_args.get("training_method", "ddpm")) == "regression":
        raise ValueError(f"{args.ckpt}: regression checkpoint; use --method regression")
    ema_state = peek.get("ema_model", {})
    weight = ema_state.get("cond_emb.mlp.0.weight")
    expected_cond_dim = int(weight.shape[1]) if weight is not None else None
    return CheckpointInspection(
        ckpt_args=ckpt_args,
        expected_cond_dim=expected_cond_dim,
        checkpoint_payload=peek,
    )


def build_model_conditioning_from_dataset_sigs(
    source_sigs: np.ndarray,
    source_meta: dict,
    ckpt_args: dict,
) -> tuple[np.ndarray, int]:
    """Build conditioning tensors directly from split-local dataset signatures."""
    train_aug_type = str(ckpt_args["aug_type"])
    source_aug_type = str(source_meta.get("aug_type", "TA"))
    expected_aug_type = resolve_data_aug_type(train_aug_type)
    if source_aug_type != expected_aug_type:
        raise ValueError(
            "Dataset signature augmentation mismatch: "
            f"checkpoint requires dataset aug_type {expected_aug_type!r}, "
            f"got {source_aug_type!r}"
        )

    sig_type = str(ckpt_args.get("sig_type", "logsig"))
    source_sig_type = str(source_meta.get("sig_type", "logsig"))
    if source_sig_type != sig_type:
        raise ValueError(
            "Dataset signature type mismatch: "
            f"checkpoint requires sig_type {sig_type!r}, got {source_sig_type!r}"
        )

    requested_sig_depth = int(ckpt_args.get("sig_depth", 0))
    dataset_sig_depth = int(source_meta.get("sig_depth", 0))
    effective_sig_depth = resolve_effective_sig_depth(requested_sig_depth, dataset_sig_depth)
    sig_transform = ckpt_args.get("sig_transform", None)
    sig_full = bool(ckpt_args.get("sig_full", False))

    cond_sigs, cond_sig_depth, _ = build_conditioning_from_signatures(
        source_sigs,
        aug_type=train_aug_type,
        sig_type=sig_type,
        sig_depth=effective_sig_depth,
        sig_transform=sig_transform,
        sig_full=sig_full,
        sig_channels=int(source_meta.get("sig_channels", 0)) or None,
    )
    return cond_sigs, cond_sig_depth


def prepare_generative_inversion(
    args: argparse.Namespace,
    *,
    source: InversionSourceContext,
    device,
    inspection: CheckpointInspection,
) -> PreparedGenerativeInversion:
    """Rebuild conditioning and load the model/process backend."""
    if args.method != "generative":
        return PreparedGenerativeInversion(
            ckpt_args={},
            cond_sigs=np.empty((source.paths.shape[0], 0), dtype=np.float32),
            cond_sig_depth=0,
            path_transform="raw",
            training_method="ddpm",
            model=None,
            process=None,
        )

    ckpt_args = inspection.ckpt_args
    train_aug_type = ckpt_args["aug_type"]
    sig_transform = ckpt_args.get("sig_transform", None)
    sig_full = ckpt_args.get("sig_full", False)
    sig_type = str(ckpt_args.get("sig_type", "logsig"))
    if sig_type != "logsig":
        raise ValueError(f"Only sig_type=\"logsig\" is supported, got {sig_type!r}")

    path_transform = ckpt_args.get("path_transform", "raw")
    training_method = normalize_training_method(ckpt_args.get("training_method", "ddpm"))
    requested_sig_depth = int(ckpt_args.get("sig_depth", 0))

    if requested_sig_depth == 0:
        cond_sigs = np.empty((source.paths.shape[0], 0), dtype=np.float32)
        cond_sig_depth = 0
        print("Model is unconditional (sig_depth=0): no conditioning")
    else:
        if source.source_sigs is None:
            raise ValueError("dataset-backed generative inversion requires source signatures")
        cond_sigs, cond_sig_depth = build_model_conditioning_from_dataset_sigs(
            source.source_sigs,
            source.meta,
            ckpt_args,
        )
        print(
            f"  Built conditioning: aug={train_aug_type}, sig_type={sig_type}, "
            f"sig_depth={cond_sig_depth}, sig_full={sig_full}, sig_transform={sig_transform}"
        )

    expected_cond_dim = inspection.expected_cond_dim
    if expected_cond_dim is not None and expected_cond_dim != cond_sigs.shape[1]:
        raise ValueError(
            "Conditioning dimension mismatch: "
            f"expected {expected_cond_dim} from checkpoint, got {cond_sigs.shape[1]}"
        )
    expected_path_dim = int(ckpt_args.get("path_dim", 1))
    if expected_path_dim != source.path_dim:
        raise ValueError(
            "Path dimension mismatch: "
            f"checkpoint path_dim={expected_path_dim}, source path_dim={source.path_dim}"
        )

    clip_x0 = tuple(args.clip_x0) if args.clip_x0 is not None else None
    print(f"  Using path_transform from checkpoint: {path_transform}")
    print(f"  Using training_method from checkpoint: {training_method}")
    model, process, _ = load_model_and_process(
        args.ckpt,
        seq_len=source.seq_len,
        cond_dim=cond_sigs.shape[1],
        device=device,
        clip_x0=clip_x0,
        checkpoint_payload=inspection.checkpoint_payload,
    )
    return PreparedGenerativeInversion(
        ckpt_args=ckpt_args,
        cond_sigs=cond_sigs,
        cond_sig_depth=cond_sig_depth,
        path_transform=path_transform,
        training_method=training_method,
        model=model,
        process=process,
    )


def resolve_generative_output_layout(
    args: argparse.Namespace,
    dataset_name: str,
    ckpt_args: dict,
) -> tuple[str, str]:
    """Resolve output paths for the generative method."""
    if args.exp_name:
        base_exp_name = args.exp_name
    else:
        conditioning = build_conditioning_label(
            str(ckpt_args.get("aug_type", getattr(args, "aug_type", "TA"))),
            int(ckpt_args.get("sig_depth", 0)),
        )
        if float(args.eta) != 0.0:
            raise ValueError(
                "Default output naming assumes eta=0.0; use --output-dir to override manually"
            )
        base_exp_name = build_output_candidate_label(
            "generative",
            conditioning=conditioning,
            sampling_steps=args.sampling_steps,
        )

    return resolve_output_layout(
        args,
        dataset_name,
        base_exp_name=base_exp_name,
    )


def _build_generative_result(
    paths: np.ndarray,
    time_grid: np.ndarray,
    *,
    conditioning: str,
    source_artifact: str,
    n_repeats: int,
    sampling_steps: int,
    eta: float,
    path_dim: int = 1,
) -> InversionResult:
    return InversionResult(
        paths=np.asarray(paths),
        time_grid=np.asarray(time_grid),
        method="generative",
        metadata={
            "conditioning": conditioning,
            "source_artifact": source_artifact,
            "n_repeats": n_repeats,
            "sampling_steps": sampling_steps,
            "eta": eta,
            "path_dim": int(path_dim),
        },
    )


def _sample_signatures(
    sigs: np.ndarray,
    model,
    process,
    sample_shape: int | tuple[int, int],
    device,
    *,
    sampling_steps: int | None = None,
    eta: float = 0.0,
) -> np.ndarray:
    """Sample one path per conditioning signature row."""
    cond = torch.from_numpy(sigs).float().to(device)
    with torch.no_grad():
        paths = process.sample_loop(
            model,
            (sigs.shape[0], sample_shape) if isinstance(sample_shape, int) else (sigs.shape[0], *sample_shape),
            cond,
            sampling_steps=sampling_steps,
            eta=eta,
        )
    return paths.cpu().numpy()


def run_generative_inversion(
    args,
    *,
    method_context: InversionMethodContext,
    prepared: PreparedGenerativeInversion,
    device,
    batch_size: int = 1024,
) -> InversionResult:
    """Sample paths from the generative backend and return the canonical candidate payload."""
    time_grid = method_context.source.time_grid
    n_paths = method_context.source.n_paths
    seq_len = method_context.source.seq_len
    path_dim = method_context.source.path_dim
    cond_rows = prepared.cond_sigs.shape[0]
    if cond_rows != n_paths:
        raise ValueError(
            "Conditioning row-count mismatch: "
            f"paths has {n_paths} rows, conditioning has {cond_rows}"
        )

    n_repeats = args.n_repeats
    total = n_paths * n_repeats
    effective_batch = min(batch_size, total)
    if batch_size < 8:
        print(f"[generative] warning: batch_size={batch_size} is very small and will be slow")
    print(
        f"[generative] generating {n_paths} x {n_repeats} paths "
        f"(sig_depth={prepared.cond_sig_depth}, "
        f"training_method={prepared.training_method}, "
        f"batch_size={effective_batch}) ..."
    )
    sigs_tiled = np.repeat(prepared.cond_sigs, n_repeats, axis=0)
    t0 = time.perf_counter()
    chunks = []
    n_chunks = (total + effective_batch - 1) // effective_batch
    iterator = tqdm(
        range(0, total, effective_batch),
        total=n_chunks,
        desc="[generative] batches",
        unit="batch",
    )
    for i in iterator:
        chunks.append(
            _sample_signatures(
                sigs_tiled[i : i + effective_batch],
                prepared.model,
                prepared.process,
                seq_len if path_dim == 1 else (path_dim, seq_len),
                device,
                sampling_steps=args.sampling_steps,
                eta=args.eta,
            )
        )
    flat = np.concatenate(chunks, axis=0)
    flat = invert_path_transform(flat, prepared.path_transform)
    elapsed = time.perf_counter() - t0
    if path_dim == 1:
        samples = flat.reshape(n_paths, n_repeats, seq_len)
    else:
        artifact_flat = from_model_paths(flat)
        samples = artifact_flat.reshape(n_paths, n_repeats, seq_len, path_dim)
    print(
        f"  done in {elapsed:.1f}s  "
        f"({elapsed / n_paths:.2f} s/path, {elapsed / (n_paths * n_repeats) * 1000:.1f} ms/sample)"
    )
    return _build_generative_result(
        samples,
        time_grid,
        conditioning=build_conditioning_label(
            str(prepared.ckpt_args.get("aug_type", "TA")),
            int(prepared.ckpt_args.get("sig_depth", 0)),
        ),
        source_artifact=str(resolve_repo_path(args.ckpt)),
        n_repeats=n_repeats,
        sampling_steps=args.sampling_steps,
        eta=args.eta,
        path_dim=path_dim,
    )


__all__ = [
    "build_model_conditioning_from_dataset_sigs",
    "inspect_generative_checkpoint",
    "prepare_generative_inversion",
    "resolve_generative_output_layout",
    "run_generative_inversion",
]
