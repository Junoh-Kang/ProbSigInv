"""Checkpoint-backed deterministic regression inversion helpers."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from probsiginv.inversion.contracts import InversionMethodContext, InversionResult, InversionSourceContext
from probsiginv.inversion.generative import build_model_conditioning_from_dataset_sigs
from probsiginv.inversion.outputs import resolve_output_layout
from probsiginv.repo import resolve_repo_path
from probsiginv.training.processes import normalize_training_method
from probsiginv.utils.contract import build_conditioning_label, build_output_candidate_label
from probsiginv.utils.path_shapes import from_model_paths
from probsiginv.utils.transforms import invert_path_transform
from probsiginv.utils.utils import load_model_and_process


@dataclass(frozen=True)
class RegressionCheckpointInspection:
    """Checkpoint metadata needed before regression backend construction."""

    ckpt_args: dict[str, Any] = field(default_factory=dict)
    expected_cond_dim: int | None = None
    checkpoint_payload: dict[str, Any] | None = None


@dataclass
class PreparedRegressionInversion:
    """Resolved conditioning tensors and backend objects for regression inversion."""

    ckpt_args: dict[str, Any]
    cond_sigs: np.ndarray
    cond_sig_depth: int
    path_transform: str
    model: object
    process: object


def inspect_regression_checkpoint(args: argparse.Namespace) -> RegressionCheckpointInspection:
    """Read checkpoint args without constructing the model backend."""
    if args.method != "regression":
        return RegressionCheckpointInspection()
    if args.ckpt is None:
        raise ValueError("--ckpt is required when --method=regression")

    peek = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ckpt_args = dict(peek.get("args", {}))
    if normalize_training_method(ckpt_args.get("training_method", "ddpm")) != "regression":
        raise ValueError(f"{args.ckpt}: checkpoint training_method must be regression")

    ema_state = peek.get("ema_model", {})
    weight = ema_state.get("cond_emb.mlp.0.weight")
    expected_cond_dim = int(weight.shape[1]) if weight is not None else None
    return RegressionCheckpointInspection(
        ckpt_args=ckpt_args,
        expected_cond_dim=expected_cond_dim,
        checkpoint_payload=peek,
    )


def prepare_regression_inversion(
    args: argparse.Namespace,
    *,
    source: InversionSourceContext,
    device,
    inspection: RegressionCheckpointInspection,
) -> PreparedRegressionInversion:
    """Build conditioning tensors and load the deterministic regression backend."""
    ckpt_args = inspection.ckpt_args
    sig_type = str(ckpt_args.get("sig_type", "logsig"))
    if sig_type != "logsig":
        raise ValueError(f"Only sig_type=\"logsig\" is supported, got {sig_type!r}")

    requested_sig_depth = int(ckpt_args.get("sig_depth", 0))
    path_transform = ckpt_args.get("path_transform", "raw")
    sig_full = bool(ckpt_args.get("sig_full", False))

    if requested_sig_depth == 0:
        cond_sigs = np.empty((source.paths.shape[0], 0), dtype=np.float32)
        cond_sig_depth = 0
        print("Regression model is unconditional (sig_depth=0): no conditioning")
    else:
        if source.source_sigs is None:
            raise ValueError("dataset-backed regression inversion requires source signatures")
        cond_sigs, cond_sig_depth = build_model_conditioning_from_dataset_sigs(
            source.source_sigs,
            source.meta,
            ckpt_args,
        )
        print(
            f"  Built conditioning: aug={ckpt_args['aug_type']}, sig_type={sig_type}, "
            f"sig_depth={cond_sig_depth}, sig_full={sig_full}, sig_transform={ckpt_args.get('sig_transform', None)}"
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

    print(f"  Using path_transform from checkpoint: {path_transform}")
    model, process, _ = load_model_and_process(
        args.ckpt,
        seq_len=source.seq_len,
        cond_dim=cond_sigs.shape[1],
        device=device,
        checkpoint_payload=inspection.checkpoint_payload,
    )
    if not hasattr(process, "decode"):
        raise TypeError("regression backend must provide a deterministic decode(...) method")

    return PreparedRegressionInversion(
        ckpt_args=ckpt_args,
        cond_sigs=cond_sigs,
        cond_sig_depth=cond_sig_depth,
        path_transform=path_transform,
        model=model,
        process=process,
    )


def resolve_regression_output_layout(
    args: argparse.Namespace,
    dataset_name: str,
    ckpt_args: dict[str, Any],
) -> tuple[str, str]:
    """Resolve output paths for the deterministic regression method."""
    if args.exp_name:
        base_exp_name = args.exp_name
    else:
        conditioning = build_conditioning_label(
            str(ckpt_args.get("aug_type", getattr(args, "aug_type", "TA"))),
            int(ckpt_args.get("sig_depth", 0)),
        )
        base_exp_name = build_output_candidate_label(
            "regression",
            conditioning=conditioning,
        )

    return resolve_output_layout(
        args,
        dataset_name,
        base_exp_name=base_exp_name,
    )


def _build_regression_result(
    paths: np.ndarray,
    time_grid: np.ndarray,
    *,
    conditioning: str,
    source_artifact: str,
    path_dim: int = 1,
) -> InversionResult:
    return InversionResult(
        paths=np.asarray(paths),
        time_grid=np.asarray(time_grid),
        method="regression",
        metadata={
            "conditioning": conditioning,
            "source_artifact": source_artifact,
            "path_dim": int(path_dim),
        },
    )


def _decode_signatures(
    sigs: np.ndarray,
    model,
    process,
    seq_len: int,
    path_dim: int,
    device,
) -> np.ndarray:
    cond = torch.from_numpy(sigs).float().to(device)
    with torch.no_grad():
        try:
            paths = process.decode(
                model,
                cond,
                seq_len=seq_len,
                path_dim=path_dim,
            )
        except TypeError:
            if int(path_dim) != 1:
                raise
            paths = process.decode(
                model,
                cond,
                seq_len=seq_len,
            )
    return paths.cpu().numpy()


def run_regression_inversion(
    args,
    *,
    method_context: InversionMethodContext,
    prepared: PreparedRegressionInversion,
    device,
    batch_size: int = 1024,
) -> InversionResult:
    """Decode one deterministic path per conditioning signature row."""
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

    effective_batch = min(int(batch_size), n_paths)
    if effective_batch < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    print(
        f"[regression] decoding {n_paths} paths "
        f"(sig_depth={prepared.cond_sig_depth}, batch_size={effective_batch}) ..."
    )
    t0 = time.perf_counter()
    chunks = [
        _decode_signatures(
            prepared.cond_sigs[i : i + effective_batch],
            prepared.model,
            prepared.process,
            seq_len,
            path_dim,
            device,
        )
        for i in range(0, n_paths, effective_batch)
    ]
    decoded = np.concatenate(chunks, axis=0)
    decoded = invert_path_transform(decoded, prepared.path_transform)
    decoded = from_model_paths(decoded)
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed:.1f}s  ({elapsed / n_paths:.2f} s/path)")
    return _build_regression_result(
        decoded,
        time_grid,
        conditioning=build_conditioning_label(
            str(prepared.ckpt_args.get("aug_type", "TA")),
            int(prepared.ckpt_args.get("sig_depth", 0)),
        ),
        source_artifact=str(resolve_repo_path(args.ckpt)),
        path_dim=path_dim,
    )


__all__ = [
    "RegressionCheckpointInspection",
    "PreparedRegressionInversion",
    "inspect_regression_checkpoint",
    "prepare_regression_inversion",
    "resolve_regression_output_layout",
    "run_regression_inversion",
]
