"""Shared output layout and generic artifact-writing helpers for inversion."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from probsiginv.repo import resolve_repo_path
from probsiginv.utils.contract import build_output_candidate_dir
from probsiginv.utils.layout import default_project_name, validate_project_name


def resolve_project_name(args: argparse.Namespace, ckpt_args: dict | None = None) -> str | None:
    explicit = getattr(args, "project", None) or getattr(args, "training_setting", None)
    if explicit:
        return validate_project_name(explicit)
    if ckpt_args is not None:
        setting = ckpt_args.get("project") or ckpt_args.get("training_setting")
        if not setting and ckpt_args.get("save_path"):
            setting = Path(ckpt_args["save_path"]).name
        return validate_project_name(setting or default_project_name(ckpt_args))
    if args.output_dir is not None:
        return None
    raise ValueError("Baseline inversion requires --project (or an explicit --output-dir)")


def resolve_output_layout(
    args: argparse.Namespace,
    dataset_name: str,
    *,
    base_exp_name: str,
) -> tuple[str, str]:
    """Resolve the experiment output dir and its parent dataset dir."""
    if args.output_dir is not None:
        out_root = os.path.normpath(str(resolve_repo_path(args.output_dir)))
        dataset_dir = os.path.dirname(out_root)
        return out_root, dataset_dir

    out_path = build_output_candidate_dir(
        dataset_name,
        base_exp_name,
        project=getattr(args, "project", None),
        base_dir=getattr(args, "output_root", None) or "_outputs",
    )
    out_root = os.path.normpath(str(out_path))
    dataset_dir = os.path.normpath(str(out_path.parent))
    return out_root, dataset_dir


def save_npz_artifact(output_dir: str, artifact_name: str, **arrays) -> str:
    """Write a named NPZ artifact under the resolved output directory."""
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, artifact_name)
    import numpy as np

    np.savez(out, **arrays)
    return out
