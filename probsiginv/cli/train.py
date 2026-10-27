"""Training CLI for conditional 1D path models."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from probsiginv.repo import resolve_repo_path
from probsiginv.training.runtime import run_training
from probsiginv.utils.layout import default_project_name, validate_project_name


def _parse_arch(value: str) -> str:
    token = str(value).strip().lower()
    if token == "transformer":
        return token
    if token in {"unet", "dit"}:
        return token
    raise argparse.ArgumentTypeError("invalid arch: choose 'unet' or 'dit'")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train conditional 1D path model")
    parser.add_argument("config", type=str, help="Path to YAML training config")
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Override dataset root from YAML",
    )
    parser.add_argument("--aug-type", type=str, default=None,
                        choices=["LS", "TA", "TLL3", "TLL"],
                        help="Override aug_type (LS|TA|TLL3|TLL)")
    parser.add_argument("--sig-type", type=str, default=None, choices=["logsig"],
                        help="Override sig_type (logsig only)")
    parser.add_argument("--sig-full", action="store_true", default=None,
                        help="Use full (unpruned) signature indices")
    parser.add_argument(
        "--sig-depth", type=int, default=None,
        help="Required conditioning signature depth (<= dataset sig_depth). Must be set in the config or via --sig-depth.",
    )
    parser.add_argument("--sig-transform", type=str, default=None,
                        choices=["raw", "symlog", "level", "level_symlog"],
                        help="Apply transform to signatures")
    parser.add_argument("--path-transform", type=str, default=None,
                        choices=["raw", "diff"],
                        help="Apply transform to model input/output paths")
    parser.add_argument("--training-method", type=str, default=None,
                        choices=["ddpm", "flow", "regression"],
                        help="Training backend (ddpm|flow|regression)")
    parser.add_argument(
        "--arch",
        type=_parse_arch,
        default=None,
        metavar="{unet,dit}",
        help="Override arch (unet|dit)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--save-every", type=int, default=None, help="Override save_every")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output-root", type=str, help="Checkpoint root (default: _checkpoints)")
    output.add_argument("--save-path", type=str, help="Legacy checkpoint root including the project name")
    parser.add_argument("--project", "--training-setting", dest="project", type=str,
                        help="Storage project name only (default: derived from method, architecture, and signature transform)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible training and monitor sampling")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume training from a saved checkpoint",
    )
    return parser


def merge_cli_overrides(cfg: dict, cli: argparse.Namespace) -> dict:
    if cli.sig_transform is not None:
        cfg["sig_transform"] = cli.sig_transform
    if cli.arch is not None:
        cfg["arch"] = cli.arch
    if cli.path_transform is not None:
        cfg["path_transform"] = cli.path_transform
    if cli.training_method is not None:
        cfg["training_method"] = cli.training_method
    if cli.data_path is not None:
        cfg["data_path"] = cli.data_path
    if cli.aug_type is not None:
        cfg["aug_type"] = cli.aug_type
    if cli.sig_type is not None:
        cfg["sig_type"] = cli.sig_type
    if cli.sig_full is not None:
        cfg["sig_full"] = cli.sig_full
    if cli.sig_depth is not None:
        cfg["sig_depth"] = cli.sig_depth
    if cli.epochs is not None:
        cfg["epochs"] = cli.epochs
    if cli.save_every is not None:
        cfg["save_every"] = cli.save_every
    if cli.save_path is not None:
        cfg["save_path"] = cli.save_path
        cfg.pop("output_root", None)
    if cli.output_root is not None:
        cfg["output_root"] = cli.output_root
        cfg.pop("save_path", None)
    if cli.project is not None:
        cfg["project"] = cli.project
    if cli.seed is not None:
        cfg["seed"] = cli.seed
    if cli.resume_from is not None:
        cfg["resume_from"] = cli.resume_from
    return cfg


def main() -> None:
    parser = build_parser()
    cli = parser.parse_args()
    cfg = yaml.safe_load(Path(resolve_repo_path(cli.config)).read_text())
    if "training_setting" in cfg:
        cfg.setdefault("project", cfg.pop("training_setting"))
    cfg = merge_cli_overrides(cfg, cli)

    if cfg.get("data_path") is not None:
        cfg["data_path"] = str(resolve_repo_path(cfg["data_path"]))
    legacy_path = cfg.get("save_path") if cfg.get("output_root") is None else None
    if legacy_path is not None:
        legacy_path = resolve_repo_path(legacy_path)
        cfg["output_root"] = str(legacy_path.parent)
        cfg.setdefault("project", legacy_path.name)
    cfg["output_root"] = str(resolve_repo_path(cfg.get("output_root", "_checkpoints")))
    cfg["project"] = validate_project_name(
        cfg.get("project") or default_project_name(cfg)
    )
    cfg["save_path"] = str(Path(cfg["output_root"]) / cfg["project"])
    if cfg.get("resume_from") is not None:
        cfg["resume_from"] = str(resolve_repo_path(cfg["resume_from"]))

    if cfg.get("sig_depth") is None:
        parser.error("sig_depth must be set explicitly in the config or via --sig-depth")

    args = argparse.Namespace(**cfg)
    run_training(args)


if __name__ == "__main__":
    main()
