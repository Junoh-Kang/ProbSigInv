"""Training runtime orchestration."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from probsiginv.dataset.dataset import PathSigDataset
from probsiginv.dataset.preprocessing import build_training_tensor_dataset
from probsiginv.models import build_model
from probsiginv.training.checkpoints import (
    build_training_checkpoint,
    load_training_checkpoint,
    resolve_training_checkpoint_layout,
    restore_training_state,
    save_training_checkpoint,
    should_save_checkpoint,
)
from probsiginv.training.processes import build_process, normalize_training_method
from probsiginv.utils.conditioning import resolve_data_aug_type
from probsiginv.utils.contract import build_split_sigs_path
from probsiginv.utils.signature_config import load_level_indices


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.995):
        self.ema_model = copy.deepcopy(model)
        self.decay = decay
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)


def _set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _model_path_dim(paths) -> int:
    return 1 if paths.ndim == 2 else int(paths.shape[1])


def compute_checkpoint_monitor_metrics(
    *,
    process,
    model: torch.nn.Module,
    paths: torch.Tensor,
    sigs: torch.Tensor,
    sampling_steps: int,
    n_samples: int,
    device: torch.device,
) -> dict[str, dict[str, float] | None]:
    """Compute compact monitor metrics for a saved checkpoint."""
    n_real = int(paths.shape[0])
    seq_len = int(paths.shape[-1])
    path_dim = _model_path_dim(paths)
    cond = sigs.to(device=device).repeat_interleave(n_samples, dim=0)
    with torch.no_grad():
        gen_paths = process.sample_loop(
            model,
            shape=(n_real * n_samples, seq_len) if path_dim == 1 and paths.ndim == 2 else (n_real * n_samples, path_dim, seq_len),
            c=cond,
            sampling_steps=sampling_steps,
        ).cpu()
    if gen_paths.ndim == 2:
        gen_paths = gen_paths.view(n_real, n_samples, seq_len)
    else:
        gen_paths = gen_paths.view(n_real, n_samples, path_dim, seq_len)

    real_paths = paths.cpu()
    diff = real_paths[:, None, :] - gen_paths
    reduce_dims = (-1,) if diff.ndim == 3 else (-1, -2)
    mse_per_real = diff.pow(2).mean(dim=reduce_dims).mean(dim=1)
    path_mse = {
        "mean": float(mse_per_real.mean().item()),
        "std": float(mse_per_real.std(unbiased=True).item()) if n_real > 1 else 0.0,
    }

    spread_mse = None
    if n_samples >= 2:
        pairwise = []
        for left in range(n_samples - 1):
            diff = gen_paths[:, left + 1 :, :] - gen_paths[:, left : left + 1, :]
            reduce_dims = (-1,) if diff.ndim == 3 else (-1, -2)
            pairwise.append(diff.pow(2).mean(dim=reduce_dims))
        if pairwise:
            per_real = torch.cat(pairwise, dim=1).mean(dim=1)
            spread_mse = {
                "mean": float(per_real.mean().item()),
                "std": float(per_real.std(unbiased=True).item()) if n_real > 1 else 0.0,
            }

    return {
        "path_mse": path_mse,
        "spread_mse": spread_mse,
    }


def _write_monitor_metrics_record(
    report_path: str | Path,
    *,
    epoch: int,
    checkpoint_path: str | Path,
    metrics: dict[str, dict[str, float] | None],
) -> None:
    """Append one compact checkpoint monitor record as JSONL."""
    record = {
        "epoch": int(epoch),
        "checkpoint_path": str(checkpoint_path),
        "path_mse": metrics["path_mse"],
        "spread_mse": metrics["spread_mse"],
    }
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record))
        handle.write("\n")


def _format_monitor_metric(name: str, stats: dict[str, float] | None) -> str:
    """Format one compact monitor metric for console output."""
    if stats is None:
        return f"{name}=None"
    return f"{name}(mean={stats['mean']:.6f}, std={stats['std']:.6f})"


def _print_monitor_metrics(*, epoch: int, metrics: dict[str, dict[str, float] | None]) -> None:
    """Print checkpoint monitor metrics in the training log."""
    path_mse = _format_monitor_metric("path_mse", metrics["path_mse"])
    spread_mse = _format_monitor_metric("spread_mse", metrics["spread_mse"])
    print(f"  Monitor epoch {epoch:04d} | {path_mse}, {spread_mse}")


def _prepare_training_data(args):
    aug_type = getattr(args, "aug_type", "TA")
    sig_type = str(getattr(args, "sig_type", "logsig"))
    if sig_type != "logsig":
        raise ValueError(f'Only sig_type="logsig" is supported, got {sig_type!r}')

    data_aug_type = resolve_data_aug_type(aug_type)
    train_sig_path = build_split_sigs_path(args.data_path, "train", data_aug_type, sig_type)
    train_raw = PathSigDataset(str(train_sig_path))

    sig_depth = getattr(args, "sig_depth", None)
    if sig_depth is None:
        raise ValueError("sig_depth must be set explicitly in the config or via --sig-depth")
    sig_depth = int(sig_depth)

    sig_transform = getattr(args, "sig_transform", None)
    path_transform = getattr(args, "path_transform", "raw")
    sig_full = getattr(args, "sig_full", False)
    training_method = normalize_training_method(getattr(args, "training_method", "ddpm"))

    level_idx = None if sig_depth == 0 else load_level_indices(
        aug_type,
        sig_type,
        full=sig_full,
        min_depth=sig_depth,
        channels=int(train_raw.sig_channels),
    )
    prepared_train = build_training_tensor_dataset(
        train_raw,
        sig_depth=sig_depth,
        sig_transform=sig_transform,
        path_transform=path_transform,
        level_indices=level_idx,
    )
    train_ds = prepared_train.tensor_dataset
    effective_sig_depth = prepared_train.effective_sig_depth
    monitor_sig_path = build_split_sigs_path(args.data_path, "val", data_aug_type, sig_type)
    if monitor_sig_path.exists():
        monitor_raw = PathSigDataset(str(monitor_sig_path))
        prepared_monitor = build_training_tensor_dataset(
            monitor_raw,
            sig_depth=effective_sig_depth,
            sig_transform=sig_transform,
            path_transform=path_transform,
            level_indices=level_idx,
        )
        monitor_ds = prepared_monitor.tensor_dataset
        if int(monitor_ds.sigs.shape[1]) != int(prepared_train.cond_dim):
            raise ValueError(
                f"monitor cond_dim {monitor_ds.sigs.shape[1]} does not match training cond_dim {prepared_train.cond_dim}"
            )
        if int(monitor_ds.paths.shape[1]) != int(train_ds.paths.shape[1]):
            raise ValueError(
                f"monitor path_dim {monitor_ds.paths.shape[1]} does not match training path_dim {train_ds.paths.shape[1]}"
            )
        if int(monitor_ds.paths.shape[-1]) != int(train_ds.paths.shape[-1]):
            raise ValueError(
                f"monitor seq_len {monitor_ds.paths.shape[-1]} does not match training seq_len {train_ds.paths.shape[-1]}"
            )
        monitor_source = "val"
    else:
        monitor_ds = train_ds
        monitor_source = "train"

    print(
        f"Dataset: {train_sig_path.parent}  "
        f"(train={len(train_ds)}, "
        f"monitor={len(monitor_ds)}[{monitor_source}], "
        f"aug={aug_type}, data_aug={data_aug_type}, sig_type={sig_type}, sig_depth={effective_sig_depth}, "
        f"sig_channels={train_raw.sig_channels}, path_dim={_model_path_dim(train_ds.paths)}, cond_dim={prepared_train.cond_dim}, "
        f"sig_full={sig_full}, sig_transform={sig_transform}, "
        f"path_transform={path_transform}, training_method={training_method})"
    )

    return {
        "aug_type": aug_type,
        "sig_type": sig_type,
        "data_aug_type": data_aug_type,
        "sig_dir": train_sig_path.parent,
        "train_raw": train_raw,
        "train_ds": train_ds,
        "monitor_ds": monitor_ds,
        "monitor_source": monitor_source,
        "sig_depth": effective_sig_depth,
        "sig_transform": sig_transform,
        "path_transform": path_transform,
        "sig_full": sig_full,
        "training_method": training_method,
    }


def run_training(args) -> None:
    """Run the end-to-end training pipeline."""
    if getattr(args, "seed", None) is not None:
        _set_training_seed(int(args.seed))
        print(f"Seed: {int(args.seed)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = _prepare_training_data(args)

    train_loader = DataLoader(
        data["train_ds"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    seq_len = data["train_ds"].paths.shape[-1]
    path_dim = _model_path_dim(data["train_ds"].paths)
    cond_dim = data["train_ds"].sigs.shape[1]
    arch = getattr(args, "arch", "unet")
    training_method = data["training_method"]
    model = build_model(
        arch=arch,
        seq_len=seq_len,
        cond_dim=cond_dim,
        path_dim=path_dim,
        channels=getattr(args, "channels", [32, 64, 128, 128]),
        num_res_blocks=getattr(args, "num_res_blocks", 2),
        emb_dim=getattr(args, "emb_dim", 128),
        patch_size=getattr(args, "patch_size", 8),
        d_model=getattr(args, "d_model", 256),
        n_heads=getattr(args, "n_heads", 8),
        n_layers=getattr(args, "n_layers", 6),
        final_deblock=(arch == "dit" and training_method == "regression"),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}  (seq_len={seq_len}, path_dim={path_dim}, cond_dim={cond_dim})")

    diffusion_steps = int(getattr(args, "diffusion_steps", 1000))
    process = build_process(
        data["training_method"],
        device=device,
        diffusion_steps=diffusion_steps,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema = EMA(model, decay=args.ema_decay)

    layout = resolve_training_checkpoint_layout(
        save_path=args.save_path,
        data_path=args.data_path,
        path_transform=data["path_transform"],
        arch=arch,
        aug_type=data["aug_type"],
        sig_depth=data["sig_depth"],
        sig_transform=data["sig_transform"],
        sig_full=data["sig_full"],
        training_method=data["training_method"],
    )
    print(f"Checkpoint dir: {layout['ckpt_dir']}")

    save_every = int(args.save_every)
    start_epoch = 1
    restored_epoch: int | None = None
    resume_from = getattr(args, "resume_from", None)
    if resume_from is not None:
        checkpoint = load_training_checkpoint(resume_from)
        restored_epoch = restore_training_state(
            checkpoint,
            model=model,
            ema_model=ema.ema_model,
            optimizer=optimizer,
        )
        start_epoch = restored_epoch + 1
        print(f"Resumed from {resume_from} at epoch {restored_epoch:04d}")
        if checkpoint.get("rng_state") is not None:
            print("Restored RNG state from checkpoint.")

    monitor_n_paths = int(getattr(args, "monitor_n_paths", 0))
    monitor_n_samples = int(getattr(args, "monitor_n_samples", 30))
    monitor_sampling_steps = int(getattr(args, "monitor_sampling_steps", diffusion_steps))
    monitor_report_path = getattr(args, "monitor_report_path", None)
    monitor_enabled = monitor_n_paths > 0 and monitor_n_samples > 0
    if data["training_method"] == "regression" and monitor_enabled:
        print("Checkpoint monitor disabled for training_method=regression; skipping monitor metrics.")
        monitor_enabled = False
    if monitor_enabled and monitor_report_path is None:
        monitor_report_path = Path(layout["exp_path"]) / "monitor_metrics.jsonl"
    if monitor_enabled and monitor_n_paths > 0 and monitor_n_samples < 2:
        raise ValueError("monitor_n_samples must be >= 2 when monitor reporting is enabled")

    if restored_epoch is not None and monitor_enabled:
        n_paths = min(monitor_n_paths, len(data["monitor_ds"]))
        metrics = compute_checkpoint_monitor_metrics(
            process=process,
            model=ema.ema_model,
            paths=data["monitor_ds"].paths[:n_paths],
            sigs=data["monitor_ds"].sigs[:n_paths],
            sampling_steps=monitor_sampling_steps,
            n_samples=monitor_n_samples,
            device=device,
        )
        _write_monitor_metrics_record(
            monitor_report_path,
            epoch=restored_epoch,
            checkpoint_path=resume_from,
            metrics=metrics,
        )
        _print_monitor_metrics(epoch=restored_epoch, metrics=metrics)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for paths, sigs in train_loader:
            paths = paths.to(device)
            sigs = sigs.to(device)

            loss = process.training_loss(model, paths, sigs)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)

            train_loss_sum += loss.item() * paths.shape[0]
            train_count += paths.shape[0]

        train_loss = train_loss_sum / train_count
        print(f"Epoch {epoch:04d} | Train Loss: {train_loss:.6f}")

        if should_save_checkpoint(epoch, int(args.epochs), save_every):
            checkpoint = build_training_checkpoint(
                args=args,
                epoch=epoch,
                model=model,
                ema_model=ema.ema_model,
                optimizer=optimizer,
                train_loss=train_loss,
                sig_depth=data["sig_depth"],
                sig_full=data["sig_full"],
                training_method=data["training_method"],
                path_dim=path_dim,
                sig_channels=int(data["train_raw"].sig_channels),
            )
            ckpt_path = save_training_checkpoint(layout["ckpt_dir"], checkpoint, epoch)
            print(f"  saved -> {ckpt_path}")
            if monitor_enabled:
                n_paths = min(monitor_n_paths, len(data["monitor_ds"]))
                metrics = compute_checkpoint_monitor_metrics(
                    process=process,
                    model=ema.ema_model,
                    paths=data["monitor_ds"].paths[:n_paths],
                    sigs=data["monitor_ds"].sigs[:n_paths],
                    sampling_steps=monitor_sampling_steps,
                    n_samples=monitor_n_samples,
                    device=device,
                )
                _write_monitor_metrics_record(
                    monitor_report_path,
                    epoch=epoch,
                    checkpoint_path=ckpt_path,
                    metrics=metrics,
                )
                _print_monitor_metrics(epoch=epoch, metrics=metrics)
