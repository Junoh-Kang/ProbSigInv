"""Compatibility utilities kept for legacy imports."""

from __future__ import annotations

import torch

from probsiginv.training.processes import build_process, normalize_training_method
from probsiginv.models import build_model
from probsiginv.utils.signature_config import load_level_indices
from probsiginv.utils.transforms import apply_path_transform, apply_sig_transform, invert_path_transform



def load_model_and_process(
    ckpt_path,
    seq_len,
    cond_dim,
    device,
    clip_x0=None,
    checkpoint_payload=None,
):
    """Load checkpoint and build model & process backend.

    All architecture hyperparameters are read exclusively from the checkpoint.
    """
    ckpt = checkpoint_payload
    if ckpt is None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})

    arch = saved_args.get("arch", "unet")
    diffusion_steps = saved_args.get("diffusion_steps", 1000)
    training_method = normalize_training_method(saved_args.get("training_method", "ddpm"))
    ema_state = ckpt.get("ema_model", {})
    final_deblock = bool(saved_args.get("final_deblock", False))
    if arch == "dit" and "out_deblock.weight" in ema_state:
        final_deblock = True

    model = build_model(
        arch=arch,
        seq_len=seq_len,
        cond_dim=cond_dim,
        path_dim=saved_args.get("path_dim", 1),
        channels=saved_args.get("channels", [32, 64, 128, 128]),
        num_res_blocks=saved_args.get("num_res_blocks", 2),
        emb_dim=saved_args.get("emb_dim", 128),
        patch_size=saved_args.get("patch_size", 8),
        d_model=saved_args.get("d_model", 256),
        n_heads=saved_args.get("n_heads", 8),
        n_layers=saved_args.get("n_layers", 6),
        final_deblock=final_deblock,
    ).to(device)
    model.load_state_dict(ema_state)
    model.eval()
    print(
        f"Loaded {arch} checkpoint from epoch {ckpt.get('epoch', '?')} "
        f"(training_method={training_method}, path_dim={int(saved_args.get('path_dim', 1))})"
    )

    process = build_process(
        training_method,
        device=device,
        diffusion_steps=diffusion_steps,
        clip_x0=clip_x0,
    )
    return model, process, ckpt



def load_model_and_diffusion(ckpt_path, seq_len, cond_dim, device, clip_x0=None):
    """Backward-compatible alias for loading model and process backend."""
    return load_model_and_process(ckpt_path, seq_len, cond_dim, device, clip_x0=clip_x0)


__all__ = [
    "apply_path_transform",
    "apply_sig_transform",
    "build_process",
    "invert_path_transform",
    "load_level_indices",
    "load_model_and_diffusion",
    "load_model_and_process",
    "normalize_training_method",
]
