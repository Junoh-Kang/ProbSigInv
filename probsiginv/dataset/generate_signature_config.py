"""Generate signature-dimension config YAMLs for analytic subsampling.

Loads precomputed log-signatures, runs level-by-level forward selection in
column index order to find independent indices, and saves the result under
configs/signature/.

level_indices[k] lists the NEW independent indices contributed by level k.
To get all independent indices for depth d: union of level_indices[1..d].

Usage
-----
    python -m probsiginv.cli.generate_signature_config \\
        --data _data/gbm_mu1.5-2.5_sig1.5-2.5 \\
        --aug-type TLL3

    python -m probsiginv.cli.generate_signature_config \\
        --data _data/gbm_mu1.5-2.5_sig1.5-2.5 \\
        --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from probsiginv.repo import repo_root
from probsiginv.utils.conditioning import resolve_data_aug_type
from probsiginv.utils.contract import build_split_sigs_path
from probsiginv.utils.signature_config import linear_time_level_indices

import iisignature
import numpy as np
import yaml

ROOT = repo_root()
CONFIG_DIR = ROOT / "configs" / "signature"


AUG_TYPES = ["TA", "TLL3", "TLL"]
SIG_TYPE  = "logsig"


def ta_linear_indices(max_depth):
    # default basis: Lyndon. Do NOT add "H" here.
    s = iisignature.prepare(2, max_depth)
    basis = iisignature.basis(s)
    idx_by_level = {
        level: indices[0]
        for level, indices in linear_time_level_indices(channels=2, depth=max_depth).items()
    }
    return s, basis, idx_by_level



def load_logsigs(data_dir: Path, aug_type: str, split: str = "train") -> tuple[np.ndarray, int, int]:
    """Load precomputed log-signatures from the canonical split artifact layout."""
    npz_path = build_split_sigs_path(data_dir, split, resolve_data_aug_type(aug_type), SIG_TYPE)
    d        = np.load(npz_path)
    return d["sigs"], int(d["sig_depth"]), int(d["sig_channels"])


def forward_selection(
    logsigs: np.ndarray,
    channels: int,
    depth: int,
    tol: float = 1e-12,
) -> dict[int, list[int]]:
    """Level-by-level forward selection in column index order.

    Processes columns level by level (k = 1, …, depth).  Within each
    level columns are checked in index order against Q (the orthonormal
    basis built from all previously accepted columns, i.e. levels 1..k-1
    plus earlier accepted columns in level k).  Once a level is done its
    selections are fixed into Q before moving to the next level.

    Satisfies:
      - prefer lower index when two columns are redundant
      - selected(depth k) ⊆ selected(depth k+1)   [subset property]
      - cross-level dependencies are handled correctly
    """
    n = logsigs.shape[0]
    D = iisignature.logsiglength(channels, depth)
    col_norms = np.linalg.norm(logsigs[:, :D], axis=0)
    Q = np.zeros((n, 0))
    level_idx: dict[int, list[int]] = {}

    for k in range(1, depth + 1):
        prev = iisignature.logsiglength(channels, k - 1) if k > 1 else 0
        curr = iisignature.logsiglength(channels, k)
        # Global scale: max column norm up to current depth (analogous to sv[0])
        global_scale = col_norms[:curr].max()
        selected_k = []

        for i in range(prev, curr):
            col = logsigs[:, i]
            if col_norms[i] == 0:
                continue
            # Two-pass (re-)orthogonalization for numerical stability
            residual = col.copy()
            if Q.shape[1] > 0:
                residual -= Q @ (Q.T @ residual)
                residual -= Q @ (Q.T @ residual)
            if np.linalg.norm(residual) > tol * global_scale:
                selected_k.append(i)
                q = residual / np.linalg.norm(residual)
                Q = np.column_stack([Q, q]) if Q.shape[1] > 0 else q[:, None]

        level_idx[k] = selected_k
        # Q is now fixed for levels 1..k

    return level_idx


def full_indices(channels: int, depth: int) -> dict[int, list[int]]:
    """Return all log-signature indices partitioned by level (no data needed)."""
    level_idx: dict[int, list[int]] = {}
    for k in range(1, depth + 1):
        prev = iisignature.logsiglength(channels, k - 1) if k > 1 else 0
        curr = iisignature.logsiglength(channels, k)
        level_idx[k] = list(range(prev, curr))
    return level_idx


def output_path(aug_type: str, sig_type: str, full: bool = False) -> Path:
    suffix = "_full" if full else ""
    return CONFIG_DIR / f"{aug_type}_{sig_type}{suffix}.yaml"


# Render level_indices lists in flow style: "1: [0, 1]"
class _FlowList(list): pass
def _flow_list_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
yaml.Dumper.add_representer(_FlowList, _flow_list_representer)


def save_yaml(
    level_idx: dict[int, list[int]],
    aug_type: str,
    sig_type: str,
    channels: int,
    depth: int,
    full: bool = False,
) -> Path:
    cfg = {
        "aug_type":      aug_type,
        "sig_type":      sig_type,
        "channels":      channels,
        "max_depth":     depth,
        "level_indices": {k: _FlowList(v) for k, v in level_idx.items()},
    }
    out = output_path(aug_type, sig_type, full=full)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        yaml.dump(cfg, fh, sort_keys=False, Dumper=yaml.Dumper)
    print(f"saved  {out}")
    return out


def generate(data_dir: Path, aug_type: str, sig_type: str = SIG_TYPE, tol: float = 1e-12) -> Path:
    logsigs, depth, channels = load_logsigs(data_dir, aug_type)
    level_idx = forward_selection(logsigs, channels, depth, tol)
    # ── sanity check (accumulated rank per depth) ─────────────────────────────
    print(f"\n{aug_type}  channels={channels}  depth={depth}")
    print(f"  {'depth':>5}  {'total_dim':>9}  {'accum_sel':>9}  {'full_rank':>9}  {'ok':>3}")
    accum = 0
    for k in range(1, depth + 1):
        accum += len(level_idx[k])
        curr  = iisignature.logsiglength(channels, k)
        sv    = np.linalg.svd(logsigs[:, :curr], compute_uv=False)
        frank = int(np.sum(sv > tol * sv[0]))
        ok    = (accum == frank)
        print(f"  {k:>5}  {curr:>9}  {accum:>9}  {frank:>9}  {'✓' if ok else '✗':>3}")

    # ── save pruned (forward selection) ──────────────────────────────────────
    out = save_yaml(level_idx, aug_type, sig_type, channels, depth, full=False)

    # ── save full (all indices, no data needed) ───────────────────────────────
    save_yaml(full_indices(channels, depth), aug_type, sig_type, channels, depth, full=True)

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     type=Path, required=True,
                        help="Dataset root, e.g. _data/gbm_mu1.5-2.5_sig1.5-2.5")
    parser.add_argument("--aug-type", choices=AUG_TYPES, default=None)
    parser.add_argument("--all",      action="store_true", help="Generate for all aug types.")
    parser.add_argument("--sig-type", default=SIG_TYPE, choices=[SIG_TYPE])
    parser.add_argument("--tol",      type=float, default=1e-12)
    args = parser.parse_args()

    data_dir = args.data.resolve()

    if args.all:
        for aug in AUG_TYPES:
            npz = build_split_sigs_path(data_dir, "train", resolve_data_aug_type(aug), args.sig_type)
            if npz.exists():
                generate(data_dir, aug, args.sig_type, args.tol)
            else:
                print(f"skip {aug}: {npz} not found")
    elif args.aug_type:
        generate(data_dir, args.aug_type, args.sig_type, args.tol)
    else:
        parser.error("specify --aug-type or --all")
