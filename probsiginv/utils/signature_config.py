"""Signature-config loading helpers."""

from __future__ import annotations

import re
import warnings

import yaml

from probsiginv.repo import repo_root


def _full_level_indices(channels: int, depth: int) -> dict[int, list[int]]:
    import iisignature

    level_idx: dict[int, list[int]] = {}
    prev = 0
    for level in range(1, depth + 1):
        curr = iisignature.logsiglength(channels, level)
        level_idx[level] = list(range(prev, curr))
        prev = curr
    return level_idx


def _basis_word(expr: str) -> tuple[int, ...]:
    return tuple(int(token) for token in re.findall(r"\d+", expr))


def linear_time_level_indices(channels: int, depth: int) -> dict[int, list[int]]:
    """Return LS indices for time-augmented logsigs with one spatial terminal."""
    import iisignature

    channels = int(channels)
    depth = int(depth)
    if depth <= 0:
        return {}
    if channels < 2:
        raise ValueError(f"LS indices require at least 2 channels, got {channels}")

    targets = {
        level: {
            tuple([1] * (level - 1) + [spatial])
            for spatial in range(2, channels + 1)
        }
        for level in range(1, depth + 1)
    }
    level_idx: dict[int, list[int]] = {level: [] for level in range(1, depth + 1)}

    basis = iisignature.basis(iisignature.prepare(channels, depth))
    for idx, expr in enumerate(basis):
        word = _basis_word(str(expr))
        level = len(word)
        if 1 <= level <= depth and word in targets[level]:
            level_idx[level].append(idx)

    missing = {
        level: len(targets[level]) - len(level_idx[level])
        for level in range(1, depth + 1)
        if len(level_idx[level]) != len(targets[level])
    }
    if missing:
        raise RuntimeError(
            f"Could not synthesize LS level indices for channels={channels}, depth={depth}: missing={missing}"
        )
    return level_idx


def load_level_indices(
    aug_type: str,
    sig_type: str,
    full: bool = False,
    *,
    min_depth: int | None = None,
    channels: int | None = None,
) -> dict[int, list[int]]:
    """Load level indices from configs/signature/{aug_type}_{sig_type}[_full].yaml."""
    if full:
        if channels is None:
            path = repo_root() / "configs" / "signature" / f"{aug_type}_{sig_type}_full.yaml"
            if not path.exists():
                raise FileNotFoundError(f"Signature config not found: {path}")
            data = yaml.safe_load(path.read_text())
            channels = int(data["channels"])
        depth = int(min_depth) if min_depth is not None else 0
        if depth <= 0:
            depth = 1
        return _full_level_indices(channels=int(channels), depth=depth)

    config_dir = repo_root() / "configs"
    path = config_dir / "signature" / f"{aug_type}_{sig_type}.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Signature config not found: {path}")
    data = yaml.safe_load(path.read_text())
    config_channels = int(data["channels"])
    level_indices = {int(k): v for k, v in data["level_indices"].items()}
    if channels is not None and int(channels) != config_channels:
        depth = int(min_depth) if min_depth is not None else data.get("max_depth", max(level_indices, default=1))
        if depth <= 0:
            depth = 1
        if aug_type == "LS":
            return linear_time_level_indices(channels=int(channels), depth=depth)

        # TA has no pruned multidim YAML; synthesize full indices for multidim TA.
        warnings.warn(
            f"{path}: pruned config has channels={config_channels} but requested sig_channels={int(channels)}; "
            f"falling back to full level indices at depth={depth}",
            stacklevel=2,
        )
        return _full_level_indices(channels=int(channels), depth=depth)
    if min_depth is None:
        return level_indices

    target_depth = int(min_depth)
    if target_depth <= 0:
        return {}

    current_depth = max(level_indices, default=0)
    if aug_type == "TA":
        return _full_level_indices(channels=config_channels, depth=max(current_depth, target_depth))

    if target_depth <= current_depth:
        return level_indices

    raise ValueError(
        f"{path}: requested depth {target_depth} exceeds configured max_depth {current_depth}"
    )


__all__ = ["linear_time_level_indices", "load_level_indices"]
