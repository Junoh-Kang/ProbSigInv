"""Training-side preprocessing for raw path/signature datasets."""

from __future__ import annotations

from dataclasses import dataclass

from probsiginv.dataset.dataset import PathSigDataset, TensorDataset
from probsiginv.utils.conditioning import build_conditioning_from_signatures, resolve_effective_sig_depth
from probsiginv.utils.path_shapes import as_model_paths
from probsiginv.utils.transforms import apply_path_transform


@dataclass
class PreparedTrainingTensorDataset:
    """Preprocessed tensors and derived metadata for training."""

    tensor_dataset: TensorDataset
    effective_sig_depth: int
    cond_dim: int


def build_training_tensor_dataset(
    raw: PathSigDataset,
    *,
    sig_depth: int,
    sig_transform: str | None = None,
    path_transform: str | None = None,
    level_indices: dict[int, list[int]] | None = None,
) -> PreparedTrainingTensorDataset:
    """Build training tensors from a raw PathSigDataset plus preprocessing policy."""
    paths = as_model_paths(apply_path_transform(raw.paths, path_transform))
    effective_depth = resolve_effective_sig_depth(sig_depth, int(raw.sig_depth))
    sigs, _, _ = build_conditioning_from_signatures(
        raw.sigs,
        aug_type=raw.aug_type,
        sig_type=raw.sig_type,
        sig_depth=effective_depth,
        sig_transform=sig_transform,
        level_indices=level_indices,
        sig_channels=int(raw.sig_channels),
    )
    tensor_dataset = TensorDataset(paths, sigs)
    return PreparedTrainingTensorDataset(
        tensor_dataset=tensor_dataset,
        effective_sig_depth=effective_depth,
        cond_dim=int(sigs.shape[1]),
    )
