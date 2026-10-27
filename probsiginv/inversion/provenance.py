"""Identity checks before reusing an inversion result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_inversion_identity(args: argparse.Namespace, source_path: str) -> dict:
    source = Path(source_path)
    signatures = {
        path.name: file_sha256(path)
        for path in sorted(source.parent.glob("*_logsigs.npz"))
    }
    controls = (
        "method", "seed", "n_repeats", "sampling_steps", "eta", "clip_x0",
        "batch_size", "fourier_sig_depth", "legendre_sig_depth",
        "insertion_sig_depth", "n_coeffs", "mirror",
    )
    checkpoint = getattr(args, "ckpt", None)
    return {
        "version": 1,
        "source_sha256": file_sha256(source),
        "signature_sha256": signatures,
        "checkpoint_sha256": file_sha256(checkpoint) if checkpoint else None,
        "settings": {key: getattr(args, key, None) for key in controls},
    }


def validate_existing_candidate(candidate_path: str | Path, identity: dict) -> bool:
    candidate = Path(candidate_path)
    if not candidate.exists():
        return False
    manifest = candidate.parent / "inversion.json"
    if not manifest.exists():
        raise ValueError(
            f"Cannot verify existing result without {manifest}; "
            "use a different --project or --output-dir"
        )
    stored = json.loads(manifest.read_text())
    if stored != identity:
        changed = sorted(key for key in set(stored) | set(identity) if stored.get(key) != identity.get(key))
        raise ValueError(
            f"Existing inversion result has different {', '.join(changed)}: {candidate}; "
            "use a different --project or --output-dir"
        )
    return True


def save_inversion_identity(output_dir: str | Path, identity: dict) -> None:
    manifest = Path(output_dir) / "inversion.json"
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest)
