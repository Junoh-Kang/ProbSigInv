from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from probsiginv.utils.contract import OUTPUT_ARTIFACT_NAME, load_candidate_contract


@dataclass(frozen=True)
class CandidateArtifact:
    key: str
    artifact_path: Path
    kind: str
    method: str
    conditioning: str | None = None
    source_artifact: str | None = None
    aug_type: str | None = None
    depth: int | None = None


def _normalize_key(value: str) -> str:
    return value.strip().casefold()


def _build_candidate(artifact_path: Path) -> CandidateArtifact:
    contract = load_candidate_contract(artifact_path)
    return CandidateArtifact(
        key=contract.key,
        artifact_path=artifact_path,
        kind=contract.kind,
        method=contract.method,
        conditioning=contract.conditioning,
        source_artifact=contract.source_artifact,
        aug_type=contract.aug_type,
        depth=contract.depth,
    )


def discover_candidates(data_dir: str | Path) -> list[CandidateArtifact]:
    root = Path(data_dir)
    candidates = [
        _build_candidate(path)
        for path in sorted(root.glob(f"**/{OUTPUT_ARTIFACT_NAME}"))
        if "results" not in path.relative_to(root).parts
    ]

    if not candidates:
        raise FileNotFoundError(f"No evaluation candidates found in {root}")
    return candidates


def resolve_candidates(
    data_dir: str | Path,
    selectors: list[str] | None = None,
) -> list[CandidateArtifact]:
    discovered = discover_candidates(data_dir)
    by_key: dict[str, list[CandidateArtifact]] = {}
    for candidate in discovered:
        by_key.setdefault(_normalize_key(candidate.key), []).append(candidate)

    normalized_selectors = [_normalize_key(selector) for selector in selectors or []]
    if not normalized_selectors:
        normalized_selectors = sorted(
            normalized_key
            for normalized_key, matches in by_key.items()
            if all(candidate.kind != "reference" for candidate in matches)
        )
        if not normalized_selectors:
            raise FileNotFoundError(f"No default evaluation candidates found in {Path(data_dir)}")

    resolved: list[CandidateArtifact] = []
    seen: set[str] = set()
    for selector in normalized_selectors:
        matches = by_key.get(selector, [])
        if not matches:
            raise ValueError(f"No match for candidate selector: {selector}")
        if len(matches) > 1:
            raise ValueError(
                f"Multiple candidates match selector '{selector}': "
                + ", ".join(sorted(candidate.artifact_path.parent.name for candidate in matches))
            )
        if selector in seen:
            continue
        seen.add(selector)
        resolved.append(matches[0])

    return sorted(resolved, key=lambda candidate: candidate.key.casefold())
