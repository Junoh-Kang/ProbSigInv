"""Helpers for resolving repo-root-relative paths."""

from __future__ import annotations

from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    return package_root().parent


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return repo_root() / resolved


def repo_relative_path(path: str | Path) -> Path:
    resolved = resolve_repo_path(path)
    try:
        return resolved.relative_to(repo_root())
    except ValueError:
        return Path(path)


__all__ = [
    "package_root",
    "repo_relative_path",
    "repo_root",
    "resolve_repo_path",
]
