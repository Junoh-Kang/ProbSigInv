"""CLI wrapper for dataset building.

Simulated processes are built by probsiginv.dataset.build_dataset. Finance
configs (process: finance) name their builder with source: sp500 | fx.
"""

from __future__ import annotations

import sys
from typing import Callable

import yaml

from probsiginv.dataset.build_dataset import main as build_dataset_main
from probsiginv.dataset.fx_data import main as fx_data_main
from probsiginv.dataset.sp500_data import main as sp500_data_main
from probsiginv.repo import resolve_repo_path


FINANCE_BUILDERS = {"sp500": sp500_data_main, "fx": fx_data_main}


def _select_builder(config_path: str) -> Callable[[], None]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    if cfg.get("process") != "finance":
        return build_dataset_main
    source = cfg.get("source")
    if source not in FINANCE_BUILDERS:
        raise SystemExit(
            f"{config_path}: process: finance requires source: {' | '.join(FINANCE_BUILDERS)}, got {source!r}"
        )
    return FINANCE_BUILDERS[source]


def main() -> None:
    argv = sys.argv[:]
    try:
        builder = build_dataset_main
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            sys.argv[1] = str(resolve_repo_path(sys.argv[1]))
            builder = _select_builder(sys.argv[1])
        builder()
    finally:
        sys.argv[:] = argv


if __name__ == "__main__":
    main()
