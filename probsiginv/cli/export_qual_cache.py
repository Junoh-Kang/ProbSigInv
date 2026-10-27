"""CLI entrypoint for reusable qualitative figure cache artifacts."""

from __future__ import annotations

import argparse

from probsiginv.evaluation.candidate_selection import resolve_candidates
from probsiginv.evaluation.qualitative import (
    DEFAULT_BASELINE_ORDER,
    DEFAULT_N_GEN,
    DEFAULT_N_SHOW,
    DEFAULT_OVERLAY_RANDOM_SEED,
    write_qualitative_cache_for_dataset,
)
from probsiginv.repo import resolve_repo_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write reusable qualitative cache artifacts.")
    parser.add_argument("dataset_dir", help="Inversion output dataset directory containing real.npz")
    parser.add_argument("--candidate", nargs="+", default=None, help="Candidate key selectors")
    parser.add_argument(
        "--out",
        default=None,
        help="Output cache directory. Defaults to <dataset_dir>/results/qual_cache.",
    )
    parser.add_argument("--n-show", type=int, default=DEFAULT_N_SHOW, help="Number of real paths to cache")
    parser.add_argument("--n-gen", type=int, default=DEFAULT_N_GEN, help="Number of candidate samples to cache for paths")
    parser.add_argument(
        "--overlay-seed",
        type=int,
        default=DEFAULT_OVERLAY_RANDOM_SEED,
        help="Seed used for deterministic path sample selection",
    )
    parser.add_argument(
        "--baseline-order",
        nargs="+",
        default=list(DEFAULT_BASELINE_ORDER),
        metavar="KEY",
        help="Deterministic baseline ordering recorded in the cache manifest",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_dir = resolve_repo_path(args.dataset_dir)
    if not dataset_dir.exists():
        parser.error(f"dataset_dir does not exist: {dataset_dir}")
    if not (dataset_dir / "real.npz").exists():
        parser.error(f"dataset_dir must contain real.npz: {dataset_dir}")

    out_dir = resolve_repo_path(args.out) if args.out is not None else None
    try:
        candidates = resolve_candidates(dataset_dir, selectors=args.candidate)
        written = write_qualitative_cache_for_dataset(
            dataset_dir,
            candidates=candidates,
            out_dir=out_dir,
            n_show=args.n_show,
            n_gen=args.n_gen,
            overlay_seed=args.overlay_seed,
            baseline_order=tuple(args.baseline_order),
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(f"{dataset_dir} -> {written}")


if __name__ == "__main__":
    main()
