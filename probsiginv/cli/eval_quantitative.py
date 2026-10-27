"""CLI entrypoint for quantitative evaluation summaries."""

from __future__ import annotations

import argparse

from probsiginv.evaluation.candidate_selection import resolve_candidates
from probsiginv.evaluation.data import load_dataset_artifacts
from probsiginv.evaluation.quantitative import (
    DEFAULT_ORACLE_SEEDS,
    DEFAULT_PATH_MSE_BOOTSTRAP_CI_LEVEL,
    DEFAULT_PATH_MSE_BOOTSTRAP_RESAMPLES,
    DEFAULT_PATH_MSE_BOOTSTRAP_SEED,
    DEFAULT_SIG_SPACE_KEYS,
    ORACLE_PRIOR_KEYS,
    SUPPORTED_METRICS,
    default_metrics_for_process,
    detect_process,
    write_quantitative_results_for_dataset,
)
from probsiginv.repo import resolve_repo_path


def _resolve_metric_names(metrics: list[str] | None, *, process: str) -> tuple[str, ...]:
    selected = list(metrics or [])
    if not selected:
        selected = list(default_metrics_for_process(process))
    return tuple(dict.fromkeys(selected))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replace candidate quant.json files and refresh results/quant/summary.csv.")
    parser.add_argument("dataset_dir", help="Dataset directory containing real.npz")
    parser.add_argument("--candidate", nargs="+", default=None, help="Candidate key selectors")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=SUPPORTED_METRICS,
        default=None,
        metavar="METRIC",
        help="Metrics replacing each selected candidate quant.json (no merge). Oracle is stored separately. Defaults are process-specific; sig-mse uses TAd6 unless overridden.",
    )
    parser.add_argument(
        "--sig-space",
        nargs="+",
        default=None,
        metavar="SPACE",
        help="Atomic signature spaces for sig-mse, e.g. LSd2 or TAd3. Defaults to TAd6 when sig-mse is selected.",
    )
    parser.add_argument(
        "--energy-reference",
        default=None,
        metavar="PATHS_NPZ",
        help="GT posterior sample artifact required by energy-distance.",
    )
    parser.add_argument(
        "--bootstrap-mse",
        action="store_true",
        help="Attach bootstrap uncertainty to generative candidates in quant.json.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_PATH_MSE_BOOTSTRAP_RESAMPLES,
        help="Number of path-MSE bootstrap resamples when --bootstrap-mse is set.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_PATH_MSE_BOOTSTRAP_SEED,
        help="Random seed for path-MSE bootstrap resampling.",
    )
    parser.add_argument(
        "--bootstrap-ci-level",
        type=float,
        default=DEFAULT_PATH_MSE_BOOTSTRAP_CI_LEVEL,
        help="Percentile CI level for path-MSE bootstrap when --bootstrap-mse is set.",
    )
    oracle = parser.add_argument_group("oracle", "Training prior of the LS oracle (uniform bounds)")
    oracle.add_argument("--mu-bounds", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"), help="Prior bounds for mu")
    oracle.add_argument("--sigma-bounds", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"), help="Prior bounds for sigma")
    oracle.add_argument("--hurst-bounds", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"), help="Prior bounds for fBM H")
    oracle.add_argument("--theta-bounds", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"), help="Prior bounds for OU theta")
    oracle.add_argument(
        "--oracle-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_ORACLE_SEEDS),
        help="Seeds of the numerical fBM / OU oracle; the reported value is their mean.",
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
    artifacts = load_dataset_artifacts(dataset_dir)
    process = detect_process(artifacts)

    metric_names = _resolve_metric_names(args.metrics, process=process)
    selected_metrics = set(metric_names)
    if args.sig_space and "sig-mse" not in selected_metrics:
        parser.error("--sig-space requires sig-mse")
    if "energy-distance" in selected_metrics and args.energy_reference is None:
        parser.error("energy-distance requires --energy-reference")
    if "energy-distance" not in selected_metrics and args.energy_reference is not None:
        parser.error("--energy-reference requires energy-distance")
    if args.bootstrap_mse and "mse" not in selected_metrics:
        parser.error("--bootstrap-mse requires mse")
    if args.bootstrap_mse and args.bootstrap_resamples < 1:
        parser.error("--bootstrap-resamples must be >= 1")
    if args.bootstrap_mse and not 0.0 < args.bootstrap_ci_level < 1.0:
        parser.error("--bootstrap-ci-level must satisfy 0 < value < 1")
    oracle_priors = {
        key: tuple(getattr(args, f"{key}_bounds"))
        for key in ("mu", "sigma", "hurst", "theta")
        if getattr(args, f"{key}_bounds") is not None
    }
    if oracle_priors and "oracle" not in selected_metrics:
        parser.error("prior bounds are only used by the oracle metric")
    oracle_needs_priors = process in ORACLE_PRIOR_KEYS and int(artifacts.get("path_dim", 1)) == 1
    if "oracle" in selected_metrics and oracle_needs_priors:
        missing = [key for key in ORACLE_PRIOR_KEYS[process] if key not in oracle_priors]
        if missing:
            flags = " ".join(f"--{key}-bounds" for key in missing)
            parser.error(f"oracle for process={process} requires {flags}")
    sig_space_keys = tuple(args.sig_space or DEFAULT_SIG_SPACE_KEYS) if "sig-mse" in selected_metrics else ()
    energy_reference = resolve_repo_path(args.energy_reference) if args.energy_reference is not None else None
    if energy_reference is not None and not energy_reference.is_file():
        parser.error(f"--energy-reference must be an existing NPZ file: {energy_reference}")

    try:
        candidates = resolve_candidates(dataset_dir, selectors=args.candidate)
        written = write_quantitative_results_for_dataset(
            dataset_dir,
            candidates=candidates,
            metric_names=metric_names,
            sig_space_keys=sig_space_keys,
            bootstrap_mse=args.bootstrap_mse,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_ci_level=args.bootstrap_ci_level,
            energy_reference=energy_reference,
            oracle_priors=oracle_priors,
            oracle_seeds=tuple(args.oracle_seeds),
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(f"{dataset_dir} [process={process}] -> {written}")


if __name__ == "__main__":
    main()
