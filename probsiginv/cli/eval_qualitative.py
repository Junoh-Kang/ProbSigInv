"""CLI entrypoint for qualitative evaluation artifacts."""

from __future__ import annotations

import argparse

from probsiginv.evaluation.candidate_selection import resolve_candidates
from probsiginv.evaluation.qualitative import (
    DEFAULT_N_GEN,
    DEFAULT_N_SHOW,
    SUPPORTED_QUALITATIVE_METRICS,
    write_qualitative_results_for_dataset,
)
from probsiginv.repo import resolve_repo_path

DEFAULT_PLOT_STYLE = {
    "real": {"color": "black", "linewidth": 1.8, "alpha": 1.0},
    "candidate": {"color": "tab:blue", "linewidth": 1.4, "alpha": 1.0},
    "candidate_sample": {"color": "tab:blue", "linewidth": 0.9, "alpha": 0.25},
    "reference": {"color": "0.45", "linewidth": 1.0, "alpha": 0.8, "linestyle": "--"},
    "baseline": {"linewidth": 1.4, "alpha": 1.0},
    "real_band": {"color": "black", "alpha": 0.08},
    "candidate_band": {"alpha": 0.15},
    "baseline_order": ("insertion", "fourier", "legendre", "reg"),
    "baseline_colors": {
        "insertion": "tab:orange",
        "fourier": "tab:red",
        "legendre": "tab:green",
        "reg": "tab:purple",
    },
    "baseline_fallback_colors": ("tab:purple", "tab:brown", "tab:pink", "tab:gray"),
    "overlay_random_seed": 42,
}


def _copy_default_plot_style() -> dict:
    copied: dict = {}
    for key, value in DEFAULT_PLOT_STYLE.items():
        if isinstance(value, dict):
            copied[key] = dict(value)
        elif isinstance(value, tuple):
            copied[key] = tuple(value)
        else:
            copied[key] = value
    return copied


def _parse_baseline_color(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError("--baseline-color entries must use KEY=COLOR")
    key, color = value.split("=", 1)
    key = key.strip()
    color = color.strip()
    if not key or not color:
        raise ValueError("--baseline-color entries must use non-empty KEY=COLOR")
    return key, color


def _build_plot_style(args: argparse.Namespace) -> dict:
    style = _copy_default_plot_style()
    style["real"]["color"] = args.real_color
    style["candidate"]["color"] = args.candidate_color
    style["candidate_sample"]["color"] = args.candidate_sample_color
    style["reference"]["color"] = args.reference_color
    style["real_band"]["color"] = args.real_color
    style["baseline_order"] = tuple(args.baseline_order)
    style["overlay_random_seed"] = int(args.overlay_seed)
    for entry in args.baseline_color or []:
        key, color = _parse_baseline_color(entry)
        style["baseline_colors"][key] = color
    return style


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write qualitative evaluation artifacts.")
    parser.add_argument("dataset_dir", help="Inversion output dataset directory containing real.npz")
    parser.add_argument("--candidate", nargs="+", default=None, help="Candidate key selectors")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=SUPPORTED_QUALITATIVE_METRICS,
        default=list(SUPPORTED_QUALITATIVE_METRICS),
        metavar="METRIC",
        help="Qualitative metric selectors",
    )
    parser.add_argument("--n-show", type=int, default=DEFAULT_N_SHOW, help="Number of real paths to export")
    parser.add_argument("--n-gen", type=int, default=DEFAULT_N_GEN, help="Number of probabilistic samples to overlay")
    parser.add_argument("--real-color", default=DEFAULT_PLOT_STYLE["real"]["color"], help="Color for real paths/curves")
    parser.add_argument(
        "--candidate-color",
        default=DEFAULT_PLOT_STYLE["candidate"]["color"],
        help="Color for probabilistic candidate QQ/ACF curves",
    )
    parser.add_argument(
        "--candidate-sample-color",
        default=DEFAULT_PLOT_STYLE["candidate_sample"]["color"],
        help="Color for sampled probabilistic overlay paths",
    )
    parser.add_argument(
        "--reference-color",
        default=DEFAULT_PLOT_STYLE["reference"]["color"],
        help="Color for QQ diagonal reference lines",
    )
    parser.add_argument(
        "--baseline-color",
        action="append",
        default=None,
        metavar="KEY=COLOR",
        help="Override deterministic baseline color, e.g. fourier=tab:red. May be repeated.",
    )
    parser.add_argument(
        "--baseline-order",
        nargs="+",
        default=list(DEFAULT_PLOT_STYLE["baseline_order"]),
        metavar="KEY",
        help="Deterministic baseline ordering shared by overlay, QQ, and ACF plots",
    )
    parser.add_argument(
        "--overlay-seed",
        type=int,
        default=DEFAULT_PLOT_STYLE["overlay_random_seed"],
        help="Seed used for deterministic overlay sample selection",
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

    try:
        candidates = resolve_candidates(dataset_dir, selectors=args.candidate)
        plot_style = _build_plot_style(args)
        written = write_qualitative_results_for_dataset(
            dataset_dir,
            candidates=candidates,
            plot_style=plot_style,
            metric_names=tuple(args.metrics),
            n_show=args.n_show,
            n_gen=args.n_gen,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(f"{dataset_dir} -> {written}")


if __name__ == "__main__":
    main()
