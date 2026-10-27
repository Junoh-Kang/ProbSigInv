"""CLI entrypoint for path inversion."""

from __future__ import annotations

import argparse

from probsiginv.inversion import run_inversion_pipeline
from probsiginv.repo import resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Invert signatures -> generate paths (no evaluation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = parser.add_argument_group("real path source")
    src.add_argument("--dataset", type=str, required=True,
                     help="PathSigDataset .npz (pre-computed paths + sigs)")

    inv = parser.add_argument_group("inversion")
    inv.add_argument("--method", type=str, default="generative",
                     choices=[
                         "generative",
                         "regression",
                         "fourier",
                         "legendre",
                         "insertion",
                         "ls_posterior",
                     ],
                     help="Inversion method")
    inv.add_argument("--n-repeats", type=int, default=30,
                     help="Samples per path for probabilistic backends (ignored by regression/baselines)")
    inv.add_argument("--ckpt", type=str, default=None,
                     help="Model checkpoint .pt (required for method=generative|regression)")
    inv.add_argument("--sampling-steps", type=int, default=1000,
                     help="Number of reverse/solver steps used by the generative model backend")
    inv.add_argument("--seed", type=int, default=None,
                     help="Random seed for reproducible stochastic inversion")
    inv.add_argument("--eta", type=float, default=0.0,
                     help="DDIM stochasticity (fixed to 0.0 in canonical experiment naming; ignored by flow matching)")
    inv.add_argument("--fourier-sig-depth", type=int, default=None,
                     help="Sig depth for Fourier inversion (default: dataset sig_depth)")
    inv.add_argument("--legendre-sig-depth", type=int, default=None,
                     help="Sig depth for Legendre inversion (default: dataset sig_depth)")
    inv.add_argument("--insertion-sig-depth", type=int, default=None,
                     help="Sig depth for Insertion inversion (default: 12)")
    inv.add_argument("--n-coeffs", type=int, default=None,
                     help="Basis coefficient count for Fourier/Legendre inversion")
    inv.add_argument("--mirror", action="store_true",
                     help="Mirror augmentation in Fourier inversion")
    inv.add_argument("--clip-x0", type=float, nargs=2, default=None,
                     metavar=("LO", "HI"),
                     help="Clamp x_0_pred to [LO, HI] during DDPM/DDIM sampling")
    inv.add_argument("--batch-size", type=int, default=1024,
                     help="GPU batch size for model inversion (samples per forward pass)")

    posterior = parser.add_argument_group("LS posterior")
    posterior.add_argument("--sig-depth", type=int, default=None,
                           help="Linear Signature depth (required for method=ls_posterior)")
    posterior.add_argument("--n-particles-per-dim", type=int, default=None,
                           help="Midpoint prior particles per non-degenerate parameter dimension")
    posterior.add_argument("--mu-bounds", type=float, nargs=2, default=None,
                           metavar=("LO", "HI"), help="Declared split prior bounds for mu")
    posterior.add_argument("--sigma-bounds", type=float, nargs=2, default=None,
                           metavar=("LO", "HI"), help="Declared split prior bounds for sigma")
    posterior.add_argument("--hurst-bounds", type=float, nargs=2, default=None,
                           metavar=("LO", "HI"), help="Declared split prior bounds for fBM H")
    posterior.add_argument("--theta-bounds", type=float, nargs=2, default=None,
                           metavar=("LO", "HI"), help="Declared split prior bounds for OU theta")

    out_grp = parser.add_argument_group("output")
    out_grp.add_argument("--project", "--training-setting", dest="project", type=str,
                         help="Output group (inferred from checkpoint; required for baselines unless --output-dir is given). --training-setting is a compatibility alias.")
    out_grp.add_argument("--exp-name", type=str, default=None,
                         help="Candidate folder name override; does not change the project or evaluation candidate key")
    location = out_grp.add_mutually_exclusive_group()
    location.add_argument("--output-root", type=str, default=None,
                         help="Output root before project/dataset/split/candidate (default: _outputs)")
    location.add_argument("--output-dir", type=str, default=None,
                         help="Full output dir override (default: canonical path under _outputs/)")

    args = parser.parse_args()

    posterior_controls = {
        "--sig-depth": args.sig_depth,
        "--n-particles-per-dim": args.n_particles_per_dim,
        "--mu-bounds": args.mu_bounds,
        "--sigma-bounds": args.sigma_bounds,
        "--hurst-bounds": args.hurst_bounds,
        "--theta-bounds": args.theta_bounds,
    }
    if args.method == "ls_posterior":
        if args.sig_depth is None:
            parser.error("--method ls_posterior requires --sig-depth")
        if args.sig_depth < 0:
            parser.error("--sig-depth must be non-negative")
        if args.ckpt is not None:
            parser.error("--method ls_posterior does not accept --ckpt")
        if args.n_particles_per_dim is None:
            args.n_particles_per_dim = 25
        if args.n_particles_per_dim < 1:
            parser.error("--n-particles-per-dim must be >= 1")
        if args.n_repeats < 1:
            parser.error("--n-repeats must be >= 1")
        if args.seed is None:
            args.seed = 2026
        if args.seed < 0:
            parser.error("--seed must be non-negative for method=ls_posterior")
    else:
        supplied = [name for name, value in posterior_controls.items() if value is not None]
        if supplied:
            parser.error(
                f"{', '.join(supplied)} require --method ls_posterior"
            )

    args.dataset = str(resolve_repo_path(args.dataset))
    if args.ckpt is not None:
        args.ckpt = str(resolve_repo_path(args.ckpt))
    if args.output_dir is not None:
        args.output_dir = str(resolve_repo_path(args.output_dir))
    args.output_root = str(resolve_repo_path(args.output_root or "_outputs"))
    return args


def main() -> None:
    run_inversion_pipeline(parse_args())


if __name__ == "__main__":
    main()
