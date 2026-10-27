"""Top-level inversion orchestration and method dispatch."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

from probsiginv.inversion.artifacts import (
    prepare_shared_dataset_artifacts,
    save_candidate_artifact,
    save_ls_posterior_artifact,
)
from probsiginv.inversion.contracts import InversionMethodContext, InversionSourceContext
from probsiginv.inversion.outputs import resolve_output_layout, resolve_project_name
from probsiginv.inversion.provenance import (
    build_inversion_identity,
    save_inversion_identity,
    validate_existing_candidate,
)
from probsiginv.inversion.sources import (
    load_real_path_source,
    resolve_fourier_sig_depth,
    resolve_insertion_sig_depth,
    resolve_legendre_sig_depth,
)
from probsiginv.utils.contract import build_output_candidate_label
from probsiginv.utils.contract import OUTPUT_ARTIFACT_NAME
from probsiginv.utils.conditioning import resolve_data_aug_type


FOURIER_METHOD = "fourier"
INSERTION_METHOD = "insertion"
GENERATIVE_METHOD = "generative"
LEGENDRE_METHOD = "legendre"
LS_POSTERIOR_METHOD = "ls_posterior"
REGRESSION_METHOD = "regression"


def _set_inversion_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_inversion_pipeline(args: argparse.Namespace) -> None:
    """CLI-facing inversion orchestration."""
    if getattr(args, "seed", None) is not None:
        _set_inversion_seed(args.seed)
        print(f"Seed: {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    inspection = None
    preferred_source_aug_type = None
    if args.method == GENERATIVE_METHOD:
        from probsiginv.inversion.generative import inspect_generative_checkpoint

        inspection = inspect_generative_checkpoint(args)
        preferred_source_aug_type = resolve_data_aug_type(
            str(inspection.ckpt_args.get("aug_type", "TA"))
        )
    elif args.method == REGRESSION_METHOD:
        from probsiginv.inversion.baselines import inspect_regression_checkpoint

        inspection = inspect_regression_checkpoint(args)
        preferred_source_aug_type = resolve_data_aug_type(
            str(inspection.ckpt_args.get("aug_type", "TA"))
        )

    args.project = resolve_project_name(
        args, inspection.ckpt_args if inspection is not None else None
    )
    paths, source_sigs, time_grid, meta, dataset_name, source_dataset_path = load_real_path_source(
        args,
        preferred_source_aug_type=preferred_source_aug_type,
    )
    source = InversionSourceContext(
        paths=paths,
        source_sigs=source_sigs,
        time_grid=time_grid,
        meta=meta,
        dataset_name=dataset_name,
    )
    print(f"  paths: {source.paths.shape}")

    if args.method == GENERATIVE_METHOD:
        from probsiginv.inversion.generative import (
            resolve_generative_output_layout,
        )

        out_root, dataset_dir = resolve_generative_output_layout(
            args,
            source.dataset_name,
            inspection.ckpt_args,
        )
    elif args.method == REGRESSION_METHOD:
        from probsiginv.inversion.baselines import (
            resolve_regression_output_layout,
        )

        out_root, dataset_dir = resolve_regression_output_layout(
            args,
            source.dataset_name,
            inspection.ckpt_args,
        )
    else:
        inspection = None
        conditioning = (
            f"LSd{int(args.sig_depth)}"
            if args.method == LS_POSTERIOR_METHOD
            else None
        )
        base_exp_name = args.exp_name or build_output_candidate_label(
            args.method,
            conditioning=conditioning,
            n_coeffs=getattr(args, "n_coeffs", None),
        )
        out_root, dataset_dir = resolve_output_layout(
            args,
            source.dataset_name,
            base_exp_name=base_exp_name,
        )

    args.output_dir = str(out_root)
    print(f"  output_dir: {args.output_dir}")

    method_context = InversionMethodContext(
        source=source,
        dataset_dir=dataset_dir,
        output_dir=args.output_dir,
    )

    candidate_path = os.path.join(method_context.output_dir, OUTPUT_ARTIFACT_NAME)
    if args.method == LS_POSTERIOR_METHOD:
        from probsiginv.inversion.ls_posterior.observation import (
            load_ls_observations,
        )
        from probsiginv.inversion.ls_posterior.runtime import (
            run_ls_posterior_method,
        )

        observation_batch = load_ls_observations(
            Path(source_dataset_path).parent,
            n_rows=source.n_paths,
            depth=int(args.sig_depth),
            T=float(source.meta["T_horizon"]),
        )
        result = run_ls_posterior_method(
            n_rows=source.n_paths,
            time_grid=source.time_grid,
            process_metadata=source.meta,
            declared_bounds={
                "mu": getattr(args, "mu_bounds", None),
                "sigma": getattr(args, "sigma_bounds", None),
                "hurst": getattr(args, "hurst_bounds", None),
                "theta": getattr(args, "theta_bounds", None),
            },
            observations=observation_batch.observations,
            source_signature_artifact=observation_batch.source_artifact,
            sig_depth=int(args.sig_depth),
            n_particles_per_dim=int(args.n_particles_per_dim),
            n_repeats=int(args.n_repeats),
            seed=int(args.seed),
            existing_artifact_path=candidate_path,
        )
        prepare_shared_dataset_artifacts(
            args,
            paths=source.paths,
            source_sigs=source.source_sigs,
            time_grid=source.time_grid,
            meta=source.meta,
            dataset_dir=method_context.dataset_dir,
            include_source_signatures=False,
            source_dataset_path=source_dataset_path,
        )
        if result is None:
            print(f"[{args.method}] compatible artifact exists, skipping -> {candidate_path}")
            print(f"\nDone. All outputs in: {args.output_dir}")
            return
        out = save_ls_posterior_artifact(method_context.output_dir, result)
        print(f"  saved -> {out}  shape={result.paths.shape}")
        print(f"\nDone. All outputs in: {args.output_dir}")
        return

    identity = build_inversion_identity(args, source_dataset_path)
    reuse_candidate = validate_existing_candidate(candidate_path, identity)
    prepare_shared_dataset_artifacts(
        args,
        paths=source.paths,
        source_sigs=source.source_sigs,
        time_grid=source.time_grid,
        meta=source.meta,
        dataset_dir=method_context.dataset_dir,
        include_source_signatures=args.method == GENERATIVE_METHOD,
        source_dataset_path=source_dataset_path,
    )

    if reuse_candidate:
        print(f"[{args.method}] already exists, skipping -> {candidate_path}")
        print(f"\nDone. All outputs in: {args.output_dir}")
        return

    if args.method == FOURIER_METHOD:
        from probsiginv.inversion.baselines.fourier import run_fourier_method

        fourier_sig_depth = resolve_fourier_sig_depth(args, source.meta)
        result = run_fourier_method(
            args,
            method_context,
            fourier_sig_depth=fourier_sig_depth,
        )
    elif args.method == LEGENDRE_METHOD:
        from probsiginv.inversion.baselines.legendre import run_legendre_method

        legendre_sig_depth = resolve_legendre_sig_depth(args, source.meta)
        result = run_legendre_method(
            args,
            method_context,
            legendre_sig_depth=legendre_sig_depth,
        )
    elif args.method == INSERTION_METHOD:
        from probsiginv.inversion.baselines.insertion import run_insertion_method

        insertion_sig_depth = resolve_insertion_sig_depth(args, source.meta)
        result = run_insertion_method(
            args,
            method_context,
            insertion_sig_depth=insertion_sig_depth,
        )
    elif args.method == REGRESSION_METHOD:
        from probsiginv.inversion.baselines import (
            prepare_regression_inversion,
            run_regression_inversion,
        )

        prepared = prepare_regression_inversion(
            args,
            source=source,
            device=device,
            inspection=inspection,
        )
        result = run_regression_inversion(
            args,
            method_context=method_context,
            prepared=prepared,
            device=device,
            batch_size=args.batch_size,
        )
    else:
        from probsiginv.inversion.generative import (
            prepare_generative_inversion,
            run_generative_inversion,
        )

        prepared = prepare_generative_inversion(
            args,
            source=source,
            device=device,
            inspection=inspection,
        )
        result = run_generative_inversion(
            args,
            method_context=method_context,
            prepared=prepared,
            device=device,
            batch_size=args.batch_size,
        )

    out = save_candidate_artifact(method_context.output_dir, result)
    save_inversion_identity(method_context.output_dir, identity)
    print(f"  saved -> {out}  shape={result.paths.shape}")

    print(f"\nDone. All outputs in: {args.output_dir}")
