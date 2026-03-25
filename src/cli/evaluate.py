# src/cli/evaluate.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.cli import eval_and_vis as synthetic_eval
from src.cli import evaluate_tsplib as tsplib_eval


def resolve_synthetic_data_path(*, synthetic_n: int, synthetic_data_root: str) -> str:
    return str(Path(synthetic_data_root) / f"N{int(synthetic_n)}" / "test_r_light_pyramid.pt")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Unified evaluation entry for synthetic standard datasets and TSPLIB."
    )
    parser.add_argument("--benchmark", type=str, default="synthetic", choices=("synthetic", "tsplib"))
    parser.add_argument("--ckpt", type=str, required=False, help="path to model checkpoint")
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lkh_exe", type=str, default="LKH", help="path to LKH executable")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for decoding / baselines")
    parser.add_argument("--pomo_ckpt", type=str, default=None, help="path to POMO checkpoint")
    parser.add_argument("--neurolkh_ckpt", type=str, default=None, help="path to NeuroLKH checkpoint")
    parser.add_argument("--run_exact", action="store_true", help="legacy flag: add exact to selected settings")
    parser.add_argument("--exact_time_limit", type=float, default=30.0, help="time limit in seconds for exact sparse decoding")
    parser.add_argument("--exact_length_weight", type=float, default=0.0, help="optional Euclidean tie-break weight for exact sparse decoding")
    parser.add_argument("--settings", type=str, default=None, help="comma-separated settings/groups")
    parser.add_argument("--list_settings", action="store_true", help="print supported settings for the chosen benchmark and exit")

    parser.add_argument("--synthetic_n", type=int, default=None, help="standard synthetic dataset size N, for example 500 or 2000")
    parser.add_argument("--synthetic_data_root", type=str, default="data/std", help="root directory for standard synthetic datasets")
    parser.add_argument("--data_pt", type=str, default=None, help="explicit synthetic dataset path; overrides --synthetic_n")
    parser.add_argument("--sample_idx", type=int, default=0, help="start sample index for synthetic evaluation")
    parser.add_argument("--sample_idx_end", type=int, default=None, help="end sample index (exclusive) for synthetic evaluation")
    parser.add_argument("--output_dir", type=str, default=None, help="output directory; synthetic uses it for plots, TSPLIB uses it for saved partial results")
    parser.add_argument("--use_iface_in_decode", type=str, default=None, help="optional override for synthetic decoding interface logits")
    parser.add_argument("--use_lkh", action="store_true", help="use LKH for synthetic teacher projection")
    parser.add_argument("--no_vis", action="store_true", help="disable synthetic visualization")
    parser.add_argument("--two_opt_passes", type=int, default=30, help="teacher tour passes for synthetic evaluation")

    parser.add_argument("--tsplib_dir", type=str, default="benchmarks/tsplib", help="directory with TSPLIB .tsp files")
    parser.add_argument("--tsplib_set", type=str, default=None, help="TSPLIB preset, for example largest10, paper, all, or largest:25")
    parser.add_argument("--tsplib_instances", type=str, default=None, help="comma-separated explicit TSPLIB instance names")
    parser.add_argument("--num_instances", type=int, default=10, help="fallback count when TSPLIB uses largest-K selection")
    parser.add_argument("--run_tag", type=str, default=None, help="optional run tag for TSPLIB saved results")
    parser.add_argument("--list_tsplib_presets", action="store_true", help="print supported TSPLIB instance presets and exit")

    args = parser.parse_args(argv)

    if args.list_tsplib_presets:
        tsplib_eval.main(["--list_instance_presets"])
        return

    if args.list_settings:
        if args.benchmark == "tsplib":
            tsplib_eval.main(["--list_settings"])
        else:
            synthetic_eval.main(["--list_settings"])
        return

    if not args.ckpt:
        parser.error("--ckpt is required unless a list command is used")

    if args.benchmark == "synthetic":
        data_pt = args.data_pt
        if not data_pt:
            if args.synthetic_n is None:
                parser.error("--synthetic_n or --data_pt is required when --benchmark synthetic")
            data_pt = resolve_synthetic_data_path(
                synthetic_n=int(args.synthetic_n),
                synthetic_data_root=str(args.synthetic_data_root),
            )

        child_argv = [
            "--ckpt", str(args.ckpt),
            "--data_pt", str(data_pt),
            "--sample_idx", str(args.sample_idx),
            "--r", str(args.r),
            "--device", str(args.device),
            "--lkh_exe", str(args.lkh_exe),
            "--two_opt_passes", str(args.two_opt_passes),
            "--num_workers", str(args.num_workers),
            "--exact_time_limit", str(args.exact_time_limit),
            "--exact_length_weight", str(args.exact_length_weight),
            "--output_dir", str(args.output_dir or "outputs/eval_lkh"),
            "--pomo_ckpt", str(args.pomo_ckpt or ""),
            "--neurolkh_ckpt", str(args.neurolkh_ckpt or ""),
        ]
        if args.sample_idx_end is not None:
            child_argv += ["--sample_idx_end", str(args.sample_idx_end)]
        if args.settings:
            child_argv += ["--settings", str(args.settings)]
        if args.run_exact:
            child_argv.append("--run_exact")
        if args.use_lkh:
            child_argv.append("--use_lkh")
        if args.no_vis:
            child_argv.append("--no_vis")
        if args.use_iface_in_decode is not None:
            child_argv += ["--use_iface_in_decode", str(args.use_iface_in_decode)]
        synthetic_eval.main(child_argv)
        return

    child_argv = [
        "--ckpt", str(args.ckpt),
        "--tsplib_dir", str(args.tsplib_dir),
        "--num_instances", str(args.num_instances),
        "--device", str(args.device),
        "--r", str(args.r),
        "--lkh_exe", str(args.lkh_exe),
        "--num_workers", str(args.num_workers),
        "--exact_time_limit", str(args.exact_time_limit),
        "--exact_length_weight", str(args.exact_length_weight),
        "--save_dir", str(args.output_dir or "outputs/eval_tsplib"),
        "--pomo_ckpt", str(args.pomo_ckpt or ""),
        "--neurolkh_ckpt", str(args.neurolkh_ckpt or ""),
    ]
    if args.settings:
        child_argv += ["--settings", str(args.settings)]
    if args.run_exact:
        child_argv.append("--run_exact")
    if args.tsplib_set:
        child_argv += ["--instance_preset", str(args.tsplib_set)]
    if args.tsplib_instances:
        child_argv += ["--instances", str(args.tsplib_instances)]
    if args.run_tag:
        child_argv += ["--run_tag", str(args.run_tag)]
    tsplib_eval.main(child_argv)


if __name__ == "__main__":
    main()
