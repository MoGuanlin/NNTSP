# src/cli/evaluate.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from typing import List

from src.cli import eval_and_vis as synthetic_eval
from src.cli import eval_training_cost as training_cost_eval
from src.cli import eval_twopass_timing as twopass_timing_eval
from src.cli import eval_onepass as onepass_eval
from src.cli import evaluate_tsplib as tsplib_eval
from src.cli.benchmark_config_builder import (
    EvaluateRequest,
    ListSettingsRequest,
    ListTSPLIBPresetsRequest,
    OnePassEvalConfig,
    SyntheticEvalConfig,
    TSPLIBEvalConfig,
    TrainingCostConfig,
    TwoPassTimingConfig,
    build_evaluate_request,
)
from src.cli.eval_profiles import AVAILABLE_SETTINGS, SETTING_GROUPS
from src.cli.guided_lkh_args import add_guided_lkh_args
from src.cli.teacher_lkh_args import add_teacher_lkh_args


def build_parser() -> argparse.ArgumentParser:
    from src.utils.lkh_solver import default_lkh_executable

    parser = argparse.ArgumentParser(
        description="Unified evaluation entry for synthetic standard datasets and TSPLIB."
    )
    default_lkh = default_lkh_executable()
    parser.add_argument(
        "--benchmark",
        type=str,
        default="synthetic",
        choices=("synthetic", "tsplib", "onepass", "twopass_timing", "training_cost"),
    )
    parser.add_argument("--ckpt", type=str, required=False, help="path to model checkpoint")
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lkh_exe", type=str, default=default_lkh, help="path to LKH executable")
    add_guided_lkh_args(parser)
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for decoding / baselines")
    parser.add_argument("--spanner_mode", type=str, default="delaunay", choices=("delaunay", "theta"))
    parser.add_argument("--theta_k", type=int, default=14, help="theta spanner cone count when --spanner_mode theta")
    parser.add_argument("--patching_mode", type=str, default="prune", choices=("prune", "arora"))
    parser.add_argument("--pomo_ckpt", type=str, default=None, help="path to POMO checkpoint")
    parser.add_argument("--neurolkh_ckpt", type=str, default=None, help="path to NeuroLKH checkpoint")
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
    parser.add_argument("--no_vis", action="store_true", help="disable synthetic visualization")
    parser.add_argument("--dp_max_sigma", type=int, default=0, help="onepass: max parent-state candidates per node; 0 disables truncation")
    parser.add_argument("--dp_child_catalog_cap", type=int, default=0, help="onepass/catalog_enum: max child states kept after C1+ranking; 0 disables truncation")
    parser.add_argument(
        "--dp_parse_mode",
        type=str,
        default="catalog_enum",
        choices=(
            "catalog_enum",
            "catalog_enum_iface_mate",
            "catalog_widening",
            "catalog_widening_iface_mate",
            "factorized_widening",
            "factorized_widening_iface_mate",
            "heuristic",
        ),
        help="onepass: parse mode for 1-pass DP",
    )
    parser.add_argument(
        "--dp_child_catalog_widening",
        type=str,
        default="8,16,32,64",
        help="onepass/catalog_widening: comma-separated widening caps; empty disables widening rounds",
    )
    parser.add_argument(
        "--dp_catalog_mate_lambda",
        type=float,
        default=1.0,
        help="onepass/catalog_enum_iface_mate: weight of mate term in child-state ranking",
    )
    parser.add_argument(
        "--dp_fallback_exact",
        type=onepass_eval.parse_bool_arg,
        default=True,
        help="onepass: enable exact fallback after catalog-enum child-cap failure",
    )
    parser.add_argument("--dp_leaf_workers", type=int, default=16, help="onepass: number of workers for leaf exact solve")
    parser.add_argument("--dp_parse_workers", type=int, default=0, help="onepass: number of CPU workers for factorized exact parse/verify")
    parser.add_argument("--teacher_data_pt", type=str, default=None, help="training_cost: explicit teacher-data dataset path")
    parser.add_argument("--teacher_sample_idx", type=int, default=0, help="training_cost: teacher timing start sample index")
    parser.add_argument("--teacher_sample_idx_end", type=int, default=None, help="training_cost: teacher timing end sample index")
    parser.add_argument("--teacher_num_workers", type=int, default=None, help="training_cost: teacher timing workers")
    add_teacher_lkh_args(
        parser,
        runs_default=None,
        timeout_default=None,
        runs_help="training_cost: override teacher LKH runs",
        timeout_help="training_cost: override teacher LKH timeout in seconds",
    )
    parser.add_argument("--num_gpus_override", type=int, default=None, help="training_cost: override GPU count for GPU-hour calculation")
    parser.add_argument("--skip_teacher_timing", action="store_true", help="training_cost: skip teacher generation timing")
    parser.add_argument("--skip_curve_plot", action="store_true", help="training_cost: skip convergence PNG generation")

    parser.add_argument("--tsplib_dir", type=str, default="benchmarks/tsplib", help="directory with TSPLIB .tsp files")
    parser.add_argument("--tsplib_set", type=str, default=None, help="TSPLIB preset, for example largest10, paper, all, or largest:25")
    parser.add_argument("--tsplib_instances", type=str, default=None, help="comma-separated explicit TSPLIB instance names")
    parser.add_argument("--num_instances", type=int, default=10, help="fallback count when TSPLIB uses largest-K selection")
    parser.add_argument("--run_tag", type=str, default=None, help="optional run tag for TSPLIB saved results")
    parser.add_argument("--list_tsplib_presets", action="store_true", help="print supported TSPLIB instance presets and exit")
    return parser


def dispatch_evaluate_request(request: EvaluateRequest) -> None:
    if isinstance(request, ListTSPLIBPresetsRequest):
        tsplib_eval.main(["--list_instance_presets"])
    elif isinstance(request, ListSettingsRequest):
        _dispatch_list_settings(request)
    elif isinstance(request, SyntheticEvalConfig):
        synthetic_eval.main(request.to_argv())
    elif isinstance(request, OnePassEvalConfig):
        onepass_eval.main(request.to_argv())
    elif isinstance(request, TwoPassTimingConfig):
        twopass_timing_eval.main(request.to_argv())
    elif isinstance(request, TrainingCostConfig):
        training_cost_eval.main(request.to_argv())
    elif isinstance(request, TSPLIBEvalConfig):
        tsplib_eval.main(request.to_argv())
    else:
        raise TypeError(f"Unsupported evaluate request: {type(request)!r}")


def _dispatch_list_settings(request: ListSettingsRequest) -> None:
    if request.benchmark == "tsplib":
        tsplib_eval.main(["--list_settings"])
    elif request.benchmark == "synthetic":
        print(
            synthetic_eval.describe_settings(
                available=AVAILABLE_SETTINGS,
                groups=SETTING_GROUPS,
            )
        )
    elif request.benchmark == "onepass":
        print(
            onepass_eval.describe_settings(
                available=AVAILABLE_SETTINGS,
                groups=SETTING_GROUPS,
            )
        )
    else:
        print(f"{request.benchmark} benchmark has no preset settings.")


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        request = build_evaluate_request(args)
    except ValueError as exc:
        parser.error(str(exc))
    dispatch_evaluate_request(request)


if __name__ == "__main__":
    main()
