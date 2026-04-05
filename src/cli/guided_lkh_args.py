# src/cli/guided_lkh_args.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from typing import Any

from src.cli.common import parse_bool_arg
from src.models.lkh_decode import DEFAULT_GUIDED_LKH_CONFIG, GuidedLKHConfig


GUIDED_LKH_UNSET = -1


def add_guided_lkh_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--guided_top_k",
        type=int,
        default=int(DEFAULT_GUIDED_LKH_CONFIG.top_k),
        help="top-k candidate neighbors per node for guided LKH",
    )
    parser.add_argument(
        "--guided_logit_scale",
        type=float,
        default=float(DEFAULT_GUIDED_LKH_CONFIG.logit_scale),
        help="scale factor for converting edge logits into LKH alpha penalties",
    )
    parser.add_argument(
        "--guided_subgradient",
        type=parse_bool_arg,
        default=bool(DEFAULT_GUIDED_LKH_CONFIG.subgradient),
        help="whether guided LKH keeps LKH subgradient optimization enabled",
    )
    parser.add_argument(
        "--guided_max_candidates",
        type=int,
        default=(
            GUIDED_LKH_UNSET
            if DEFAULT_GUIDED_LKH_CONFIG.max_candidates is None
            else int(DEFAULT_GUIDED_LKH_CONFIG.max_candidates)
        ),
        help="guided LKH MAX_CANDIDATES; use -1 to keep the LKH default, 0 to keep all provided candidates",
    )
    parser.add_argument(
        "--guided_max_trials",
        type=int,
        default=(
            GUIDED_LKH_UNSET
            if DEFAULT_GUIDED_LKH_CONFIG.max_trials is None
            else int(DEFAULT_GUIDED_LKH_CONFIG.max_trials)
        ),
        help="guided LKH MAX_TRIALS; use -1 to keep the LKH default",
    )
    parser.add_argument(
        "--guided_use_initial_tour",
        type=parse_bool_arg,
        default=bool(DEFAULT_GUIDED_LKH_CONFIG.use_initial_tour),
        help="reuse greedy/direct output as INPUT_TOUR_FILE when available",
    )


def guided_lkh_config_from_args(args: Any) -> GuidedLKHConfig:
    top_k = int(args.guided_top_k)
    if top_k <= 0:
        raise ValueError(f"--guided_top_k must be positive, got {top_k}")

    return GuidedLKHConfig(
        top_k=top_k,
        logit_scale=float(args.guided_logit_scale),
        subgradient=bool(args.guided_subgradient),
        max_candidates=_optional_limit(args.guided_max_candidates),
        max_trials=_optional_limit(args.guided_max_trials),
        use_initial_tour=bool(args.guided_use_initial_tour),
    )


def _optional_limit(value: Any) -> int | None:
    parsed = int(value)
    if parsed < 0:
        return None
    return parsed


__all__ = [
    "GUIDED_LKH_UNSET",
    "add_guided_lkh_args",
    "guided_lkh_config_from_args",
]
