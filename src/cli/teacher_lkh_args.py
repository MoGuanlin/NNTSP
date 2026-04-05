# src/cli/teacher_lkh_args.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TeacherLKHConfig:
    runs: int = 1
    timeout: float | None = None

    def __post_init__(self) -> None:
        if int(self.runs) <= 0:
            raise ValueError(f"teacher LKH runs must be positive, got {self.runs}")


def add_teacher_lkh_args(
    parser: argparse.ArgumentParser,
    *,
    runs_default: int | None = 1,
    timeout_default: float | None = 0.0,
    runs_help: str = "number of LKH runs for sparse-spanner teacher generation",
    timeout_help: str = "timeout in seconds for one sparse-spanner teacher solve (0 disables timeout)",
) -> None:
    parser.add_argument("--teacher_lkh_runs", type=int, default=runs_default, help=runs_help)
    parser.add_argument("--teacher_lkh_timeout", type=float, default=timeout_default, help=timeout_help)


def normalize_teacher_lkh_timeout(value: Any) -> float | None:
    if value is None:
        return None
    timeout = float(value)
    if timeout <= 0.0:
        return None
    return timeout


def teacher_lkh_config_from_args(
    args: Any,
    *,
    runs_default: int = 1,
    timeout_default: float | None = 0.0,
) -> TeacherLKHConfig:
    raw_runs = getattr(args, "teacher_lkh_runs", runs_default)
    raw_timeout = getattr(args, "teacher_lkh_timeout", timeout_default)
    runs = runs_default if raw_runs is None else int(raw_runs)
    timeout = timeout_default if raw_timeout is None else raw_timeout
    return TeacherLKHConfig(
        runs=runs,
        timeout=normalize_teacher_lkh_timeout(timeout),
    )


def build_spanner_teacher_labeler(
    *,
    lkh_exe: str,
    config: TeacherLKHConfig,
    prefer_cpu: bool = True,
):
    from src.models.labeler import PseudoLabeler

    return PseudoLabeler(
        lkh_exe=str(lkh_exe),
        prefer_cpu=bool(prefer_cpu),
        teacher_mode="spanner_lkh",
        teacher_lkh_runs=int(config.runs),
        teacher_lkh_timeout=config.timeout,
    )


__all__ = [
    "TeacherLKHConfig",
    "add_teacher_lkh_args",
    "build_spanner_teacher_labeler",
    "normalize_teacher_lkh_timeout",
    "teacher_lkh_config_from_args",
]
