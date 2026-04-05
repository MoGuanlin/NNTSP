# src/cli/eval_profiles.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Mapping, Sequence


@dataclass(frozen=True)
class EvalProfile:
    available: tuple[str, ...]
    default: tuple[str, ...]
    groups: Mapping[str, tuple[str, ...]]
    aliases: Mapping[str, str]
    display_names: Mapping[str, str]
    colors: Mapping[str, str]


def placeholder_result() -> SimpleNamespace:
    return SimpleNamespace(feasible=False, length=float("inf"), duration=0.0, order=[])


STANDARD_EVAL_PROFILE = EvalProfile(
    available=(
        "greedy",
        "exact",
        "spanner_uniform_lkh",
        "guided_lkh",
        "pure_lkh",
        "pomo",
        "neurolkh",
    ),
    default=(
        "greedy",
        "guided_lkh",
        "pure_lkh",
        "pomo",
        "neurolkh",
    ),
    groups={
        "default": (
            "greedy",
            "guided_lkh",
            "pure_lkh",
            "pomo",
            "neurolkh",
        ),
        "all": ("greedy", "exact", "spanner_uniform_lkh", "guided_lkh", "pure_lkh", "pomo", "neurolkh"),
        "ours": ("greedy", "exact", "guided_lkh"),
        "baselines": ("pomo", "neurolkh"),
        "reference": ("pure_lkh",),
        "lkh": ("spanner_uniform_lkh", "guided_lkh", "pure_lkh"),
        "practical_lkh": ("spanner_uniform_lkh", "guided_lkh", "pure_lkh"),
    },
    aliases={
        "spanner": "spanner_uniform_lkh",
        "spanner_only": "spanner_uniform_lkh",
        "spanner_lkh": "spanner_uniform_lkh",
        "uniform_lkh": "spanner_uniform_lkh",
        "guided": "guided_lkh",
        "guidedlkh": "guided_lkh",
        "pure": "pure_lkh",
        "lkh_pure": "pure_lkh",
        "ex": "exact",
    },
    display_names={
        "greedy": "Greedy",
        "exact": "Exact Sparse",
        "spanner_uniform_lkh": "Spanner-only LKH",
        "guided_lkh": "Guided LKH",
        "pure_lkh": "Pure LKH",
        "pomo": "POMO",
        "neurolkh": "NeuroLKH",
    },
    colors={
        "greedy": "red",
        "exact": "darkorange",
        "spanner_uniform_lkh": "teal",
        "guided_lkh": "blue",
        "pure_lkh": "green",
        "pomo": "purple",
        "neurolkh": "brown",
    },
)


TSPLIB_EVAL_PROFILE = EvalProfile(
    available=(
        "greedy",
        "exact",
        "spanner_uniform_lkh",
        "guided_lkh",
        "pomo",
        "neurolkh",
        "paper_lkh",
    ),
    default=(
        "greedy",
        "guided_lkh",
        "pomo",
        "neurolkh",
        "paper_lkh",
    ),
    groups={
        "default": (
            "greedy",
            "guided_lkh",
            "pomo",
            "neurolkh",
            "paper_lkh",
        ),
        "all": ("greedy", "exact", "spanner_uniform_lkh", "guided_lkh", "pomo", "neurolkh", "paper_lkh"),
        "ours": ("greedy", "exact", "guided_lkh"),
        "baselines": ("pomo", "neurolkh"),
        "reference": ("paper_lkh",),
        "lkh": ("spanner_uniform_lkh", "guided_lkh", "paper_lkh"),
        "practical_lkh": ("spanner_uniform_lkh", "guided_lkh", "paper_lkh"),
    },
    aliases={
        "spanner": "spanner_uniform_lkh",
        "spanner_only": "spanner_uniform_lkh",
        "spanner_lkh": "spanner_uniform_lkh",
        "uniform_lkh": "spanner_uniform_lkh",
        "guided": "guided_lkh",
        "guidedlkh": "guided_lkh",
        "paper": "paper_lkh",
        "pure": "paper_lkh",
        "ref": "paper_lkh",
        "reference_lkh": "paper_lkh",
        "ex": "exact",
    },
    display_names={
        "greedy": "Greedy",
        "exact": "Exact Sparse",
        "spanner_uniform_lkh": "Spanner-only LKH",
        "guided_lkh": "Guided LKH",
        "pomo": "POMO",
        "neurolkh": "NeuroLKH",
        "paper_lkh": "Paper LKH",
    },
    colors={},
)


AVAILABLE_SETTINGS = STANDARD_EVAL_PROFILE.available
DEFAULT_SETTINGS = STANDARD_EVAL_PROFILE.default
SETTING_GROUPS = STANDARD_EVAL_PROFILE.groups
SETTING_ALIASES = STANDARD_EVAL_PROFILE.aliases
SETTING_DISPLAY_NAMES = STANDARD_EVAL_PROFILE.display_names
SETTING_COLORS = STANDARD_EVAL_PROFILE.colors

TSPLIB_AVAILABLE_SETTINGS = TSPLIB_EVAL_PROFILE.available
TSPLIB_DEFAULT_SETTINGS = TSPLIB_EVAL_PROFILE.default
TSPLIB_SETTING_GROUPS = TSPLIB_EVAL_PROFILE.groups
TSPLIB_SETTING_ALIASES = TSPLIB_EVAL_PROFILE.aliases


__all__ = [
    "AVAILABLE_SETTINGS",
    "DEFAULT_SETTINGS",
    "EvalProfile",
    "SETTING_ALIASES",
    "SETTING_COLORS",
    "SETTING_DISPLAY_NAMES",
    "SETTING_GROUPS",
    "STANDARD_EVAL_PROFILE",
    "TSPLIB_AVAILABLE_SETTINGS",
    "TSPLIB_DEFAULT_SETTINGS",
    "TSPLIB_EVAL_PROFILE",
    "TSPLIB_SETTING_ALIASES",
    "TSPLIB_SETTING_GROUPS",
    "placeholder_result",
]
