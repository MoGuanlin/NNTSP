# tests/test_child_catalog_cap.py
# -*- coding: utf-8 -*-
"""
Regression tests for post-C1 post-sort child-catalog truncation in catalog_enum.

Usage:
  python tests/test_child_catalog_cap.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.dp_core import _rank_child_catalog_states_for_parse
from src.models.dp_runner import OnePassDPRunner


def test_truncation_happens_after_c1_and_sort() -> None:
    cat_used = torch.tensor(
        [
            [False, True, True],   # invalid under C1 on slot 0
            [True, False, False],  # valid, medium score
            [True, True, False],   # valid, best score
            [True, False, True],   # valid, worst score
        ],
        dtype=torch.bool,
    )
    child_cost_q = torch.zeros(4, dtype=torch.float32)
    scores_q = torch.tensor([0.9, 0.8, 0.2], dtype=torch.float32)
    child_iface_mask_q = torch.tensor([True, True, True], dtype=torch.bool)
    constrained_q = torch.tensor([True, False, False], dtype=torch.bool)
    required_q = torch.tensor([True, False, False], dtype=torch.bool)

    ranked = _rank_child_catalog_states_for_parse(
        scores_q=scores_q,
        child_iface_mask_q=child_iface_mask_q,
        constrained_q=constrained_q,
        required_q=required_q,
        cat_used=cat_used,
        child_cost_q=child_cost_q,
        max_child_states=2,
    )

    assert ranked.tolist() == [2, 1], ranked.tolist()


def test_nonpositive_cap_disables_child_truncation() -> None:
    runner = OnePassDPRunner(r=4, max_used=4, max_child_catalog_states=0)
    assert runner.max_child_catalog_states is None

    runner = OnePassDPRunner(r=4, max_used=4, max_child_catalog_states=-3)
    assert runner.max_child_catalog_states is None


def test_catalog_enum_cap_can_fallback_to_exact() -> None:
    runner = OnePassDPRunner(
        r=4,
        max_used=4,
        max_child_catalog_states=1,
        fallback_exact=True,
    )

    scores = torch.zeros(4, 2, dtype=torch.float32)
    parent_a = torch.zeros(2, dtype=torch.bool)
    parent_mate = torch.full((2,), -1, dtype=torch.long)
    parent_iface_mask = torch.ones(2, dtype=torch.bool)
    child_iface_mask = torch.ones(4, 2, dtype=torch.bool)
    child_exists = torch.ones(4, dtype=torch.bool)
    maps = mock.Mock()
    cat_used = torch.zeros(3, 2, dtype=torch.bool)
    cat_mate = torch.full((3, 2), -1, dtype=torch.long)
    child_costs = [torch.zeros(3, dtype=torch.float32) for _ in range(4)]

    with mock.patch("src.models.dp_runner.parse_by_catalog_enum") as patched:
        patched.side_effect = [
            (float("inf"), (-1, -1, -1, -1)),
            (7.5, (0, 1, 2, 0)),
        ]
        cost, child_si, used_fallback = runner._parse_catalog_enum_with_optional_fallback(
            scores=scores,
            parent_a=parent_a,
            parent_mate=parent_mate,
            parent_iface_mask=parent_iface_mask,
            child_iface_mask=child_iface_mask,
            child_exists=child_exists,
            maps=maps,
            cat_used=cat_used,
            cat_mate=cat_mate,
            child_costs=child_costs,
        )

    assert patched.call_count == 2
    assert patched.call_args_list[0].kwargs["max_child_states"] == 1
    assert patched.call_args_list[1].kwargs["max_child_states"] is None
    assert cost == 7.5
    assert child_si == (0, 1, 2, 0)
    assert used_fallback is True


if __name__ == "__main__":
    tests = [
        ("truncate after c1 and sort", test_truncation_happens_after_c1_and_sort),
        ("nonpositive cap disables truncation", test_nonpositive_cap_disables_child_truncation),
        ("catalog cap can fallback to exact", test_catalog_enum_cap_can_fallback_to_exact),
    ]

    for name, fn in tests:
        print(f"Test: {name}...")
        fn()
        print("  PASS\n")

    print("All child-catalog cap tests passed!")
