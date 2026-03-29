# tests/test_sigma_cap.py
# -*- coding: utf-8 -*-
"""
Regression tests for sigma-cap semantics in the 1-pass DP runner.

Usage:
  python tests/test_sigma_cap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.bc_state_catalog import build_boundary_state_catalog
from src.models.dp_runner import OnePassDPRunner


def _make_candidates(Ti: int = 16, max_used: int = 4) -> torch.Tensor:
    catalog = build_boundary_state_catalog(
        num_slots=Ti,
        max_used=max_used,
        device=torch.device("cpu"),
    )
    cands = torch.arange(catalog.used_iface.shape[0], dtype=torch.long)
    assert cands.numel() > 500, "Test requires a catalog larger than the historical 500 cap."
    return cands


def test_default_disables_truncation() -> None:
    cands = _make_candidates()
    runner = OnePassDPRunner(r=4, max_used=4)
    kept = runner._apply_sigma_cap(cands)
    assert runner.max_sigma_enumerate is None
    assert kept.numel() == cands.numel()
    assert torch.equal(kept, cands)


def test_nonpositive_cap_disables_truncation() -> None:
    cands = _make_candidates()
    for cap in (0, -1):
        runner = OnePassDPRunner(r=4, max_used=4, max_sigma_enumerate=cap)
        kept = runner._apply_sigma_cap(cands)
        assert runner.max_sigma_enumerate is None
        assert kept.numel() == cands.numel()
        assert torch.equal(kept, cands)


def test_positive_cap_is_explicit_heuristic_truncation() -> None:
    cands = _make_candidates()
    runner = OnePassDPRunner(r=4, max_used=4, max_sigma_enumerate=500)
    kept = runner._apply_sigma_cap(cands)
    assert runner.max_sigma_enumerate == 500
    assert kept.numel() == 500
    assert torch.equal(kept, cands[:500])


if __name__ == "__main__":
    tests = [
        ("default disables truncation", test_default_disables_truncation),
        ("nonpositive cap disables truncation", test_nonpositive_cap_disables_truncation),
        ("positive cap truncates explicitly", test_positive_cap_is_explicit_heuristic_truncation),
    ]

    for name, fn in tests:
        print(f"Test: {name}...")
        fn()
        print("  PASS\n")

    print("All sigma-cap tests passed!")
