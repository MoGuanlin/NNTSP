# tests/test_depth_fallback_stats.py
# -*- coding: utf-8 -*-
"""
Regression test for per-depth fallback statistics in the 1-pass DP runner.

Usage:
  python tests/test_depth_fallback_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.models.dp_runner import OnePassDPRunner
from src.models.merge_decoder import MergeDecoder
from test_dp_runner import StubLeafEncoder, StubMergeEncoder, build_synthetic_tree


def test_depth_fallback_stats_are_recorded():
    """Each merge depth should expose raw counts and fallback rates."""
    Ti, Tc, P, d = 8, 4, 4, 64
    tokens, leaves = build_synthetic_tree(Ti=Ti, Tc=Tc, P=P)

    leaf_enc = StubLeafEncoder(d)
    merge_enc = StubMergeEncoder(d)
    merge_dec = MergeDecoder(
        d_model=d,
        n_heads=4,
        num_iface_slots=Ti,
        parent_num_layers=2,
        cross_num_layers=1,
        max_depth=16,
    )
    leaf_enc.eval()
    merge_enc.eval()
    merge_dec.eval()

    runner = OnePassDPRunner(r=4, max_used=4, topk=5, fallback_exact=True)
    result = runner.run_single(
        tokens=tokens,
        leaves=leaves,
        leaf_encoder=leaf_enc,
        merge_encoder=merge_enc,
        merge_decoder=merge_dec,
    )

    depth_stats = result.stats.get("depth_stats")
    assert isinstance(depth_stats, dict), "depth_stats should be a dict"
    assert "0" in depth_stats, "synthetic tree should record the root merge depth"

    bucket = depth_stats["0"]
    assert int(bucket["num_internal_nodes"]) == 1, "root depth should contain one internal node"

    total = float(bucket["num_sigma_total"])
    classified = (
        float(bucket["num_parse_ok"])
        + float(bucket["num_topk_ok"])
        + float(bucket["num_fallback"])
        + float(bucket["num_infeasible"])
    )
    assert abs(total - classified) < 1e-6, "per-depth sigma accounting should close"

    depth_rates = result.stats.get("depth_fallback_rates")
    assert isinstance(depth_rates, dict), "depth_fallback_rates should be a dict"
    assert "0" in depth_rates, "depth_fallback_rates should mirror depth_stats"
    assert abs(float(depth_rates["0"]) - float(bucket["fallback_rate"])) < 1e-9

    print("depth_stats[0] =", bucket)
    print("depth_fallback_rates =", depth_rates)


if __name__ == "__main__":
    test_depth_fallback_stats_are_recorded()
    print("test_depth_fallback_stats.py: PASS")
