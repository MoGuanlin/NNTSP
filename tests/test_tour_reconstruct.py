# tests/test_tour_reconstruct.py
# -*- coding: utf-8 -*-
"""
Unit tests for src/models/tour_reconstruct.py — DP result → logits → edge scores.

Usage:
  python tests/test_tour_reconstruct.py
  python -m pytest tests/test_tour_reconstruct.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.tour_reconstruct import (
    dp_result_to_logits,
    dp_result_to_edge_scores,
    reconstruct_tour_direct,
)
from src.models.dp_runner import OnePassDPRunner, OnePassDPResult, CostTableEntry
from src.models.merge_decoder import MergeDecoder
from src.models.bc_state_catalog import build_boundary_state_catalog
from src.models.edge_decode import decode_tour_from_edge_logits
from src.models.dp_core import leaf_exact_solve


# ─── Reuse synthetic tree from test_dp_runner ────────────────────────────────

class StubLeafEncoder(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.d = d_model
        self.proj = nn.Linear(4, d_model)

    def forward(self, *, node_feat_rel, node_depth, iface_feat6, iface_mask,
                iface_boundary_dir, iface_inside_endpoint, iface_inside_quadrant,
                cross_feat6, cross_mask, cross_child_pair, cross_is_leaf_internal,
                leaf_points_xy, leaf_points_mask):
        B = node_feat_rel.shape[0]
        return self.proj(node_feat_rel)


class StubMergeEncoder(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.d = d_model
        self.proj = nn.Linear(4, d_model)

    def forward(self, *, node_feat_rel, node_depth, iface_feat6, iface_mask,
                iface_boundary_dir, iface_inside_endpoint, iface_inside_quadrant,
                cross_feat6, cross_mask, cross_child_pair, cross_is_leaf_internal,
                child_z, child_mask):
        B = node_feat_rel.shape[0]
        return self.proj(node_feat_rel)


def build_synthetic_tree(Ti: int = 8, Tc: int = 4, P: int = 4):
    """Build a minimal 2-level tree: 1 root + 4 leaf children.
    Same as test_dp_runner.build_synthetic_tree but imported here to be self-contained.
    """
    M = 5
    device = torch.device("cpu")

    tree_parent_index = torch.tensor([-1, 0, 0, 0, 0], dtype=torch.long, device=device)
    tree_children_index = torch.full((M, 4), -1, dtype=torch.long, device=device)
    tree_children_index[0] = torch.tensor([1, 2, 3, 4])
    tree_node_depth = torch.tensor([0, 1, 1, 1, 1], dtype=torch.long, device=device)
    is_leaf = torch.tensor([False, True, True, True, True], dtype=torch.bool, device=device)

    tree_node_feat_rel = torch.tensor([
        [0.5, 0.5, 1.0, 1.0],
        [0.25, 0.75, 0.5, 0.5],
        [0.75, 0.75, 0.5, 0.5],
        [0.25, 0.25, 0.5, 0.5],
        [0.75, 0.25, 0.5, 0.5],
    ], dtype=torch.float32, device=device)

    iface_mask = torch.zeros(M, Ti, dtype=torch.bool, device=device)
    iface_eid = torch.full((M, Ti), -1, dtype=torch.long, device=device)
    iface_feat6 = torch.zeros(M, Ti, 6, dtype=torch.float32, device=device)
    iface_boundary_dir = torch.full((M, Ti), -1, dtype=torch.long, device=device)
    iface_inside_endpoint = torch.full((M, Ti), -1, dtype=torch.long, device=device)
    iface_inside_quadrant = torch.full((M, Ti), -1, dtype=torch.long, device=device)

    # e0: parent LEFT, child TL
    iface_mask[0, 0] = True; iface_eid[0, 0] = 0; iface_boundary_dir[0, 0] = 0
    iface_inside_endpoint[0, 0] = 0; iface_inside_quadrant[0, 0] = 0
    iface_mask[1, 0] = True; iface_eid[1, 0] = 0; iface_boundary_dir[1, 0] = 0
    iface_inside_endpoint[1, 0] = 0; iface_inside_quadrant[1, 0] = 0

    # e1: parent RIGHT, child TR
    iface_mask[0, 1] = True; iface_eid[0, 1] = 1; iface_boundary_dir[0, 1] = 1
    iface_inside_endpoint[0, 1] = 0; iface_inside_quadrant[0, 1] = 1
    iface_mask[2, 0] = True; iface_eid[2, 0] = 1; iface_boundary_dir[2, 0] = 1
    iface_inside_endpoint[2, 0] = 0; iface_inside_quadrant[2, 0] = 1

    # e2: crossing TL<->TR
    iface_mask[1, 1] = True; iface_eid[1, 1] = 2; iface_boundary_dir[1, 1] = 1
    iface_inside_endpoint[1, 1] = 0; iface_inside_quadrant[1, 1] = 0
    iface_mask[2, 1] = True; iface_eid[2, 1] = 2; iface_boundary_dir[2, 1] = 0
    iface_inside_endpoint[2, 1] = 0; iface_inside_quadrant[2, 1] = 1

    # e3: crossing BL<->BR
    iface_mask[3, 0] = True; iface_eid[3, 0] = 3; iface_boundary_dir[3, 0] = 1
    iface_inside_endpoint[3, 0] = 0; iface_inside_quadrant[3, 0] = 2
    iface_mask[4, 0] = True; iface_eid[4, 0] = 3; iface_boundary_dir[4, 0] = 0
    iface_inside_endpoint[4, 0] = 0; iface_inside_quadrant[4, 0] = 3

    cross_mask = torch.zeros(M, Tc, dtype=torch.bool, device=device)
    cross_eid = torch.full((M, Tc), -1, dtype=torch.long, device=device)
    cross_feat6 = torch.zeros(M, Tc, 6, dtype=torch.float32, device=device)
    cross_child_pair = torch.full((M, Tc, 2), -1, dtype=torch.long, device=device)
    cross_is_leaf_internal = torch.zeros(M, Tc, dtype=torch.bool, device=device)

    cross_mask[0, 0] = True; cross_eid[0, 0] = 2
    cross_child_pair[0, 0] = torch.tensor([0, 1])
    cross_mask[0, 1] = True; cross_eid[0, 1] = 3
    cross_child_pair[0, 1] = torch.tensor([2, 3])

    class Tokens:
        pass

    tokens = Tokens()
    tokens.tree_parent_index = tree_parent_index
    tokens.tree_children_index = tree_children_index
    tokens.tree_node_depth = tree_node_depth
    tokens.tree_node_feat_rel = tree_node_feat_rel
    tokens.is_leaf = is_leaf
    tokens.root_id = torch.tensor([0], dtype=torch.long)
    tokens.root_scale_s = torch.tensor([1.0])
    tokens.iface_mask = iface_mask
    tokens.iface_eid = iface_eid
    tokens.iface_feat6 = iface_feat6
    tokens.iface_boundary_dir = iface_boundary_dir
    tokens.iface_inside_endpoint = iface_inside_endpoint
    tokens.iface_inside_quadrant = iface_inside_quadrant
    tokens.cross_mask = cross_mask
    tokens.cross_eid = cross_eid
    tokens.cross_feat6 = cross_feat6
    tokens.cross_child_pair = cross_child_pair
    tokens.cross_is_leaf_internal = cross_is_leaf_internal

    class Leaves:
        pass

    leaves = Leaves()
    leaves.leaf_node_id = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    leaves.point_idx = torch.tensor([
        [0, 1, -1, -1],
        [2, 3, -1, -1],
        [4, 5, -1, -1],
        [6, 7, -1, -1],
    ], dtype=torch.long)
    leaves.point_xy = torch.rand(4, P, 2)
    leaves.point_mask = torch.ones(4, P, dtype=torch.bool)
    leaves.point_mask[:, 2:] = False

    return tokens, leaves


def build_manual_cycle_tree():
    """Build a small 2-level tree whose traceback is a single direct cycle."""
    device = torch.device("cpu")
    M, Ti, Tc, P = 5, 2, 4, 1

    class Tokens:
        pass

    tokens = Tokens()
    tokens.tree_parent_index = torch.tensor([-1, 0, 0, 0, 0], dtype=torch.long, device=device)
    tokens.tree_children_index = torch.full((M, 4), -1, dtype=torch.long, device=device)
    tokens.tree_children_index[0] = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    tokens.tree_node_depth = torch.tensor([0, 1, 1, 1, 1], dtype=torch.long, device=device)
    tokens.is_leaf = torch.tensor([False, True, True, True, True], dtype=torch.bool, device=device)
    tokens.root_id = torch.tensor([0], dtype=torch.long, device=device)
    tokens.root_scale_s = torch.tensor([1.0], dtype=torch.float32, device=device)
    tokens.tree_node_feat_rel = torch.tensor([
        [0.5, 0.5, 1.0, 1.0],
        [0.25, 0.75, 0.5, 0.5],
        [0.75, 0.75, 0.5, 0.5],
        [0.25, 0.25, 0.5, 0.5],
        [0.75, 0.25, 0.5, 0.5],
    ], dtype=torch.float32, device=device)

    tokens.iface_mask = torch.zeros(M, Ti, dtype=torch.bool, device=device)
    tokens.iface_eid = torch.full((M, Ti), -1, dtype=torch.long, device=device)
    tokens.iface_feat6 = torch.zeros(M, Ti, 6, dtype=torch.float32, device=device)
    tokens.iface_boundary_dir = torch.full((M, Ti), -1, dtype=torch.long, device=device)
    tokens.iface_inside_endpoint = torch.full((M, Ti), -1, dtype=torch.long, device=device)
    tokens.iface_inside_quadrant = torch.full((M, Ti), -1, dtype=torch.long, device=device)

    # Leaf 1 (TL): right boundary e0, bottom boundary e3
    tokens.iface_mask[1] = torch.tensor([True, True])
    tokens.iface_eid[1] = torch.tensor([0, 3])
    tokens.iface_boundary_dir[1] = torch.tensor([1, 2])  # RIGHT, BOTTOM

    # Leaf 2 (TR): left boundary e0, bottom boundary e1
    tokens.iface_mask[2] = torch.tensor([True, True])
    tokens.iface_eid[2] = torch.tensor([0, 1])
    tokens.iface_boundary_dir[2] = torch.tensor([0, 2])  # LEFT, BOTTOM

    # Leaf 3 (BL): right boundary e2, top boundary e3
    tokens.iface_mask[3] = torch.tensor([True, True])
    tokens.iface_eid[3] = torch.tensor([2, 3])
    tokens.iface_boundary_dir[3] = torch.tensor([1, 3])  # RIGHT, TOP

    # Leaf 4 (BR): top boundary e1, left boundary e2
    tokens.iface_mask[4] = torch.tensor([True, True])
    tokens.iface_eid[4] = torch.tensor([1, 2])
    tokens.iface_boundary_dir[4] = torch.tensor([3, 0])  # TOP, LEFT

    tokens.cross_mask = torch.zeros(M, Tc, dtype=torch.bool, device=device)
    tokens.cross_eid = torch.full((M, Tc), -1, dtype=torch.long, device=device)
    tokens.cross_feat6 = torch.zeros(M, Tc, 6, dtype=torch.float32, device=device)
    tokens.cross_child_pair = torch.full((M, Tc, 2), -1, dtype=torch.long, device=device)
    tokens.cross_is_leaf_internal = torch.zeros(M, Tc, dtype=torch.bool, device=device)

    tokens.cross_mask[0, :4] = True
    tokens.cross_eid[0, :4] = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    tokens.cross_child_pair[0, 0] = torch.tensor([0, 1], dtype=torch.long)  # TL <-> TR
    tokens.cross_child_pair[0, 1] = torch.tensor([1, 3], dtype=torch.long)  # TR <-> BR
    tokens.cross_child_pair[0, 2] = torch.tensor([2, 3], dtype=torch.long)  # BL <-> BR
    tokens.cross_child_pair[0, 3] = torch.tensor([0, 2], dtype=torch.long)  # TL <-> BL

    # Crossing sites at side centers.
    tokens.iface_feat6[1, 0, 2:4] = torch.tensor([1.0, 0.0])   # TL right
    tokens.iface_feat6[1, 1, 2:4] = torch.tensor([0.0, -1.0])  # TL bottom
    tokens.iface_feat6[2, 0, 2:4] = torch.tensor([-1.0, 0.0])  # TR left
    tokens.iface_feat6[2, 1, 2:4] = torch.tensor([0.0, -1.0])  # TR bottom
    tokens.iface_feat6[3, 0, 2:4] = torch.tensor([1.0, 0.0])   # BL right
    tokens.iface_feat6[3, 1, 2:4] = torch.tensor([0.0, 1.0])   # BL top
    tokens.iface_feat6[4, 0, 2:4] = torch.tensor([0.0, 1.0])   # BR top
    tokens.iface_feat6[4, 1, 2:4] = torch.tensor([-1.0, 0.0])  # BR left

    class Leaves:
        pass

    leaves = Leaves()
    leaves.leaf_node_id = torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device)
    leaves.point_idx = torch.tensor([[0], [1], [2], [3]], dtype=torch.long, device=device)
    leaves.point_mask = torch.tensor([[True], [True], [True], [True]], dtype=torch.bool, device=device)
    leaves.point_xy = torch.tensor([
        [[0.0, 0.0]],
        [[0.0, 0.0]],
        [[0.0, 0.0]],
        [[0.0, 0.0]],
    ], dtype=torch.float32, device=device)

    return tokens, leaves


def _run_dp(Ti=8, Tc=4, P=4, d=64):
    """Run 1-pass DP and return all needed objects."""
    tokens, leaves = build_synthetic_tree(Ti=Ti, Tc=Tc, P=P)

    leaf_enc = StubLeafEncoder(d)
    merge_enc = StubMergeEncoder(d)
    merge_dec = MergeDecoder(
        d_model=d, n_heads=4, num_iface_slots=Ti,
        parent_num_layers=2, cross_num_layers=1, max_depth=16,
    )
    leaf_enc.eval(); merge_enc.eval(); merge_dec.eval()

    catalog = build_boundary_state_catalog(num_slots=Ti, max_used=4, device=torch.device("cpu"))

    runner = OnePassDPRunner(r=4, max_used=4, topk=5, fallback_exact=True)
    result = runner.run_single(
        tokens=tokens, leaves=leaves,
        leaf_encoder=leaf_enc, merge_encoder=merge_enc, merge_decoder=merge_dec,
        catalog=catalog,
    )
    return result, tokens, leaves, catalog


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_dp_result_to_logits_shape():
    """dp_result_to_logits should return tensors with correct shapes."""
    result, tokens, leaves, catalog = _run_dp()

    iface_logit, cross_logit = dp_result_to_logits(
        result=result, tokens=tokens, catalog=catalog,
    )

    M = 5
    Ti = 8
    Tc = 4
    assert iface_logit.shape == (M, Ti), f"iface_logit shape: {iface_logit.shape}"
    assert cross_logit.shape == (M, Tc), f"cross_logit shape: {cross_logit.shape}"
    print(f"  iface_logit shape: {iface_logit.shape}")
    print(f"  cross_logit shape: {cross_logit.shape}")


def test_dp_result_to_logits_values():
    """Logit values should be either _LOGIT_ON (+10) or _LOGIT_OFF (-10)."""
    result, tokens, leaves, catalog = _run_dp()

    iface_logit, cross_logit = dp_result_to_logits(
        result=result, tokens=tokens, catalog=catalog,
    )

    # All values should be exactly +10 or -10
    unique_iface = set(iface_logit.unique().tolist())
    unique_cross = set(cross_logit.unique().tolist())
    assert unique_iface.issubset({10.0, -10.0}), f"Unexpected iface values: {unique_iface}"
    assert unique_cross.issubset({10.0, -10.0}), f"Unexpected cross values: {unique_cross}"
    print(f"  iface unique values: {unique_iface}")
    print(f"  cross unique values: {unique_cross}")


def test_dp_result_to_edge_scores_shape():
    """dp_result_to_edge_scores should return EdgeScores with correct shape."""
    result, tokens, leaves, catalog = _run_dp()

    edge_scores = dp_result_to_edge_scores(
        result=result, tokens=tokens, catalog=catalog,
    )

    # We have 4 edges (eid 0-3)
    assert edge_scores.edge_logit.shape[0] == 4, f"Expected 4 edges, got {edge_scores.edge_logit.shape[0]}"
    assert edge_scores.edge_mask.shape[0] == 4
    print(f"  edge_logit: {edge_scores.edge_logit}")
    print(f"  edge_mask: {edge_scores.edge_mask}")


def test_dp_result_to_edge_scores_with_num_edges():
    """Explicit num_edges should be respected."""
    result, tokens, leaves, catalog = _run_dp()

    edge_scores = dp_result_to_edge_scores(
        result=result, tokens=tokens, catalog=catalog, num_edges=10,
    )

    assert edge_scores.edge_logit.shape[0] == 10
    assert edge_scores.edge_mask.shape[0] == 10
    # First 4 edges may have real values, rest should be -inf/masked
    assert not edge_scores.edge_mask[4:].any(), "Extra edges should be masked"
    print(f"  edge_logit shape with num_edges=10: {edge_scores.edge_logit.shape}")


def test_infeasible_result_produces_all_off():
    """An infeasible DP result (root_sigma=-1) should produce all-off logits."""
    _, tokens, leaves, catalog = _run_dp()

    # Create an infeasible result
    infeasible = OnePassDPResult(
        tour_cost=float("inf"),
        root_sigma=-1,
        leaf_states={},
        cost_tables={},
        stats={},
    )

    iface_logit, cross_logit = dp_result_to_logits(
        result=infeasible, tokens=tokens, catalog=catalog,
    )

    # All logits should be OFF (-10)
    assert (iface_logit == -10.0).all(), "Infeasible result should have all-off iface logits"
    assert (cross_logit == -10.0).all(), "Infeasible result should have all-off cross logits"
    print("  All logits correctly OFF for infeasible result")


def test_direct_reconstruction_manual_cycle_is_legal():
    """Direct traceback reconstruction should yield one legal Hamiltonian cycle."""
    tokens, leaves = build_manual_cycle_tree()
    catalog = build_boundary_state_catalog(num_slots=2, max_used=2, device=torch.device("cpu"))

    # State 1 is the only non-empty state for Ti=2: slots 0 and 1 paired together.
    pair_state = 1

    cost_tables = {}
    leaf_states = {}
    total = 0.0
    for row, nid in enumerate([1, 2, 3, 4]):
        costs = leaf_exact_solve(
            points_xy=leaves.point_xy[row],
            point_mask=leaves.point_mask[row],
            iface_eid=tokens.iface_eid[nid],
            iface_mask=tokens.iface_mask[nid],
            iface_boundary_dir=tokens.iface_boundary_dir[nid],
            iface_feat6=tokens.iface_feat6[nid],
            state_used_iface=catalog.used_iface,
            state_mate=catalog.mate,
            state_mask=torch.tensor([True, True], dtype=torch.bool),
            box_xy=tokens.tree_node_feat_rel[nid],
            is_root=False,
        )
        cost_tables[nid] = CostTableEntry(costs=costs, backptr={})
        leaf_states[nid] = pair_state
        total += float(costs[pair_state].item())

    root_costs = torch.full((catalog.used_iface.shape[0],), float("inf"), dtype=torch.float32)
    root_costs[0] = total
    cost_tables[0] = CostTableEntry(costs=root_costs, backptr={0: (pair_state, pair_state, pair_state, pair_state)})

    result = OnePassDPResult(
        tour_cost=total,
        root_sigma=0,
        leaf_states=leaf_states,
        cost_tables=cost_tables,
        stats={},
    )

    direct = reconstruct_tour_direct(
        result=result,
        tokens=tokens,
        leaves=leaves,
        catalog=catalog,
    )

    assert direct.feasible, direct.stats
    assert len(direct.order) == 4, direct.order
    assert set(direct.order) == {0, 1, 2, 3}, direct.order
    assert direct.length > 0.0
    print(f"  direct order: {direct.order}, length={direct.length:.4f}")


def test_end_to_end_dp_to_tour():
    """Full pipeline: DP → logits → edge scores → greedy decode → tour.

    This uses the actual `edge_scores.edge_logit` emitted by reconstruction
    instead of a dummy replacement, so it exercises the real glue code.
    """
    Ti, Tc, P, d = 8, 4, 4, 64
    result, tokens, leaves, catalog = _run_dp(Ti=Ti, Tc=Tc, P=P, d=d)

    # Step 1: DP result → edge scores
    edge_scores = dp_result_to_edge_scores(
        result=result, tokens=tokens, catalog=catalog,
    )

    # Step 2: Build a tiny local spanner with exactly one edge per eid.
    # The synthetic reconstruction above emits 4 edge logits (eid 0..3), so we
    # map them onto a 4-cycle over a square.
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    spanner_edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
        ],
        dtype=torch.long,
    )
    edge_logit_full = edge_scores.edge_logit.clone()
    assert edge_logit_full.shape[0] == spanner_edge_index.shape[1]

    # Step 3: Greedy decode
    tour_result = decode_tour_from_edge_logits(
        pos=pos,
        spanner_edge_index=spanner_edge_index,
        edge_logit=edge_logit_full,
        prefer_spanner_only=True,
        allow_off_spanner_patch=False,
    )

    N = pos.shape[0]
    assert len(tour_result.order) == N, f"Tour should visit all {N} points, got {len(tour_result.order)}"
    assert tour_result.length > 0, "Tour length should be positive"
    print(f"  Tour decoded: N={N}, length={tour_result.length:.4f}, feasible={tour_result.feasible}")
    print(f"  Order: {tour_result.order}")


# ─── CLI runner ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("dp_result_to_logits shape", test_dp_result_to_logits_shape),
        ("dp_result_to_logits values", test_dp_result_to_logits_values),
        ("dp_result_to_edge_scores shape", test_dp_result_to_edge_scores_shape),
        ("dp_result_to_edge_scores with num_edges", test_dp_result_to_edge_scores_with_num_edges),
        ("infeasible result → all off", test_infeasible_result_produces_all_off),
        ("direct reconstruction manual cycle", test_direct_reconstruction_manual_cycle_is_legal),
        ("end-to-end DP → tour", test_end_to_end_dp_to_tour),
    ]

    for name, fn in tests:
        print(f"Test: {name}...")
        fn()
        print("  PASS\n")

    print("All tour_reconstruct tests passed!")
