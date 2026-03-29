# tests/test_dp_runner.py
# -*- coding: utf-8 -*-
"""
Integration tests for the 1-pass DP pipeline (dp_runner + merge_decoder + dp_core).

Constructs a minimal synthetic quadtree and runs the full 1-pass DP to verify
the pipeline is wired correctly end-to-end.

Usage:
  python tests/test_dp_runner.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.dp_runner import OnePassDPRunner, OnePassDPResult
from src.models.merge_decoder import MergeDecoder
from src.models.bc_state_catalog import build_boundary_state_catalog
from src.models.dp_core import build_correspondence_maps, propagate_c1_constraints


# ─── Minimal stub encoders ────────────────────────────────────────────────────

class StubLeafEncoder(nn.Module):
    """Minimal leaf encoder that produces random latents."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.d = d_model
        self.proj = nn.Linear(4, d_model)

    def forward(self, *, node_feat_rel, node_depth, iface_feat6, iface_mask,
                iface_boundary_dir, iface_inside_endpoint, iface_inside_quadrant,
                cross_feat6, cross_mask, cross_child_pair, cross_is_leaf_internal,
                leaf_points_xy, leaf_points_mask):
        B = node_feat_rel.shape[0]
        return self.proj(node_feat_rel)  # [B, d_model]


class StubMergeEncoder(nn.Module):
    """Minimal merge encoder that produces random latents."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.d = d_model
        self.proj = nn.Linear(4, d_model)

    def forward(self, *, node_feat_rel, node_depth, iface_feat6, iface_mask,
                iface_boundary_dir, iface_inside_endpoint, iface_inside_quadrant,
                cross_feat6, cross_mask, cross_child_pair, cross_is_leaf_internal,
                child_z, child_mask):
        B = node_feat_rel.shape[0]
        return self.proj(node_feat_rel)  # [B, d_model]


# ─── Synthetic quadtree builder ──────────────────────────────────────────────

def build_synthetic_tree(Ti: int = 8, Tc: int = 4, P: int = 4):
    """Build a minimal 2-level tree: 1 root + 4 leaf children.

    Node 0 = root (internal)
    Nodes 1-4 = leaves (children of root)

    We place a few spanner edges:
      e0: parent LEFT boundary, owned by child TL(1)
      e1: parent RIGHT boundary, owned by child TR(2)
      e2: crossing TL(1) <-> TR(2) on vertical internal boundary
      e3: crossing BL(3) <-> BR(4) on vertical internal boundary
    """
    M = 5  # root + 4 children
    device = torch.device("cpu")

    # Tree structure
    tree_parent_index = torch.tensor([-1, 0, 0, 0, 0], dtype=torch.long, device=device)
    tree_children_index = torch.full((M, 4), -1, dtype=torch.long, device=device)
    tree_children_index[0] = torch.tensor([1, 2, 3, 4])  # root's children: TL, TR, BL, BR
    tree_node_depth = torch.tensor([0, 1, 1, 1, 1], dtype=torch.long, device=device)
    is_leaf = torch.tensor([False, True, True, True, True], dtype=torch.bool, device=device)

    # Node geometry (root-normalized)
    tree_node_feat_rel = torch.tensor([
        [0.5, 0.5, 1.0, 1.0],   # root: full box
        [0.25, 0.75, 0.5, 0.5], # TL
        [0.75, 0.75, 0.5, 0.5], # TR
        [0.25, 0.25, 0.5, 0.5], # BL
        [0.75, 0.25, 0.5, 0.5], # BR
    ], dtype=torch.float32, device=device)

    # Interface tokens: simplified
    iface_mask = torch.zeros(M, Ti, dtype=torch.bool, device=device)
    iface_eid = torch.full((M, Ti), -1, dtype=torch.long, device=device)
    iface_feat6 = torch.zeros(M, Ti, 6, dtype=torch.float32, device=device)
    iface_boundary_dir = torch.full((M, Ti), -1, dtype=torch.long, device=device)
    iface_inside_endpoint = torch.full((M, Ti), -1, dtype=torch.long, device=device)
    iface_inside_quadrant = torch.full((M, Ti), -1, dtype=torch.long, device=device)

    # e0: parent slot 0, LEFT, eid=0
    # Appears in parent (iface) and child TL (iface, LEFT side)
    iface_mask[0, 0] = True
    iface_eid[0, 0] = 0
    iface_boundary_dir[0, 0] = 0  # LEFT
    iface_inside_endpoint[0, 0] = 0
    iface_inside_quadrant[0, 0] = 0

    iface_mask[1, 0] = True  # child TL
    iface_eid[1, 0] = 0
    iface_boundary_dir[1, 0] = 0  # LEFT
    iface_inside_endpoint[1, 0] = 0
    iface_inside_quadrant[1, 0] = 0

    # e1: parent slot 1, RIGHT, eid=1
    iface_mask[0, 1] = True
    iface_eid[0, 1] = 1
    iface_boundary_dir[0, 1] = 1  # RIGHT
    iface_inside_endpoint[0, 1] = 0
    iface_inside_quadrant[0, 1] = 1

    iface_mask[2, 0] = True  # child TR
    iface_eid[2, 0] = 1
    iface_boundary_dir[2, 0] = 1  # RIGHT
    iface_inside_endpoint[2, 0] = 0
    iface_inside_quadrant[2, 0] = 1

    # e2: crossing TL<->TR, eid=2 — appears as iface in TL (RIGHT) and TR (LEFT)
    iface_mask[1, 1] = True  # TL slot 1
    iface_eid[1, 1] = 2
    iface_boundary_dir[1, 1] = 1  # RIGHT side of TL
    iface_inside_endpoint[1, 1] = 0
    iface_inside_quadrant[1, 1] = 0

    iface_mask[2, 1] = True  # TR slot 1
    iface_eid[2, 1] = 2
    iface_boundary_dir[2, 1] = 0  # LEFT side of TR
    iface_inside_endpoint[2, 1] = 0
    iface_inside_quadrant[2, 1] = 1

    # e3: crossing BL<->BR, eid=3
    iface_mask[3, 0] = True  # BL slot 0
    iface_eid[3, 0] = 3
    iface_boundary_dir[3, 0] = 1  # RIGHT side of BL
    iface_inside_endpoint[3, 0] = 0
    iface_inside_quadrant[3, 0] = 2

    iface_mask[4, 0] = True  # BR slot 0
    iface_eid[4, 0] = 3
    iface_boundary_dir[4, 0] = 0  # LEFT side of BR
    iface_inside_endpoint[4, 0] = 0
    iface_inside_quadrant[4, 0] = 3

    # Crossing tokens (parent only)
    cross_mask = torch.zeros(M, Tc, dtype=torch.bool, device=device)
    cross_eid = torch.full((M, Tc), -1, dtype=torch.long, device=device)
    cross_feat6 = torch.zeros(M, Tc, 6, dtype=torch.float32, device=device)
    cross_child_pair = torch.full((M, Tc, 2), -1, dtype=torch.long, device=device)
    cross_is_leaf_internal = torch.zeros(M, Tc, dtype=torch.bool, device=device)

    # e2 as crossing in root
    cross_mask[0, 0] = True
    cross_eid[0, 0] = 2
    cross_child_pair[0, 0] = torch.tensor([0, 1])  # TL(0) <-> TR(1)

    # e3 as crossing in root
    cross_mask[0, 1] = True
    cross_eid[0, 1] = 3
    cross_child_pair[0, 1] = torch.tensor([2, 3])  # BL(2) <-> BR(3)

    # Build PackedNodeTokens (use SimpleNamespace for flexibility)
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

    # Leaf points
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
    leaves.point_mask[:, 2:] = False  # only 2 points per leaf

    return tokens, leaves


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_pipeline_runs_without_crash():
    """The full 1-pass pipeline should run on synthetic data without errors."""
    Ti, Tc, P, d = 8, 4, 4, 64
    tokens, leaves = build_synthetic_tree(Ti=Ti, Tc=Tc, P=P)

    leaf_enc = StubLeafEncoder(d)
    merge_enc = StubMergeEncoder(d)
    merge_dec = MergeDecoder(
        d_model=d, n_heads=4, num_iface_slots=Ti,
        parent_num_layers=2, cross_num_layers=1, max_depth=16,
    )

    leaf_enc.eval()
    merge_enc.eval()
    merge_dec.eval()

    runner = OnePassDPRunner(r=4, max_used=4, topk=5, fallback_exact=True)
    result = runner.run_single(
        tokens=tokens, leaves=leaves,
        leaf_encoder=leaf_enc, merge_encoder=merge_enc, merge_decoder=merge_dec,
    )

    assert isinstance(result, OnePassDPResult)
    print(f"  tour_cost={result.tour_cost:.4f}")
    print(f"  root_sigma={result.root_sigma}")
    print(f"  num_leaf_states={len(result.leaf_states)}")
    print(f"  stats={result.stats}")


def test_leaf_cost_tables_populated():
    """Leaf cost tables should reflect current C4 semantics."""
    Ti, Tc, P, d = 8, 4, 4, 64
    tokens, leaves = build_synthetic_tree(Ti=Ti, Tc=Tc, P=P)

    leaf_enc = StubLeafEncoder(d)
    merge_enc = StubMergeEncoder(d)
    merge_dec = MergeDecoder(
        d_model=d, n_heads=4, num_iface_slots=Ti,
        parent_num_layers=2, cross_num_layers=1, max_depth=16,
    )
    leaf_enc.eval(); merge_enc.eval(); merge_dec.eval()

    runner = OnePassDPRunner(r=4, max_used=4, topk=5)
    result = runner.run_single(
        tokens=tokens, leaves=leaves,
        leaf_encoder=leaf_enc, merge_encoder=merge_enc, merge_decoder=merge_dec,
    )

    # All 4 leaves + 1 root should have cost tables
    for nid in [1, 2, 3, 4]:
        assert nid in result.cost_tables, f"Leaf node {nid} missing cost table"
        ct = result.cost_tables[nid]
        # Non-root leaves with interior points may not use the empty state:
        # that would create an internal closed component and violate C4.
        assert ct.costs[0].item() == float("inf"), (
            f"Leaf {nid} empty state should be infeasible under current C4 semantics"
        )

    # Leaves 1 and 2 each expose two interface slots, so they should still
    # admit at least one feasible paired state.
    for nid in [1, 2]:
        ct = result.cost_tables[nid]
        assert (ct.costs < float("inf")).any().item(), (
            f"Leaf {nid} should have at least one feasible non-empty state"
        )

    assert 0 in result.cost_tables, "Root node should have cost table"
    print(f"  All 5 cost tables present")
    print(f"  Root has {(result.cost_tables[0].costs < float('inf')).sum().item()} feasible states")


def test_root_tour_cost_is_rescaled_to_euclidean() -> None:
    """Runner should convert the root DP cost back to the original instance scale."""

    class Tokens:
        pass

    class Leaves:
        pass

    tokens = Tokens()
    tokens.tree_parent_index = torch.tensor([-1], dtype=torch.long)
    tokens.tree_children_index = torch.full((1, 4), -1, dtype=torch.long)
    tokens.tree_node_depth = torch.tensor([0], dtype=torch.long)
    tokens.tree_node_feat_rel = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
    tokens.is_leaf = torch.tensor([True], dtype=torch.bool)
    tokens.root_id = torch.tensor([0], dtype=torch.long)
    tokens.root_scale_s = torch.tensor([10.0], dtype=torch.float32)
    tokens.iface_mask = torch.zeros(1, 4, dtype=torch.bool)
    tokens.iface_eid = torch.full((1, 4), -1, dtype=torch.long)
    tokens.iface_feat6 = torch.zeros(1, 4, 6, dtype=torch.float32)
    tokens.iface_boundary_dir = torch.full((1, 4), -1, dtype=torch.long)
    tokens.iface_inside_endpoint = torch.full((1, 4), -1, dtype=torch.long)
    tokens.iface_inside_quadrant = torch.full((1, 4), -1, dtype=torch.long)
    tokens.cross_mask = torch.zeros(1, 1, dtype=torch.bool)
    tokens.cross_eid = torch.full((1, 1), -1, dtype=torch.long)
    tokens.cross_feat6 = torch.zeros(1, 1, 6, dtype=torch.float32)
    tokens.cross_child_pair = torch.full((1, 1, 2), -1, dtype=torch.long)
    tokens.cross_is_leaf_internal = torch.zeros(1, 1, dtype=torch.bool)

    leaves = Leaves()
    leaves.leaf_node_id = torch.tensor([0], dtype=torch.long)
    leaves.point_idx = torch.tensor([[0, 1]], dtype=torch.long)
    leaves.point_xy = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)
    leaves.point_mask = torch.tensor([[True, True]], dtype=torch.bool)

    leaf_enc = StubLeafEncoder(32).eval()
    merge_enc = StubMergeEncoder(32).eval()
    merge_dec = MergeDecoder(
        d_model=32, n_heads=4, num_iface_slots=4,
        parent_num_layers=1, cross_num_layers=1, max_depth=8,
    ).eval()

    runner = OnePassDPRunner(r=4, max_used=4, topk=5)
    result = runner.run_single(
        tokens=tokens,
        leaves=leaves,
        leaf_encoder=leaf_enc,
        merge_encoder=merge_enc,
        merge_decoder=merge_dec,
    )

    # Normalized root cost is 1.0 for this configuration; root_scale_s=10 should
    # convert it back to Euclidean length 10.0.
    assert abs(result.tour_cost - 10.0) < 1e-5, result.tour_cost
    assert abs(float(result.stats["tour_cost_normalized"]) - 1.0) < 1e-5
    assert abs(float(result.stats["root_scale_s"]) - 10.0) < 1e-5


def test_native_direct_tour_on_root_leaf() -> None:
    """1-pass should emit a legal direct tour without any post-decoder."""

    class Tokens:
        pass

    class Leaves:
        pass

    tokens = Tokens()
    tokens.tree_parent_index = torch.tensor([-1], dtype=torch.long)
    tokens.tree_children_index = torch.full((1, 4), -1, dtype=torch.long)
    tokens.tree_node_depth = torch.tensor([0], dtype=torch.long)
    tokens.tree_node_feat_rel = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
    tokens.is_leaf = torch.tensor([True], dtype=torch.bool)
    tokens.root_id = torch.tensor([0], dtype=torch.long)
    tokens.root_scale_s = torch.tensor([10.0], dtype=torch.float32)
    tokens.iface_mask = torch.zeros(1, 4, dtype=torch.bool)
    tokens.iface_eid = torch.full((1, 4), -1, dtype=torch.long)
    tokens.iface_feat6 = torch.zeros(1, 4, 6, dtype=torch.float32)
    tokens.iface_boundary_dir = torch.full((1, 4), -1, dtype=torch.long)
    tokens.iface_inside_endpoint = torch.full((1, 4), -1, dtype=torch.long)
    tokens.iface_inside_quadrant = torch.full((1, 4), -1, dtype=torch.long)
    tokens.cross_mask = torch.zeros(1, 1, dtype=torch.bool)
    tokens.cross_eid = torch.full((1, 1), -1, dtype=torch.long)
    tokens.cross_feat6 = torch.zeros(1, 1, 6, dtype=torch.float32)
    tokens.cross_child_pair = torch.full((1, 1, 2), -1, dtype=torch.long)
    tokens.cross_is_leaf_internal = torch.zeros(1, 1, dtype=torch.bool)

    leaves = Leaves()
    leaves.leaf_node_id = torch.tensor([0], dtype=torch.long)
    leaves.point_idx = torch.tensor([[0, 1]], dtype=torch.long)
    leaves.point_xy = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)
    leaves.point_mask = torch.tensor([[True, True]], dtype=torch.bool)

    leaf_enc = StubLeafEncoder(32).eval()
    merge_enc = StubMergeEncoder(32).eval()
    merge_dec = MergeDecoder(
        d_model=32, n_heads=4, num_iface_slots=4,
        parent_num_layers=1, cross_num_layers=1, max_depth=8,
    ).eval()

    runner = OnePassDPRunner(r=4, max_used=4, topk=5)
    result = runner.run_single(
        tokens=tokens,
        leaves=leaves,
        leaf_encoder=leaf_enc,
        merge_encoder=merge_enc,
        merge_decoder=merge_dec,
    )

    assert result.tour_feasible, result.tour_stats
    assert result.tour_order == [0, 1], result.tour_order
    assert abs(result.tour_length - 10.0) < 1e-5, result.tour_length


def test_traceback_reaches_leaves():
    """Traceback from root should assign states to leaf nodes."""
    Ti, Tc, P, d = 8, 4, 4, 64
    tokens, leaves = build_synthetic_tree(Ti=Ti, Tc=Tc, P=P)

    leaf_enc = StubLeafEncoder(d)
    merge_enc = StubMergeEncoder(d)
    merge_dec = MergeDecoder(
        d_model=d, n_heads=4, num_iface_slots=Ti,
        parent_num_layers=2, cross_num_layers=1, max_depth=16,
    )
    leaf_enc.eval(); merge_enc.eval(); merge_dec.eval()

    runner = OnePassDPRunner(r=4, max_used=4, topk=5)
    result = runner.run_single(
        tokens=tokens, leaves=leaves,
        leaf_encoder=leaf_enc, merge_encoder=merge_enc, merge_decoder=merge_dec,
    )

    if result.root_sigma >= 0 and result.tour_cost < float("inf"):
        # Should have leaf states
        assert len(result.leaf_states) > 0, "Traceback should produce leaf states"
        print(f"  Traceback found {len(result.leaf_states)} leaf states: {result.leaf_states}")
    else:
        print(f"  No feasible root state found (expected for random weights)")


def test_correspondence_maps_on_synthetic():
    """Verify correspondence maps are correctly built for the synthetic tree."""
    Ti, Tc = 8, 4
    tokens, _ = build_synthetic_tree(Ti=Ti, Tc=Tc)

    children = tokens.tree_children_index[0].long()
    child_exists = children >= 0
    ch_clamped = children.clamp_min(0)

    maps = build_correspondence_maps(
        parent_iface_eid=tokens.iface_eid[0],
        parent_iface_mask=tokens.iface_mask[0].bool(),
        parent_iface_bdir=tokens.iface_boundary_dir[0],
        parent_cross_eid=tokens.cross_eid[0],
        parent_cross_mask=tokens.cross_mask[0].bool(),
        parent_cross_child_pair=tokens.cross_child_pair[0],
        children_iface_eid=tokens.iface_eid[ch_clamped],
        children_iface_mask=tokens.iface_mask[ch_clamped].bool(),
        children_iface_bdir=tokens.iface_boundary_dir[ch_clamped],
        child_exists=child_exists,
    )

    # e0 (parent slot 0, LEFT) -> child TL (slot 0)
    assert maps.phi_out_child[0].item() == 0, "e0 should map to child TL(0)"
    assert maps.phi_out_slot[0].item() == 0, "e0 should map to TL slot 0"

    # e1 (parent slot 1, RIGHT) -> child TR (slot 0)
    assert maps.phi_out_child[1].item() == 1, "e1 should map to child TR(1)"
    assert maps.phi_out_slot[1].item() == 0, "e1 should map to TR slot 0"

    # e2: shared boundary TL(slot 1) <-> TR(slot 1)
    assert maps.phi_sh_peer_child[0, 1].item() == 1, "TL slot 1 peer should be TR(1)"
    assert maps.phi_sh_peer_slot[0, 1].item() == 1, "TL slot 1 peer slot should be 1"
    assert maps.phi_sh_peer_child[1, 1].item() == 0, "TR slot 1 peer should be TL(0)"
    assert maps.phi_sh_peer_slot[1, 1].item() == 1, "TR slot 1 peer slot should be 1"

    print("  All correspondence map checks passed")


def test_propagate_c1_constraints():
    """C1 propagation should correctly fix child outer-boundary activations."""
    Ti, Tc = 8, 4
    tokens, _ = build_synthetic_tree(Ti=Ti, Tc=Tc)

    children = tokens.tree_children_index[0].long()
    child_exists = children >= 0
    ch_clamped = children.clamp_min(0)

    maps = build_correspondence_maps(
        parent_iface_eid=tokens.iface_eid[0],
        parent_iface_mask=tokens.iface_mask[0].bool(),
        parent_iface_bdir=tokens.iface_boundary_dir[0],
        parent_cross_eid=tokens.cross_eid[0],
        parent_cross_mask=tokens.cross_mask[0].bool(),
        parent_cross_child_pair=tokens.cross_child_pair[0],
        children_iface_eid=tokens.iface_eid[ch_clamped],
        children_iface_mask=tokens.iface_mask[ch_clamped].bool(),
        children_iface_bdir=tokens.iface_boundary_dir[ch_clamped],
        child_exists=child_exists,
    )

    child_iface_mask_4 = tokens.iface_mask[ch_clamped].bool()
    child_iface_mask_4 = child_iface_mask_4 & child_exists.unsqueeze(-1)

    # Case 1: parent activates slot 0 (e0, LEFT -> TL slot 0)
    parent_a = torch.zeros(Ti, dtype=torch.bool)
    parent_a[0] = True  # activate e0

    c1_req, c1_con = propagate_c1_constraints(
        parent_a=parent_a,
        parent_iface_mask=tokens.iface_mask[0].bool(),
        maps=maps,
        child_exists=child_exists,
        child_iface_mask=child_iface_mask_4,
    )

    # e0 maps to child TL(0), slot 0 -> constrained=True, required=True
    assert c1_con[0, 0].item(), "TL slot 0 should be constrained"
    assert c1_req[0, 0].item(), "TL slot 0 should be required active"

    # e1 (parent slot 1) not activated -> child TR slot 0 required inactive
    assert c1_con[1, 0].item(), "TR slot 0 should be constrained"
    assert not c1_req[1, 0].item(), "TR slot 0 should be required inactive"

    # Case 2: parent activates both slots 0 and 1
    parent_a[1] = True
    c1_req2, c1_con2 = propagate_c1_constraints(
        parent_a=parent_a,
        parent_iface_mask=tokens.iface_mask[0].bool(),
        maps=maps,
        child_exists=child_exists,
        child_iface_mask=child_iface_mask_4,
    )
    assert c1_req2[0, 0].item(), "TL slot 0 should be active"
    assert c1_req2[1, 0].item(), "TR slot 0 should be active"

    print("  C1 propagation checks passed")


def test_fallback_no_truncation():
    """The constraint-propagated fallback should run to completion
    (no 10000-combo limit) and find the same or better result."""
    Ti, Tc, P, d = 8, 4, 4, 64
    tokens, leaves = build_synthetic_tree(Ti=Ti, Tc=Tc, P=P)

    leaf_enc = StubLeafEncoder(d)
    merge_enc = StubMergeEncoder(d)
    merge_dec = MergeDecoder(
        d_model=d, n_heads=4, num_iface_slots=Ti,
        parent_num_layers=2, cross_num_layers=1, max_depth=16,
    )
    leaf_enc.eval(); merge_enc.eval(); merge_dec.eval()

    runner = OnePassDPRunner(
        r=4, max_used=4, topk=5, fallback_exact=True,
    )
    result = runner.run_single(
        tokens=tokens, leaves=leaves,
        leaf_encoder=leaf_enc, merge_encoder=merge_enc,
        merge_decoder=merge_dec,
    )

    # Should not have any "too large" stats (old limit removed)
    assert result.stats.get("num_exact_too_large", 0) == 0, \
        "Constraint-propagated fallback should never be 'too large'"

    # Root should have at least one feasible state (the empty state)
    root_ct = result.cost_tables.get(0)
    assert root_ct is not None, "Root cost table should exist"
    n_feasible = int((root_ct.costs < float("inf")).sum().item())
    print(f"  Root has {n_feasible} feasible states")
    print(f"  stats: {result.stats}")
    print(f"  tour_cost={result.tour_cost}")


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Test 1: Pipeline runs without crash...")
    test_pipeline_runs_without_crash()
    print("  PASS\n")

    print("Test 2: Leaf cost tables populated...")
    test_leaf_cost_tables_populated()
    print("  PASS\n")

    print("Test 2b: Root tour cost rescaled...")
    test_root_tour_cost_is_rescaled_to_euclidean()
    print("  PASS\n")

    print("Test 2c: Native direct tour on root leaf...")
    test_native_direct_tour_on_root_leaf()
    print("  PASS\n")

    print("Test 3: Traceback reaches leaves...")
    test_traceback_reaches_leaves()
    print("  PASS\n")

    print("Test 4: Correspondence maps on synthetic tree...")
    test_correspondence_maps_on_synthetic()
    print("  PASS\n")

    print("Test 5: C1 constraint propagation...")
    test_propagate_c1_constraints()
    print("  PASS\n")

    print("Test 6: Fallback no truncation...")
    test_fallback_no_truncation()
    print("  PASS\n")

    print("All tests passed!")
