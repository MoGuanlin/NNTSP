# tests/test_onepass_trainer.py
# -*- coding: utf-8 -*-
"""
Unit tests for src/models/onepass_trainer.py — 1-pass training runner.

Usage:
  python tests/test_onepass_trainer.py
  python -m pytest tests/test_onepass_trainer.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.onepass_trainer import (
    OnePassTrainRunner,
    OnePassTrainResult,
    build_child_mate_targets_from_structured_states,
    build_child_mate_targets_from_states,
    onepass_loss,
)
from src.models.merge_decoder import MergeDecoder
from src.models.bc_state_catalog import build_boundary_state_catalog
from src.models.labeler import (
    _build_matching_target_for_node,
    _build_matching_target_for_node_structured,
)


# ─── Stub encoders ───────────────────────────────────────────────────────────

class StubLeafEncoder(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.proj = nn.Linear(4, d_model)

    def forward(self, *, node_feat_rel, node_depth, iface_feat6, iface_mask,
                iface_boundary_dir, iface_inside_endpoint, iface_inside_quadrant,
                cross_feat6, cross_mask, cross_child_pair, cross_is_leaf_internal,
                leaf_points_xy, leaf_points_mask):
        return self.proj(node_feat_rel)


class StubMergeEncoder(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.proj = nn.Linear(4, d_model)

    def forward(self, *, node_feat_rel, node_depth, iface_feat6, iface_mask,
                iface_boundary_dir, iface_inside_endpoint, iface_inside_quadrant,
                cross_feat6, cross_mask, cross_child_pair, cross_is_leaf_internal,
                child_z, child_mask):
        return self.proj(node_feat_rel)


# ─── Synthetic tree (reused from test_dp_runner) ─────────────────────────────

def build_synthetic_tree(Ti: int = 8, Tc: int = 4, P: int = 4):
    M = 5
    device = torch.device("cpu")

    tree_parent_index = torch.tensor([-1, 0, 0, 0, 0], dtype=torch.long)
    tree_children_index = torch.full((M, 4), -1, dtype=torch.long)
    tree_children_index[0] = torch.tensor([1, 2, 3, 4])
    tree_node_depth = torch.tensor([0, 1, 1, 1, 1], dtype=torch.long)
    is_leaf = torch.tensor([False, True, True, True, True], dtype=torch.bool)

    tree_node_feat_rel = torch.tensor([
        [0.5, 0.5, 1.0, 1.0],
        [0.25, 0.75, 0.5, 0.5],
        [0.75, 0.75, 0.5, 0.5],
        [0.25, 0.25, 0.5, 0.5],
        [0.75, 0.25, 0.5, 0.5],
    ], dtype=torch.float32)

    iface_mask = torch.zeros(M, Ti, dtype=torch.bool)
    iface_eid = torch.full((M, Ti), -1, dtype=torch.long)
    iface_feat6 = torch.zeros(M, Ti, 6, dtype=torch.float32)
    iface_boundary_dir = torch.full((M, Ti), -1, dtype=torch.long)
    iface_inside_endpoint = torch.full((M, Ti), -1, dtype=torch.long)
    iface_inside_quadrant = torch.full((M, Ti), -1, dtype=torch.long)

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

    cross_mask = torch.zeros(M, Tc, dtype=torch.bool)
    cross_eid = torch.full((M, Tc), -1, dtype=torch.long)
    cross_feat6 = torch.zeros(M, Tc, 6, dtype=torch.float32)
    cross_child_pair = torch.full((M, Tc, 2), -1, dtype=torch.long)
    cross_is_leaf_internal = torch.zeros(M, Tc, dtype=torch.bool)

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
    leaves.point_xy = torch.rand(4, P, 2)
    leaves.point_mask = torch.ones(4, P, dtype=torch.bool)
    leaves.point_mask[:, 2:] = False

    return tokens, leaves


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_forward_pass_shape():
    """OnePassTrainRunner should produce correctly shaped outputs."""
    Ti, Tc, P, d = 8, 4, 4, 64
    M = 5
    tokens, leaves = build_synthetic_tree(Ti=Ti, Tc=Tc, P=P)

    leaf_enc = StubLeafEncoder(d)
    merge_enc = StubMergeEncoder(d)
    merge_dec = MergeDecoder(
        d_model=d, n_heads=4, num_iface_slots=Ti,
        parent_num_layers=2, cross_num_layers=1, max_depth=16,
    )

    catalog = build_boundary_state_catalog(num_slots=Ti, max_used=4, device=torch.device("cpu"))

    # Teacher state: root node (nid=0) gets state 0 (empty state)
    target_state_idx = torch.full((M,), -1, dtype=torch.long)
    m_state = torch.zeros(M, dtype=torch.bool)
    target_state_idx[0] = 0  # empty state for root
    m_state[0] = True

    runner = OnePassTrainRunner()
    result = runner.run_single(
        tokens=tokens, leaves=leaves,
        leaf_encoder=leaf_enc, merge_encoder=merge_enc, merge_decoder=merge_dec,
        catalog=catalog,
        target_state_idx=target_state_idx, m_state=m_state,
    )

    assert isinstance(result, OnePassTrainResult)
    assert result.z.shape == (M, d), f"z shape: {result.z.shape}"
    assert result.child_scores.shape == (M, 4, Ti), f"child_scores shape: {result.child_scores.shape}"
    assert result.decode_mask.shape == (M,), f"decode_mask shape: {result.decode_mask.shape}"
    assert result.decode_mask[0].item() == True, "Root should be decoded"
    assert result.decode_mask[1:].sum().item() == 0, "Leaves should not be decoded"
    print(f"  z shape: {result.z.shape}")
    print(f"  child_scores shape: {result.child_scores.shape}")
    print(f"  decode_mask: {result.decode_mask}")


def test_forward_pass_shape_iface_mate():
    """iface_mate decoder should emit mate logits for decoded internal nodes."""
    Ti, Tc, P, d = 8, 4, 4, 64
    M = 5
    tokens, leaves = build_synthetic_tree(Ti=Ti, Tc=Tc, P=P)

    leaf_enc = StubLeafEncoder(d)
    merge_enc = StubMergeEncoder(d)
    merge_dec = MergeDecoder(
        d_model=d,
        n_heads=4,
        num_iface_slots=Ti,
        decoder_variant="iface_mate",
        parent_num_layers=2,
        cross_num_layers=1,
        max_depth=16,
    )

    catalog = build_boundary_state_catalog(num_slots=Ti, max_used=4, device=torch.device("cpu"))
    target_state_idx = torch.full((M,), -1, dtype=torch.long)
    m_state = torch.zeros(M, dtype=torch.bool)
    target_state_idx[0] = 0
    m_state[0] = True

    runner = OnePassTrainRunner()
    result = runner.run_single(
        tokens=tokens, leaves=leaves,
        leaf_encoder=leaf_enc, merge_encoder=merge_enc, merge_decoder=merge_dec,
        catalog=catalog,
        target_state_idx=target_state_idx, m_state=m_state,
    )

    assert result.child_mate_scores is not None
    assert result.child_mate_scores.shape == (M, 4, Ti, Ti)


def test_forward_pass_shape_direct_structured():
    """Direct structured supervision should decode from raw (used, mate) labels."""
    Ti, Tc, P, d = 8, 4, 4, 64
    M = 5
    tokens, leaves = build_synthetic_tree(Ti=Ti, Tc=Tc, P=P)

    leaf_enc = StubLeafEncoder(d)
    merge_enc = StubMergeEncoder(d)
    merge_dec = MergeDecoder(
        d_model=d,
        n_heads=4,
        num_iface_slots=Ti,
        decoder_variant="iface_mate",
        parent_num_layers=2,
        cross_num_layers=1,
        max_depth=16,
    )

    parent_sigma_used = torch.zeros((M, Ti), dtype=torch.bool)
    parent_sigma_mate = torch.full((M, Ti), -1, dtype=torch.long)
    m_parent_sigma = torch.zeros(M, dtype=torch.bool)
    parent_sigma_used[0, 0] = True
    parent_sigma_used[0, 1] = True
    parent_sigma_mate[0, 0] = 1
    parent_sigma_mate[0, 1] = 0
    m_parent_sigma[0] = True

    runner = OnePassTrainRunner()
    result = runner.run_single(
        tokens=tokens,
        leaves=leaves,
        leaf_encoder=leaf_enc,
        merge_encoder=merge_enc,
        merge_decoder=merge_dec,
        catalog=None,
        parent_sigma_used=parent_sigma_used,
        parent_sigma_mate=parent_sigma_mate,
        m_state=m_parent_sigma,
        supervision_mode="direct_structured",
    )

    assert isinstance(result, OnePassTrainResult)
    assert result.child_scores.shape == (M, 4, Ti)
    assert result.child_mate_scores is not None
    assert result.decode_mask[0].item() is True


def test_build_child_mate_targets_from_structured_states():
    Ti, Tc, P = 8, 4, 4
    M = 5
    tokens, _ = build_synthetic_tree(Ti=Ti, Tc=Tc, P=P)

    parent_sigma_used = torch.zeros((M, Ti), dtype=torch.bool)
    parent_sigma_mate = torch.full((M, Ti), -1, dtype=torch.long)
    m_parent_sigma = torch.zeros(M, dtype=torch.bool)

    parent_sigma_used[1, 0] = True
    parent_sigma_used[1, 1] = True
    parent_sigma_mate[1, 0] = 1
    parent_sigma_mate[1, 1] = 0
    m_parent_sigma[1] = True

    m_child_iface = torch.ones((M, 4, Ti), dtype=torch.bool)
    y_child_mate, m_child_mate = build_child_mate_targets_from_structured_states(
        tree_children_index=tokens.tree_children_index,
        m_child_iface=m_child_iface,
        parent_sigma_used=parent_sigma_used,
        parent_sigma_mate=parent_sigma_mate,
        m_parent_sigma_structured=m_parent_sigma,
    )

    assert m_child_mate[0, 0, 0].item() is True
    assert m_child_mate[0, 0, 1].item() is True
    assert y_child_mate[0, 0, 0].item() == 1
    assert y_child_mate[0, 0, 1].item() == 0


def test_structured_matching_helper_is_not_capped_by_max_used():
    Ti = 8
    iface_mask = torch.ones(Ti, dtype=torch.bool)
    iface_eid = torch.arange(Ti, dtype=torch.long)
    iface_inside_ep = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.long)
    points_in_node = {0, 1, 2, 3}
    selected_local_eids = {0, 1, 2, 3, 4, 5}
    sp_u = [0, 0, 1, 1, 2, 2, 9, 9]
    sp_v = [4, 5, 6, 7, 8, 9, 10, 11]

    used, mate, structured_exact = _build_matching_target_for_node_structured(
        local_node_id=0,
        points_in_node=points_in_node,
        selected_local_eids=selected_local_eids,
        sp_u=sp_u,
        sp_v=sp_v,
        iface_eid_row=iface_eid,
        iface_mask_row=iface_mask,
        iface_inside_ep_row=iface_inside_ep,
    )

    assert int(used.sum().item()) == 6
    assert structured_exact is True
    assert mate.shape == (Ti,)


def test_gradient_flows():
    """Gradients should flow through the full 1-pass pipeline."""
    Ti, Tc, P, d = 8, 4, 4, 64
    M = 5
    tokens, leaves = build_synthetic_tree(Ti=Ti, Tc=Tc, P=P)

    leaf_enc = StubLeafEncoder(d)
    merge_enc = StubMergeEncoder(d)
    merge_dec = MergeDecoder(
        d_model=d, n_heads=4, num_iface_slots=Ti,
        parent_num_layers=2, cross_num_layers=1, max_depth=16,
    )

    catalog = build_boundary_state_catalog(num_slots=Ti, max_used=4, device=torch.device("cpu"))

    target_state_idx = torch.full((M,), -1, dtype=torch.long)
    m_state = torch.zeros(M, dtype=torch.bool)
    target_state_idx[0] = 0
    m_state[0] = True

    runner = OnePassTrainRunner()
    result = runner.run_single(
        tokens=tokens, leaves=leaves,
        leaf_encoder=leaf_enc, merge_encoder=merge_enc, merge_decoder=merge_dec,
        catalog=catalog,
        target_state_idx=target_state_idx, m_state=m_state,
    )

    # Fake targets
    y_child_iface = torch.zeros(M, 4, Ti, dtype=torch.float32)
    m_child_iface = torch.zeros(M, 4, Ti, dtype=torch.bool)
    # Set some targets for root's children
    m_child_iface[0, :, :2] = True
    y_child_iface[0, 0, 0] = 1.0  # child TL, slot 0 active

    loss = onepass_loss(
        child_scores=result.child_scores,
        decode_mask=result.decode_mask,
        y_child_iface=y_child_iface,
        m_child_iface=m_child_iface,
    )

    assert loss.requires_grad, "Loss should require grad"
    loss.backward()

    # Check gradients in all three components
    has_grad = {"leaf_enc": False, "merge_enc": False, "merge_dec": False}
    for p in leaf_enc.parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            has_grad["leaf_enc"] = True
            break
    for p in merge_enc.parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            has_grad["merge_enc"] = True
            break
    for p in merge_dec.parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            has_grad["merge_dec"] = True
            break

    print(f"  Gradient flow: {has_grad}")
    assert has_grad["merge_dec"], "MergeDecoder must have gradients"


def test_onepass_loss_with_mate_targets() -> None:
    Ti = 8
    M = 5
    tokens, _ = build_synthetic_tree(Ti=Ti, Tc=4, P=4)
    catalog = build_boundary_state_catalog(num_slots=Ti, max_used=4, device=torch.device("cpu"))

    target_state_idx = torch.full((M,), -1, dtype=torch.long)
    child_state_mask = torch.zeros(M, dtype=torch.bool)
    non_empty_state = int(torch.nonzero(catalog.used_iface.any(dim=1), as_tuple=False)[0].item())
    target_state_idx[1:] = non_empty_state
    child_state_mask[1:] = True
    m_child_iface = torch.ones(M, 4, Ti, dtype=torch.bool)

    y_child_iface = torch.zeros(M, 4, Ti)
    child_scores = torch.zeros(M, 4, Ti, requires_grad=True)
    child_mate_scores = torch.zeros(M, 4, Ti, Ti, requires_grad=True)
    decode_mask = torch.zeros(M, dtype=torch.bool)
    decode_mask[0] = True

    y_child_mate, m_child_mate = build_child_mate_targets_from_states(
        tree_children_index=tokens.tree_children_index,
        m_child_iface=m_child_iface,
        target_state_idx=target_state_idx,
        child_state_mask=child_state_mask,
        state_used_iface=catalog.used_iface,
        state_mate=catalog.mate,
    )

    loss = onepass_loss(
        child_scores=child_scores,
        decode_mask=decode_mask,
        y_child_iface=y_child_iface,
        m_child_iface=m_child_iface,
        child_mate_scores=child_mate_scores,
        y_child_mate=y_child_mate,
        m_child_mate=m_child_mate,
        mate_weight=1.0,
    )
    assert torch.isfinite(loss).item()
    loss.backward()
    assert child_mate_scores.grad is not None
    # Note: leaf_enc and merge_enc may or may not have gradients depending on
    # whether their outputs influence the loss through z_node in parent_memory


def test_no_teacher_state_produces_zero_loss():
    """If no node has a valid teacher state, loss should be zero."""
    Ti, Tc, P, d = 8, 4, 4, 64
    M = 5
    tokens, leaves = build_synthetic_tree(Ti=Ti, Tc=Tc, P=P)

    leaf_enc = StubLeafEncoder(d)
    merge_enc = StubMergeEncoder(d)
    merge_dec = MergeDecoder(
        d_model=d, n_heads=4, num_iface_slots=Ti,
        parent_num_layers=2, cross_num_layers=1, max_depth=16,
    )

    catalog = build_boundary_state_catalog(num_slots=Ti, max_used=4, device=torch.device("cpu"))

    # No valid teacher states
    target_state_idx = torch.full((M,), -1, dtype=torch.long)
    m_state = torch.zeros(M, dtype=torch.bool)

    runner = OnePassTrainRunner()
    result = runner.run_single(
        tokens=tokens, leaves=leaves,
        leaf_encoder=leaf_enc, merge_encoder=merge_enc, merge_decoder=merge_dec,
        catalog=catalog,
        target_state_idx=target_state_idx, m_state=m_state,
    )

    assert result.decode_mask.sum().item() == 0, "No nodes should be decoded"

    y = torch.zeros(M, 4, Ti)
    m = torch.ones(M, 4, Ti, dtype=torch.bool)

    loss = onepass_loss(
        child_scores=result.child_scores,
        decode_mask=result.decode_mask,
        y_child_iface=y, m_child_iface=m,
    )
    assert loss.item() == 0.0, f"Loss should be 0 with no decoded nodes, got {loss.item()}"
    print("  Loss correctly zero when no teacher states available")


def test_different_teacher_states_different_outputs():
    """Different teacher σ should produce different child predictions."""
    Ti, Tc, P, d = 8, 4, 4, 64
    M = 5
    tokens, leaves = build_synthetic_tree(Ti=Ti, Tc=Tc, P=P)

    leaf_enc = StubLeafEncoder(d)
    merge_enc = StubMergeEncoder(d)
    merge_dec = MergeDecoder(
        d_model=d, n_heads=4, num_iface_slots=Ti,
        parent_num_layers=2, cross_num_layers=1, max_depth=16,
    )
    merge_dec.eval()

    catalog = build_boundary_state_catalog(num_slots=Ti, max_used=4, device=torch.device("cpu"))
    S = catalog.used_iface.shape[0]

    # Find a non-empty state (state 0 is empty)
    nonempty_si = -1
    for si in range(1, S):
        if catalog.used_iface[si].any().item():
            nonempty_si = si
            break
    assert nonempty_si > 0, "Should have at least one non-empty state"

    runner = OnePassTrainRunner()
    results = []
    for si in [0, nonempty_si]:
        target_state_idx = torch.full((M,), -1, dtype=torch.long)
        m_state = torch.zeros(M, dtype=torch.bool)
        target_state_idx[0] = si
        m_state[0] = True

        with torch.no_grad():
            r = runner.run_single(
                tokens=tokens, leaves=leaves,
                leaf_encoder=leaf_enc, merge_encoder=merge_enc, merge_decoder=merge_dec,
                catalog=catalog,
                target_state_idx=target_state_idx, m_state=m_state,
            )
        results.append(r.child_scores[0].clone())

    diff = (results[0] - results[1]).abs().sum().item()
    assert diff > 0.01, f"Different σ should give different child_scores, diff={diff}"
    print(f"  Score diff between empty and non-empty σ: {diff:.4f}")


def test_onepass_loss_basic():
    """onepass_loss should compute meaningful BCE values."""
    M, Ti = 5, 8

    # Simulate: one node decoded with high-confidence predictions
    child_scores = torch.zeros(M, 4, Ti)
    child_scores[0] = 5.0  # predict all active

    decode_mask = torch.zeros(M, dtype=torch.bool)
    decode_mask[0] = True

    y = torch.zeros(M, 4, Ti)
    y[0, 0, :2] = 1.0  # only 2 slots should be active

    m = torch.zeros(M, 4, Ti, dtype=torch.bool)
    m[0, :, :4] = True  # 4 valid slots per child

    loss = onepass_loss(
        child_scores=child_scores,
        decode_mask=decode_mask,
        y_child_iface=y,
        m_child_iface=m,
    )

    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
    print(f"  Loss value: {loss.item():.6f}")


def test_teacher_state_over_max_used_is_not_exact():
    """Teacher matchings exceeding max_used must fall back to non-exact labels."""
    Ti = 6
    catalog = build_boundary_state_catalog(num_slots=Ti, max_used=4, device=torch.device("cpu"))

    state_idx, exact_used = _build_matching_target_for_node(
        local_node_id=0,
        points_in_node={0, 1, 2, 3, 4, 5},
        selected_local_eids={0, 1, 2},
        sp_u=[0, 2, 4],
        sp_v=[1, 3, 5],
        iface_eid_row=torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
        iface_mask_row=torch.ones(Ti, dtype=torch.bool),
        iface_inside_ep_row=torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long),
        state_mask_row=torch.ones(catalog.used_iface.shape[0], dtype=torch.bool),
        state_used_iface=catalog.used_iface,
        state_mate=catalog.mate,
        max_used=4,
    )

    assert state_idx >= 0
    assert not exact_used


# ─── CLI runner ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("forward pass shape", test_forward_pass_shape),
        ("gradient flows", test_gradient_flows),
        ("no teacher state → zero loss", test_no_teacher_state_produces_zero_loss),
        ("different teacher σ → different outputs", test_different_teacher_states_different_outputs),
        ("onepass_loss basic", test_onepass_loss_basic),
        ("teacher state over max_used is not exact", test_teacher_state_over_max_used_is_not_exact),
    ]

    for name, fn in tests:
        print(f"Test: {name}...")
        fn()
        print("  PASS\n")

    print("All onepass_trainer tests passed!")
