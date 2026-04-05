# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.boundary_state_structured import (
    enumerate_structured_states_for_iface_mask,
    stack_state_tensors,
)
from src.models.dp_core import build_correspondence_maps
from src.models.dp_parse_factorized import (
    _generate_factorized_child_candidates,
    rank_parent_states_by_child_lower_bound,
)
from src.models.dp_runner import CostTableEntry
from tests.test_dp_runner import build_synthetic_tree


def test_generate_factorized_child_candidates_interleaves_used_masks() -> None:
    iface_scores = torch.tensor([2.0, 2.0, 2.0, 2.0], dtype=torch.float32)
    mate_scores = torch.zeros(4, 4, dtype=torch.float32)
    mate_scores[0, 3] = 3.0
    mate_scores[1, 2] = 3.0
    mate_scores[0, 1] = 1.0
    mate_scores[2, 3] = 1.0

    child_iface_mask = torch.tensor([True, True, True, True], dtype=torch.bool)
    constrained = torch.zeros(4, dtype=torch.bool)
    required = torch.zeros(4, dtype=torch.bool)
    child_state_used = torch.tensor(
        [
            [True, True, True, True],   # bucket A, lower mate score
            [True, True, False, False], # bucket B
            [True, True, True, True],   # bucket A, higher mate score
        ],
        dtype=torch.bool,
    )
    child_state_mate = torch.tensor(
        [
            [1, 0, 3, 2],
            [1, 0, -1, -1],
            [3, 2, 1, 0],
        ],
        dtype=torch.long,
    )
    child_cost = torch.tensor([1.0, 2.0, 1.5], dtype=torch.float32)

    ranked = _generate_factorized_child_candidates(
        iface_scores_q=iface_scores,
        mate_scores_q=mate_scores,
        child_iface_mask_q=child_iface_mask,
        constrained_q=constrained,
        required_q=required,
        child_state_used_q=child_state_used,
        child_state_mate_q=child_state_mate,
        child_cost_q=child_cost,
        ranking_mode="iface_mate",
        lambda_mate=1.0,
    )

    assert ranked == [2, 1, 0]


def test_generate_factorized_child_candidates_respects_lambda_mate() -> None:
    iface_scores = torch.tensor([0.0, 0.0], dtype=torch.float32)
    mate_scores = torch.tensor(
        [
            [0.0, 10.0],
            [1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    child_iface_mask = torch.tensor([True, True], dtype=torch.bool)
    constrained = torch.zeros(2, dtype=torch.bool)
    required = torch.zeros(2, dtype=torch.bool)
    child_state_used = torch.tensor(
        [
            [True, True],
            [True, True],
        ],
        dtype=torch.bool,
    )
    child_state_mate = torch.tensor(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=torch.long,
    )
    child_cost = torch.tensor([5.0, 1.0], dtype=torch.float32)

    ranked_no_mate = _generate_factorized_child_candidates(
        iface_scores_q=iface_scores,
        mate_scores_q=mate_scores,
        child_iface_mask_q=child_iface_mask,
        constrained_q=constrained,
        required_q=required,
        child_state_used_q=child_state_used,
        child_state_mate_q=child_state_mate,
        child_cost_q=child_cost,
        ranking_mode="iface_mate",
        lambda_mate=0.0,
    )
    ranked_with_mate = _generate_factorized_child_candidates(
        iface_scores_q=iface_scores,
        mate_scores_q=mate_scores,
        child_iface_mask_q=child_iface_mask,
        constrained_q=constrained,
        required_q=required,
        child_state_used_q=child_state_used,
        child_state_mate_q=child_state_mate,
        child_cost_q=child_cost,
        ranking_mode="iface_mate",
        lambda_mate=10.0,
    )

    assert ranked_no_mate == [1, 0]
    assert ranked_with_mate == [0, 1]


def test_rank_parent_states_by_child_lower_bound_prunes_infeasible_bucket() -> None:
    tokens, _ = build_synthetic_tree(Ti=8, Tc=4, P=4)
    root_id = 0
    children = tokens.tree_children_index[root_id].long()
    child_exists = children >= 0
    ch_clamped = children.clamp_min(0)
    maps = build_correspondence_maps(
        parent_iface_eid=tokens.iface_eid[root_id],
        parent_iface_mask=tokens.iface_mask[root_id].bool(),
        parent_iface_bdir=tokens.iface_boundary_dir[root_id],
        parent_cross_eid=tokens.cross_eid[root_id],
        parent_cross_mask=tokens.cross_mask[root_id].bool(),
        parent_cross_child_pair=tokens.cross_child_pair[root_id],
        children_iface_eid=tokens.iface_eid[ch_clamped],
        children_iface_mask=tokens.iface_mask[ch_clamped].bool(),
        children_iface_bdir=tokens.iface_boundary_dir[ch_clamped],
        child_exists=child_exists,
    )

    parent_states = enumerate_structured_states_for_iface_mask(
        iface_mask=tokens.iface_mask[root_id].bool()
    )
    parent_used, _ = stack_state_tensors(states=parent_states, num_slots=tokens.iface_mask.shape[1])

    child_tables = [None] * 4
    for q in range(4):
        if not child_exists[q].item():
            continue
        cid = int(children[q].item())
        states_q = enumerate_structured_states_for_iface_mask(
            iface_mask=tokens.iface_mask[cid].bool()
        )
        used_q, mate_q = stack_state_tensors(states=states_q, num_slots=tokens.iface_mask.shape[1])
        costs_q = torch.full((len(states_q),), float("inf"), dtype=torch.float32)
        costs_q[0] = 0.0  # only the empty child state is feasible
        child_tables[q] = CostTableEntry(
            costs=costs_q,
            backptr={},
            state_list=states_q,
            state_used_iface=used_q,
            state_mate=mate_q,
        )

    ordered_indices, lower_bounds = rank_parent_states_by_child_lower_bound(
        parent_state_used=parent_used,
        parent_iface_mask=tokens.iface_mask[root_id].bool(),
        child_iface_mask=tokens.iface_mask[ch_clamped].bool() & child_exists.unsqueeze(-1),
        child_exists=child_exists,
        maps=maps,
        child_tables=child_tables,
    )

    assert ordered_indices == [0]
    assert lower_bounds == [0.0]
