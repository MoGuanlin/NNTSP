# -*- coding: utf-8 -*-
"""Structured leaf exact solve for uncapped 1-pass DP."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor

from .boundary_state_structured import (
    StructuredBoundaryState,
    build_state_index,
    enumerate_structured_states_for_iface_mask,
    stack_state_tensors,
)
from .dp_leaf_solver import leaf_solve_state


def leaf_exact_solve_structured(
    *,
    points_xy: Tensor,
    point_mask: Tensor,
    iface_mask: Tensor,
    iface_feat6: Tensor,
    box_xy: Tensor,
    is_root: bool = False,
) -> Tuple[Tensor, List[StructuredBoundaryState], Tensor, Tensor, Dict[StructuredBoundaryState, int]]:
    """Enumerate all feasible node-local structured states for one leaf."""
    device = points_xy.device
    num_slots = int(iface_mask.numel())
    states = enumerate_structured_states_for_iface_mask(iface_mask=iface_mask)
    used_rows, mate_rows = stack_state_tensors(
        states=states,
        num_slots=num_slots,
        device=device,
    )
    costs = torch.full((len(states),), float("inf"), dtype=torch.float32, device=device)

    for idx in range(len(states)):
        cost, _ = leaf_solve_state(
            points_xy=points_xy,
            point_mask=point_mask,
            iface_mask=iface_mask,
            iface_feat6=iface_feat6,
            state_used=used_rows[idx],
            state_mate=mate_rows[idx],
            box_xy=box_xy,
            is_root=is_root,
        )
        costs[idx] = float(cost)

    return costs, states, used_rows, mate_rows, build_state_index(states)


__all__ = ["leaf_exact_solve_structured"]
