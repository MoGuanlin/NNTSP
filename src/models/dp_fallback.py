# -*- coding: utf-8 -*-
"""Fallback and child-cost lookup helpers for the 1-pass DP runner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

import torch
from torch import Tensor

from .bc_state_catalog import BoundaryStateCatalog
from .dp_correspondence import CorrespondenceMaps
from .dp_verify import verify_tuple
from .node_token_packer import PackedNodeTokens
from .dp_core import propagate_c1_constraints

if TYPE_CHECKING:
    from .dp_runner import CostTableEntry


def find_state_index(
    child_a: Tensor,
    child_mate: Tensor,
    catalog: BoundaryStateCatalog,
) -> int:
    """Find the catalog index matching a discrete (a, mate) state."""
    device = child_a.device
    cat_used = catalog.used_iface
    cat_mate = catalog.mate
    if cat_used.device != device:
        cat_used = cat_used.to(device)
        cat_mate = cat_mate.to(device)

    match_a = (cat_used == child_a.unsqueeze(0)).all(dim=1)
    match_m = (cat_mate == child_mate.unsqueeze(0)).all(dim=1)
    match = match_a & match_m
    indices = torch.nonzero(match, as_tuple=False).flatten()
    if indices.numel() > 0:
        return int(indices[0].item())
    return -1


def lookup_child_costs(
    *,
    child_a: Tensor,
    child_mate: Tensor,
    children: Tensor,
    child_exists: Tensor,
    cost_tables: Dict[int, "CostTableEntry"],
    catalog: BoundaryStateCatalog,
) -> Tuple[float, Tuple[int, int, int, int]]:
    """Look up total cost from child cost tables."""
    total_cost = 0.0
    child_state_indices = [-1, -1, -1, -1]

    for q in range(4):
        if not child_exists[q].item():
            child_state_indices[q] = -1
            continue

        cid = int(children[q].item())
        ct = cost_tables.get(cid)
        if ct is None:
            return float("inf"), (-1, -1, -1, -1)

        si = find_state_index(child_a[q], child_mate[q], catalog)
        if si < 0:
            return float("inf"), (-1, -1, -1, -1)

        cost = float(ct.costs[si].item())
        if cost == float("inf"):
            return float("inf"), (-1, -1, -1, -1)

        total_cost += cost
        child_state_indices[q] = si

    return total_cost, tuple(child_state_indices)


def exact_fallback(
    *,
    si: int,
    catalog: BoundaryStateCatalog,
    tokens: PackedNodeTokens,
    nid: int,
    children: Tensor,
    child_exists: Tensor,
    maps: CorrespondenceMaps,
    cost_tables: Dict[int, "CostTableEntry"],
) -> Tuple[float, Tuple[int, int, int, int]]:
    """Constraint-propagated exact fallback for one parent state."""
    device = tokens.tree_parent_index.device
    num_slots = int(tokens.iface_mask.shape[1])

    cat_used = catalog.used_iface if catalog.used_iface.device == device else catalog.used_iface.to(device)
    cat_mate = catalog.mate if catalog.mate.device == device else catalog.mate.to(device)

    parent_a = cat_used[si]
    parent_mate = cat_mate[si]
    parent_iface_mask = tokens.iface_mask[nid].bool()
    clamped_children = children.clamp_min(0)
    child_iface_mask = tokens.iface_mask[clamped_children].bool()
    child_iface_mask = child_iface_mask & child_exists.unsqueeze(-1)

    c1_required, c1_constrained = propagate_c1_constraints(
        parent_a=parent_a.bool(),
        parent_iface_mask=parent_iface_mask,
        maps=maps,
        child_exists=child_exists,
        child_iface_mask=child_iface_mask,
    )

    child_valid_states: List[List[int]] = []
    for q in range(4):
        if not child_exists[q].item():
            child_valid_states.append([-1])
            continue
        cid = int(children[q].item())
        ct = cost_tables.get(cid)
        if ct is None:
            child_valid_states.append([])
            continue

        constrained_q = c1_constrained[q]
        finite_mask = ct.costs < float("inf")
        if constrained_q.any():
            required_q = c1_required[q]
            state_vals = cat_used[:, constrained_q].bool()
            required_vals = required_q[constrained_q].unsqueeze(0)
            c1_match = (state_vals == required_vals).all(dim=1)
            valid_mask = c1_match & finite_mask
        else:
            valid_mask = finite_mask

        valid_indices = valid_mask.nonzero(as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            return float("inf"), (-1, -1, -1, -1)
        child_valid_states.append(valid_indices.tolist())

    sh_links: List[List[Tuple[int, int, int]]] = [[] for _ in range(4)]
    for q in range(4):
        if not child_exists[q].item():
            continue
        for s in range(num_slots):
            if not child_iface_mask[q, s].item():
                continue
            peer_q = int(maps.phi_glue_peer_child[q, s].item())
            peer_s = int(maps.phi_glue_peer_slot[q, s].item())
            if peer_q < 0 or peer_s < 0:
                continue
            if peer_q < q and child_exists[peer_q].item():
                sh_links[q].append((s, peer_q, peer_s))

    best_cost = float("inf")
    best_indices: Tuple[int, int, int, int] = (-1, -1, -1, -1)

    def _enum(q: int, current_cost: float, current_indices: List[int]) -> None:
        nonlocal best_cost, best_indices

        if current_cost >= best_cost:
            return

        if q == 4:
            child_a = torch.zeros(4, num_slots, dtype=torch.bool, device=device)
            child_mate_t = torch.full((4, num_slots), -1, dtype=torch.long, device=device)
            for qq in range(4):
                if child_exists[qq].item() and current_indices[qq] >= 0:
                    child_a[qq] = cat_used[current_indices[qq]]
                    child_mate_t[qq] = cat_mate[current_indices[qq]]

            ok = verify_tuple(
                parent_a=parent_a.bool(),
                parent_mate=parent_mate,
                parent_iface_mask=parent_iface_mask,
                child_a=child_a,
                child_mate=child_mate_t,
                child_iface_mask=child_iface_mask,
                child_exists=child_exists,
                maps=maps,
            )
            if ok and current_cost < best_cost:
                best_cost = current_cost
                best_indices = tuple(current_indices)
            return

        if not child_exists[q].item():
            current_indices.append(-1)
            _enum(q + 1, current_cost, current_indices)
            current_indices.pop()
            return

        cid = int(children[q].item())
        ct = cost_tables.get(cid)
        if ct is None:
            return

        for s in child_valid_states[q]:
            c2_ok = True
            for my_slot, peer_q, peer_slot in sh_links[q]:
                peer_si = current_indices[peer_q]
                if peer_si < 0:
                    continue
                if bool(cat_used[s, my_slot].item()) != bool(cat_used[peer_si, peer_slot].item()):
                    c2_ok = False
                    break
            if not c2_ok:
                continue

            cost = float(ct.costs[s].item())
            current_indices.append(s)
            _enum(q + 1, current_cost + cost, current_indices)
            current_indices.pop()

    _enum(0, 0.0, [])
    return best_cost, best_indices


__all__ = ["exact_fallback", "find_state_index", "lookup_child_costs"]
