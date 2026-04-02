# src/models/dp_parse_catalog.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from .dp_correspondence import CorrespondenceMaps, propagate_c1_constraints
from .dp_verify import verify_tuple


def _rank_child_catalog_states_for_parse(
    *,
    scores_q: Tensor,
    child_iface_mask_q: Tensor,
    constrained_q: Tensor,
    required_q: Tensor,
    cat_used: Tensor,
    child_cost_q: Tensor,
    max_child_states: Optional[int] = None,
) -> Tensor:
    """Return child-state indices after C1 filtering, ranking, and optional truncation."""
    finite_mask = child_cost_q < float("inf")
    if constrained_q.any():
        state_vals = cat_used[:, constrained_q].bool()
        required_vals = required_q[constrained_q].unsqueeze(0)
        c1_match = (state_vals == required_vals).all(dim=1)
        valid_mask = c1_match & finite_mask
    else:
        valid_mask = finite_mask

    valid_indices = valid_mask.nonzero(as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return valid_indices

    valid_used = cat_used[valid_indices].float()
    sq = scores_q.unsqueeze(0)
    mask_q = child_iface_mask_q.float().unsqueeze(0)
    agree = valid_used * sq + (1.0 - valid_used) * (1.0 - sq)
    state_scores = (agree * mask_q).sum(dim=1)

    if max_child_states is not None:
        cap = int(max_child_states)
        if cap > 0 and valid_indices.numel() > cap:
            top_pos = torch.topk(state_scores, k=cap, largest=True, sorted=True).indices
            return valid_indices[top_pos]

    sort_order = state_scores.argsort(descending=True)
    return valid_indices[sort_order]


def parse_by_catalog_enum(
    *,
    scores: Tensor,
    parent_a: Tensor,
    parent_mate: Tensor,
    parent_iface_mask: Tensor,
    child_iface_mask: Tensor,
    child_exists: Tensor,
    maps: CorrespondenceMaps,
    cat_used: Tensor,
    cat_mate: Tensor,
    child_costs: List[Optional[Tensor]],
    max_child_states: Optional[int] = None,
) -> Tuple[float, Tuple[int, int, int, int]]:
    """Catalog-enumeration PARSE."""
    Ti = int(parent_a.shape[0])
    device = parent_a.device

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
        if child_costs[q] is None:
            child_valid_states.append([])
            continue

        ranked_indices = _rank_child_catalog_states_for_parse(
            scores_q=scores[q],
            child_iface_mask_q=child_iface_mask[q],
            constrained_q=c1_constrained[q],
            required_q=c1_required[q],
            cat_used=cat_used,
            child_cost_q=child_costs[q],
            max_child_states=max_child_states,
        )
        if ranked_indices.numel() == 0:
            return float("inf"), (-1, -1, -1, -1)
        child_valid_states.append(ranked_indices.tolist())

    sh_links: List[List[Tuple[int, int, int]]] = [[] for _ in range(4)]
    for q in range(4):
        if not child_exists[q].item():
            continue
        for s in range(Ti):
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
            child_a = torch.zeros(4, Ti, dtype=torch.bool, device=device)
            child_mate_t = torch.full((4, Ti), -1, dtype=torch.long, device=device)
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

        if child_costs[q] is None:
            return

        ct_costs = child_costs[q]
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

            c = float(ct_costs[s].item())
            current_indices.append(s)
            _enum(q + 1, current_cost + c, current_indices)
            current_indices.pop()

    _enum(0, 0.0, [])
    return best_cost, best_indices


__all__ = ["_rank_child_catalog_states_for_parse", "parse_by_catalog_enum"]
