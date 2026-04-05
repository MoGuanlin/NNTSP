# -*- coding: utf-8 -*-
"""Factorized structured parse helpers for uncapped 1-pass DP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .dp_correspondence import CorrespondenceMaps, propagate_c1_constraints
from .dp_verify import verify_tuple

if TYPE_CHECKING:
    from .dp_runner import CostTableEntry


def _rank_child_structured_states_for_parse(
    *,
    iface_scores_q: Tensor,
    mate_scores_q: Optional[Tensor],
    child_iface_mask_q: Tensor,
    constrained_q: Tensor,
    required_q: Tensor,
    child_state_used_q: Tensor,
    child_state_mate_q: Tensor,
    child_cost_q: Tensor,
    ranking_mode: str,
    lambda_mate: float = 1.0,
) -> Tensor:
    """Rank one child's local structured states under current parent constraints."""
    finite_mask = child_cost_q < float("inf")
    if constrained_q.any():
        state_vals = child_state_used_q[:, constrained_q].bool()
        required_vals = required_q[constrained_q].unsqueeze(0)
        c1_match = (state_vals == required_vals).all(dim=1)
        valid_mask = c1_match & finite_mask
    else:
        valid_mask = finite_mask

    valid_indices = valid_mask.nonzero(as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return valid_indices

    valid_used = child_state_used_q[valid_indices].bool()
    valid_mate = child_state_mate_q[valid_indices].long()
    slot_mask = child_iface_mask_q.bool().unsqueeze(0)

    pos = F.logsigmoid(iface_scores_q).unsqueeze(0)
    neg = F.logsigmoid(-iface_scores_q).unsqueeze(0)
    iface_term = torch.where(valid_used, pos, neg)
    state_scores = (iface_term * slot_mask.float()).sum(dim=1)

    if ranking_mode == "iface_mate":
        if mate_scores_q is None:
            raise ValueError("mate_scores_q must be provided when ranking_mode='iface_mate'")
        lam = float(lambda_mate)
        if lam != 0.0:
            ti = int(iface_scores_q.shape[0])
            slot_ids = torch.arange(ti, device=iface_scores_q.device, dtype=torch.long).unsqueeze(0)
            mate_clamped = valid_mate.clamp(min=0)
            gathered = mate_scores_q[slot_ids.expand_as(mate_clamped), mate_clamped]
            pair_mask = (
                valid_used
                & slot_mask
                & (valid_mate >= 0)
                & (slot_ids < valid_mate)
            )
            mate_term = (gathered * pair_mask.float()).sum(dim=1)
            state_scores = state_scores + lam * mate_term

    sort_order = state_scores.argsort(descending=True)
    return valid_indices[sort_order]


def _used_mask_from_row(used_row: Tensor) -> int:
    """Encode one bool used row as a compact bitmask."""
    mask = 0
    used_b = used_row.bool()
    for slot in range(int(used_b.numel())):
        if bool(used_b[slot].item()):
            mask |= (1 << slot)
    return mask


def rank_parent_states_by_child_lower_bound(
    *,
    parent_state_used: Tensor,
    parent_iface_mask: Tensor,
    child_iface_mask: Tensor,
    child_exists: Tensor,
    maps: CorrespondenceMaps,
    child_tables: Sequence[Optional["CostTableEntry"]],
) -> Tuple[List[int], List[float]]:
    """Order parent local states by a cheap exact lower bound from child tables.

    The lower bound only uses C1 propagation from the parent used-mask into the
    children plus each child's exact local DP table. Parent matching is ignored
    here, so every state in the same parent used-mask bucket shares the same
    lower bound.

    Returns:
      - ordered parent local state indices
      - aligned lower-bound values for those indices

    Parent states whose child-based lower bound is already `inf` are filtered
    out entirely, because they cannot be feasible for any matching.
    """
    num_states = int(parent_state_used.shape[0])
    if num_states == 0:
        return [], []

    buckets: Dict[int, List[int]] = {}
    for si in range(num_states):
        used_mask = _used_mask_from_row(parent_state_used[si])
        buckets.setdefault(used_mask, []).append(int(si))

    bucket_lb: Dict[int, float] = {}
    for used_mask, indices in buckets.items():
        rep_si = int(indices[0])
        parent_a = parent_state_used[rep_si].bool()
        c1_required, c1_constrained = propagate_c1_constraints(
            parent_a=parent_a,
            parent_iface_mask=parent_iface_mask,
            maps=maps,
            child_exists=child_exists,
            child_iface_mask=child_iface_mask,
        )

        total_lb = 0.0
        feasible = True
        for q in range(4):
            if not bool(child_exists[q].item()):
                continue
            ct = child_tables[q]
            if ct is None or ct.state_used_iface is None or ct.costs is None:
                feasible = False
                break
            finite_mask = ct.costs < float("inf")
            if c1_constrained[q].any():
                required = c1_required[q, c1_constrained[q]].bool().unsqueeze(0)
                state_vals = ct.state_used_iface[:, c1_constrained[q]].bool()
                c1_match = (state_vals == required).all(dim=1)
                valid_mask = finite_mask & c1_match
            else:
                valid_mask = finite_mask
            if not bool(valid_mask.any().item()):
                feasible = False
                break
            total_lb += float(ct.costs[valid_mask].min().item())

        bucket_lb[used_mask] = total_lb if feasible else float("inf")

    ordered_masks = [
        used_mask
        for used_mask, lb in sorted(bucket_lb.items(), key=lambda kv: (kv[1], kv[0]))
        if lb < float("inf")
    ]
    ordered_indices: List[int] = []
    ordered_lb: List[float] = []
    for used_mask in ordered_masks:
        indices = buckets[used_mask]
        indices.sort()
        lb = float(bucket_lb[used_mask])
        ordered_indices.extend(indices)
        ordered_lb.extend([lb] * len(indices))
    return ordered_indices, ordered_lb


def _generate_factorized_child_candidates(
    *,
    iface_scores_q: Tensor,
    mate_scores_q: Optional[Tensor],
    child_iface_mask_q: Tensor,
    constrained_q: Tensor,
    required_q: Tensor,
    child_state_used_q: Tensor,
    child_state_mate_q: Tensor,
    child_cost_q: Tensor,
    ranking_mode: str,
    lambda_mate: float = 1.0,
) -> List[int]:
    """Generate a diversified candidate ordering by first ranking used-masks.

    This is more factorized than globally ranking every full `(used, mate)` state:
    we first rank activation patterns, then rank pairings within each activation
    bucket, and finally interleave buckets round-robin to keep early candidates
    diverse.
    """
    finite_mask = child_cost_q < float("inf")
    num_states = int(child_cost_q.numel())
    if num_states == 0:
        return []

    slot_mask = child_iface_mask_q.bool()
    pos = F.logsigmoid(iface_scores_q)
    neg = F.logsigmoid(-iface_scores_q)

    bucket_items: Dict[int, List[int]] = {}
    bucket_score: Dict[int, float] = {}
    bucket_min_cost: Dict[int, float] = {}

    for si in range(num_states):
        if not bool(finite_mask[si].item()):
            continue
        used = child_state_used_q[si].bool()
        if constrained_q.any():
            if not torch.equal(used[constrained_q], required_q[constrained_q].bool()):
                continue

        used_mask = _used_mask_from_row(used)
        bucket_items.setdefault(used_mask, []).append(int(si))
        if used_mask not in bucket_score:
            iface_term = torch.where(used, pos, neg)
            bucket_score[used_mask] = float((iface_term * slot_mask.float()).sum().item())
            bucket_min_cost[used_mask] = float(child_cost_q[si].item())
        else:
            bucket_min_cost[used_mask] = min(bucket_min_cost[used_mask], float(child_cost_q[si].item()))

    if not bucket_items:
        return []

    ordered_masks = sorted(
        bucket_items.keys(),
        key=lambda mask: (-bucket_score[mask], bucket_min_cost[mask], mask),
    )

    ranked_buckets: List[List[int]] = []
    for used_mask in ordered_masks:
        indices = bucket_items[used_mask]
        if ranking_mode == "iface_mate":
            if mate_scores_q is None:
                raise ValueError("mate_scores_q must be provided when ranking_mode='iface_mate'")
            lam = float(lambda_mate)
            score_pairs: List[Tuple[float, float, int]] = []
            for si in indices:
                used = child_state_used_q[si].bool()
                mate = child_state_mate_q[si].long()
                ti = int(used.numel())
                mate_term = 0.0
                for slot in range(ti):
                    if not bool(used[slot].item()):
                        continue
                    partner = int(mate[slot].item())
                    if partner < 0 or slot >= partner:
                        continue
                    mate_term += float(mate_scores_q[slot, partner].item())
                score_pairs.append((lam * mate_term, -float(child_cost_q[si].item()), int(si)))
            score_pairs.sort(reverse=True)
            ranked_buckets.append([si for _, _, si in score_pairs])
        else:
            ranked = sorted(indices, key=lambda si: float(child_cost_q[si].item()))
            ranked_buckets.append([int(si) for si in ranked])

    interleaved: List[int] = []
    max_bucket_len = max(len(bucket) for bucket in ranked_buckets)
    for depth in range(max_bucket_len):
        for bucket in ranked_buckets:
            if depth < len(bucket):
                interleaved.append(bucket[depth])
    return interleaved


def _enum_child_tuples(
    *,
    candidate_lists: Sequence[Sequence[int]],
    child_tables: Sequence[Optional["CostTableEntry"]],
    parent_a: Tensor,
    parent_mate: Tensor,
    parent_iface_mask: Tensor,
    child_iface_mask: Tensor,
    child_exists: Tensor,
    maps: CorrespondenceMaps,
) -> Tuple[float, Tuple[int, int, int, int]]:
    """Exact tuple enumeration over pre-ranked child local-state indices."""
    device = parent_a.device
    num_slots = int(parent_a.shape[0])

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
    min_cost_suffix = [0.0] * 5
    for q in range(3, -1, -1):
        if not child_exists[q].item():
            min_cost_suffix[q] = min_cost_suffix[q + 1]
            continue
        ct = child_tables[q]
        if ct is None or ct.costs is None or len(candidate_lists[q]) == 0:
            min_cost_suffix[q] = float("inf")
            continue
        best_child_cost = min(float(ct.costs[int(si)].item()) for si in candidate_lists[q])
        min_cost_suffix[q] = best_child_cost + min_cost_suffix[q + 1]

    def _enum(q: int, current_cost: float, current_indices: List[int]) -> None:
        nonlocal best_cost, best_indices
        if current_cost >= best_cost:
            return
        if current_cost + min_cost_suffix[q] >= best_cost:
            return

        if q == 4:
            child_a = torch.zeros(4, num_slots, dtype=torch.bool, device=device)
            child_mate_t = torch.full((4, num_slots), -1, dtype=torch.long, device=device)
            for qq in range(4):
                if not child_exists[qq].item():
                    continue
                ct = child_tables[qq]
                local_si = current_indices[qq]
                if ct is None or local_si < 0 or ct.state_used_iface is None or ct.state_mate is None:
                    return
                used_rows = ct.state_used_iface.to(device=device)
                mate_rows = ct.state_mate.to(device=device)
                child_a[qq] = used_rows[local_si]
                child_mate_t[qq] = mate_rows[local_si]

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

        ct = child_tables[q]
        if ct is None or ct.state_used_iface is None or ct.costs is None:
            return
        used_rows = ct.state_used_iface
        local_costs = ct.costs
        for s in candidate_lists[q]:
            c2_ok = True
            for my_slot, peer_q, peer_slot in sh_links[q]:
                peer_si = current_indices[peer_q]
                if peer_si < 0:
                    continue
                peer_ct = child_tables[peer_q]
                if peer_ct is None or peer_ct.state_used_iface is None:
                    c2_ok = False
                    break
                if bool(used_rows[s, my_slot].item()) != bool(peer_ct.state_used_iface[peer_si, peer_slot].item()):
                    c2_ok = False
                    break
            if not c2_ok:
                continue

            c = float(local_costs[s].item())
            if current_cost + c + min_cost_suffix[q + 1] >= best_cost:
                continue
            current_indices.append(int(s))
            _enum(q + 1, current_cost + c, current_indices)
            current_indices.pop()

    _enum(0, 0.0, [])
    return best_cost, best_indices


def prepare_factorized_child_rankings(
    *,
    scores: Tensor,
    parent_a: Tensor,
    parent_iface_mask: Tensor,
    child_iface_mask: Tensor,
    child_exists: Tensor,
    maps: CorrespondenceMaps,
    child_tables: Sequence[Optional["CostTableEntry"]],
    mate_scores: Optional[Tensor] = None,
    ranking_mode: str = "iface",
    lambda_mate: float = 1.0,
    child_state_used_dev: Optional[Sequence[Optional[Tensor]]] = None,
    child_state_mate_dev: Optional[Sequence[Optional[Tensor]]] = None,
    child_cost_dev: Optional[Sequence[Optional[Tensor]]] = None,
) -> Tuple[Optional[List[List[int]]], str]:
    """Prepare ranked child-state candidate lists for one fixed parent sigma.

    This stage is purely about ranking/filtering. It is the right place to keep
    tensor-heavy work on GPU before the exact tuple enumeration step.
    """
    if ranking_mode not in ("iface", "iface_mate"):
        raise ValueError(f"ranking_mode must be 'iface' or 'iface_mate', got {ranking_mode!r}")

    c1_required, c1_constrained = propagate_c1_constraints(
        parent_a=parent_a.bool(),
        parent_iface_mask=parent_iface_mask,
        maps=maps,
        child_exists=child_exists,
        child_iface_mask=child_iface_mask,
    )

    ranked_child_states: List[List[int]] = []
    for q in range(4):
        if not child_exists[q].item():
            ranked_child_states.append([-1])
            continue
        ct = child_tables[q]
        if ct is None or ct.state_used_iface is None or ct.state_mate is None or ct.costs is None:
            ranked_child_states.append([])
            continue
        dev = scores[q].device
        used_q = (
            child_state_used_dev[q]
            if child_state_used_dev is not None and child_state_used_dev[q] is not None
            else ct.state_used_iface.to(device=dev)
        )
        mate_q = (
            child_state_mate_dev[q]
            if child_state_mate_dev is not None and child_state_mate_dev[q] is not None
            else ct.state_mate.to(device=dev)
        )
        cost_q = (
            child_cost_dev[q]
            if child_cost_dev is not None and child_cost_dev[q] is not None
            else ct.costs.to(device=dev)
        )
        candidate_list = _generate_factorized_child_candidates(
            iface_scores_q=scores[q],
            mate_scores_q=(mate_scores[q] if mate_scores is not None else None),
            child_iface_mask_q=child_iface_mask[q],
            constrained_q=c1_constrained[q],
            required_q=c1_required[q],
            child_state_used_q=used_q,
            child_state_mate_q=mate_q,
            child_cost_q=cost_q,
            ranking_mode=ranking_mode,
            lambda_mate=lambda_mate,
        )
        if len(candidate_list) == 0:
            return None, "num_infeasible"
        ranked_child_states.append(candidate_list)
    return ranked_child_states, "ok"


def solve_factorized_widening_from_ranked_child_states(
    *,
    ranked_child_states: Sequence[Sequence[int]],
    parent_a: Tensor,
    parent_mate: Tensor,
    parent_iface_mask: Tensor,
    child_iface_mask: Tensor,
    child_exists: Tensor,
    maps: CorrespondenceMaps,
    child_tables: Sequence[Optional["CostTableEntry"]],
    widening_schedule: Optional[Sequence[int]] = None,
    fallback_exact: bool = True,
) -> Tuple[float, Tuple[int, int, int, int], str]:
    """Run widening + exact tuple enumeration from pre-ranked child lists."""
    schedule = [int(v) for v in (widening_schedule or []) if int(v) > 0]
    if not schedule:
        schedule = [max(len(lst), 1) for lst in ranked_child_states if lst]
        schedule = [max(schedule)] if schedule else [1]

    for attempt_idx, cap in enumerate(schedule):
        round_lists: List[List[int]] = []
        for q in range(4):
            if not child_exists[q].item():
                round_lists.append([-1])
                continue
            round_lists.append(list(ranked_child_states[q][: min(int(cap), len(ranked_child_states[q]))]))
        cost, child_si = _enum_child_tuples(
            candidate_lists=round_lists,
            child_tables=child_tables,
            parent_a=parent_a,
            parent_mate=parent_mate,
            parent_iface_mask=parent_iface_mask,
            child_iface_mask=child_iface_mask,
            child_exists=child_exists,
            maps=maps,
        )
        if cost < float("inf"):
            outcome = "num_parse_ok" if attempt_idx == 0 else "num_widening_ok"
            return cost, child_si, outcome

    if not fallback_exact:
        return float("inf"), (-1, -1, -1, -1), "num_infeasible"

    exact_cost, exact_child_si = _enum_child_tuples(
        candidate_lists=ranked_child_states,
        child_tables=child_tables,
        parent_a=parent_a,
        parent_mate=parent_mate,
        parent_iface_mask=parent_iface_mask,
        child_iface_mask=child_iface_mask,
        child_exists=child_exists,
        maps=maps,
    )
    if exact_cost < float("inf"):
        return exact_cost, exact_child_si, "num_fallback"
    return float("inf"), (-1, -1, -1, -1), "num_infeasible"


def parse_by_factorized_widening(
    *,
    scores: Tensor,
    parent_a: Tensor,
    parent_mate: Tensor,
    parent_iface_mask: Tensor,
    child_iface_mask: Tensor,
    child_exists: Tensor,
    maps: CorrespondenceMaps,
    child_tables: Sequence[Optional["CostTableEntry"]],
    mate_scores: Optional[Tensor] = None,
    ranking_mode: str = "iface",
    lambda_mate: float = 1.0,
    widening_schedule: Optional[Sequence[int]] = None,
    fallback_exact: bool = True,
) -> Tuple[float, Tuple[int, int, int, int], str]:
    """Structured child-state ranking + widening over local child state tables."""
    ranked_child_states, status = prepare_factorized_child_rankings(
        scores=scores,
        parent_a=parent_a,
        parent_iface_mask=parent_iface_mask,
        child_iface_mask=child_iface_mask,
        child_exists=child_exists,
        maps=maps,
        child_tables=child_tables,
        mate_scores=mate_scores,
        ranking_mode=ranking_mode,
        lambda_mate=lambda_mate,
    )
    if ranked_child_states is None:
        return float("inf"), (-1, -1, -1, -1), status
    return solve_factorized_widening_from_ranked_child_states(
        ranked_child_states=ranked_child_states,
        parent_a=parent_a,
        parent_mate=parent_mate,
        parent_iface_mask=parent_iface_mask,
        child_iface_mask=child_iface_mask,
        child_exists=child_exists,
        maps=maps,
        child_tables=child_tables,
        widening_schedule=widening_schedule,
        fallback_exact=fallback_exact,
    )


__all__ = [
    "parse_by_factorized_widening",
    "prepare_factorized_child_rankings",
    "rank_parent_states_by_child_lower_bound",
    "solve_factorized_widening_from_ranked_child_states",
]
