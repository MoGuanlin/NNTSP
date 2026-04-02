# src/models/dp_parse_heuristic.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .dp_correspondence import CorrespondenceMaps, propagate_c1_constraints
from .dp_verify import verify_tuple


def _noncrossing_min_cost_matching(
    active_slots: List[int],
    cost_matrix: Tensor,
) -> Optional[List[Tuple[int, int]]]:
    """Find minimum-cost noncrossing perfect matching on slots in clockwise order."""
    k = len(active_slots)
    if k == 0:
        return []
    if k % 2 != 0:
        return None

    INF = float("inf")
    dp = [[INF] * k for _ in range(k)]
    choice = [[-1] * k for _ in range(k)]

    for i in range(0, k - 1, 2):
        pass

    for i in range(k):
        dp[i][i] = INF

    for length in range(2, k + 1, 2):
        for i in range(k - length + 1):
            j = i + length - 1
            best = INF
            best_m = -1
            for m in range(i + 1, j + 1, 2):
                p, q = active_slots[i], active_slots[m]
                c = float(cost_matrix[p, q].item())
                left = dp[i + 1][m - 1] if m > i + 1 else 0.0
                right = dp[m + 1][j] if m < j else 0.0
                total = c + left + right
                if total < best:
                    best = total
                    best_m = m
            dp[i][j] = best
            choice[i][j] = best_m

    if dp[0][k - 1] >= INF:
        return None

    pairs: List[Tuple[int, int]] = []

    def _traceback(i: int, j: int) -> None:
        if i > j:
            return
        m = choice[i][j]
        if m < 0:
            return
        pairs.append((active_slots[i], active_slots[m]))
        if m > i + 1:
            _traceback(i + 1, m - 1)
        if m < j:
            _traceback(m + 1, j)

    _traceback(0, k - 1)
    return pairs


def parse_continuous(
    *,
    scores: Tensor,
    child_iface_mask: Tensor,
    child_iface_bdir: Tensor,
    child_exists: Tensor,
    maps: CorrespondenceMaps,
    r: int = 4,
    threshold: float = 0.5,
    parent_a: Optional[Tensor] = None,
    parent_iface_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Convert continuous child-state scores to discrete (a, mate) per child."""
    Ti = int(scores.shape[1])
    device = scores.device
    child_a = torch.zeros(4, Ti, dtype=torch.bool, device=device)
    child_mate = torch.full((4, Ti), -1, dtype=torch.long, device=device)

    def _is_c1_required_active(qi: int, slot: int) -> bool:
        return bool(
            c1_constrained is not None
            and c1_required is not None
            and c1_constrained[qi, slot].item()
            and c1_required[qi, slot].item()
        )

    c1_required: Optional[Tensor] = None
    c1_constrained: Optional[Tensor] = None
    if parent_a is not None and parent_iface_mask is not None:
        c1_required, c1_constrained = propagate_c1_constraints(
            parent_a=parent_a.bool(),
            parent_iface_mask=parent_iface_mask,
            maps=maps,
            child_exists=child_exists,
            child_iface_mask=child_iface_mask,
        )

    glue_avg = scores.clone()
    for qi in range(4):
        if not child_exists[qi].item():
            continue
        for s in range(Ti):
            if not child_iface_mask[qi, s].item():
                continue
            pq = int(maps.phi_glue_peer_child[qi, s].item())
            ps = int(maps.phi_glue_peer_slot[qi, s].item())
            if pq < 0 or ps < 0:
                continue
            if qi < pq or (qi == pq and s < ps):
                avg = (scores[qi, s] + scores[pq, ps]) / 2.0
                glue_avg[qi, s] = avg
                glue_avg[pq, ps] = avg

    for qi in range(4):
        if not child_exists[qi].item():
            continue

        side_slots: Dict[int, List[Tuple[float, int]]] = {0: [], 1: [], 2: [], 3: []}
        for s in range(Ti):
            if not child_iface_mask[qi, s].item():
                continue
            if c1_constrained is not None and c1_constrained[qi, s].item():
                continue
            bdir = int(child_iface_bdir[qi, s].item())
            side_slots[bdir].append((float(glue_avg[qi, s].item()), s))

        activated_slots: List[int] = []

        if c1_constrained is not None:
            for s in range(Ti):
                if not child_iface_mask[qi, s].item():
                    continue
                if c1_constrained[qi, s].item():
                    if c1_required[qi, s].item():
                        child_a[qi, s] = True
                        activated_slots.append(s)

        for bdir in range(4):
            slots = side_slots[bdir]
            slots.sort(key=lambda x: -x[0])
            count = 0
            for score, s in slots:
                if count < r and score >= threshold:
                    child_a[qi, s] = True
                    activated_slots.append(s)
                    count += 1

        if sum(child_a[qi].tolist()) % 2 != 0:
            active_with_score = [
                (abs(float(glue_avg[qi, s].item()) - threshold), s)
                for s in activated_slots
                if child_a[qi, s].item() and not _is_c1_required_active(qi, s)
            ]
            if active_with_score:
                active_with_score.sort()
                flip_slot = active_with_score[0][1]
                child_a[qi, flip_slot] = False

    for qi in range(4):
        if not child_exists[qi].item():
            continue
        for s in range(Ti):
            if not child_iface_mask[qi, s].item():
                continue
            pq = int(maps.phi_glue_peer_child[qi, s].item())
            ps = int(maps.phi_glue_peer_slot[qi, s].item())
            if pq < 0 or ps < 0:
                continue
            if qi < pq or (qi == pq and s < ps):
                either = child_a[qi, s].item() or child_a[pq, ps].item()
                child_a[qi, s] = either
                child_a[pq, ps] = either

    for _parity_round in range(4):
        any_odd = False
        for qi in range(4):
            if not child_exists[qi].item():
                continue
            if sum(child_a[qi].tolist()) % 2 == 0:
                continue
            any_odd = True
            best_conf = float("inf")
            best_pair = None
            for s in range(Ti):
                if not child_a[qi, s].item():
                    continue
                pq = int(maps.phi_glue_peer_child[qi, s].item())
                ps = int(maps.phi_glue_peer_slot[qi, s].item())
                if pq >= 0 and ps >= 0:
                    if _is_c1_required_active(qi, s) or _is_c1_required_active(pq, ps):
                        continue
                    conf = abs(float(glue_avg[qi, s].item()) - threshold)
                    if conf < best_conf:
                        best_conf = conf
                        best_pair = (qi, s, pq, ps)
            if best_pair is not None:
                child_a[best_pair[0], best_pair[1]] = False
                child_a[best_pair[2], best_pair[3]] = False
            else:
                active_with_score = [
                    (abs(float(glue_avg[qi, s].item()) - threshold), s)
                    for s in range(Ti)
                    if child_a[qi, s].item()
                    and child_iface_mask[qi, s].item()
                    and not _is_c1_required_active(qi, s)
                ]
                if active_with_score:
                    active_with_score.sort()
                    child_a[qi, active_with_score[0][1]] = False
        if not any_odd:
            break

    for qi in range(4):
        if not child_exists[qi].item():
            continue
        active_slots = [
            s
            for s in range(Ti)
            if child_a[qi, s].item() and child_iface_mask[qi, s].item()
        ]

        if len(active_slots) == 0:
            continue
        if len(active_slots) % 2 != 0:
            continue

        cost_matrix = torch.zeros(Ti, Ti, device=scores.device)
        for i, si in enumerate(active_slots):
            for j, sj in enumerate(active_slots):
                if i < j:
                    cost_matrix[si, sj] = -(glue_avg[qi, si] * glue_avg[qi, sj])
                    cost_matrix[sj, si] = cost_matrix[si, sj]

        pairs = _noncrossing_min_cost_matching(active_slots, cost_matrix)
        if pairs is not None:
            for sa, sb in pairs:
                child_mate[qi, sa] = sb
                child_mate[qi, sb] = sa

    return child_a, child_mate


def parse_continuous_topk(
    *,
    scores: Tensor,
    child_iface_mask: Tensor,
    child_iface_bdir: Tensor,
    child_exists: Tensor,
    maps: CorrespondenceMaps,
    parent_a: Tensor,
    parent_mate: Tensor,
    parent_iface_mask: Tensor,
    r: int = 4,
    K: int = 5,
) -> List[Tuple[Tensor, Tensor]]:
    """Emit up to K candidate discrete child tuples by varying threshold."""
    results: List[Tuple[Tensor, Tensor]] = []
    seen: set = set()

    for thr in [0.5, 0.4, 0.3, 0.6, 0.7, 0.2, 0.8]:
        if len(results) >= K:
            break
        ca, cm = parse_continuous(
            scores=scores,
            child_iface_mask=child_iface_mask,
            child_iface_bdir=child_iface_bdir,
            child_exists=child_exists,
            maps=maps,
            r=r,
            threshold=thr,
            parent_a=parent_a,
            parent_iface_mask=parent_iface_mask,
        )
        key = (ca.bool().tolist().__repr__(), cm.tolist().__repr__())
        if key in seen:
            continue
        seen.add(key)

        ok = verify_tuple(
            parent_a=parent_a,
            parent_mate=parent_mate,
            parent_iface_mask=parent_iface_mask,
            child_a=ca,
            child_mate=cm,
            child_iface_mask=child_iface_mask,
            child_exists=child_exists,
            maps=maps,
        )
        if ok:
            results.append((ca.clone(), cm.clone()))

    return results


def parse_activation_batch(
    *,
    scores_batch: Tensor,
    child_iface_mask: Tensor,
    child_iface_bdir: Tensor,
    child_exists: Tensor,
    maps: CorrespondenceMaps,
    r: int = 4,
    threshold: float = 0.5,
) -> Tensor:
    """Batch PARSE Phase A: continuous scores -> discrete activations (no matching)."""
    K = scores_batch.shape[0]
    Ti = scores_batch.shape[2]
    device = scores_batch.device

    glue_avg = scores_batch.clone()
    phi_glue_c = maps.phi_glue_peer_child
    phi_glue_s = maps.phi_glue_peer_slot

    for qi in range(4):
        if not child_exists[qi].item():
            continue
        for s in range(Ti):
            if not child_iface_mask[qi, s].item():
                continue
            pq = int(phi_glue_c[qi, s].item())
            ps = int(phi_glue_s[qi, s].item())
            if pq < 0 or ps < 0:
                continue
            if qi < pq or (qi == pq and s < ps):
                avg = (scores_batch[:, qi, s] + scores_batch[:, pq, ps]) / 2.0
                glue_avg[:, qi, s] = avg
                glue_avg[:, pq, ps] = avg

    child_a = torch.zeros(K, 4, Ti, dtype=torch.bool, device=device)
    for qi in range(4):
        if not child_exists[qi].item():
            continue
        for bdir in range(4):
            side_mask = child_iface_mask[qi] & (child_iface_bdir[qi] == bdir)
            side_slots = torch.nonzero(side_mask, as_tuple=False).flatten()
            if side_slots.numel() == 0:
                continue
            side_scores = glue_avg[:, qi, side_slots]
            keep = min(r, side_slots.numel())
            if keep == 0:
                continue
            topk_vals, topk_idx = side_scores.topk(keep, dim=1)
            above = topk_vals >= threshold
            slot_ids = side_slots[topk_idx]
            child_a[:, qi, :].scatter_(1, slot_ids, above)

    for qi in range(4):
        if not child_exists[qi].item():
            continue
        counts = child_a[:, qi, :].sum(dim=1)
        odd_mask = (counts % 2) != 0
        if not odd_mask.any():
            continue
        for k_i in torch.nonzero(odd_mask, as_tuple=False).flatten():
            ki = int(k_i.item())
            active_slots = torch.nonzero(child_a[ki, qi] & child_iface_mask[qi], as_tuple=False).flatten()
            if active_slots.numel() == 0:
                continue
            confs = (glue_avg[ki, qi, active_slots] - threshold).abs()
            flip_idx = active_slots[confs.argmin()]
            child_a[ki, qi, flip_idx] = False

    for qi in range(4):
        if not child_exists[qi].item():
            continue
        for s in range(Ti):
            if not child_iface_mask[qi, s].item():
                continue
            pq = int(phi_glue_c[qi, s].item())
            ps = int(phi_glue_s[qi, s].item())
            if pq < 0 or ps < 0:
                continue
            if qi < pq or (qi == pq and s < ps):
                either = child_a[:, qi, s] | child_a[:, pq, ps]
                child_a[:, qi, s] = either
                child_a[:, pq, ps] = either

    for qi in range(4):
        if not child_exists[qi].item():
            continue
        counts = child_a[:, qi, :].sum(dim=1)
        odd_mask = (counts % 2) != 0
        if not odd_mask.any():
            continue
        for k_i in torch.nonzero(odd_mask, as_tuple=False).flatten():
            ki = int(k_i.item())
            best_conf = float("inf")
            best_pair = None
            for s in range(Ti):
                if not child_a[ki, qi, s].item():
                    continue
                pq = int(phi_glue_c[qi, s].item())
                ps = int(phi_glue_s[qi, s].item())
                if pq >= 0 and ps >= 0:
                    conf = abs(float(glue_avg[ki, qi, s].item()) - threshold)
                    if conf < best_conf:
                        best_conf = conf
                        best_pair = (qi, s, pq, ps)
            if best_pair is not None:
                child_a[ki, best_pair[0], best_pair[1]] = False
                child_a[ki, best_pair[2], best_pair[3]] = False
            else:
                active_slots = torch.nonzero(child_a[ki, qi] & child_iface_mask[qi], as_tuple=False).flatten()
                if active_slots.numel() > 0:
                    confs = (glue_avg[ki, qi, active_slots] - threshold).abs()
                    child_a[ki, qi, active_slots[confs.argmin()]] = False

    return child_a


__all__ = [
    "_noncrossing_min_cost_matching",
    "parse_activation_batch",
    "parse_continuous",
    "parse_continuous_topk",
]
