# src/models/dp_core.py
# -*- coding: utf-8 -*-
"""
Core DP utilities for the 1-pass bottom-up algorithm.

Implements:
  1. CorrespondenceMaps  — parent↔child interface slot correspondence (φ^out, φ^sh)
  2. verify_tuple        — feasibility check for a child-state 4-tuple (Algorithm 1)
  3. parse_continuous     — continuous activation scores → discrete boundary state
  4. leaf_exact_solve     — exact cost table for leaf boxes

All routines operate on a *single* quadtree node at a time (not batched),
using the packed-token index space.  Batched wrappers can be added later.

Notation (aligned with paper Appendix D–E):
  - Ti: padded interface slot count per node
  - a[p] ∈ {0,1}: activation of slot p
  - mate[p] ∈ {0..Ti-1, -1}: partner slot (noncrossing pairing), -1 = inactive
  - boundary_dir: 0=Left, 1=Right, 2=Bottom, 3=Top
  - Quadrant: TL=0, TR=1, BL=2, BR=3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


# ─── constants ────────────────────────────────────────────────────────────────
TL, TR, BL, BR = 0, 1, 2, 3
LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 3

# Which children own each side of the parent's outer boundary.
# parent side → list of (child_quad, child_side)
# e.g. parent LEFT boundary is shared by TL (its LEFT) and BL (its LEFT).
_PARENT_SIDE_TO_CHILDREN: Dict[int, List[Tuple[int, int]]] = {
    LEFT:   [(TL, LEFT),  (BL, LEFT)],
    RIGHT:  [(TR, RIGHT), (BR, RIGHT)],
    BOTTOM: [(BL, BOTTOM),(BR, BOTTOM)],
    TOP:    [(TL, TOP),   (TR, TOP)],
}

# Adjacent child pairs sharing an internal boundary:
# (child_i, child_j, side_in_i, side_in_j)
_SHARED_BOUNDARIES: List[Tuple[int, int, int, int]] = [
    (TL, TR, RIGHT, LEFT),    # vertical boundary, upper half
    (BL, BR, RIGHT, LEFT),    # vertical boundary, lower half
    (TL, BL, BOTTOM, TOP),    # horizontal boundary, left half
    (TR, BR, BOTTOM, TOP),    # horizontal boundary, right half
]


# ─── CorrespondenceMaps ───────────────────────────────────────────────────────

@dataclass
class CorrespondenceMaps:
    """Parent–child interface correspondence for one internal node.

    All tensors are indexed by the *parent's* or *child's* padded iface slot.

    phi_out_child[p]:  which child (0–3) owns parent outer-boundary slot p, or -1
    phi_out_slot[p]:   the corresponding iface slot index in that child, or -1

    phi_sh_peer_child[i, p]:  for child i's iface slot p on a shared boundary,
                              the peer child index, or -1
    phi_sh_peer_slot[i, p]:   the peer child's iface slot index, or -1
    """
    phi_out_child: Tensor    # [Ti] long
    phi_out_slot: Tensor     # [Ti] long
    phi_sh_peer_child: Tensor  # [4, Ti] long
    phi_sh_peer_slot: Tensor   # [4, Ti] long


def build_correspondence_maps(
    *,
    parent_iface_eid: Tensor,       # [Ti] long, -1 = padding
    parent_iface_mask: Tensor,      # [Ti] bool
    parent_iface_bdir: Tensor,      # [Ti] long (boundary_dir)
    parent_cross_eid: Tensor,       # [Tc] long, -1 = padding
    parent_cross_mask: Tensor,      # [Tc] bool
    parent_cross_child_pair: Tensor, # [Tc, 2] long (quadrant indices)
    children_iface_eid: Tensor,     # [4, Ti] long, -1 = padding
    children_iface_mask: Tensor,    # [4, Ti] bool
    children_iface_bdir: Tensor,    # [4, Ti] long
    child_exists: Tensor,           # [4] bool
) -> CorrespondenceMaps:
    """Build φ^out and φ^sh by eid matching.

    This works because a single spanner edge has the same eid wherever it
    appears (crossing at LCA, interface at every ancestor below LCA).
    """
    Ti = int(parent_iface_eid.shape[0])

    phi_out_child = torch.full((Ti,), -1, dtype=torch.long)
    phi_out_slot = torch.full((Ti,), -1, dtype=torch.long)
    phi_sh_peer_child = torch.full((4, Ti), -1, dtype=torch.long)
    phi_sh_peer_slot = torch.full((4, Ti), -1, dtype=torch.long)

    # ── Build eid → (child, slot) lookup for all children ──
    # child_eid_map[eid] = list of (child_q, slot_idx)
    child_eid_map: Dict[int, List[Tuple[int, int]]] = {}
    for q in range(4):
        if not child_exists[q].item():
            continue
        for s in range(Ti):
            if not children_iface_mask[q, s].item():
                continue
            eid = int(children_iface_eid[q, s].item())
            if eid < 0:
                continue
            child_eid_map.setdefault(eid, []).append((q, s))

    # ── φ^out: parent outer-boundary iface → child iface ──
    # A parent's interface on ∂B appears as an interface in exactly one child
    # (the child whose outer boundary includes that segment of ∂B).
    for p in range(Ti):
        if not parent_iface_mask[p].item():
            continue
        eid = int(parent_iface_eid[p].item())
        if eid < 0:
            continue
        bdir = int(parent_iface_bdir[p].item())
        # Find the child that has an iface with the same eid AND whose
        # boundary_dir matches (outer boundary preserved in child).
        candidates = child_eid_map.get(eid, [])
        for (cq, cs) in candidates:
            child_bdir = int(children_iface_bdir[cq, cs].item())
            if child_bdir == bdir:
                phi_out_child[p] = cq
                phi_out_slot[p] = cs
                break

    # ── φ^sh: parent crossing → child↔child shared-boundary iface ──
    # A parent's crossing token (eid=e, child_pair=(i,j)) corresponds to
    # interface tokens in child i and child j with the same eid.
    Tc = int(parent_cross_eid.shape[0])
    for t in range(Tc):
        if not parent_cross_mask[t].item():
            continue
        eid = int(parent_cross_eid[t].item())
        if eid < 0:
            continue
        qi = int(parent_cross_child_pair[t, 0].item())
        qj = int(parent_cross_child_pair[t, 1].item())
        if qi < 0 or qj < 0:
            continue

        # Find child i's slot and child j's slot for this eid
        slot_i, slot_j = -1, -1
        for (cq, cs) in child_eid_map.get(eid, []):
            if cq == qi:
                slot_i = cs
            elif cq == qj:
                slot_j = cs
        if slot_i >= 0 and slot_j >= 0:
            phi_sh_peer_child[qi, slot_i] = qj
            phi_sh_peer_slot[qi, slot_i] = slot_j
            phi_sh_peer_child[qj, slot_j] = qi
            phi_sh_peer_slot[qj, slot_j] = slot_i

    return CorrespondenceMaps(
        phi_out_child=phi_out_child,
        phi_out_slot=phi_out_slot,
        phi_sh_peer_child=phi_sh_peer_child,
        phi_sh_peer_slot=phi_sh_peer_slot,
    )


# ─── VERIFYTUPLE ──────────────────────────────────────────────────────────────

def verify_tuple(
    *,
    parent_a: Tensor,          # [Ti] bool — parent activation vector
    parent_mate: Tensor,       # [Ti] long — parent mate map (-1 = inactive)
    parent_iface_mask: Tensor, # [Ti] bool
    child_a: Tensor,           # [4, Ti] bool — child activation vectors
    child_mate: Tensor,        # [4, Ti] long — child mate maps
    child_iface_mask: Tensor,  # [4, Ti] bool
    child_exists: Tensor,      # [4] bool
    maps: CorrespondenceMaps,
) -> bool:
    """Check feasibility of a child-state 4-tuple w.r.t. parent state.

    Implements the three checks from Algorithm 1 (paper Appendix E):
      C1: outer-boundary activation agreement
      C2: shared-boundary activation agreement
      C3+C4: connectivity composition yields parent pairing, no internal cycles

    Returns True iff the tuple is feasible.
    """
    Ti = int(parent_a.shape[0])

    # ── C1: outer-boundary activation agreement ──
    for p in range(Ti):
        if not parent_iface_mask[p].item():
            continue
        cq = int(maps.phi_out_child[p].item())
        cs = int(maps.phi_out_slot[p].item())
        if cq < 0 or cs < 0:
            # Slot has no corresponding child (e.g., pruned edge).
            # If parent activates it, that's infeasible.
            if parent_a[p].item():
                return False
            continue
        if not child_exists[cq].item():
            if parent_a[p].item():
                return False
            continue
        if bool(parent_a[p].item()) != bool(child_a[cq, cs].item()):
            return False

    # ── C2: shared-boundary activation agreement ──
    for qi in range(4):
        if not child_exists[qi].item():
            continue
        for s in range(Ti):
            if not child_iface_mask[qi, s].item():
                continue
            peer_q = int(maps.phi_sh_peer_child[qi, s].item())
            peer_s = int(maps.phi_sh_peer_slot[qi, s].item())
            if peer_q < 0 or peer_s < 0:
                continue
            if not child_exists[peer_q].item():
                continue
            # Only check each pair once (qi < peer_q or (qi==peer_q and s < peer_s))
            if qi > peer_q or (qi == peer_q and s > peer_s):
                continue
            if bool(child_a[qi, s].item()) != bool(child_a[peer_q, peer_s].item()):
                return False

    # ── C3+C4: Build gluing graph and trace ──
    # Vertices: (child_q, slot_p) for all active child-side slots.
    # Two edge types:
    #   Mate edges: (i,p) — (i, mate^(i)(p))   [inside child]
    #   Glue edges: (i,p) — (j, phi_sh(p))     [across shared boundary]

    # Collect outer-boundary active parent slots and their child mapping
    outer_parent_slots: List[int] = []
    outer_child_vertex: Dict[int, Tuple[int, int]] = {}  # parent_slot → (child_q, child_slot)
    for p in range(Ti):
        if not parent_iface_mask[p].item():
            continue
        if not parent_a[p].item():
            continue
        cq = int(maps.phi_out_child[p].item())
        cs = int(maps.phi_out_slot[p].item())
        if cq < 0 or cs < 0:
            return False
        outer_parent_slots.append(p)
        outer_child_vertex[p] = (cq, cs)

    # Build adjacency for the gluing graph
    # vertex = (child_q, slot) — only active slots
    def _neighbor_mate(q: int, s: int) -> Optional[Tuple[int, int]]:
        """Follow mate edge inside child q."""
        m = int(child_mate[q, s].item())
        if m < 0:
            return None
        return (q, m)

    def _neighbor_glue(q: int, s: int) -> Optional[Tuple[int, int]]:
        """Follow glue edge across shared boundary."""
        pq = int(maps.phi_sh_peer_child[q, s].item())
        ps = int(maps.phi_sh_peer_slot[q, s].item())
        if pq < 0 or ps < 0:
            return None
        return (pq, ps)

    # For each active outer-boundary parent slot, trace through the
    # glued system until we reach another outer-boundary slot.
    induced_mate: Dict[int, int] = {}
    traced_starts: set = set()

    for p_start in outer_parent_slots:
        if p_start in traced_starts:
            continue
        # Start at the child vertex corresponding to p_start
        cur = outer_child_vertex[p_start]
        visited: set = set()
        visited.add(cur)

        # Alternate: mate → glue → mate → glue → ... until outer boundary
        while True:
            # Step: follow mate edge
            nxt = _neighbor_mate(cur[0], cur[1])
            if nxt is None:
                return False  # active slot with no mate — invalid child state

            if nxt in visited:
                return False  # C4: internal cycle

            visited.add(nxt)

            # Check if nxt is an outer-boundary vertex
            nxt_is_outer = False
            p_end = -1
            for p_cand in outer_parent_slots:
                if outer_child_vertex[p_cand] == nxt:
                    nxt_is_outer = True
                    p_end = p_cand
                    break

            if nxt_is_outer:
                # Reached another outer-boundary slot — record pairing
                induced_mate[p_start] = p_end
                induced_mate[p_end] = p_start
                traced_starts.add(p_start)
                traced_starts.add(p_end)
                break

            # nxt is on a shared boundary — follow glue edge
            glue_nxt = _neighbor_glue(nxt[0], nxt[1])
            if glue_nxt is None:
                return False  # shared-boundary slot with no glue partner

            if glue_nxt in visited:
                return False  # C4: internal cycle

            visited.add(glue_nxt)
            cur = glue_nxt

    # ── C3: Verify induced pairing matches parent mate ──
    for p in outer_parent_slots:
        expected_partner = int(parent_mate[p].item())
        if expected_partner < 0:
            return False  # active parent slot should have a mate
        actual_partner = induced_mate.get(p, -1)
        if actual_partner != expected_partner:
            return False

    return True


# ─── PARSE ────────────────────────────────────────────────────────────────────

def _noncrossing_min_cost_matching(
    active_slots: List[int],
    cost_matrix: Tensor,
) -> Optional[List[Tuple[int, int]]]:
    """Find minimum-cost noncrossing perfect matching on slots in clockwise order.

    Uses interval DP: dp[i][j] = min cost to match the substring active_slots[i..j].
    Requires even length. cost_matrix[p][q] is the cost of pairing slots p and q.

    Returns list of (slot_a, slot_b) pairs, or None if no valid matching exists.
    """
    k = len(active_slots)
    if k == 0:
        return []
    if k % 2 != 0:
        return None

    INF = float("inf")
    dp = [[INF] * k for _ in range(k)]
    choice = [[-1] * k for _ in range(k)]  # split point

    # Base case: pairs
    for i in range(0, k - 1, 2):
        # Actually, for the DP we consider all even-length substrings
        pass

    # dp[i][j] = min cost to match active_slots[i..j] (inclusive, j-i+1 is even)
    # Recurrence: match active_slots[i] with active_slots[m] for some m in {i+1, i+3, ...},
    #   then dp[i][j] = min over m of: cost(i,m) + dp[i+1][m-1] + dp[m+1][j]
    for i in range(k):
        dp[i][i] = INF  # single element can't be matched

    for length in range(2, k + 1, 2):
        for i in range(k - length + 1):
            j = i + length - 1
            # Match active_slots[i] with some active_slots[m], m = i+1, i+3, ...
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

    # Traceback
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
    scores: Tensor,                # [4, Ti] float — sigmoid activation scores per child
    child_iface_mask: Tensor,      # [4, Ti] bool
    child_iface_bdir: Tensor,      # [4, Ti] long
    child_exists: Tensor,          # [4] bool
    maps: CorrespondenceMaps,
    r: int = 4,
    threshold: float = 0.5,
) -> Tuple[Tensor, Tensor]:
    """Convert continuous child-state scores to discrete (a, mate) per child.

    Returns:
      child_a:    [4, Ti] bool  — rounded activation
      child_mate: [4, Ti] long  — noncrossing pairing (-1 = inactive)
    """
    Ti = int(scores.shape[1])
    child_a = torch.zeros(4, Ti, dtype=torch.bool)
    child_mate = torch.full((4, Ti), -1, dtype=torch.long)

    # ── Step 1: Activation rounding with constraints ──

    # 1a: Shared-boundary consistency — average scores and round jointly
    shared_avg = scores.clone()
    for qi in range(4):
        if not child_exists[qi].item():
            continue
        for s in range(Ti):
            if not child_iface_mask[qi, s].item():
                continue
            pq = int(maps.phi_sh_peer_child[qi, s].item())
            ps = int(maps.phi_sh_peer_slot[qi, s].item())
            if pq < 0 or ps < 0:
                continue
            if qi < pq or (qi == pq and s < ps):
                avg = (scores[qi, s] + scores[pq, ps]) / 2.0
                shared_avg[qi, s] = avg
                shared_avg[pq, ps] = avg

    # 1b: Per-child, per-side budget (at most r per side), then threshold
    for qi in range(4):
        if not child_exists[qi].item():
            continue

        # Group slots by boundary_dir
        side_slots: Dict[int, List[Tuple[float, int]]] = {0: [], 1: [], 2: [], 3: []}
        for s in range(Ti):
            if not child_iface_mask[qi, s].item():
                continue
            bdir = int(child_iface_bdir[qi, s].item())
            side_slots[bdir].append((float(shared_avg[qi, s].item()), s))

        activated_slots: List[int] = []
        for bdir in range(4):
            slots = side_slots[bdir]
            # Sort by score descending
            slots.sort(key=lambda x: -x[0])
            # Keep top-r above threshold
            count = 0
            for score, s in slots:
                if count < r and score >= threshold:
                    child_a[qi, s] = True
                    activated_slots.append(s)
                    count += 1

        # 1c: Parity enforcement — must be even
        if sum(child_a[qi].tolist()) % 2 != 0:
            # Flip the lowest-confidence active slot
            active_with_score = [
                (abs(float(shared_avg[qi, s].item()) - threshold), s)
                for s in activated_slots
                if child_a[qi, s].item()
            ]
            if active_with_score:
                active_with_score.sort()
                flip_slot = active_with_score[0][1]
                child_a[qi, flip_slot] = False

    # ── Enforce shared-boundary consistency after rounding ──
    for qi in range(4):
        if not child_exists[qi].item():
            continue
        for s in range(Ti):
            if not child_iface_mask[qi, s].item():
                continue
            pq = int(maps.phi_sh_peer_child[qi, s].item())
            ps = int(maps.phi_sh_peer_slot[qi, s].item())
            if pq < 0 or ps < 0:
                continue
            if qi < pq or (qi == pq and s < ps):
                # Take OR (if either is active, both must be)
                either = child_a[qi, s].item() or child_a[pq, ps].item()
                child_a[qi, s] = either
                child_a[pq, ps] = either

    # Re-check parity after consistency enforcement
    for qi in range(4):
        if not child_exists[qi].item():
            continue
        if sum(child_a[qi].tolist()) % 2 != 0:
            # Find the shared-boundary slot with lowest confidence and deactivate both
            best_conf = float("inf")
            best_pair = None
            for s in range(Ti):
                if not child_a[qi, s].item():
                    continue
                pq = int(maps.phi_sh_peer_child[qi, s].item())
                ps = int(maps.phi_sh_peer_slot[qi, s].item())
                if pq >= 0 and ps >= 0:
                    conf = abs(float(shared_avg[qi, s].item()) - threshold)
                    if conf < best_conf:
                        best_conf = conf
                        best_pair = (qi, s, pq, ps)
            if best_pair is not None:
                child_a[best_pair[0], best_pair[1]] = False
                child_a[best_pair[2], best_pair[3]] = False
            else:
                # No shared-boundary slot; flip the lowest-confidence active slot
                active_with_score = [
                    (abs(float(shared_avg[qi, s].item()) - threshold), s)
                    for s in range(Ti)
                    if child_a[qi, s].item() and child_iface_mask[qi, s].item()
                ]
                if active_with_score:
                    active_with_score.sort()
                    child_a[qi, active_with_score[0][1]] = False

    # ── Step 2: Pairing inference ──
    for qi in range(4):
        if not child_exists[qi].item():
            continue

        # Collect active slots in clockwise order (by boundary_dir then position)
        # Clockwise for a rectangle: Top → Right → Bottom(reversed) → Left(reversed)
        # We approximate by sorting: TOP slots by x asc, RIGHT by y desc,
        # BOTTOM by x desc, LEFT by y asc.
        # For simplicity, use boundary_dir * Ti + slot_index as a proxy for
        # clockwise order (since slots within a side are already sorted by
        # coordinate during pyramid construction).
        active = []
        for s in range(Ti):
            if child_a[qi, s].item() and child_iface_mask[qi, s].item():
                bdir = int(child_iface_bdir[qi, s].item())
                # Clockwise ordering: TOP(3)=0, RIGHT(1)=1, BOTTOM(2)=2, LEFT(0)=3
                cw_order = {TOP: 0, RIGHT: 1, BOTTOM: 2, LEFT: 3}
                active.append((cw_order.get(bdir, bdir), s, s))

        active.sort(key=lambda x: (x[0], x[1]))
        active_slots = [x[2] for x in active]

        if len(active_slots) == 0:
            continue
        if len(active_slots) % 2 != 0:
            # Should not happen after parity enforcement; skip
            continue

        # Build cost matrix for pairing: prefer pairing slots with high scores
        # Cost = negative product of activation scores (lower = better)
        cost_matrix = torch.zeros(Ti, Ti)
        for i, si in enumerate(active_slots):
            for j, sj in enumerate(active_slots):
                if i < j:
                    # Lower cost for higher confidence pairs
                    cost_matrix[si, sj] = -(shared_avg[qi, si] * shared_avg[qi, sj])
                    cost_matrix[sj, si] = cost_matrix[si, sj]

        pairs = _noncrossing_min_cost_matching(active_slots, cost_matrix)
        if pairs is not None:
            for (sa, sb) in pairs:
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
    """Emit up to K candidate discrete child tuples by varying threshold.

    Returns list of (child_a, child_mate) tuples that pass VERIFYTUPLE.
    """
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
        )
        # Deduplicate
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


# ─── Leaf exact solver ────────────────────────────────────────────────────────

def _all_permutations(n: int) -> List[List[int]]:
    """Generate all permutations of [0..n-1]."""
    if n == 0:
        return [[]]
    result: List[List[int]] = []

    def _perm(arr: List[int], start: int) -> None:
        if start == len(arr):
            result.append(arr[:])
            return
        for i in range(start, len(arr)):
            arr[start], arr[i] = arr[i], arr[start]
            _perm(arr, start + 1)
            arr[start], arr[i] = arr[i], arr[start]

    _perm(list(range(n)), 0)
    return result


def leaf_exact_solve(
    *,
    points_xy: Tensor,          # [P, 2] float — points inside this leaf (absolute coords)
    point_mask: Tensor,         # [P] bool
    iface_eid: Tensor,          # [Ti] long
    iface_mask: Tensor,         # [Ti] bool
    iface_boundary_dir: Tensor, # [Ti] long
    iface_feat6: Tensor,        # [Ti, 6] float (for crossing-site coordinates)
    state_used_iface: Tensor,   # [S, Ti] bool — catalog of all states' activation patterns
    state_mate: Tensor,         # [S, Ti] long — catalog of all states' mate maps
    state_mask: Tensor,         # [S] bool — which states are valid for this node
    box_xy: Tensor,             # [4] float — (x, y, w, h) of this leaf box
) -> Tensor:
    """Compute exact cost table for a leaf box.

    For each valid boundary state σ, compute the minimum total path length
    of a legal partial solution inside the leaf.

    For leaf boxes with very few points (≤4), we enumerate all possible
    routings.

    Returns:
      costs: [S] float — cost for each state (+inf if infeasible)
    """
    S = int(state_used_iface.shape[0])
    INF = float("inf")
    costs = torch.full((S,), INF)

    num_points = int(point_mask.sum().item())
    valid_points = points_xy[point_mask]  # [K, 2]

    # Interface crossing-site absolute positions (reconstructed from feat6)
    # feat6 = [inside_rel_x, inside_rel_y, inter_rel_x, inter_rel_y, norm_len, angle]
    # inter_rel is center-relative: abs_x = cx + inter_rel_x * hw
    cx = float(box_xy[0].item()) + float(box_xy[2].item()) * 0.5
    cy = float(box_xy[1].item()) + float(box_xy[3].item()) * 0.5
    hw = float(box_xy[2].item()) * 0.5
    hh = float(box_xy[3].item()) * 0.5

    Ti = int(iface_mask.shape[0])
    iface_pos = torch.zeros(Ti, 2)
    for s in range(Ti):
        if not iface_mask[s].item():
            continue
        inter_rx = float(iface_feat6[s, 2].item())
        inter_ry = float(iface_feat6[s, 3].item())
        iface_pos[s, 0] = cx + inter_rx * hw
        iface_pos[s, 1] = cy + inter_ry * hh

    for si in range(S):
        if not state_mask[si].item():
            continue

        used = state_used_iface[si]  # [Ti] bool
        mate = state_mate[si]         # [Ti] long

        # Count active boundary crossings
        active_slots = [s for s in range(Ti) if used[s].item() and iface_mask[s].item()]
        num_active = len(active_slots)

        if num_active % 2 != 0:
            continue  # invalid state

        # Determine path endpoints from the pairing
        # Each pair (p, mate(p)) defines two endpoints of one path segment
        # that must pass through boundary sites iface_pos[p] and iface_pos[mate(p)]
        pairs: List[Tuple[int, int]] = []
        seen_slots: set = set()
        for p in active_slots:
            if p in seen_slots:
                continue
            m = int(mate[p].item())
            if m < 0:
                break  # invalid
            pairs.append((p, m))
            seen_slots.add(p)
            seen_slots.add(m)

        if len(seen_slots) != num_active:
            continue  # invalid pairing

        num_paths = len(pairs)

        # Special case: no active crossings (empty state)
        if num_paths == 0:
            if num_points == 0:
                costs[si] = 0.0
            else:
                # Must visit all points with internal paths only
                # For the root box with p=0 this means a full tour (single cycle)
                # For non-root empty state: all points must be visited by paths
                # that don't cross the boundary — this means a single closed tour
                # inside the box. For leaf boxes this is a simple TSP on the points.
                if num_points <= 1:
                    costs[si] = 0.0
                else:
                    # Enumerate all permutations for small point sets
                    best = INF
                    for perm in _all_permutations(num_points):
                        length = 0.0
                        for i in range(num_points - 1):
                            d = (valid_points[perm[i]] - valid_points[perm[i + 1]]).norm().item()
                            length += d
                        # Close the tour
                        d = (valid_points[perm[-1]] - valid_points[perm[0]]).norm().item()
                        length += d
                        best = min(best, length)
                    costs[si] = best
            continue

        # General case: num_paths path segments, each connecting two boundary sites
        # and collectively visiting all interior points.
        #
        # This is a constrained routing problem. For small point counts (<=4),
        # we enumerate all assignments of points to paths and all orderings.
        #
        # Simplified approach for now:
        # - All points must be visited by the paths.
        # - Enumerate all ways to distribute points among paths and all orderings.
        # - Each path goes: iface_pos[pair_a] → [assigned points in order] → iface_pos[pair_b]
        #
        # For very small instances this is exact but exponential.
        # For larger ones we'd need a smarter solver.

        if num_points > 6:
            # Too many points for brute force; skip (will be handled by heuristic)
            continue

        # Enumerate all distributions of points to paths
        all_pts = list(range(num_points))
        best_cost = INF

        def _enumerate_distributions(
            remaining: List[int],
            assignment: List[List[int]],
            depth: int,
        ) -> None:
            nonlocal best_cost
            if depth == len(remaining):
                # Evaluate this distribution
                total = 0.0
                for path_idx, (pa, pb) in enumerate(pairs):
                    pts_in_path = assignment[path_idx]
                    start = iface_pos[pa]
                    end = iface_pos[pb]
                    if len(pts_in_path) == 0:
                        total += (start - end).norm().item()
                    else:
                        # Enumerate all orderings of pts_in_path
                        best_path = INF
                        for perm in _all_permutations(len(pts_in_path)):
                            path_len = 0.0
                            prev = start
                            for pi in perm:
                                cur = valid_points[pts_in_path[pi]]
                                path_len += (prev - cur).norm().item()
                                prev = cur
                            path_len += (prev - end).norm().item()
                            best_path = min(best_path, path_len)
                        total += best_path
                if total < best_cost:
                    best_cost = total
                return

            pt = remaining[depth]
            for path_idx in range(num_paths):
                assignment[path_idx].append(pt)
                _enumerate_distributions(remaining, assignment, depth + 1)
                assignment[path_idx].pop()

        init_assignment: List[List[int]] = [[] for _ in range(num_paths)]
        _enumerate_distributions(all_pts, init_assignment, 0)
        costs[si] = best_cost

    return costs


__all__ = [
    "CorrespondenceMaps",
    "build_correspondence_maps",
    "verify_tuple",
    "parse_continuous",
    "parse_continuous_topk",
    "leaf_exact_solve",
]
