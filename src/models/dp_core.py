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

_SHARED_BOUNDARY_DIR_LOOKUP: Dict[Tuple[int, int], Tuple[int, int]] = {}
for _qi, _qj, _si, _sj in _SHARED_BOUNDARIES:
    _SHARED_BOUNDARY_DIR_LOOKUP[(_qi, _qj)] = (_si, _sj)
    _SHARED_BOUNDARY_DIR_LOOKUP[(_qj, _qi)] = (_sj, _si)


# ─── CorrespondenceMaps ───────────────────────────────────────────────────────

@dataclass
class CorrespondenceMaps:
    """Parent–child interface correspondence for one internal node.

    All tensors are indexed by the *parent's* or *child's* padded iface slot.

    phi_out_child[p]:  which child (0–3) owns parent outer-boundary slot p, or -1
    phi_out_slot[p]:   the corresponding iface slot index in that child, or -1

    phi_glue_peer_child[i, p]:  for child i's iface slot p participating in a
                                parent crossing (adjacent or diagonal),
                                the peer child index, or -1
    phi_glue_peer_slot[i, p]:   the peer child's iface slot index, or -1

    phi_sh_peer_child[i, p]:    subset of phi_glue on a true adjacent shared
                                boundary; diagonal parent crossings stay -1
    phi_sh_peer_slot[i, p]:     the peer child's iface slot index, or -1
    """
    phi_out_child: Tensor    # [Ti] long
    phi_out_slot: Tensor     # [Ti] long
    phi_glue_peer_child: Tensor  # [4, Ti] long
    phi_glue_peer_slot: Tensor   # [4, Ti] long
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
    """Build φ^out, generic parent-crossing glue, and true shared-boundary maps.

    This works because a single spanner edge has the same eid wherever it
    appears (crossing at LCA, interface at every ancestor below LCA).
    """
    Ti = int(parent_iface_eid.shape[0])

    device = parent_iface_eid.device
    phi_out_child = torch.full((Ti,), -1, dtype=torch.long, device=device)
    phi_out_slot = torch.full((Ti,), -1, dtype=torch.long, device=device)
    phi_glue_peer_child = torch.full((4, Ti), -1, dtype=torch.long, device=device)
    phi_glue_peer_slot = torch.full((4, Ti), -1, dtype=torch.long, device=device)
    phi_sh_peer_child = torch.full((4, Ti), -1, dtype=torch.long, device=device)
    phi_sh_peer_slot = torch.full((4, Ti), -1, dtype=torch.long, device=device)

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

    # ── Parent crossing → child↔child glue iface ──
    # A parent's crossing token (eid=e, child_pair=(i,j)) corresponds to
    # interface tokens in child i and child j with the same eid. This is a
    # generic glue relation: adjacent child pairs are true shared boundaries,
    # diagonal child pairs are still valid glue but not physical side-sharing.
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
            phi_glue_peer_child[qi, slot_i] = qj
            phi_glue_peer_slot[qi, slot_i] = slot_j
            phi_glue_peer_child[qj, slot_j] = qi
            phi_glue_peer_slot[qj, slot_j] = slot_i

            shared_dirs = _SHARED_BOUNDARY_DIR_LOOKUP.get((qi, qj))
            if shared_dirs is not None:
                dir_i = int(children_iface_bdir[qi, slot_i].item())
                dir_j = int(children_iface_bdir[qj, slot_j].item())
                if dir_i == shared_dirs[0] and dir_j == shared_dirs[1]:
                    phi_sh_peer_child[qi, slot_i] = qj
                    phi_sh_peer_slot[qi, slot_i] = slot_j
                    phi_sh_peer_child[qj, slot_j] = qi
                    phi_sh_peer_slot[qj, slot_j] = slot_i

    return CorrespondenceMaps(
        phi_out_child=phi_out_child,
        phi_out_slot=phi_out_slot,
        phi_glue_peer_child=phi_glue_peer_child,
        phi_glue_peer_slot=phi_glue_peer_slot,
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
      C2: child↔child glue activation agreement
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

    # ── C2: child↔child glue activation agreement ──
    for qi in range(4):
        if not child_exists[qi].item():
            continue
        for s in range(Ti):
            if not child_iface_mask[qi, s].item():
                continue
            peer_q = int(maps.phi_glue_peer_child[qi, s].item())
            peer_s = int(maps.phi_glue_peer_slot[qi, s].item())
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
    #   Glue edges: (i,p) — (j, phi_glue(p))   [across parent crossing]

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
        """Follow glue edge induced by the parent's crossing relation."""
        pq = int(maps.phi_glue_peer_child[q, s].item())
        ps = int(maps.phi_glue_peer_slot[q, s].item())
        if pq < 0 or ps < 0:
            return None
        return (pq, ps)

    # For each active outer-boundary parent slot, trace through the
    # glued system until we reach another outer-boundary slot.
    induced_mate: Dict[int, int] = {}
    traced_starts: set = set()
    all_traced: set = set()  # global set of all (child_q, slot) visited across ALL traces

    # ── Root-node special case ──────────────────────────────────────────
    # At the root box, parent_iface_mask is all-False (no spanner edges
    # cross the outermost boundary), so outer_parent_slots is empty and
    # the standard trace-based C3+C4 has no starting points.
    #
    # Instead of rejecting every child configuration with active shared-
    # boundary slots (which would make the DP return +inf on every real
    # instance), we verify that all active child interface slots form a
    # SINGLE connected component in the gluing graph.  Multiple disjoint
    # components correspond to premature sub-tours and are rejected.
    if not parent_iface_mask.any().item() and len(outer_parent_slots) == 0:
        all_active: set = set()
        for qi in range(4):
            if not child_exists[qi].item():
                continue
            for s in range(Ti):
                if not child_a[qi, s].item():
                    continue
                if not child_iface_mask[qi, s].item():
                    continue
                all_active.add((qi, s))

        if not all_active:
            return True  # no crossings at root — valid (costs govern feasibility)

        # Check every active slot has a valid mate (basic sanity)
        for qi, s in all_active:
            m = int(child_mate[qi, s].item())
            if m < 0:
                return False  # active slot without mate — invalid state

        # BFS on mate+glue edges: must reach every active slot from any start
        visited_root: set = set()
        stack: List[Tuple[int, int]] = [next(iter(all_active))]
        while stack:
            cur = stack.pop()
            if cur in visited_root:
                continue
            visited_root.add(cur)
            nxt = _neighbor_mate(cur[0], cur[1])
            if nxt is not None and nxt not in visited_root:
                stack.append(nxt)
            nxt = _neighbor_glue(cur[0], cur[1])
            if nxt is not None and nxt not in visited_root:
                stack.append(nxt)

        if not visited_root >= all_active:
            return False  # multiple disconnected components — premature sub-tours
        return True

    for p_start in outer_parent_slots:
        if p_start in traced_starts:
            continue
        # Start at the child vertex corresponding to p_start
        cur = outer_child_vertex[p_start]
        visited: set = set()
        visited.add(cur)
        all_traced.add(cur)

        # Alternate: mate → glue → mate → glue → ... until outer boundary
        while True:
            # Step: follow mate edge
            nxt = _neighbor_mate(cur[0], cur[1])
            if nxt is None:
                return False  # active slot with no mate — invalid child state

            if nxt in visited:
                return False  # C4: internal cycle

            visited.add(nxt)
            all_traced.add(nxt)

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

            # nxt participates in a parent-induced glue relation — follow it
            glue_nxt = _neighbor_glue(nxt[0], nxt[1])
            if glue_nxt is None:
                return False  # glued slot with no glue partner

            if glue_nxt in visited:
                return False  # C4: internal cycle

            visited.add(glue_nxt)
            all_traced.add(glue_nxt)
            cur = glue_nxt

    # ── C3: Verify induced pairing matches parent mate ──
    for p in outer_parent_slots:
        expected_partner = int(parent_mate[p].item())
        if expected_partner < 0:
            return False  # active parent slot should have a mate
        actual_partner = induced_mate.get(p, -1)
        if actual_partner != expected_partner:
            return False

    # ── C4 (complete): Reject internal cycles not reachable from outer boundary ──
    # Any active glued child slot not visited by the traces above
    # must belong to an internal closed cycle (disconnected from ∂B).
    for qi in range(4):
        if not child_exists[qi].item():
            continue
        for s in range(Ti):
            if not child_a[qi, s].item():
                continue
            if not child_iface_mask[qi, s].item():
                continue
            # Only check glued slots (those participating in a parent crossing)
            if int(maps.phi_glue_peer_child[qi, s].item()) < 0:
                continue
            if (qi, s) not in all_traced:
                return False  # active glued slot unreached → internal cycle

    return True


# ─── C1 Constraint Propagation ───────────────────────────────────────────────

def propagate_c1_constraints(
    *,
    parent_a: Tensor,           # [Ti] bool — parent activation vector
    parent_iface_mask: Tensor,  # [Ti] bool
    maps: CorrespondenceMaps,
    child_exists: Tensor,       # [4] bool
    child_iface_mask: Tensor,   # [4, Ti] bool
) -> Tuple[Tensor, Tensor]:
    """Propagate C1 (outer-boundary agreement) from parent σ to children.

    Given parent state σ, C1 requires that for every parent outer-boundary
    slot p mapped to child (cq, cs) via φ^out, the child's activation at cs
    must equal the parent's activation at p.

    Returns:
        c1_required:    [4, Ti] bool — required activation value for constrained slots
        c1_constrained: [4, Ti] bool — which child slots are constrained by C1
    """
    Ti = int(parent_a.shape[0])
    device = parent_a.device

    c1_required = torch.zeros(4, Ti, dtype=torch.bool, device=device)
    c1_constrained = torch.zeros(4, Ti, dtype=torch.bool, device=device)

    for p in range(Ti):
        if not parent_iface_mask[p].item():
            continue
        cq = int(maps.phi_out_child[p].item())
        cs = int(maps.phi_out_slot[p].item())
        if cq < 0 or cs < 0:
            continue
        if not child_exists[cq].item():
            continue
        if not child_iface_mask[cq, cs].item():
            continue
        c1_constrained[cq, cs] = True
        c1_required[cq, cs] = parent_a[p].bool()

    return c1_required, c1_constrained


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
    parent_a: Optional[Tensor] = None,       # [Ti] bool — parent activation (C1)
    parent_iface_mask: Optional[Tensor] = None,  # [Ti] bool
) -> Tuple[Tensor, Tensor]:
    """Convert continuous child-state scores to discrete (a, mate) per child.

    If parent_a and parent_iface_mask are provided, C1 constraints are enforced:
    outer-boundary child slots mapped via φ^out are forced to match the parent.

    Returns:
      child_a:    [4, Ti] bool  — rounded activation
      child_mate: [4, Ti] long  — noncrossing pairing (-1 = inactive)
    """
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

    # ── Step 0: Propagate C1 constraints (outer-boundary agreement) ──
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

    # ── Step 1: Activation rounding with constraints ──

    # 1a: Glue consistency — average parent-crossing peers and round jointly
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

    # 1b: Per-child, per-side budget (at most r per side), then threshold
    for qi in range(4):
        if not child_exists[qi].item():
            continue

        # Group slots by boundary_dir
        side_slots: Dict[int, List[Tuple[float, int]]] = {0: [], 1: [], 2: [], 3: []}
        for s in range(Ti):
            if not child_iface_mask[qi, s].item():
                continue
            # Skip C1-constrained slots (handled separately below)
            if c1_constrained is not None and c1_constrained[qi, s].item():
                continue
            bdir = int(child_iface_bdir[qi, s].item())
            side_slots[bdir].append((float(glue_avg[qi, s].item()), s))

        activated_slots: List[int] = []

        # First: force C1-constrained slots to match parent activation
        if c1_constrained is not None:
            for s in range(Ti):
                if not child_iface_mask[qi, s].item():
                    continue
                if c1_constrained[qi, s].item():
                    if c1_required[qi, s].item():
                        child_a[qi, s] = True
                        activated_slots.append(s)
                    # else: leave False (parent says inactive)

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
                (abs(float(glue_avg[qi, s].item()) - threshold), s)
                for s in activated_slots
                if child_a[qi, s].item()
                and not _is_c1_required_active(qi, s)
            ]
            if active_with_score:
                active_with_score.sort()
                flip_slot = active_with_score[0][1]
                child_a[qi, flip_slot] = False

    # ── Enforce glue consistency after rounding ──
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
                # Take OR (if either is active, both must be)
                either = child_a[qi, s].item() or child_a[pq, ps].item()
                child_a[qi, s] = either
                child_a[pq, ps] = either

    # Re-check parity after consistency enforcement.
    # Deactivating a glue pair (qi, s) and (pq, ps) changes the
    # parity of BOTH children qi and pq.  A single pass may fix qi but
    # break pq.  Iterate until all children have even parity.
    for _parity_round in range(4):
        any_odd = False
        for qi in range(4):
            if not child_exists[qi].item():
                continue
            if sum(child_a[qi].tolist()) % 2 == 0:
                continue
            any_odd = True
            # Find the glue slot with lowest confidence and deactivate both
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
                # No glued slot; flip the lowest-confidence active slot
                active_with_score = [
                    (abs(float(glue_avg[qi, s].item()) - threshold), s)
                    for s in range(Ti)
                    if child_a[qi, s].item() and child_iface_mask[qi, s].item()
                    and not _is_c1_required_active(qi, s)
                ]
                if active_with_score:
                    active_with_score.sort()
                    child_a[qi, active_with_score[0][1]] = False
        if not any_odd:
            break

    # ── Step 2: Pairing inference ──
    for qi in range(4):
        if not child_exists[qi].item():
            continue

        # Matching-mode packing orders slots clockwise around ∂B_i already, so
        # the active slot indices themselves define the noncrossing order.
        active_slots = [
            s
            for s in range(Ti)
            if child_a[qi, s].item() and child_iface_mask[qi, s].item()
        ]

        if len(active_slots) == 0:
            continue
        if len(active_slots) % 2 != 0:
            # Should not happen after parity enforcement; skip
            continue

        # Build cost matrix for pairing: prefer pairing slots with high scores
        # Cost = negative product of activation scores (lower = better)
        cost_matrix = torch.zeros(Ti, Ti, device=scores.device)
        for i, si in enumerate(active_slots):
            for j, sj in enumerate(active_slots):
                if i < j:
                    # Lower cost for higher confidence pairs
                    cost_matrix[si, sj] = -(glue_avg[qi, si] * glue_avg[qi, sj])
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
            parent_a=parent_a,
            parent_iface_mask=parent_iface_mask,
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


@dataclass(frozen=True)
class LeafPathWitness:
    """One open path fragment inside a leaf, oriented from start_slot to end_slot."""

    start_slot: int
    end_slot: int
    point_indices: Tuple[int, ...]


@dataclass(frozen=True)
class LeafStateWitness:
    """Exact/heuristic witness for one leaf boundary state."""

    open_paths: Tuple[LeafPathWitness, ...] = ()
    closed_cycle: Tuple[int, ...] = ()


def _held_karp_tsp(dist: List[List[float]]) -> float:
    """Exact TSP (closed tour) via Held-Karp DP. O(2^n * n^2). Handles n <= ~20."""
    n = len(dist)
    if n <= 1:
        return 0.0
    if n == 2:
        return dist[0][1] + dist[1][0]
    INF = float("inf")
    # dp[S][i] = min cost to visit subset S ending at i, starting from 0
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0.0  # start at node 0
    for S in range(1, 1 << n):
        if not (S & 1):
            continue  # must include node 0
        for i in range(n):
            if not (S & (1 << i)):
                continue
            if dp[S][i] >= INF:
                continue
            for j in range(n):
                if S & (1 << j):
                    continue
                S2 = S | (1 << j)
                c = dp[S][i] + dist[i][j]
                if c < dp[S2][j]:
                    dp[S2][j] = c
    full = (1 << n) - 1
    return min(dp[full][i] + dist[i][0] for i in range(1, n))


def _held_karp_tsp_with_order(dist: List[List[float]]) -> Tuple[float, List[int]]:
    """Exact TSP via Held-Karp with a concrete tour order."""
    n = len(dist)
    if n <= 0:
        return 0.0, []
    if n == 1:
        return 0.0, [0]
    if n == 2:
        return dist[0][1] + dist[1][0], [0, 1]

    INF = float("inf")
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    dp[1][0] = 0.0

    for S in range(1, 1 << n):
        if not (S & 1):
            continue
        for i in range(n):
            if not (S & (1 << i)):
                continue
            if dp[S][i] >= INF:
                continue
            for j in range(n):
                if S & (1 << j):
                    continue
                S2 = S | (1 << j)
                c = dp[S][i] + dist[i][j]
                if c < dp[S2][j]:
                    dp[S2][j] = c
                    parent[S2][j] = i

    full = (1 << n) - 1
    best_cost = INF
    best_last = -1
    for i in range(1, n):
        c = dp[full][i] + dist[i][0]
        if c < best_cost:
            best_cost = c
            best_last = i

    if best_last < 0:
        return INF, []

    rev: List[int] = [best_last]
    S = full
    cur = best_last
    while cur != 0:
        prev = parent[S][cur]
        S ^= (1 << cur)
        cur = prev
        if cur < 0:
            return INF, []
        rev.append(cur)

    order = list(reversed(rev))
    return best_cost, order


def _held_karp_path(dist: List[List[float]], start: int, end: int) -> float:
    """Exact shortest Hamiltonian PATH from start to end via Held-Karp.
    dist is (n+2)x(n+2) where start and end are indices into dist."""
    n = len(dist)
    if n <= 1:
        return 0.0
    if n == 2:
        return dist[start][end]
    INF = float("inf")
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1 << start][start] = 0.0
    for S in range(1, 1 << n):
        if not (S & (1 << start)):
            continue
        for i in range(n):
            if not (S & (1 << i)):
                continue
            if dp[S][i] >= INF:
                continue
            for j in range(n):
                if S & (1 << j):
                    continue
                S2 = S | (1 << j)
                c = dp[S][i] + dist[i][j]
                if c < dp[S2][j]:
                    dp[S2][j] = c
    full = (1 << n) - 1
    return dp[full][end]


def _nn_tsp(points: Tensor) -> float:
    """Nearest-neighbor heuristic for TSP. Returns tour length."""
    n = points.shape[0]
    if n <= 1:
        return 0.0
    visited = [False] * n
    tour_len = 0.0
    cur = 0
    visited[0] = True
    for _ in range(n - 1):
        best_d, best_j = float("inf"), -1
        for j in range(n):
            if visited[j]:
                continue
            d = (points[cur] - points[j]).norm().item()
            if d < best_d:
                best_d, best_j = d, j
        tour_len += best_d
        visited[best_j] = True
        cur = best_j
    tour_len += (points[cur] - points[0]).norm().item()
    return tour_len


def _nn_tsp_with_order(points: Tensor) -> Tuple[float, List[int]]:
    """Nearest-neighbor heuristic for TSP with a concrete tour order."""
    n = points.shape[0]
    if n <= 0:
        return 0.0, []
    if n == 1:
        return 0.0, [0]

    visited = [False] * n
    order = [0]
    tour_len = 0.0
    cur = 0
    visited[0] = True
    for _ in range(n - 1):
        best_d, best_j = float("inf"), -1
        for j in range(n):
            if visited[j]:
                continue
            d = (points[cur] - points[j]).norm().item()
            if d < best_d:
                best_d, best_j = d, j
        if best_j < 0:
            return float("inf"), []
        order.append(best_j)
        tour_len += best_d
        visited[best_j] = True
        cur = best_j
    tour_len += (points[cur] - points[0]).norm().item()
    return tour_len, order


def _nn_path(all_points: Tensor, start_pos: Tensor, end_pos: Tensor,
             interior_indices: List[int]) -> float:
    """Nearest-neighbor heuristic for Hamiltonian path: start → interior → end."""
    if not interior_indices:
        return (start_pos - end_pos).norm().item()
    path_len = 0.0
    cur = start_pos
    remaining = set(interior_indices)
    for _ in range(len(interior_indices)):
        best_d, best_j = float("inf"), -1
        for j in remaining:
            d = (cur - all_points[j]).norm().item()
            if d < best_d:
                best_d, best_j = d, j
        path_len += best_d
        cur = all_points[best_j]
        remaining.remove(best_j)
    path_len += (cur - end_pos).norm().item()
    return path_len


def _nn_path_with_order(
    all_points: Tensor,
    start_pos: Tensor,
    end_pos: Tensor,
    interior_indices: List[int],
) -> Tuple[float, List[int]]:
    """Nearest-neighbor heuristic for Hamiltonian path with an explicit order."""
    if not interior_indices:
        return (start_pos - end_pos).norm().item(), []

    path_len = 0.0
    order: List[int] = []
    cur = start_pos
    remaining = set(int(i) for i in interior_indices)
    for _ in range(len(interior_indices)):
        best_d, best_j = float("inf"), -1
        for j in remaining:
            d = (cur - all_points[j]).norm().item()
            if d < best_d:
                best_d, best_j = d, j
        if best_j < 0:
            return float("inf"), []
        path_len += best_d
        cur = all_points[best_j]
        remaining.remove(best_j)
        order.append(best_j)
    path_len += (cur - end_pos).norm().item()
    return path_len, order


def _prepare_leaf_geometry(
    *,
    points_xy: Tensor,
    point_mask: Tensor,
    iface_mask: Tensor,
    iface_feat6: Tensor,
    box_xy: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Convert leaf-local coordinates to root-normalized geometry once."""
    device = points_xy.device
    valid_points_rel = points_xy[point_mask]
    num_points = int(point_mask.sum().item())

    cx = float(box_xy[0].item())
    cy = float(box_xy[1].item())
    hw = float(box_xy[2].item()) * 0.5
    hh = float(box_xy[3].item()) * 0.5

    valid_points = torch.zeros_like(valid_points_rel)
    if num_points > 0:
        valid_points[:, 0] = cx + valid_points_rel[:, 0] * hw
        valid_points[:, 1] = cy + valid_points_rel[:, 1] * hh

    Ti = int(iface_mask.shape[0])
    iface_pos = torch.zeros(Ti, 2, device=device)
    for s in range(Ti):
        if not iface_mask[s].item():
            continue
        inter_rx = float(iface_feat6[s, 2].item())
        inter_ry = float(iface_feat6[s, 3].item())
        iface_pos[s, 0] = cx + inter_rx * hw
        iface_pos[s, 1] = cy + inter_ry * hh

    return valid_points, iface_pos


def _solve_leaf_state(
    *,
    valid_points: Tensor,
    iface_pos: Tensor,
    iface_mask: Tensor,
    used: Tensor,
    mate: Tensor,
    is_root: bool,
    return_witness: bool = False,
) -> Tuple[float, Optional[LeafStateWitness]]:
    """Solve one leaf boundary state, optionally returning a concrete witness."""
    INF = float("inf")
    num_points = int(valid_points.shape[0])
    Ti = int(iface_mask.shape[0])

    active_slots = [s for s in range(Ti) if used[s].item() and iface_mask[s].item()]
    num_active = len(active_slots)
    if num_active % 2 != 0:
        return INF, None

    pairs: List[Tuple[int, int]] = []
    seen_slots: set[int] = set()
    for p in active_slots:
        if p in seen_slots:
            continue
        m = int(mate[p].item())
        if m < 0 or m in seen_slots or not used[m].item() or not iface_mask[m].item():
            return INF, None
        pairs.append((p, m))
        seen_slots.add(p)
        seen_slots.add(m)

    if len(seen_slots) != num_active:
        return INF, None

    num_paths = len(pairs)

    if num_paths == 0:
        if num_points == 0:
            witness = LeafStateWitness() if return_witness else None
            return 0.0, witness
        if not is_root:
            return INF, None

        if num_points == 1:
            witness = LeafStateWitness(closed_cycle=(0,)) if return_witness else None
            return 0.0, witness

        if num_points <= 18:
            dist = [[0.0] * num_points for _ in range(num_points)]
            for a in range(num_points):
                for b in range(a + 1, num_points):
                    d = (valid_points[a] - valid_points[b]).norm().item()
                    dist[a][b] = dist[b][a] = d
            if return_witness:
                cost, order = _held_karp_tsp_with_order(dist)
                return cost, LeafStateWitness(closed_cycle=tuple(order))
            return _held_karp_tsp(dist), None

        if return_witness:
            cost, order = _nn_tsp_with_order(valid_points)
            return cost, LeafStateWitness(closed_cycle=tuple(order))
        return _nn_tsp(valid_points), None

    if num_paths == 1 and num_points == 0:
        cost = (iface_pos[pairs[0][0]] - iface_pos[pairs[0][1]]).norm().item()
        witness = None
        if return_witness:
            witness = LeafStateWitness(
                open_paths=(LeafPathWitness(pairs[0][0], pairs[0][1], ()),),
            )
        return cost, witness

    if num_points <= 6:
        all_pts = list(range(num_points))
        best_cost = INF
        best_orders: Optional[List[List[int]]] = None

        def _enumerate_distributions(
            remaining: List[int],
            assignment: List[List[int]],
            depth: int,
        ) -> None:
            nonlocal best_cost, best_orders
            if depth == len(remaining):
                total = 0.0
                chosen_orders: List[List[int]] = []
                for path_idx, (pa, pb) in enumerate(pairs):
                    pts_in_path = assignment[path_idx]
                    start = iface_pos[pa]
                    end = iface_pos[pb]
                    if len(pts_in_path) == 0:
                        total += (start - end).norm().item()
                        chosen_orders.append([])
                        continue

                    best_path = INF
                    best_order: List[int] = []
                    for perm in _all_permutations(len(pts_in_path)):
                        path_len = 0.0
                        prev = start
                        order_local: List[int] = []
                        for pi in perm:
                            cur_idx = pts_in_path[pi]
                            cur = valid_points[cur_idx]
                            path_len += (prev - cur).norm().item()
                            prev = cur
                            order_local.append(cur_idx)
                        path_len += (prev - end).norm().item()
                        if path_len < best_path:
                            best_path = path_len
                            best_order = order_local
                    total += best_path
                    chosen_orders.append(best_order)

                if total < best_cost:
                    best_cost = total
                    if return_witness:
                        best_orders = [o[:] for o in chosen_orders]
                return

            pt = remaining[depth]
            for path_idx in range(num_paths):
                assignment[path_idx].append(pt)
                _enumerate_distributions(remaining, assignment, depth + 1)
                assignment[path_idx].pop()

        init_assignment: List[List[int]] = [[] for _ in range(num_paths)]
        _enumerate_distributions(all_pts, init_assignment, 0)

        if not return_witness:
            return best_cost, None

        if best_orders is None:
            return best_cost, None

        open_paths = tuple(
            LeafPathWitness(pa, pb, tuple(best_orders[path_idx]))
            for path_idx, (pa, pb) in enumerate(pairs)
        )
        return best_cost, LeafStateWitness(open_paths=open_paths)

    path_assignments: List[List[int]] = [[] for _ in range(num_paths)]
    for pi in range(num_points):
        best_d, best_path = float("inf"), 0
        for pidx, (pa, pb) in enumerate(pairs):
            d = min(
                (valid_points[pi] - iface_pos[pa]).norm().item(),
                (valid_points[pi] - iface_pos[pb]).norm().item(),
            )
            if d < best_d:
                best_d, best_path = d, pidx
        path_assignments[best_path].append(pi)

    total = 0.0
    orders: List[List[int]] = []
    for pidx, (pa, pb) in enumerate(pairs):
        if return_witness:
            path_cost, order = _nn_path_with_order(
                valid_points,
                iface_pos[pa],
                iface_pos[pb],
                path_assignments[pidx],
            )
            total += path_cost
            orders.append(order)
        else:
            total += _nn_path(
                valid_points,
                iface_pos[pa],
                iface_pos[pb],
                path_assignments[pidx],
            )

    if not return_witness:
        return total, None

    open_paths = tuple(
        LeafPathWitness(pa, pb, tuple(orders[pidx]))
        for pidx, (pa, pb) in enumerate(pairs)
    )
    return total, LeafStateWitness(open_paths=open_paths)


def leaf_solve_state(
    *,
    points_xy: Tensor,
    point_mask: Tensor,
    iface_mask: Tensor,
    iface_feat6: Tensor,
    state_used: Tensor,
    state_mate: Tensor,
    box_xy: Tensor,
    is_root: bool = False,
) -> Tuple[float, Optional[LeafStateWitness]]:
    """Solve a single leaf state and recover its concrete witness."""
    valid_points, iface_pos = _prepare_leaf_geometry(
        points_xy=points_xy,
        point_mask=point_mask,
        iface_mask=iface_mask,
        iface_feat6=iface_feat6,
        box_xy=box_xy,
    )
    return _solve_leaf_state(
        valid_points=valid_points,
        iface_pos=iface_pos,
        iface_mask=iface_mask,
        used=state_used,
        mate=state_mate,
        is_root=is_root,
        return_witness=True,
    )


def leaf_exact_solve(
    *,
    points_xy: Tensor,          # [P, 2] float — points inside this leaf (node-relative coords, [-1,1])
    point_mask: Tensor,         # [P] bool
    iface_eid: Tensor,          # [Ti] long
    iface_mask: Tensor,         # [Ti] bool
    iface_boundary_dir: Tensor, # [Ti] long
    iface_feat6: Tensor,        # [Ti, 6] float (for crossing-site coordinates)
    state_used_iface: Tensor,   # [S, Ti] bool — catalog of all states' activation patterns
    state_mate: Tensor,         # [S, Ti] long — catalog of all states' mate maps
    state_mask: Tensor,         # [S] bool — which states are valid for this node
    box_xy: Tensor,             # [4] float — (x, y, w, h) of this leaf box
    is_root: bool = False,      # True only for the root box (allows closed-tour empty state)
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
    device = points_xy.device
    costs = torch.full((S,), float("inf"), device=device)
    valid_points, iface_pos = _prepare_leaf_geometry(
        points_xy=points_xy,
        point_mask=point_mask,
        iface_mask=iface_mask,
        iface_feat6=iface_feat6,
        box_xy=box_xy,
    )

    for si in range(S):
        if not state_mask[si].item():
            continue
        cost, _ = _solve_leaf_state(
            valid_points=valid_points,
            iface_pos=iface_pos,
            iface_mask=iface_mask,
            used=state_used_iface[si],
            mate=state_mate[si],
            is_root=is_root,
            return_witness=False,
        )
        costs[si] = cost

    return costs


# ─── Batch C1+C2 check (vectorized) ─────────────────────────────────────────

def batch_check_c1c2(
    *,
    parent_a_batch: Tensor,        # [K, Ti] bool — parent activation per candidate
    child_a_batch: Tensor,         # [K, 4, Ti] bool — child activation per candidate
    child_iface_mask: Tensor,      # [4, Ti] bool (shared across candidates)
    child_exists: Tensor,          # [4] bool (shared)
    maps: CorrespondenceMaps,
) -> Tensor:
    """Vectorized C1 + C2 feasibility check for K candidates at once.

    Returns: [K] bool — True if candidate passes both C1 and C2.
    """
    K, Ti = parent_a_batch.shape
    device = parent_a_batch.device

    # ── C1: outer-boundary activation agreement ──
    # For each parent slot p with phi_out mapping → child (cq, cs):
    #   parent_a[p] must equal child_a[cq, cs]
    c1_ok = torch.ones(K, dtype=torch.bool, device=device)

    phi_c = maps.phi_out_child   # [Ti]
    phi_s = maps.phi_out_slot    # [Ti]

    # Build a mask of parent iface slots that have valid phi_out mapping
    has_map = phi_c >= 0  # [Ti]

    if has_map.any():
        mapped_p = torch.nonzero(has_map, as_tuple=False).flatten()  # [P]
        cq = phi_c[mapped_p]  # [P] child quadrant
        cs = phi_s[mapped_p]  # [P] child slot

        # Gather parent activations at mapped slots: [K, P]
        pa = parent_a_batch[:, mapped_p]

        # Gather child activations at corresponding (cq, cs): [K, P]
        # child_a_batch is [K, 4, Ti], we need child_a_batch[:, cq[j], cs[j]] for each j
        ca = child_a_batch[:, cq, cs]  # advanced indexing → [K, P]

        # C1: all mapped slots must agree
        c1_ok = (pa == ca).all(dim=1)  # [K]

    # ── C2: child↔child glue activation agreement ──
    # For each (qi, s) with phi_glue peer (pq, ps):
    #   child_a[qi, s] must equal child_a[pq, ps]
    c2_ok = torch.ones(K, dtype=torch.bool, device=device)

    phi_glue_c = maps.phi_glue_peer_child  # [4, Ti]
    phi_glue_s = maps.phi_glue_peer_slot   # [4, Ti]

    # Collect all glue pairs (only qi < pq to avoid double-checking)
    sh_qi_list, sh_si_list, sh_pq_list, sh_ps_list = [], [], [], []
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
                sh_qi_list.append(qi)
                sh_si_list.append(s)
                sh_pq_list.append(pq)
                sh_ps_list.append(ps)

    if sh_qi_list:
        sh_qi = torch.tensor(sh_qi_list, device=device, dtype=torch.long)
        sh_si = torch.tensor(sh_si_list, device=device, dtype=torch.long)
        sh_pq = torch.tensor(sh_pq_list, device=device, dtype=torch.long)
        sh_ps = torch.tensor(sh_ps_list, device=device, dtype=torch.long)

        # Gather: child_a_batch[:, sh_qi, sh_si] vs child_a_batch[:, sh_pq, sh_ps]
        a_left = child_a_batch[:, sh_qi, sh_si]    # [K, num_pairs]
        a_right = child_a_batch[:, sh_pq, sh_ps]   # [K, num_pairs]
        c2_ok = (a_left == a_right).all(dim=1)      # [K]

    return c1_ok & c2_ok


def parse_activation_batch(
    *,
    scores_batch: Tensor,          # [K, 4, Ti] float — sigmoid scores
    child_iface_mask: Tensor,      # [4, Ti] bool (shared)
    child_iface_bdir: Tensor,      # [4, Ti] long (shared)
    child_exists: Tensor,          # [4] bool (shared)
    maps: CorrespondenceMaps,
    r: int = 4,
    threshold: float = 0.5,
) -> Tensor:
    """Batch PARSE Phase A: continuous scores → discrete activations (no matching).

    Returns:
      child_a: [K, 4, Ti] bool — rounded activation vectors
    """
    K = scores_batch.shape[0]
    Ti = scores_batch.shape[2]
    device = scores_batch.device

    # ── Step 1a: Glue consistency — average jointly ──
    glue_avg = scores_batch.clone()  # [K, 4, Ti]

    phi_glue_c = maps.phi_glue_peer_child  # [4, Ti]
    phi_glue_s = maps.phi_glue_peer_slot   # [4, Ti]

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
                avg = (scores_batch[:, qi, s] + scores_batch[:, pq, ps]) / 2.0  # [K]
                glue_avg[:, qi, s] = avg
                glue_avg[:, pq, ps] = avg

    # ── Step 1b: Per-child, per-side budget + threshold → child_a ──
    child_a = torch.zeros(K, 4, Ti, dtype=torch.bool, device=device)

    for qi in range(4):
        if not child_exists[qi].item():
            continue
        for bdir in range(4):
            # Find slots on this side
            side_mask = child_iface_mask[qi] & (child_iface_bdir[qi] == bdir)
            side_slots = torch.nonzero(side_mask, as_tuple=False).flatten()
            if side_slots.numel() == 0:
                continue
            # Scores for this side: [K, num_side_slots]
            side_scores = glue_avg[:, qi, side_slots]
            # Top-r above threshold
            n_side = side_slots.numel()
            keep = min(r, n_side)
            if keep == 0:
                continue
            topk_vals, topk_idx = side_scores.topk(keep, dim=1)  # [K, keep]
            above = topk_vals >= threshold  # [K, keep]
            # Map back to original slot indices
            slot_ids = side_slots[topk_idx]  # [K, keep]
            # Set activations: child_a[:, qi, slot_ids] = above
            child_a[:, qi, :].scatter_(1, slot_ids, above)

    # ── Step 1c: Parity enforcement — must be even per child ──
    for qi in range(4):
        if not child_exists[qi].item():
            continue
        counts = child_a[:, qi, :].sum(dim=1)  # [K]
        odd_mask = (counts % 2) != 0  # [K]
        if not odd_mask.any():
            continue
        # For odd candidates, flip lowest-confidence active slot
        for k_i in torch.nonzero(odd_mask, as_tuple=False).flatten():
            ki = int(k_i.item())
            active_slots = torch.nonzero(
                child_a[ki, qi] & child_iface_mask[qi], as_tuple=False
            ).flatten()
            if active_slots.numel() == 0:
                continue
            confs = (glue_avg[ki, qi, active_slots] - threshold).abs()
            flip_idx = active_slots[confs.argmin()]
            child_a[ki, qi, flip_idx] = False

    # ── Enforce glue consistency after rounding (OR rule) ──
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
                either = child_a[:, qi, s] | child_a[:, pq, ps]  # [K]
                child_a[:, qi, s] = either
                child_a[:, pq, ps] = either

    # ── Re-check parity after consistency enforcement ──
    for qi in range(4):
        if not child_exists[qi].item():
            continue
        counts = child_a[:, qi, :].sum(dim=1)  # [K]
        odd_mask = (counts % 2) != 0
        if not odd_mask.any():
            continue
        for k_i in torch.nonzero(odd_mask, as_tuple=False).flatten():
            ki = int(k_i.item())
            # Find glue slot with lowest confidence, deactivate pair
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
                active_slots = torch.nonzero(
                    child_a[ki, qi] & child_iface_mask[qi], as_tuple=False
                ).flatten()
                if active_slots.numel() > 0:
                    confs = (glue_avg[ki, qi, active_slots] - threshold).abs()
                    child_a[ki, qi, active_slots[confs.argmin()]] = False

    return child_a


# ─── Catalog-enumeration PARSE ────────────────────────────────────────────────

def _rank_child_catalog_states_for_parse(
    *,
    scores_q: Tensor,             # [Ti] float — neural scores for one child
    child_iface_mask_q: Tensor,   # [Ti] bool
    constrained_q: Tensor,        # [Ti] bool
    required_q: Tensor,           # [Ti] bool
    cat_used: Tensor,             # [S, Ti] bool
    child_cost_q: Tensor,         # [S] float
    max_child_states: Optional[int] = None,
) -> Tensor:
    """Return child-state indices after C1 filtering, ranking, and optional truncation."""
    finite_mask = child_cost_q < float("inf")  # [S]
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

    valid_used = cat_used[valid_indices].float()               # [V, Ti]
    sq = scores_q.unsqueeze(0)                                 # [1, Ti]
    mask_q = child_iface_mask_q.float().unsqueeze(0)           # [1, Ti]
    agree = valid_used * sq + (1.0 - valid_used) * (1.0 - sq)  # [V, Ti]
    state_scores = (agree * mask_q).sum(dim=1)                 # [V]

    if max_child_states is not None:
        cap = int(max_child_states)
        if cap > 0 and valid_indices.numel() > cap:
            top_pos = torch.topk(
                state_scores,
                k=cap,
                largest=True,
                sorted=True,
            ).indices
            return valid_indices[top_pos]

    sort_order = state_scores.argsort(descending=True)
    ranked_indices = valid_indices[sort_order]

    return ranked_indices


def parse_by_catalog_enum(
    *,
    scores: Tensor,                # [4, Ti] float — sigmoid activation scores per child
    parent_a: Tensor,              # [Ti] bool — parent activation
    parent_mate: Tensor,           # [Ti] long — parent mate
    parent_iface_mask: Tensor,     # [Ti] bool
    child_iface_mask: Tensor,      # [4, Ti] bool
    child_exists: Tensor,          # [4] bool
    maps: CorrespondenceMaps,
    cat_used: Tensor,              # [S, Ti] bool — catalog used_iface (on device)
    cat_mate: Tensor,              # [S, Ti] long — catalog mate (on device)
    child_costs: List[Optional[Tensor]],  # 4-list; each [S] float or None
    max_child_states: Optional[int] = None,
) -> Tuple[float, Tuple[int, int, int, int]]:
    """Catalog-enumeration PARSE: replace heuristic rounding with direct enumeration.

    Since max_used is small (4–6), the catalog per child is small enough to
    enumerate.  We score each catalog state by its agreement with the neural
    prediction, sort candidates best-first, and backtrack with C1+C2 pruning
    and C3+C4 verification.

    All constraints (even parity, noncrossing matching, C1–C4) are guaranteed
    by construction — no fragile heuristic rounding needed.

    Returns:
      (best_cost, (s0, s1, s2, s3))  — total child DP cost and state indices,
      or (inf, (-1,-1,-1,-1)) if infeasible.
    """
    Ti = int(parent_a.shape[0])
    device = parent_a.device

    # ── C1: propagate parent constraints to children ──
    c1_required, c1_constrained = propagate_c1_constraints(
        parent_a=parent_a.bool(),
        parent_iface_mask=parent_iface_mask,
        maps=maps,
        child_exists=child_exists,
        child_iface_mask=child_iface_mask,
    )

    # ── Per-child: filter by C1 + finite cost, sort by neural score ──
    # Optional heuristic mode: after C1 filtering and ranking, keep only the
    # top-L child states. This does NOT preserve certified exactness and should
    # only be used as an explicit speed/quality trade-off.
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

    # ── Glue links for C2 pruning ──
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

    # ── Backtracking with C2 pruning + C3/C4 verify ──
    best_cost = float("inf")
    best_indices: Tuple[int, int, int, int] = (-1, -1, -1, -1)

    def _enum(q: int, current_cost: float, current_indices: List[int]) -> None:
        nonlocal best_cost, best_indices

        if current_cost >= best_cost:
            return  # cost pruning

        if q == 4:
            # All children chosen — full C3+C4 verify
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
            # C2 pruning: glue agreement with already-chosen peers
            c2_ok = True
            for my_slot, peer_q, peer_slot in sh_links[q]:
                peer_si = current_indices[peer_q]
                if peer_si < 0:
                    continue
                if bool(cat_used[s, my_slot].item()) != \
                   bool(cat_used[peer_si, peer_slot].item()):
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


__all__ = [
    "CorrespondenceMaps",
    "LeafPathWitness",
    "LeafStateWitness",
    "build_correspondence_maps",
    "verify_tuple",
    "parse_continuous",
    "parse_continuous_topk",
    "leaf_exact_solve",
    "leaf_solve_state",
    "batch_check_c1c2",
    "parse_activation_batch",
    "_rank_child_catalog_states_for_parse",
    "parse_by_catalog_enum",
]
