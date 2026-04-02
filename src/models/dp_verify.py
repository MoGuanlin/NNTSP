# src/models/dp_verify.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .dp_correspondence import CorrespondenceMaps


def verify_tuple(
    *,
    parent_a: Tensor,
    parent_mate: Tensor,
    parent_iface_mask: Tensor,
    child_a: Tensor,
    child_mate: Tensor,
    child_iface_mask: Tensor,
    child_exists: Tensor,
    maps: CorrespondenceMaps,
) -> bool:
    """Check feasibility of a child-state 4-tuple w.r.t. parent state."""
    Ti = int(parent_a.shape[0])

    for p in range(Ti):
        if not parent_iface_mask[p].item():
            continue
        cq = int(maps.phi_out_child[p].item())
        cs = int(maps.phi_out_slot[p].item())
        if cq < 0 or cs < 0:
            if parent_a[p].item():
                return False
            continue
        if not child_exists[cq].item():
            if parent_a[p].item():
                return False
            continue
        if bool(parent_a[p].item()) != bool(child_a[cq, cs].item()):
            return False

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
            if qi > peer_q or (qi == peer_q and s > peer_s):
                continue
            if bool(child_a[qi, s].item()) != bool(child_a[peer_q, peer_s].item()):
                return False

    outer_parent_slots: List[int] = []
    outer_child_vertex: Dict[int, Tuple[int, int]] = {}
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

    def _neighbor_mate(q: int, s: int) -> Optional[Tuple[int, int]]:
        m = int(child_mate[q, s].item())
        if m < 0:
            return None
        return (q, m)

    def _neighbor_glue(q: int, s: int) -> Optional[Tuple[int, int]]:
        pq = int(maps.phi_glue_peer_child[q, s].item())
        ps = int(maps.phi_glue_peer_slot[q, s].item())
        if pq < 0 or ps < 0:
            return None
        return (pq, ps)

    induced_mate: Dict[int, int] = {}
    traced_starts: set = set()
    all_traced: set = set()

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
            return True

        for qi, s in all_active:
            m = int(child_mate[qi, s].item())
            if m < 0:
                return False
            # At the root there is no outer boundary, so every active child slot
            # must be consumed by a glue edge to another child. Otherwise the
            # tuple encodes an open path with exposed endpoints, not a closed tour.
            if _neighbor_glue(qi, s) is None:
                return False

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
            return False
        return True

    for p_start in outer_parent_slots:
        if p_start in traced_starts:
            continue
        cur = outer_child_vertex[p_start]
        visited: set = set()
        visited.add(cur)
        all_traced.add(cur)

        while True:
            nxt = _neighbor_mate(cur[0], cur[1])
            if nxt is None:
                return False

            if nxt in visited:
                return False

            visited.add(nxt)
            all_traced.add(nxt)

            nxt_is_outer = False
            p_end = -1
            for p_cand in outer_parent_slots:
                if outer_child_vertex[p_cand] == nxt:
                    nxt_is_outer = True
                    p_end = p_cand
                    break

            if nxt_is_outer:
                induced_mate[p_start] = p_end
                induced_mate[p_end] = p_start
                traced_starts.add(p_start)
                traced_starts.add(p_end)
                break

            glue_nxt = _neighbor_glue(nxt[0], nxt[1])
            if glue_nxt is None:
                return False

            if glue_nxt in visited:
                return False

            visited.add(glue_nxt)
            all_traced.add(glue_nxt)
            cur = glue_nxt

    for p in outer_parent_slots:
        expected_partner = int(parent_mate[p].item())
        if expected_partner < 0:
            return False
        actual_partner = induced_mate.get(p, -1)
        if actual_partner != expected_partner:
            return False

    # Every active child slot must be accounted for by the traced parent paths.
    # If an active slot is neither reached from an outer parent slot nor connected
    # through the glue graph, this tuple leaves an orphan endpoint/component that
    # direct traceback cannot place on the parent boundary.
    for qi in range(4):
        if not child_exists[qi].item():
            continue
        for s in range(Ti):
            if not child_a[qi, s].item():
                continue
            if not child_iface_mask[qi, s].item():
                continue
            if (qi, s) not in all_traced:
                return False

    return True


def batch_check_c1c2(
    *,
    parent_a_batch: Tensor,
    child_a_batch: Tensor,
    child_iface_mask: Tensor,
    child_exists: Tensor,
    maps: CorrespondenceMaps,
) -> Tensor:
    """Vectorized C1 + C2 feasibility check for K candidates at once."""
    K, Ti = parent_a_batch.shape
    device = parent_a_batch.device

    c1_ok = torch.ones(K, dtype=torch.bool, device=device)
    phi_c = maps.phi_out_child
    phi_s = maps.phi_out_slot
    has_map = phi_c >= 0

    if has_map.any():
        mapped_p = torch.nonzero(has_map, as_tuple=False).flatten()
        cq = phi_c[mapped_p]
        cs = phi_s[mapped_p]
        pa = parent_a_batch[:, mapped_p]
        ca = child_a_batch[:, cq, cs]
        c1_ok = (pa == ca).all(dim=1)

    c2_ok = torch.ones(K, dtype=torch.bool, device=device)
    phi_glue_c = maps.phi_glue_peer_child
    phi_glue_s = maps.phi_glue_peer_slot

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
        a_left = child_a_batch[:, sh_qi, sh_si]
        a_right = child_a_batch[:, sh_pq, sh_ps]
        c2_ok = (a_left == a_right).all(dim=1)

    return c1_ok & c2_ok


__all__ = ["batch_check_c1c2", "verify_tuple"]
