# src/models/dp_correspondence.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import Tensor


TL, TR, BL, BR = 0, 1, 2, 3
LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 3

_SHARED_BOUNDARIES: List[Tuple[int, int, int, int]] = [
    (TL, TR, RIGHT, LEFT),
    (BL, BR, RIGHT, LEFT),
    (TL, BL, BOTTOM, TOP),
    (TR, BR, BOTTOM, TOP),
]

_SHARED_BOUNDARY_DIR_LOOKUP: Dict[Tuple[int, int], Tuple[int, int]] = {}
for _qi, _qj, _si, _sj in _SHARED_BOUNDARIES:
    _SHARED_BOUNDARY_DIR_LOOKUP[(_qi, _qj)] = (_si, _sj)
    _SHARED_BOUNDARY_DIR_LOOKUP[(_qj, _qi)] = (_sj, _si)


@dataclass
class CorrespondenceMaps:
    """Parent–child interface correspondence for one internal node."""

    phi_out_child: Tensor
    phi_out_slot: Tensor
    phi_glue_peer_child: Tensor
    phi_glue_peer_slot: Tensor
    phi_sh_peer_child: Tensor
    phi_sh_peer_slot: Tensor


def build_correspondence_maps(
    *,
    parent_iface_eid: Tensor,
    parent_iface_mask: Tensor,
    parent_iface_bdir: Tensor,
    parent_cross_eid: Tensor,
    parent_cross_mask: Tensor,
    parent_cross_child_pair: Tensor,
    children_iface_eid: Tensor,
    children_iface_mask: Tensor,
    children_iface_bdir: Tensor,
    child_exists: Tensor,
) -> CorrespondenceMaps:
    """Build φ^out, generic parent-crossing glue, and true shared-boundary maps."""
    Ti = int(parent_iface_eid.shape[0])
    device = parent_iface_eid.device

    phi_out_child = torch.full((Ti,), -1, dtype=torch.long, device=device)
    phi_out_slot = torch.full((Ti,), -1, dtype=torch.long, device=device)
    phi_glue_peer_child = torch.full((4, Ti), -1, dtype=torch.long, device=device)
    phi_glue_peer_slot = torch.full((4, Ti), -1, dtype=torch.long, device=device)
    phi_sh_peer_child = torch.full((4, Ti), -1, dtype=torch.long, device=device)
    phi_sh_peer_slot = torch.full((4, Ti), -1, dtype=torch.long, device=device)

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

    for p in range(Ti):
        if not parent_iface_mask[p].item():
            continue
        eid = int(parent_iface_eid[p].item())
        if eid < 0:
            continue
        bdir = int(parent_iface_bdir[p].item())
        for cq, cs in child_eid_map.get(eid, []):
            child_bdir = int(children_iface_bdir[cq, cs].item())
            if child_bdir == bdir:
                phi_out_child[p] = cq
                phi_out_slot[p] = cs
                break

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

        slot_i, slot_j = -1, -1
        for cq, cs in child_eid_map.get(eid, []):
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


def propagate_c1_constraints(
    *,
    parent_a: Tensor,
    parent_iface_mask: Tensor,
    maps: CorrespondenceMaps,
    child_exists: Tensor,
    child_iface_mask: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Propagate C1 (outer-boundary agreement) from parent σ to children."""
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


__all__ = ["CorrespondenceMaps", "build_correspondence_maps", "propagate_c1_constraints"]
