# src/models/tour_reconstruct.py
# -*- coding: utf-8 -*-
"""
Tour reconstruction utilities for 1-pass DP results.

This module supports two reconstruction modes:

1. Native direct reconstruction:
   DP traceback -> leaf witnesses -> internal gluing -> explicit Hamiltonian tour

2. Legacy logit projection:
   DP traceback -> hard iface/cross logits -> edge decoder

The direct path is the primary 1-pass output. The logit projection remains as an
engineering baseline / compatibility path for greedy / exact / LKH post-decoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .bc_state_catalog import BoundaryStateCatalog
from .dp_core import (
    LeafStateWitness,
    build_correspondence_maps,
    leaf_solve_state,
)
from .dp_runner import CostTableEntry, OnePassDPResult
from .node_token_packer import PackedLeafPoints, PackedNodeTokens
from .shared_tree import build_leaf_row_for_node
from .tour_reconstruct_legacy import dp_result_to_edge_scores, dp_result_to_logits


class _ReconstructionError(RuntimeError):
    """Raised when traceback-selected states cannot be turned into a legal tour."""

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.details: Dict[str, Any] = dict(details) if details is not None else {}


@dataclass
class DirectTourResult:
    order: List[int]
    length: float
    feasible: bool
    num_points: int
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _OpenFragment:
    points: List[int]
    start_slot: int
    end_slot: int


@dataclass
class _NodeWitness:
    open_fragments: List[_OpenFragment] = field(default_factory=list)
    closed_cycles: List[List[int]] = field(default_factory=list)


@dataclass
class _TaggedFragment:
    points: List[int]
    start_tag: Any
    end_tag: Any


@dataclass
class _AssemblyState:
    fragments: Dict[int, _TaggedFragment] = field(default_factory=dict)
    endpoint_owner: Dict[Any, Tuple[int, str]] = field(default_factory=dict)
    closed_cycles: List[List[int]] = field(default_factory=list)
    next_id: int = 0

    def add_fragment(self, frag: _TaggedFragment) -> int:
        fid = self.next_id
        self.next_id += 1
        self.fragments[fid] = frag
        if frag.start_tag in self.endpoint_owner or frag.end_tag in self.endpoint_owner:
            raise _ReconstructionError("duplicate exposed boundary endpoint during gluing")
        self.endpoint_owner[frag.start_tag] = (fid, "start")
        self.endpoint_owner[frag.end_tag] = (fid, "end")
        return fid

    def _remove_fragment(self, fid: int) -> _TaggedFragment:
        frag = self.fragments.pop(fid)
        self.endpoint_owner.pop(frag.start_tag, None)
        self.endpoint_owner.pop(frag.end_tag, None)
        return frag

    def has_endpoint(self, tag: Any) -> bool:
        return tag in self.endpoint_owner

    def _orient_to_end(self, frag: _TaggedFragment, tag: Any) -> Tuple[List[int], Any]:
        if tag == frag.end_tag:
            return frag.points[:], frag.start_tag
        if tag == frag.start_tag:
            return list(reversed(frag.points)), frag.end_tag
        raise _ReconstructionError("endpoint does not belong to the selected fragment")

    def _orient_to_start(self, frag: _TaggedFragment, tag: Any) -> Tuple[List[int], Any]:
        if tag == frag.start_tag:
            return frag.points[:], frag.end_tag
        if tag == frag.end_tag:
            return list(reversed(frag.points)), frag.start_tag
        raise _ReconstructionError("endpoint does not belong to the selected fragment")

    def glue(self, tag_a: Any, tag_b: Any) -> None:
        owner_a = self.endpoint_owner.get(tag_a)
        owner_b = self.endpoint_owner.get(tag_b)
        if owner_a is None or owner_b is None:
            raise _ReconstructionError("glue endpoint missing during reconstruction")

        fid_a, _ = owner_a
        fid_b, _ = owner_b

        if fid_a == fid_b:
            frag = self._remove_fragment(fid_a)
            expected = {frag.start_tag, frag.end_tag}
            actual = {tag_a, tag_b}
            if expected != actual:
                raise _ReconstructionError("attempted to close a fragment through an interior endpoint")
            self.closed_cycles.append(frag.points[:])
            return

        frag_a = self._remove_fragment(fid_a)
        frag_b = self._remove_fragment(fid_b)
        seq_a, new_start = self._orient_to_end(frag_a, tag_a)
        seq_b, new_end = self._orient_to_start(frag_b, tag_b)
        self.add_fragment(
            _TaggedFragment(
                points=seq_a + seq_b,
                start_tag=new_start,
                end_tag=new_end,
            )
        )


def _to_cpu_tensor(x: Tensor) -> Tensor:
    return x.detach().cpu()


def _to_cpu_tokens(tokens: PackedNodeTokens) -> SimpleNamespace:
    return SimpleNamespace(
        tree_children_index=_to_cpu_tensor(tokens.tree_children_index),
        tree_node_feat_rel=_to_cpu_tensor(tokens.tree_node_feat_rel),
        is_leaf=_to_cpu_tensor(tokens.is_leaf),
        root_id=_to_cpu_tensor(tokens.root_id),
        root_scale_s=_to_cpu_tensor(tokens.root_scale_s),
        iface_mask=_to_cpu_tensor(tokens.iface_mask),
        iface_eid=_to_cpu_tensor(tokens.iface_eid),
        iface_feat6=_to_cpu_tensor(tokens.iface_feat6),
        iface_boundary_dir=_to_cpu_tensor(tokens.iface_boundary_dir),
        cross_mask=_to_cpu_tensor(tokens.cross_mask),
        cross_eid=_to_cpu_tensor(tokens.cross_eid),
        cross_child_pair=_to_cpu_tensor(tokens.cross_child_pair),
    )


def _to_cpu_leaves(leaves: PackedLeafPoints) -> SimpleNamespace:
    return SimpleNamespace(
        leaf_node_id=_to_cpu_tensor(leaves.leaf_node_id),
        point_idx=_to_cpu_tensor(leaves.point_idx),
        point_mask=_to_cpu_tensor(leaves.point_mask),
        point_xy=_to_cpu_tensor(leaves.point_xy),
    )


def _to_cpu_catalog(catalog: BoundaryStateCatalog) -> BoundaryStateCatalog:
    return BoundaryStateCatalog(
        used_iface=_to_cpu_tensor(catalog.used_iface),
        mate=_to_cpu_tensor(catalog.mate),
        num_used=_to_cpu_tensor(catalog.num_used),
        empty_index=int(catalog.empty_index),
        max_used=int(catalog.max_used),
    )

def _root_scale(tokens: PackedNodeTokens | SimpleNamespace) -> float:
    if hasattr(tokens, "root_scale_s") and torch.is_tensor(tokens.root_scale_s) and tokens.root_scale_s.numel() > 0:
        return float(tokens.root_scale_s[0].item())
    return 1.0


def _build_point_position_map(
    *,
    tokens: SimpleNamespace,
    leaves: SimpleNamespace,
    leaf_row_for_node: Tensor,
) -> Dict[int, Tuple[float, float]]:
    point_pos: Dict[int, Tuple[float, float]] = {}
    total_m = int(tokens.is_leaf.shape[0])
    for nid in range(total_m):
        if not tokens.is_leaf[nid].item():
            continue
        row = int(leaf_row_for_node[nid].item())
        if row < 0:
            continue
        point_mask = leaves.point_mask[row].bool()
        point_idx = leaves.point_idx[row][point_mask].long()
        point_xy = leaves.point_xy[row][point_mask]
        box_xy = tokens.tree_node_feat_rel[nid]
        cx = float(box_xy[0].item())
        cy = float(box_xy[1].item())
        hw = float(box_xy[2].item()) * 0.5
        hh = float(box_xy[3].item()) * 0.5
        for local_i in range(int(point_idx.numel())):
            pid = int(point_idx[local_i].item())
            if pid < 0:
                continue
            x = cx + float(point_xy[local_i, 0].item()) * hw
            y = cy + float(point_xy[local_i, 1].item()) * hh
            if pid in point_pos:
                raise _ReconstructionError(f"duplicate point id {pid} across leaf packs")
            point_pos[pid] = (x, y)
    return point_pos


def _canonicalize_cycle(order: List[int]) -> List[int]:
    if not order:
        return []
    anchor = 0 if 0 in order else min(order)

    def _rotate(seq: List[int], start: int) -> List[int]:
        idx = seq.index(start)
        return seq[idx:] + seq[:idx]

    cand_a = _rotate(order[:], anchor)
    cand_b = _rotate(list(reversed(order)), anchor)
    return cand_a if tuple(cand_a) <= tuple(cand_b) else cand_b


def _cycle_length(order: List[int], point_pos: Dict[int, Tuple[float, float]], root_scale: float) -> float:
    if len(order) <= 1:
        return 0.0
    total = 0.0
    n = len(order)
    for i in range(n):
        a = order[i]
        b = order[(i + 1) % n]
        ax, ay = point_pos[a]
        bx, by = point_pos[b]
        dx = ax - bx
        dy = ay - by
        total += (dx * dx + dy * dy) ** 0.5
    return total * root_scale


def _convert_leaf_witness_to_node_witness(
    *,
    witness: LeafStateWitness,
    point_ids: List[int],
) -> _NodeWitness:
    node_witness = _NodeWitness()
    for path in witness.open_paths:
        node_witness.open_fragments.append(
            _OpenFragment(
                points=[point_ids[i] for i in path.point_indices],
                start_slot=int(path.start_slot),
                end_slot=int(path.end_slot),
            )
        )
    if witness.closed_cycle:
        node_witness.closed_cycles.append([point_ids[i] for i in witness.closed_cycle])
    return node_witness


def _validate_nonroot_witness(
    *,
    nid: int,
    sigma_idx: int,
    witness: _NodeWitness,
    tokens: SimpleNamespace,
    catalog: BoundaryStateCatalog,
) -> None:
    if witness.closed_cycles:
        raise _ReconstructionError(
            f"node {nid} reconstructed an internal closed cycle",
            details={
                "reason": "internal_closed_cycle",
                "node_id": int(nid),
                "sigma_idx": int(sigma_idx),
                "node_depth": int(tokens.tree_node_depth[nid].item()),
            },
        )

    used = catalog.used_iface[sigma_idx]
    mate = catalog.mate[sigma_idx]
    iface_mask = tokens.iface_mask[nid].bool()

    expected_active = {
        s for s in range(int(iface_mask.shape[0]))
        if iface_mask[s].item() and used[s].item()
    }
    actual_counts: Dict[int, int] = {}
    for frag in witness.open_fragments:
        actual_counts[frag.start_slot] = actual_counts.get(frag.start_slot, 0) + 1
        actual_counts[frag.end_slot] = actual_counts.get(frag.end_slot, 0) + 1
        if int(mate[frag.start_slot].item()) != frag.end_slot:
            raise _ReconstructionError(
                f"node {nid} start-slot pairing does not match catalog state",
                details={
                    "reason": "start_slot_pairing_mismatch",
                    "node_id": int(nid),
                    "sigma_idx": int(sigma_idx),
                    "node_depth": int(tokens.tree_node_depth[nid].item()),
                    "fragment_start_slot": int(frag.start_slot),
                    "fragment_end_slot": int(frag.end_slot),
                    "catalog_mate": int(mate[frag.start_slot].item()),
                },
            )
        if int(mate[frag.end_slot].item()) != frag.start_slot:
            raise _ReconstructionError(
                f"node {nid} end-slot pairing does not match catalog state",
                details={
                    "reason": "end_slot_pairing_mismatch",
                    "node_id": int(nid),
                    "sigma_idx": int(sigma_idx),
                    "node_depth": int(tokens.tree_node_depth[nid].item()),
                    "fragment_start_slot": int(frag.start_slot),
                    "fragment_end_slot": int(frag.end_slot),
                    "catalog_mate": int(mate[frag.end_slot].item()),
                },
            )

    if set(actual_counts.keys()) != expected_active:
        raise _ReconstructionError(
            f"node {nid} exposed boundary slots do not match traceback state",
            details={
                "reason": "boundary_slots_do_not_match_traceback_state",
                "node_id": int(nid),
                "sigma_idx": int(sigma_idx),
                "node_depth": int(tokens.tree_node_depth[nid].item()),
                "expected_active_slots": sorted(int(s) for s in expected_active),
                "actual_slot_counts": {str(int(k)): int(v) for k, v in actual_counts.items()},
            },
        )
    for slot, count in actual_counts.items():
        if count != 1:
            raise _ReconstructionError(
                f"node {nid} boundary slot {slot} appears {count} times",
                details={
                    "reason": "boundary_slot_multiplicity",
                    "node_id": int(nid),
                    "sigma_idx": int(sigma_idx),
                    "node_depth": int(tokens.tree_node_depth[nid].item()),
                    "slot": int(slot),
                    "count": int(count),
                },
            )


def _reconstruct_leaf_node(
    *,
    nid: int,
    sigma_idx: int,
    tokens: SimpleNamespace,
    leaves: SimpleNamespace,
    leaf_row_for_node: Tensor,
    catalog: BoundaryStateCatalog,
    cost_tables: Dict[int, CostTableEntry],
    root_id: int,
) -> _NodeWitness:
    row = int(leaf_row_for_node[nid].item())
    if row < 0:
        raise _ReconstructionError(f"missing leaf row for node {nid}")

    point_mask = leaves.point_mask[row].bool()
    point_ids = leaves.point_idx[row][point_mask].long().tolist()

    cost, witness = leaf_solve_state(
        points_xy=leaves.point_xy[row],
        point_mask=leaves.point_mask[row],
        iface_mask=tokens.iface_mask[nid],
        iface_feat6=tokens.iface_feat6[nid],
        state_used=catalog.used_iface[sigma_idx],
        state_mate=catalog.mate[sigma_idx],
        box_xy=tokens.tree_node_feat_rel[nid],
        is_root=(nid == root_id),
    )
    if witness is None or cost == float("inf"):
        raise _ReconstructionError(f"selected leaf state {sigma_idx} at node {nid} is not reconstructible")

    ct = cost_tables.get(nid)
    if ct is None:
        raise _ReconstructionError(f"missing cost table for leaf node {nid}")
    dp_cost = float(ct.costs[sigma_idx].item())
    if abs(cost - dp_cost) > 1e-5:
        raise _ReconstructionError(
            f"leaf witness cost mismatch at node {nid}: witness={cost:.6f}, dp={dp_cost:.6f}"
        )

    node_witness = _convert_leaf_witness_to_node_witness(
        witness=witness,
        point_ids=point_ids,
    )
    if nid != root_id:
        _validate_nonroot_witness(
            nid=nid,
            sigma_idx=sigma_idx,
            witness=node_witness,
            tokens=tokens,
            catalog=catalog,
        )
    return node_witness


def _reconstruct_node_witness(
    *,
    nid: int,
    sigma_idx: int,
    tokens: SimpleNamespace,
    leaves: SimpleNamespace,
    leaf_row_for_node: Tensor,
    catalog: BoundaryStateCatalog,
    cost_tables: Dict[int, CostTableEntry],
    root_id: int,
) -> _NodeWitness:
    if tokens.is_leaf[nid].item():
        return _reconstruct_leaf_node(
            nid=nid,
            sigma_idx=sigma_idx,
            tokens=tokens,
            leaves=leaves,
            leaf_row_for_node=leaf_row_for_node,
            catalog=catalog,
            cost_tables=cost_tables,
            root_id=root_id,
        )

    ct = cost_tables.get(nid)
    if ct is None:
        raise _ReconstructionError(f"missing cost table for internal node {nid}")
    child_states = ct.backptr.get(sigma_idx)
    if child_states is None:
        raise _ReconstructionError(f"missing backpointer for node {nid}, state {sigma_idx}")

    children = tokens.tree_children_index[nid].long()
    child_exists = children >= 0
    ch_clamped = children.clamp_min(0)

    maps = build_correspondence_maps(
        parent_iface_eid=tokens.iface_eid[nid],
        parent_iface_mask=tokens.iface_mask[nid].bool(),
        parent_iface_bdir=tokens.iface_boundary_dir[nid],
        parent_cross_eid=tokens.cross_eid[nid],
        parent_cross_mask=tokens.cross_mask[nid].bool(),
        parent_cross_child_pair=tokens.cross_child_pair[nid],
        children_iface_eid=tokens.iface_eid[ch_clamped],
        children_iface_mask=tokens.iface_mask[ch_clamped].bool(),
        children_iface_bdir=tokens.iface_boundary_dir[ch_clamped],
        child_exists=child_exists,
    )

    assembly = _AssemblyState()
    for q in range(4):
        cid = int(children[q].item())
        c_sigma = int(child_states[q])
        if cid < 0 or c_sigma < 0:
            continue
        child_witness = _reconstruct_node_witness(
            nid=cid,
            sigma_idx=c_sigma,
            tokens=tokens,
            leaves=leaves,
            leaf_row_for_node=leaf_row_for_node,
            catalog=catalog,
            cost_tables=cost_tables,
            root_id=root_id,
        )
        if child_witness.closed_cycles:
            raise _ReconstructionError(f"child node {cid} produced a closed cycle before reaching the root")
        for frag in child_witness.open_fragments:
            assembly.add_fragment(
                _TaggedFragment(
                    points=frag.points[:],
                    start_tag=(q, frag.start_slot),
                    end_tag=(q, frag.end_slot),
                )
            )

    ti = int(tokens.iface_mask.shape[1])
    for q in range(4):
        if not child_exists[q].item():
            continue
        for s in range(ti):
            peer_q = int(maps.phi_glue_peer_child[q, s].item())
            peer_s = int(maps.phi_glue_peer_slot[q, s].item())
            if peer_q < 0 or peer_s < 0:
                continue
            if q > peer_q or (q == peer_q and s > peer_s):
                continue
            tag_a = (q, s)
            tag_b = (peer_q, peer_s)
            has_a = assembly.has_endpoint(tag_a)
            has_b = assembly.has_endpoint(tag_b)
            if has_a != has_b:
                raise _ReconstructionError(
                    f"node {nid} has unmatched glue endpoint between {(q, s)} and {(peer_q, peer_s)}",
                    details={
                        "reason": "unmatched_glue_endpoint",
                        "node_id": int(nid),
                        "sigma_idx": int(sigma_idx),
                        "node_depth": int(tokens.tree_node_depth[nid].item()),
                        "child_global_ids": [int(x) for x in children.tolist()],
                        "child_states": [int(x) for x in child_states],
                        "glue_tag_a": {"child": int(q), "slot": int(s), "present": bool(has_a)},
                        "glue_tag_b": {"child": int(peer_q), "slot": int(peer_s), "present": bool(has_b)},
                    },
                )
            if has_a and has_b:
                assembly.glue(tag_a, tag_b)

    child_to_parent: Dict[Tuple[int, int], int] = {}
    for p in range(ti):
        if not tokens.iface_mask[nid, p].bool().item():
            continue
        cq = int(maps.phi_out_child[p].item())
        cs = int(maps.phi_out_slot[p].item())
        if cq >= 0 and cs >= 0:
            child_to_parent[(cq, cs)] = p

    node_witness = _NodeWitness(closed_cycles=[c[:] for c in assembly.closed_cycles])
    for frag in assembly.fragments.values():
        start_key = frag.start_tag
        end_key = frag.end_tag
        if start_key not in child_to_parent or end_key not in child_to_parent:
            open_fragments = []
            for frag_idx, rem_frag in enumerate(assembly.fragments.values()):
                open_fragments.append(
                    {
                        "fragment_idx": int(frag_idx),
                        "num_points": int(len(rem_frag.points)),
                        "start_child": int(rem_frag.start_tag[0]),
                        "start_slot": int(rem_frag.start_tag[1]),
                        "end_child": int(rem_frag.end_tag[0]),
                        "end_slot": int(rem_frag.end_tag[1]),
                        "start_maps_to_parent": rem_frag.start_tag in child_to_parent,
                        "end_maps_to_parent": rem_frag.end_tag in child_to_parent,
                    }
                )
            raise _ReconstructionError(
                f"node {nid} has an open fragment that does not map to the parent boundary",
                details={
                    "reason": "open_fragment_not_mapped_to_parent_boundary",
                    "node_id": int(nid),
                    "sigma_idx": int(sigma_idx),
                    "node_depth": int(tokens.tree_node_depth[nid].item()),
                    "child_global_ids": [int(x) for x in children.tolist()],
                    "child_states": [int(x) for x in child_states],
                    "parent_boundary_slots": sorted(int(p) for p in child_to_parent.values()),
                    "child_to_parent": [
                        {
                            "child": int(cq),
                            "child_slot": int(cs),
                            "parent_slot": int(p),
                        }
                        for (cq, cs), p in sorted(child_to_parent.items())
                    ],
                    "open_fragments": open_fragments,
                    "offending_fragment": {
                        "start_child": int(start_key[0]),
                        "start_slot": int(start_key[1]),
                        "end_child": int(end_key[0]),
                        "end_slot": int(end_key[1]),
                    },
                },
            )
        node_witness.open_fragments.append(
            _OpenFragment(
                points=frag.points[:],
                start_slot=child_to_parent[start_key],
                end_slot=child_to_parent[end_key],
            )
        )

    if nid != root_id:
        _validate_nonroot_witness(
            nid=nid,
            sigma_idx=sigma_idx,
            witness=node_witness,
            tokens=tokens,
            catalog=catalog,
        )
    return node_witness


def reconstruct_tour_direct(
    *,
    result: OnePassDPResult,
    tokens: PackedNodeTokens,
    leaves: PackedLeafPoints,
    catalog: BoundaryStateCatalog,
) -> DirectTourResult:
    """Reconstruct an explicit tour directly from DP traceback witnesses."""
    if result.root_sigma < 0:
        return DirectTourResult(
            order=[],
            length=float("inf"),
            feasible=False,
            num_points=0,
            stats={"error": "infeasible_root"},
        )
    if not hasattr(leaves, "point_idx"):
        return DirectTourResult(
            order=[],
            length=float("inf"),
            feasible=False,
            num_points=0,
            stats={"error": "missing_point_idx"},
        )

    cpu_tokens = _to_cpu_tokens(tokens)
    cpu_leaves = _to_cpu_leaves(leaves)
    cpu_catalog = _to_cpu_catalog(catalog)

    root_id = int(cpu_tokens.root_id[0].item())
    total_m = int(cpu_tokens.is_leaf.shape[0])
    leaf_row_for_node = build_leaf_row_for_node(total_m, cpu_leaves.leaf_node_id)
    point_pos = _build_point_position_map(
        tokens=cpu_tokens,
        leaves=cpu_leaves,
        leaf_row_for_node=leaf_row_for_node,
    )
    num_points = len(point_pos)

    try:
        root_witness = _reconstruct_node_witness(
            nid=root_id,
            sigma_idx=int(result.root_sigma),
            tokens=cpu_tokens,
            leaves=cpu_leaves,
            leaf_row_for_node=leaf_row_for_node,
            catalog=cpu_catalog,
            cost_tables=result.cost_tables,
            root_id=root_id,
        )
        if root_witness.open_fragments:
            raise _ReconstructionError("root reconstruction left exposed endpoints")

        nonempty_cycles = [cyc for cyc in root_witness.closed_cycles if len(cyc) > 0]
        if num_points == 0:
            order: List[int] = []
        else:
            if len(nonempty_cycles) != 1:
                raise _ReconstructionError(
                    f"root reconstruction produced {len(nonempty_cycles)} non-empty cycles"
                )
            order = nonempty_cycles[0]
            if len(order) != num_points:
                raise _ReconstructionError(
                    f"reconstructed cycle length {len(order)} does not match num_points={num_points}"
                )
            if len(set(order)) != num_points:
                raise _ReconstructionError("reconstructed cycle repeats points")
            if set(order) != set(point_pos.keys()):
                raise _ReconstructionError("reconstructed cycle does not cover exactly the leaf points")
            order = _canonicalize_cycle(order)

        length = _cycle_length(order, point_pos, _root_scale(cpu_tokens))
        stats = {
            "num_cycles": float(len(root_witness.closed_cycles)),
            "num_points": float(num_points),
            "dp_vs_tour_abs_gap": abs(length - result.tour_cost) if result.tour_cost < float("inf") else float("inf"),
        }
        return DirectTourResult(
            order=order,
            length=length,
            feasible=True,
            num_points=num_points,
            stats=stats,
        )
    except _ReconstructionError as exc:
        stats: Dict[str, Any] = {"error": str(exc), "num_points": float(num_points)}
        if getattr(exc, "details", None):
            stats["error_details"] = exc.details
        return DirectTourResult(
            order=[],
            length=float("inf"),
            feasible=False,
            num_points=num_points,
            stats=stats,
        )

__all__ = [
    "DirectTourResult",
    "reconstruct_tour_direct",
    "dp_result_to_logits",
    "dp_result_to_edge_scores",
]
