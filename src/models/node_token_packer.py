# src/models/node_token_packer.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor


# -----------------------------
# Dataclasses
# -----------------------------

@dataclass(frozen=True)
class PackedLeafPoints:
    """
    Leaf-level point sets, packed with a fixed cap max_points_per_leaf.

    Shapes:
      leaf_node_id: [L]                   (tree node id of each leaf)
      point_idx:    [L, P]                (pid; -1 padded)
      point_mask:   [L, P]                (bool)
      point_xy:     [L, P, 2]             (float32, leaf-cell-relative coords)
    """
    leaf_node_id: Tensor
    point_idx: Tensor
    point_mask: Tensor
    point_xy: Tensor


@dataclass(frozen=True)
class PackedNodeTokens:
    """
    Per-tree-node packed tokens (padded). Leading dim is num_tree_nodes M (or total_M for batch).

    Tree:
      tree_node_feat_rel:  [M,4]     root-normalized center-box [(cx-cx0)/s, (cy-cy0)/s, w/s, h/s]
      tree_node_depth:     [M]
      tree_children_index: [M,4]
      tree_parent_index:   [M]
      is_leaf:             [M] bool
      root_id:             [B] long   (batch-global node id per graph)
      root_scale_s:        [B] float

    Interface (Ti cap):
      iface_mask:            [M, Ti] bool
      iface_eid:             [M, Ti] long (batch-global eid after pack_batch offset)
      iface_feat6:           [M, Ti, 6] float
      iface_boundary_dir:    [M, Ti] long
      iface_inside_endpoint: [M, Ti] long
      iface_inside_quadrant: [M, Ti] long

    Crossing (Tc cap):
      cross_mask:             [M, Tc] bool
      cross_eid:              [M, Tc] long (batch-global eid after pack_batch offset)
      cross_feat6:            [M, Tc, 6] float
      cross_child_pair:       [M, Tc, 2] long
      cross_is_leaf_internal: [M, Tc] bool
    """
    tree_node_feat_rel: Tensor
    tree_node_depth: Tensor
    tree_children_index: Tensor
    tree_parent_index: Tensor
    is_leaf: Tensor
    root_id: Tensor
    root_scale_s: Tensor

    iface_mask: Tensor
    iface_eid: Tensor
    iface_feat6: Tensor
    iface_boundary_dir: Tensor
    iface_inside_endpoint: Tensor
    iface_inside_quadrant: Tensor

    cross_mask: Tensor
    cross_eid: Tensor
    cross_feat6: Tensor
    cross_child_pair: Tensor
    cross_is_leaf_internal: Tensor


@dataclass(frozen=True)
class PackedBatch:
    """
    Batch-level packed structure.

    node_ptr: [B+1] prefix sums of tree nodes
    leaf_ptr: [B+1] prefix sums of leaves
    edge_ptr: [B+1] prefix sums of spanner edges (for eid offset)

    graph_id_for_node: [total_M] mapping node -> graph id
    tokens: PackedNodeTokens
    leaves: PackedLeafPoints
    """
    node_ptr: Tensor
    leaf_ptr: Tensor
    edge_ptr: Tensor
    graph_id_for_node: Tensor
    tokens: PackedNodeTokens
    leaves: PackedLeafPoints


# -----------------------------
# Utilities
# -----------------------------

def _require(data: Any, name: str) -> Any:
    if not hasattr(data, name):
        raise KeyError(f"Data is missing required field: {name}")
    return getattr(data, name)


def _as_long(x: Tensor) -> Tensor:
    return x.to(dtype=torch.long)


def _as_float(x: Tensor) -> Tensor:
    return x.to(dtype=torch.float32)


def _safe_device(data: Any) -> torch.device:
    for k in ["pos", "tree_node_feat", "spanner_edge_index"]:
        if hasattr(data, k) and torch.is_tensor(getattr(data, k)):
            return getattr(data, k).device
    return torch.device("cpu")


def _require_contract_v2(data: Any) -> None:
    if not hasattr(data, "coord_contract_version"):
        raise ValueError("Missing coord_contract_version; regenerate dataset with Contract v2.")
    v_raw = getattr(data, "coord_contract_version")
    v = int(v_raw.item()) if torch.is_tensor(v_raw) else int(v_raw)
    if v != 2:
        raise ValueError(f"coord_contract_version={v} != 2; expected Contract v2 dataset.")

    if not hasattr(data, "tree_node_feat_box_mode"):
        raise ValueError("Missing tree_node_feat_box_mode; regenerate dataset with Contract v2.")
    m_raw = getattr(data, "tree_node_feat_box_mode")
    m = int(m_raw.item()) if torch.is_tensor(m_raw) else int(m_raw)
    if m != 0:
        raise ValueError(f"tree_node_feat_box_mode={m} != 0; expected 0 := llwh(abs) under Contract v2.")


def _to_center_box_from_ll(box_ll: Tensor) -> Tensor:
    # [x_ll,y_ll,w,h] -> [cx,cy,w,h]
    x0, y0, w, h = box_ll[:, 0], box_ll[:, 1], box_ll[:, 2], box_ll[:, 3]
    cx = x0 + 0.5 * w
    cy = y0 + 0.5 * h
    return torch.stack([cx, cy, w, h], dim=1)


def _node_rel_xy_from_center_box(xy_abs: Tensor, center_box: Tensor) -> Tensor:
    # rel=(x-cx)/(w/2),(y-cy)/(h/2)
    cx, cy, w, h = center_box[..., 0], center_box[..., 1], center_box[..., 2], center_box[..., 3]
    hw = (0.5 * w).clamp_min(1e-8)
    hh = (0.5 * h).clamp_min(1e-8)
    rel = torch.empty_like(xy_abs, dtype=torch.float32)
    rel[..., 0] = (xy_abs[..., 0] - cx) / hw
    rel[..., 1] = (xy_abs[..., 1] - cy) / hh
    return rel


def _find_root_id(tree_parent_index: Tensor) -> int:
    roots = torch.nonzero(tree_parent_index < 0, as_tuple=False).flatten()
    if roots.numel() != 1:
        raise ValueError(f"Expected exactly 1 root (parent < 0), found {int(roots.numel())}.")
    return int(roots[0].item())


def _group_pad_by_nid(
    nids: Tensor,
    payloads: Dict[str, Tensor],
    num_nodes: int,
    cap: int,
    pad_values: Dict[str, float | int | bool],
) -> Tuple[Dict[str, Tensor], Tensor]:
    """
    Group flat token lists by node id and pad/truncate to cap. Preserves within-node order
    given by the input (nids/payloads).
    """
    if cap <= 0:
        raise ValueError("cap must be positive.")

    nids = _as_long(nids)
    device = nids.device

    if nids.numel() == 0:
        out: Dict[str, Tensor] = {}
        mask = torch.zeros((num_nodes, cap), dtype=torch.bool, device=device)
        for k, v in payloads.items():
            shape_tail = v.shape[1:]
            fill = pad_values.get(k, 0)
            out[k] = torch.full((num_nodes, cap, *shape_tail), fill_value=fill, dtype=v.dtype, device=device)
        return out, mask

    T = int(nids.numel())

    # stable sort by nids; keeps within-node relative order
    perm = torch.argsort(nids, stable=True)
    nids_sorted = nids[perm]

    counts = torch.bincount(nids_sorted, minlength=num_nodes)
    starts = torch.cumsum(counts, dim=0) - counts
    idx_sorted = torch.arange(T, device=device, dtype=torch.long)
    rank = idx_sorted - starts[nids_sorted]

    keep = rank < cap
    nids_kept = nids_sorted[keep]
    rank_kept = rank[keep]
    perm_kept = perm[keep]

    mask = torch.zeros((num_nodes, cap), dtype=torch.bool, device=device)
    mask[nids_kept, rank_kept] = True

    out: Dict[str, Tensor] = {}
    for k, v in payloads.items():
        v_kept = v[perm_kept]
        shape_tail = v.shape[1:]
        fill = pad_values.get(k, 0)
        out_k = torch.full((num_nodes, cap, *shape_tail), fill_value=fill, dtype=v.dtype, device=device)
        out_k[nids_kept, rank_kept] = v_kept
        out[k] = out_k

    return out, mask


def _stable_sort_interfaces(
    *,
    iface_nid: Tensor,
    iface_eid: Tensor,
    iface_feat6: Tensor,
    iface_dir: Tensor,
    iface_inside_ep: Tensor,
    iface_inside_quad: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Deterministic ordering of interface tokens within each node:
      sort by (nid, boundary_dir, t_along_boundary, eid)

    t_along_boundary:
      - Left/Right (dir 0/1): use inter_rel_y = feat6[:,3]
      - Bottom/Top (dir 2/3): use inter_rel_x = feat6[:,2]
    Then quantize to int in [0,4096] for stability.
    """
    if iface_nid.numel() == 0:
        return iface_nid, iface_eid, iface_feat6, iface_dir, iface_inside_ep, iface_inside_quad

    device = iface_nid.device
    dir_safe = torch.where(
        (iface_dir >= 0) & (iface_dir <= 3),
        iface_dir,
        torch.full_like(iface_dir, 4),
    )

    inter_rel_x = iface_feat6[:, 2]
    inter_rel_y = iface_feat6[:, 3]
    is_lr = (dir_safe == 0) | (dir_safe == 1)
    t = torch.where(is_lr, inter_rel_y, inter_rel_x)
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)

    # quantize: (t in [-1,1]) -> [0,4096]
    t_q = torch.round((t + 1.0) * 2048.0).to(dtype=torch.long).clamp(0, 4096)

    perm = torch.arange(int(iface_nid.numel()), device=device, dtype=torch.long)

    # stable lexicographic sort: lowest key first
    perm = perm[torch.argsort(iface_eid[perm], stable=True)]
    perm = perm[torch.argsort(t_q[perm], stable=True)]
    perm = perm[torch.argsort(dir_safe[perm], stable=True)]
    perm = perm[torch.argsort(iface_nid[perm], stable=True)]

    return (
        iface_nid[perm],
        iface_eid[perm],
        iface_feat6[perm],
        iface_dir[perm],
        iface_inside_ep[perm],
        iface_inside_quad[perm],
    )


# -----------------------------
# Main packer
# -----------------------------

class NodeTokenPacker:
    def __init__(
        self,
        r: int | None = None,
        *,
        max_iface_per_node: int | None = None,
        max_cross_per_node: int | None = None,
        max_points_per_leaf: int = 20,
    ) -> None:
        self.r = None if r is None else int(r)
        self.max_iface_per_node = None if max_iface_per_node is None else int(max_iface_per_node)
        self.max_cross_per_node = None if max_cross_per_node is None else int(max_cross_per_node)
        self.max_points_per_leaf = int(max_points_per_leaf)

        if self.r is not None and self.r <= 0:
            raise ValueError("r must be positive.")
        if self.max_points_per_leaf <= 0:
            raise ValueError("max_points_per_leaf must be positive.")

    def _pack_single(self, data: Any) -> Tuple[PackedNodeTokens, PackedLeafPoints, int, int, int, int, float]:
        """
        Pack a single graph without any batch offset.
        Returns:
          tokens_single, leaves_single,
          M, L, E, root_id_local, root_scale_s(float)
        """
        device = _safe_device(data)
        _require_contract_v2(data)

        # Tree structure
        tree_node_feat_ll = _as_float(_require(data, "tree_node_feat")).to(device)  # [M,4] llwh(abs)
        tree_children_index = _as_long(_require(data, "tree_children_index")).to(device)  # [M,4]
        tree_parent_index = _as_long(_require(data, "tree_parent_index")).to(device)      # [M]
        tree_node_depth = _as_long(_require(data, "tree_node_depth")).to(device)          # [M]
        is_leaf = _require(data, "is_leaf")
        is_leaf = is_leaf.to(device=device, dtype=torch.bool) if torch.is_tensor(is_leaf) else torch.as_tensor(is_leaf, device=device, dtype=torch.bool)

        M = int(tree_node_feat_ll.shape[0])

        # Root-normalized node feature
        node_center_box = _to_center_box_from_ll(tree_node_feat_ll)  # [M,4] center-box abs
        root_id_local = _find_root_id(tree_parent_index)
        root_box = node_center_box[root_id_local]
        s = torch.maximum(root_box[2], root_box[3]).clamp_min(1e-8)
        root_scale_s = float(s.detach().item())

        tree_node_feat_rel = node_center_box.clone()
        tree_node_feat_rel[:, 0] = (node_center_box[:, 0] - root_box[0]) / s
        tree_node_feat_rel[:, 1] = (node_center_box[:, 1] - root_box[1]) / s
        tree_node_feat_rel[:, 2] = node_center_box[:, 2] / s
        tree_node_feat_rel[:, 3] = node_center_box[:, 3] / s

        # Spanner edges count (for later eid offset)
        spanner_edge_index = _as_long(_require(data, "spanner_edge_index")).to(device)
        if spanner_edge_index.dim() != 2 or spanner_edge_index.shape[0] != 2:
            raise ValueError("spanner_edge_index must be [2,E].")
        E = int(spanner_edge_index.shape[1])

        # ---- Interface tokens (flat) ----
        ia = _as_long(_require(data, "interface_assign_index")).to(device)  # [2,I]
        iface_nid = ia[0]
        iface_eid = ia[1]
        iface_feat6 = _as_float(_require(data, "interface_edge_attr")).to(device)  # [I,6]
        iface_dir = _as_long(_require(data, "interface_boundary_dir")).to(device)
        iface_inside_ep = _as_long(_require(data, "interface_inside_endpoint")).to(device)
        iface_inside_quad = _as_long(_require(data, "interface_inside_quadrant")).to(device)

        # stable ordering (node-local semantics)
        iface_nid, iface_eid, iface_feat6, iface_dir, iface_inside_ep, iface_inside_quad = _stable_sort_interfaces(
            iface_nid=iface_nid,
            iface_eid=iface_eid,
            iface_feat6=iface_feat6,
            iface_dir=iface_dir,
            iface_inside_ep=iface_inside_ep,
            iface_inside_quad=iface_inside_quad,
        )

        # caps
        if self.max_iface_per_node is not None:
            Ti = int(self.max_iface_per_node)
        elif self.r is not None:
            Ti = int(4 * self.r)
        else:
            # fallback: per-node max
            counts = torch.bincount(iface_nid, minlength=M) if iface_nid.numel() > 0 else torch.zeros((M,), device=device, dtype=torch.long)
            Ti = int(counts.max().item()) if counts.numel() > 0 else 1
            Ti = max(Ti, 1)

        iface_payloads = {
            "eid": iface_eid,
            "feat6": iface_feat6,
            "dir": iface_dir,
            "inside_ep": iface_inside_ep,
            "inside_quad": iface_inside_quad,
        }
        iface_pad = {"eid": -1, "feat6": 0.0, "dir": -1, "inside_ep": -1, "inside_quad": -1}
        iface_out, iface_mask = _group_pad_by_nid(iface_nid, iface_payloads, M, Ti, iface_pad)

        # ---- Crossing tokens (flat) ----
        ca = _as_long(_require(data, "crossing_assign_index")).to(device)  # [2,C]
        cross_nid = ca[0]
        cross_eid = ca[1]
        cross_feat6 = _as_float(_require(data, "crossing_edge_attr")).to(device)  # [C,6]
        cross_pair = _as_long(_require(data, "crossing_child_pair")).to(device)   # [C,2]
        cross_leaf_internal = _require(data, "crossing_is_leaf_internal")
        cross_leaf_internal = cross_leaf_internal.to(device=device, dtype=torch.bool) if torch.is_tensor(cross_leaf_internal) else torch.as_tensor(cross_leaf_internal, device=device, dtype=torch.bool)

        if self.max_cross_per_node is not None:
            Tc = int(self.max_cross_per_node)
        elif self.r is not None:
            Tc = int(8 * self.r)
        else:
            counts = torch.bincount(cross_nid, minlength=M) if cross_nid.numel() > 0 else torch.zeros((M,), device=device, dtype=torch.long)
            Tc = int(counts.max().item()) if counts.numel() > 0 else 1
            Tc = max(Tc, 1)

        cross_payloads = {
            "eid": cross_eid,
            "feat6": cross_feat6,
            "pair": cross_pair,
            "leaf_internal": cross_leaf_internal.to(dtype=torch.uint8),
        }
        cross_pad = {"eid": -1, "feat6": 0.0, "pair": -1, "leaf_internal": 0}
        cross_out, cross_mask = _group_pad_by_nid(cross_nid, cross_payloads, M, Tc, cross_pad)

        # ---- Leaf points (cell-relative) ----
        pos_abs = _as_float(_require(data, "pos")).to(device)  # [N,2]
        leaf_ids = _as_long(_require(data, "leaf_ids")).to(device)       # [L]
        leaf_ptr = _as_long(_require(data, "leaf_ptr")).to(device)       # [L+1]
        leaf_points = _as_long(_require(data, "leaf_points")).to(device) # [sum]
        L = int(leaf_ids.numel())
        P = int(self.max_points_per_leaf)

        if L == 0 or leaf_points.numel() == 0:
            point_idx = torch.full((L, P), -1, dtype=torch.long, device=device)
            point_mask = torch.zeros((L, P), dtype=torch.bool, device=device)
            point_xy = torch.zeros((L, P, 2), dtype=torch.float32, device=device)
        else:
            seg_len = (leaf_ptr[1:] - leaf_ptr[:-1]).clamp_min(0)
            total = int(leaf_points.numel())

            leaf_index_for_each = torch.repeat_interleave(torch.arange(L, device=device), seg_len)
            start_for_each = torch.repeat_interleave(leaf_ptr[:-1], seg_len)
            pos_in_leaf = torch.arange(total, device=device, dtype=torch.long) - start_for_each

            keep = pos_in_leaf < P
            leaf_kept = leaf_index_for_each[keep]
            pos_kept = pos_in_leaf[keep]
            pid_kept = leaf_points[keep]

            point_idx = torch.full((L, P), -1, dtype=torch.long, device=device)
            point_mask = torch.zeros((L, P), dtype=torch.bool, device=device)
            point_idx[leaf_kept, pos_kept] = pid_kept
            point_mask[leaf_kept, pos_kept] = True

            point_xy = torch.zeros((L, P, 2), dtype=torch.float32, device=device)
            leaf_nid_kept = leaf_ids[leaf_kept]
            leaf_box_kept = node_center_box[leaf_nid_kept]
            point_xy[leaf_kept, pos_kept] = _node_rel_xy_from_center_box(pos_abs[pid_kept], leaf_box_kept)

        leaves = PackedLeafPoints(
            leaf_node_id=leaf_ids,
            point_idx=point_idx,
            point_mask=point_mask,
            point_xy=point_xy,
        )

        # NOTE: root_id/root_scale_s are set by pack_batch (as [B]) after offset
        tokens = PackedNodeTokens(
            tree_node_feat_rel=tree_node_feat_rel,
            tree_node_depth=tree_node_depth,
            tree_children_index=tree_children_index,
            tree_parent_index=tree_parent_index,
            is_leaf=is_leaf,
            root_id=torch.empty((0,), dtype=torch.long, device=device),        # placeholder
            root_scale_s=torch.empty((0,), dtype=torch.float32, device=device),# placeholder

            iface_mask=iface_mask,
            iface_eid=iface_out["eid"],
            iface_feat6=iface_out["feat6"],
            iface_boundary_dir=iface_out["dir"],
            iface_inside_endpoint=iface_out["inside_ep"],
            iface_inside_quadrant=iface_out["inside_quad"],

            cross_mask=cross_mask,
            cross_eid=cross_out["eid"],
            cross_feat6=cross_out["feat6"],
            cross_child_pair=cross_out["pair"],
            cross_is_leaf_internal=cross_out["leaf_internal"].to(dtype=torch.bool),
        )

        return tokens, leaves, M, L, E, root_id_local, root_scale_s

    def pack_batch(self, datas: List[Any]) -> PackedBatch:
        """
        Unified entry point. For a single graph, call pack_batch([data]).
        """
        if len(datas) == 0:
            raise ValueError("pack_batch expects a non-empty list of Data.")

        # pack singles
        single_tokens: List[PackedNodeTokens] = []
        single_leaves: List[PackedLeafPoints] = []
        Ms: List[int] = []
        Ls: List[int] = []
        Es: List[int] = []
        roots_local: List[int] = []
        root_scales: List[float] = []
        devices: List[torch.device] = []

        for d in datas:
            t, l, M, L, E, rid, rs = self._pack_single(d)
            single_tokens.append(t)
            single_leaves.append(l)
            Ms.append(M)
            Ls.append(L)
            Es.append(E)
            roots_local.append(rid)
            root_scales.append(rs)
            devices.append(t.tree_node_feat_rel.device)

        device = devices[0]
        B = len(datas)

        # ptrs
        node_ptr = torch.zeros((B + 1,), dtype=torch.long, device=device)
        leaf_ptr = torch.zeros((B + 1,), dtype=torch.long, device=device)
        edge_ptr = torch.zeros((B + 1,), dtype=torch.long, device=device)
        node_ptr[1:] = torch.cumsum(torch.tensor(Ms, dtype=torch.long, device=device), dim=0)
        leaf_ptr[1:] = torch.cumsum(torch.tensor(Ls, dtype=torch.long, device=device), dim=0)
        edge_ptr[1:] = torch.cumsum(torch.tensor(Es, dtype=torch.long, device=device), dim=0)

        total_M = int(node_ptr[-1].item())
        total_L = int(leaf_ptr[-1].item())

        graph_id_for_node = torch.empty((total_M,), dtype=torch.long, device=device)
        for b in range(B):
            graph_id_for_node[node_ptr[b]: node_ptr[b + 1]] = b

        # concat + offset
        tree_node_feat_rel_all: List[Tensor] = []
        tree_node_depth_all: List[Tensor] = []
        tree_children_all: List[Tensor] = []
        tree_parent_all: List[Tensor] = []
        is_leaf_all: List[Tensor] = []

        iface_mask_all: List[Tensor] = []
        iface_eid_all: List[Tensor] = []
        iface_feat6_all: List[Tensor] = []
        iface_dir_all: List[Tensor] = []
        iface_ep_all: List[Tensor] = []
        iface_quad_all: List[Tensor] = []

        cross_mask_all: List[Tensor] = []
        cross_eid_all: List[Tensor] = []
        cross_feat6_all: List[Tensor] = []
        cross_pair_all: List[Tensor] = []
        cross_leafint_all: List[Tensor] = []

        leaf_node_id_all: List[Tensor] = []
        point_idx_all: List[Tensor] = []
        point_mask_all: List[Tensor] = []
        point_xy_all: List[Tensor] = []

        root_id_global = torch.empty((B,), dtype=torch.long, device=device)
        root_scale_s = torch.tensor(root_scales, dtype=torch.float32, device=device)

        for b in range(B):
            t = single_tokens[b]
            l = single_leaves[b]
            off_n = int(node_ptr[b].item())
            off_e = int(edge_ptr[b].item())

            # root id (global, per graph)
            root_id_global[b] = off_n + int(roots_local[b])

            tree_node_feat_rel_all.append(t.tree_node_feat_rel)
            tree_node_depth_all.append(t.tree_node_depth)
            is_leaf_all.append(t.is_leaf)

            ch = t.tree_children_index
            pa = t.tree_parent_index
            ch = torch.where(ch >= 0, ch + off_n, ch)
            pa = torch.where(pa >= 0, pa + off_n, pa)
            tree_children_all.append(ch)
            tree_parent_all.append(pa)

            iface_mask_all.append(t.iface_mask)
            iface_feat6_all.append(t.iface_feat6)
            iface_dir_all.append(t.iface_boundary_dir)
            iface_ep_all.append(t.iface_inside_endpoint)
            iface_quad_all.append(t.iface_inside_quadrant)
            iface_eid_all.append(torch.where(t.iface_eid >= 0, t.iface_eid + off_e, t.iface_eid))

            cross_mask_all.append(t.cross_mask)
            cross_feat6_all.append(t.cross_feat6)
            cross_pair_all.append(t.cross_child_pair)
            cross_leafint_all.append(t.cross_is_leaf_internal)
            cross_eid_all.append(torch.where(t.cross_eid >= 0, t.cross_eid + off_e, t.cross_eid))

            # leaves
            if int(l.leaf_node_id.numel()) > 0:
                leaf_node_id_all.append(torch.where(l.leaf_node_id >= 0, l.leaf_node_id + off_n, l.leaf_node_id))
                point_idx_all.append(l.point_idx)
                point_mask_all.append(l.point_mask)
                point_xy_all.append(l.point_xy)

        # concat leaves (may be empty)
        if total_L == 0:
            leaves = PackedLeafPoints(
                leaf_node_id=torch.empty((0,), dtype=torch.long, device=device),
                point_idx=torch.empty((0, self.max_points_per_leaf), dtype=torch.long, device=device),
                point_mask=torch.empty((0, self.max_points_per_leaf), dtype=torch.bool, device=device),
                point_xy=torch.empty((0, self.max_points_per_leaf, 2), dtype=torch.float32, device=device),
            )
        else:
            leaves = PackedLeafPoints(
                leaf_node_id=torch.cat(leaf_node_id_all, dim=0),
                point_idx=torch.cat(point_idx_all, dim=0),
                point_mask=torch.cat(point_mask_all, dim=0),
                point_xy=torch.cat(point_xy_all, dim=0),
            )

        tokens = PackedNodeTokens(
            tree_node_feat_rel=torch.cat(tree_node_feat_rel_all, dim=0),
            tree_node_depth=torch.cat(tree_node_depth_all, dim=0),
            tree_children_index=torch.cat(tree_children_all, dim=0),
            tree_parent_index=torch.cat(tree_parent_all, dim=0),
            is_leaf=torch.cat(is_leaf_all, dim=0),
            root_id=root_id_global,          # [B]
            root_scale_s=root_scale_s,       # [B]

            iface_mask=torch.cat(iface_mask_all, dim=0),
            iface_eid=torch.cat(iface_eid_all, dim=0),
            iface_feat6=torch.cat(iface_feat6_all, dim=0),
            iface_boundary_dir=torch.cat(iface_dir_all, dim=0),
            iface_inside_endpoint=torch.cat(iface_ep_all, dim=0),
            iface_inside_quadrant=torch.cat(iface_quad_all, dim=0),

            cross_mask=torch.cat(cross_mask_all, dim=0),
            cross_eid=torch.cat(cross_eid_all, dim=0),
            cross_feat6=torch.cat(cross_feat6_all, dim=0),
            cross_child_pair=torch.cat(cross_pair_all, dim=0),
            cross_is_leaf_internal=torch.cat(cross_leafint_all, dim=0),
        )

        return PackedBatch(
            node_ptr=node_ptr,
            leaf_ptr=leaf_ptr,
            edge_ptr=edge_ptr,
            graph_id_for_node=graph_id_for_node,
            tokens=tokens,
            leaves=leaves,
        )
