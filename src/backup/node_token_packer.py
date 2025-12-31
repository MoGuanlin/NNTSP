# src/models/node_token_packer.py
# -*- coding: utf-8 -*-
"""
NodeTokenPacker: pack a PyG Data (one TSP instance with quadtree + r-light tokens)
into padded, per-tree-node tensors with STRICT scale-invariant geometry features.

Key scale-invariance requirements:
- Neural I/O per node is O(r): token caps Ti,Tc,P are constants independent of N.
- Geometric features fed to the network must be relative/normalized (no absolute coords).

Critical fix:
Your dataset may store data.tree_node_feat either as
  (A) center-box: [cx, cy, w, h]
or
  (B) lower-left box: [x0, y0, w, h]
If we misinterpret (B) as (A), node-relative coordinates become ~[0,2] instead of [-1,1],
breaking coordinate-frame alignment.

This packer AUTO-DETECTS the box mode by sampling leaf points, then internally converts
all boxes to center-box representation for consistent downstream usage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor


# -----------------------------
# Dataclasses for packed outputs
# -----------------------------

@dataclass(frozen=True)
class PackedLeafPoints:
    """
    Leaf-level point sets, packed with a fixed cap max_points_per_leaf.

    Shapes:
      leaf_node_id: [L]
      point_idx:    [L, P]      (pid; -1 padded)
      point_mask:   [L, P]      (bool)
      point_xy:     [L, P, 2]   (float32, cell-relative coords: (x-cx)/(w/2),(y-cy)/(h/2))
    """
    leaf_node_id: Tensor
    point_idx: Tensor
    point_mask: Tensor
    point_xy: Tensor


@dataclass(frozen=True)
class PackedNodeTokens:
    """
    Per-tree-node packed tokens (padded). Leading dim is num_tree_nodes M.

    Tree:
      tree_node_feat_raw: [M,4] center-box ABSOLUTE [cx,cy,w,h] (debug only)
      tree_node_feat_rel: [M,4] root-normalized center-box [(cx-cx0)/s, (cy-cy0)/s, w/s, h/s]
      tree_node_depth:    [M]
      tree_children_index:[M,4]
      tree_parent_index:  [M]
      is_leaf:            [M] bool
      root_id:            scalar long
      root_scale_s:       scalar float  s=max(root_w,root_h)

    Interface:
      iface_mask:             [M, Ti]
      iface_eid:              [M, Ti]
      iface_feat6:            [M, Ti, 6]   (node-relative by construction)
      iface_boundary_dir:     [M, Ti]
      iface_inside_endpoint:  [M, Ti]
      iface_inter_rel_xy:     [M, Ti, 2]   (node-relative intersection; computed if absent)
      iface_inside_quadrant:  [M, Ti]

    Crossing:
      cross_mask:             [M, Tc]
      cross_eid:              [M, Tc]
      cross_feat6:            [M, Tc, 6]
      cross_child_pair:       [M, Tc, 2]
      cross_is_leaf_internal: [M, Tc]
    """
    tree_node_feat_raw: Tensor
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
    iface_inter_rel_xy: Tensor
    iface_inside_quadrant: Tensor

    cross_mask: Tensor
    cross_eid: Tensor
    cross_feat6: Tensor
    cross_child_pair: Tensor
    cross_is_leaf_internal: Tensor


@dataclass(frozen=True)
class PackedBatch:
    node_ptr: Tensor
    graph_id_for_node: Tensor
    tokens: PackedNodeTokens
    leaves: PackedLeafPoints
    leaf_ptr: Tensor


# -----------------------------
# Helpers
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


def _find_root_id(tree_parent_index: Tensor) -> int:
    root = torch.nonzero(tree_parent_index < 0, as_tuple=False)
    if root.numel() == 0:
        return 0
    return int(root[0, 0].item())


def _to_center_box_from_ll(box_ll: Tensor) -> Tensor:
    """
    Convert [x0,y0,w,h] -> [cx,cy,w,h]
    """
    x0, y0, w, h = box_ll[:, 0], box_ll[:, 1], box_ll[:, 2], box_ll[:, 3]
    cx = x0 + 0.5 * w
    cy = y0 + 0.5 * h
    return torch.stack([cx, cy, w, h], dim=1)


def _node_rel_xy_from_center_box(xy_abs: Tensor, center_box: Tensor) -> Tensor:
    """
    xy_abs: [...,2]
    center_box: [...,4] broadcastable, in center form [cx,cy,w,h]
    rel = (x-cx)/(w/2), (y-cy)/(h/2)
    """
    cx = center_box[..., 0]
    cy = center_box[..., 1]
    w = center_box[..., 2]
    h = center_box[..., 3]
    hw = (0.5 * w).clamp_min(1e-8)
    hh = (0.5 * h).clamp_min(1e-8)

    rel = torch.empty_like(xy_abs, dtype=torch.float32)
    rel[..., 0] = (xy_abs[..., 0] - cx) / hw
    rel[..., 1] = (xy_abs[..., 1] - cy) / hh
    return rel


def _inside_quadrant_from_center(inside_xy_abs: Tensor, center_box: Tensor) -> Tensor:
    cx = center_box[..., 0]
    cy = center_box[..., 1]
    right = inside_xy_abs[..., 0] >= cx
    top = inside_xy_abs[..., 1] >= cy
    q = torch.empty(inside_xy_abs.shape[:-1], dtype=torch.long, device=inside_xy_abs.device)
    q[(top & (~right))] = 0
    q[(top & right)] = 1
    q[((~top) & (~right))] = 2
    q[((~top) & right)] = 3
    return q


def _score_box_mode_by_leaf_points(
    tree_node_feat_in: Tensor,
    pos_abs: Tensor,
    leaf_ids: Tensor,
    leaf_ptr: Tensor,
    leaf_points: Tensor,
    mode: str,
    sample_cap_per_leaf: int = 4,
) -> float:
    """
    Score how well a box mode explains that leaf points are inside leaf cells.
    Returns fraction of sampled points whose node-relative coords lie within [-1.05,1.05] in both dims.
    """
    if leaf_ids.numel() == 0 or leaf_points.numel() == 0:
        return 0.0

    if mode not in ("center", "ll"):
        raise ValueError("mode must be 'center' or 'll'")

    # Build a center-box view under this mode
    if mode == "center":
        center_box = tree_node_feat_in
    else:
        center_box = _to_center_box_from_ll(tree_node_feat_in)

    # Sample: take first <= sample_cap_per_leaf points per leaf (vectorized, O(N) preprocessing is ok here)
    seg_len = (leaf_ptr[1:] - leaf_ptr[:-1]).clamp_min(0)
    L = int(leaf_ids.numel())
    if L == 0:
        return 0.0

    leaf_index_for_each = torch.repeat_interleave(torch.arange(L, device=leaf_ids.device), seg_len)
    start_for_each = torch.repeat_interleave(leaf_ptr[:-1], seg_len)
    pos_in_leaf = torch.arange(int(leaf_points.numel()), device=leaf_ids.device, dtype=torch.long) - start_for_each
    keep = pos_in_leaf < sample_cap_per_leaf
    if keep.sum().item() == 0:
        return 0.0

    leaf_idx = leaf_index_for_each[keep]
    pid = leaf_points[keep]
    nid = leaf_ids[leaf_idx]  # [K]
    box = center_box[nid]     # [K,4]
    rel = _node_rel_xy_from_center_box(pos_abs[pid], box)  # [K,2]

    inside = (rel.abs() <= 1.05).all(dim=-1)
    return float(inside.float().mean().item())


def _infer_and_convert_tree_boxes_to_center(
    tree_node_feat_in: Tensor,
    pos_abs: Tensor,
    leaf_ids: Tensor,
    leaf_ptr: Tensor,
    leaf_points: Tensor,
) -> Tuple[Tensor, str]:
    """
    Return (tree_node_feat_center, mode_chosen).
    """
    # If no leaf info, default to center (matches build_raw_pyramid).
    if leaf_ids.numel() == 0 or leaf_points.numel() == 0:
        return tree_node_feat_in, "center(default)"

    s_center = _score_box_mode_by_leaf_points(tree_node_feat_in, pos_abs, leaf_ids, leaf_ptr, leaf_points, mode="center")
    s_ll = _score_box_mode_by_leaf_points(tree_node_feat_in, pos_abs, leaf_ids, leaf_ptr, leaf_points, mode="ll")

    if s_ll > s_center + 1e-6:
        return _to_center_box_from_ll(tree_node_feat_in), f"ll(auto, score_ll={s_ll:.3f} > score_center={s_center:.3f})"
    else:
        return tree_node_feat_in, f"center(auto, score_center={s_center:.3f} >= score_ll={s_ll:.3f})"


def _compute_pos_norm_from_root(pos_abs: Tensor, root_center_box: Tensor) -> Tensor:
    """
    pos_norm = (pos - root_ll) / s, s=max(root_w,root_h), root_ll = (cx-w/2, cy-h/2)
    """
    cx0, cy0, w0, h0 = root_center_box[0], root_center_box[1], root_center_box[2], root_center_box[3]
    s = torch.maximum(w0, h0).clamp_min(1e-8)
    ll = torch.stack([cx0 - 0.5 * w0, cy0 - 0.5 * h0], dim=0)
    return (pos_abs - ll) / s


def _group_pad_by_nid(
    nids: Tensor,
    payloads: Dict[str, Tensor],
    num_nodes: int,
    cap: int,
    pad_values: Dict[str, float | int | bool],
) -> Tuple[Dict[str, Tensor], Tensor]:
    if cap <= 0:
        raise ValueError("cap must be positive.")

    if nids.numel() == 0:
        out: Dict[str, Tensor] = {}
        device = nids.device
        mask = torch.zeros((num_nodes, cap), dtype=torch.bool, device=device)
        for k, v in payloads.items():
            shape_tail = v.shape[1:]
            fill = pad_values.get(k, 0)
            out[k] = torch.full((num_nodes, cap, *shape_tail), fill_value=fill, dtype=v.dtype, device=device)
        return out, mask

    nids = _as_long(nids)
    device = nids.device
    T = nids.numel()

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


def _validate_r_light_interface(nids: Tensor, boundary_dir: Tensor, num_tree_nodes: int, r: Optional[int]) -> Dict[str, Tensor]:
    device = nids.device
    counts_bd = torch.zeros((num_tree_nodes, 4), dtype=torch.long, device=device)

    if nids.numel() > 0:
        nids = _as_long(nids)
        boundary_dir = _as_long(boundary_dir)
        ones = torch.ones_like(nids, dtype=torch.long)
        counts_bd.index_put_((nids, boundary_dir), ones, accumulate=True)

    if r is not None:
        if (counts_bd > r).any().item():
            bad = torch.nonzero(counts_bd > r, as_tuple=False)[:10].tolist()
            raise ValueError(
                f"r-light violated: found (nid,dir) buckets with count > r (r={r}). "
                f"Example offending buckets (up to 10): {bad}"
            )
    return {"counts_per_node_boundary": counts_bd}


# -----------------------------
# Main packer
# -----------------------------

class NodeTokenPacker:
    def __init__(
        self,
        r: Optional[int] = None,
        *,
        max_iface_per_node: Optional[int] = None,
        max_cross_per_node: Optional[int] = None,
        max_points_per_leaf: int = 20,
    ) -> None:
        if r is not None and r <= 0:
            raise ValueError("r must be positive when provided.")
        if max_points_per_leaf <= 0:
            raise ValueError("max_points_per_leaf must be positive.")
        self.r = r
        self.max_iface_per_node = max_iface_per_node
        self.max_cross_per_node = max_cross_per_node
        self.max_points_per_leaf = int(max_points_per_leaf)

    def pack_one(self, data: Any) -> Tuple[PackedNodeTokens, PackedLeafPoints, Dict[str, Tensor]]:
        device = _safe_device(data)

        # Raw input fields
        tree_node_feat_in = _as_float(_require(data, "tree_node_feat")).to(device)  # [M,4] but may be center or ll
        M = int(tree_node_feat_in.shape[0])

        tree_children_index = _as_long(_require(data, "tree_children_index")).to(device)  # [M,4]
        tree_parent_index = _as_long(_require(data, "tree_parent_index")).to(device)      # [M]
        tree_node_depth = _as_long(_require(data, "tree_node_depth")).to(device)          # [M]
        is_leaf = _require(data, "is_leaf")
        is_leaf = is_leaf.to(device=device, dtype=torch.bool) if torch.is_tensor(is_leaf) else torch.as_tensor(is_leaf, device=device, dtype=torch.bool)

        pos_abs = _as_float(_require(data, "pos")).to(device)  # [N,2]

        # Leaf CSR structures (used both for packing and for box-mode autodetect)
        leaf_ids = _as_long(_require(data, "leaf_ids")).to(device)            # [L]
        leaf_ptr = _as_long(_require(data, "leaf_ptr")).to(device)            # [L+1]
        leaf_points = _as_long(_require(data, "leaf_points")).to(device)      # [sum]

        # Infer & convert to center-box representation
        tree_node_feat_raw_center, box_mode = _infer_and_convert_tree_boxes_to_center(
            tree_node_feat_in, pos_abs, leaf_ids, leaf_ptr, leaf_points
        )

        # Root info
        root_id_int = _find_root_id(tree_parent_index)
        root_id = torch.tensor(root_id_int, dtype=torch.long, device=device)
        root_box = tree_node_feat_raw_center[root_id_int]  # [cx,cy,w,h]
        w0, h0 = root_box[2], root_box[3]
        s = torch.maximum(w0, h0).clamp_min(1e-8)
        root_scale_s = s.detach()

        # Root-normalized center-box
        tree_node_feat_rel = tree_node_feat_raw_center.clone()
        tree_node_feat_rel[:, 0] = (tree_node_feat_raw_center[:, 0] - root_box[0]) / s
        tree_node_feat_rel[:, 1] = (tree_node_feat_raw_center[:, 1] - root_box[1]) / s
        tree_node_feat_rel[:, 2] = tree_node_feat_raw_center[:, 2] / s
        tree_node_feat_rel[:, 3] = tree_node_feat_raw_center[:, 3] / s

        # pos_norm (optional; not used by NN modules, but keep for completeness)
        if hasattr(data, "pos_norm"):
            pos_norm = _as_float(getattr(data, "pos_norm")).to(device)
        else:
            pos_norm = _compute_pos_norm_from_root(pos_abs, root_box)

        # Spanner topology
        sp_edge_index = _as_long(_require(data, "spanner_edge_index")).to(device)  # [2,E]

        # --------------------
        # Interface tokens
        # --------------------
        ia = _as_long(_require(data, "interface_assign_index")).to(device)  # [2,I]
        iface_nid = ia[0]
        iface_eid = ia[1]

        iface_feat6 = _as_float(_require(data, "interface_edge_attr")).to(device)  # [I,6] (node-relative by dataset)
        iface_dir = _as_long(_require(data, "interface_boundary_dir")).to(device)  # [I]
        iface_inside_ep = _as_long(_require(data, "interface_inside_endpoint")).to(device)  # [I]

        stats = _validate_r_light_interface(iface_nid, iface_dir, M, self.r)

        # Intersection rel xy
        if hasattr(data, "interface_intersection_rel_xy"):
            iface_inter_rel = _as_float(getattr(data, "interface_intersection_rel_xy")).to(device)
            mode = getattr(data, "interface_intersection_rel_xy_mode", None)
            if mode in ("ll", "lower_left"):
                # convert [0,1] -> [-1,1] in center coordinates
                iface_inter_rel = iface_inter_rel * 2.0 - 1.0
        else:
            if hasattr(data, "interface_intersection_xy"):
                inter_abs = _as_float(getattr(data, "interface_intersection_xy")).to(device)
                node_box = tree_node_feat_raw_center[iface_nid]  # [I,4] center
                iface_inter_rel = _node_rel_xy_from_center_box(inter_abs, node_box)
            else:
                iface_inter_rel = torch.zeros((iface_feat6.shape[0], 2), dtype=torch.float32, device=device)

        # inside quadrant
        if hasattr(data, "interface_inside_quadrant"):
            iface_inside_quad = _as_long(getattr(data, "interface_inside_quadrant")).to(device)
        else:
            if iface_feat6.shape[0] > 0:
                u = sp_edge_index[0, iface_eid]  # [I]
                v = sp_edge_index[1, iface_eid]  # [I]
                inside_pid = torch.where(iface_inside_ep == 0, u, v)
                inside_xy_abs = pos_abs[inside_pid]
                node_box = tree_node_feat_raw_center[iface_nid]
                iface_inside_quad = _inside_quadrant_from_center(inside_xy_abs, node_box)
            else:
                iface_inside_quad = torch.empty((0,), dtype=torch.long, device=device)

        # Caps
        if self.max_iface_per_node is not None:
            Ti = int(self.max_iface_per_node)
        elif self.r is not None:
            Ti = int(4 * self.r)
        else:
            counts_iface = torch.bincount(_as_long(iface_nid), minlength=M)
            Ti = int(counts_iface.max().item()) if counts_iface.numel() > 0 else 1
            Ti = max(Ti, 1)

        iface_payloads = {
            "eid": iface_eid,
            "feat6": iface_feat6,
            "dir": iface_dir,
            "inside_ep": iface_inside_ep,
            "inter_rel": iface_inter_rel,
            "inside_quad": iface_inside_quad,
        }
        iface_pad = {
            "eid": -1,
            "feat6": 0.0,
            "dir": -1,
            "inside_ep": -1,
            "inter_rel": 0.0,
            "inside_quad": -1,
        }
        iface_out, iface_mask = _group_pad_by_nid(iface_nid, iface_payloads, M, Ti, iface_pad)

        # --------------------
        # Crossing tokens
        # --------------------
        ca = _as_long(_require(data, "crossing_assign_index")).to(device)  # [2,C]
        cross_nid = ca[0]
        cross_eid = ca[1]
        cross_feat6 = _as_float(_require(data, "crossing_edge_attr")).to(device)  # [C,6]

        if hasattr(data, "crossing_child_pair"):
            cross_pair = _as_long(getattr(data, "crossing_child_pair")).to(device)
        else:
            cross_pair = torch.full((cross_feat6.shape[0], 2), -1, dtype=torch.long, device=device)

        if hasattr(data, "crossing_is_leaf_internal"):
            x = getattr(data, "crossing_is_leaf_internal")
            cross_leaf_internal = x.to(device=device, dtype=torch.bool) if torch.is_tensor(x) else torch.as_tensor(x, device=device, dtype=torch.bool)
        else:
            cross_leaf_internal = torch.zeros((cross_feat6.shape[0],), dtype=torch.bool, device=device)

        if self.max_cross_per_node is not None:
            Tc = int(self.max_cross_per_node)
        elif self.r is not None:
            Tc = int(8 * self.r)
        else:
            counts_cross = torch.bincount(_as_long(cross_nid), minlength=M)
            Tc = int(counts_cross.max().item()) if counts_cross.numel() > 0 else 1
            Tc = max(Tc, 1)

        cross_payloads = {
            "eid": cross_eid,
            "feat6": cross_feat6,
            "pair": cross_pair,
            "leaf_internal": cross_leaf_internal.to(dtype=torch.uint8),
        }
        cross_pad = {"eid": -1, "feat6": 0.0, "pair": -1, "leaf_internal": 0}
        cross_out, cross_mask = _group_pad_by_nid(cross_nid, cross_payloads, M, Tc, cross_pad)
        cross_leaf_internal_padded = cross_out["leaf_internal"].to(dtype=torch.bool)

        # --------------------
        # Leaf points packing (CELL-RELATIVE)
        # --------------------
        L = int(leaf_ids.numel())
        P = int(self.max_points_per_leaf)

        seg_len = (leaf_ptr[1:] - leaf_ptr[:-1]).clamp_min(0)
        total = int(leaf_points.numel())

        if total == 0 or L == 0:
            point_idx = torch.full((L, P), -1, dtype=torch.long, device=device)
            point_mask = torch.zeros((L, P), dtype=torch.bool, device=device)
            point_xy = torch.zeros((L, P, 2), dtype=torch.float32, device=device)
        else:
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
            leaf_box_kept = tree_node_feat_raw_center[leaf_nid_kept]  # center-box
            point_xy[leaf_kept, pos_kept] = _node_rel_xy_from_center_box(pos_abs[pid_kept], leaf_box_kept)

        leaves = PackedLeafPoints(
            leaf_node_id=leaf_ids,
            point_idx=point_idx,
            point_mask=point_mask,
            point_xy=point_xy,
        )

        tokens = PackedNodeTokens(
            tree_node_feat_raw=tree_node_feat_raw_center,
            tree_node_feat_rel=tree_node_feat_rel,
            tree_node_depth=tree_node_depth,
            tree_children_index=tree_children_index,
            tree_parent_index=tree_parent_index,
            is_leaf=is_leaf,
            root_id=root_id,
            root_scale_s=root_scale_s,

            iface_mask=iface_mask,
            iface_eid=iface_out["eid"],
            iface_feat6=iface_out["feat6"],
            iface_boundary_dir=iface_out["dir"],
            iface_inside_endpoint=iface_out["inside_ep"],
            iface_inter_rel_xy=iface_out["inter_rel"],
            iface_inside_quadrant=iface_out["inside_quad"],

            cross_mask=cross_mask,
            cross_eid=cross_out["eid"],
            cross_feat6=cross_out["feat6"],
            cross_child_pair=cross_out["pair"],
            cross_is_leaf_internal=cross_leaf_internal_padded,
        )

        stats["num_tree_nodes"] = torch.tensor(M, device=device)
        stats["iface_cap"] = torch.tensor(Ti, device=device)
        stats["cross_cap"] = torch.tensor(Tc, device=device)
        stats["num_iface_tokens"] = torch.tensor(int(iface_feat6.shape[0]), device=device)
        stats["num_cross_tokens"] = torch.tensor(int(cross_feat6.shape[0]), device=device)
        stats["num_leaves"] = torch.tensor(int(L), device=device)
        stats["root_scale_s"] = root_scale_s.detach()
        stats["tree_box_mode"] = torch.tensor([0], device=device)  # placeholder numeric
        # store mode string in Python-side print (not tensor), for clarity:
        print(f"[packer] tree_node_feat box_mode = {box_mode}")

        return tokens, leaves, stats

    def pack_batch(self, datas: List[Any]) -> PackedBatch:
        if len(datas) == 0:
            raise ValueError("pack_batch expects a non-empty list of Data.")

        tokens_list: List[PackedNodeTokens] = []
        leaves_list: List[PackedLeafPoints] = []
        Ms: List[int] = []
        Ls: List[int] = []
        devices: List[torch.device] = []

        for d in datas:
            t, l, _ = self.pack_one(d)
            tokens_list.append(t)
            leaves_list.append(l)
            Ms.append(int(t.tree_node_feat_rel.shape[0]))
            Ls.append(int(l.leaf_node_id.shape[0]))
            devices.append(t.tree_node_feat_rel.device)

        device = devices[0]

        def _mv(x: Tensor) -> Tensor:
            return x.to(device)

        tokens_list = [
            PackedNodeTokens(**{k: _mv(getattr(t, k)) for k in t.__dataclass_fields__.keys()})
            for t in tokens_list
        ]
        leaves_list = [
            PackedLeafPoints(**{k: _mv(getattr(l, k)) for k in l.__dataclass_fields__.keys()})
            for l in leaves_list
        ]

        B = len(datas)
        node_ptr = torch.zeros((B + 1,), dtype=torch.long, device=device)
        leaf_ptr = torch.zeros((B + 1,), dtype=torch.long, device=device)
        node_ptr[1:] = torch.cumsum(torch.tensor(Ms, dtype=torch.long, device=device), dim=0)
        leaf_ptr[1:] = torch.cumsum(torch.tensor(Ls, dtype=torch.long, device=device), dim=0)
        total_M = int(node_ptr[-1].item())
        total_L = int(leaf_ptr[-1].item())

        graph_id_for_node = torch.empty((total_M,), dtype=torch.long, device=device)
        for b in range(B):
            graph_id_for_node[node_ptr[b]:node_ptr[b + 1]] = b

        tree_node_feat_raw = torch.cat([t.tree_node_feat_raw for t in tokens_list], dim=0)
        tree_node_feat_rel = torch.cat([t.tree_node_feat_rel for t in tokens_list], dim=0)
        tree_node_depth = torch.cat([t.tree_node_depth for t in tokens_list], dim=0)
        is_leaf = torch.cat([t.is_leaf for t in tokens_list], dim=0)

        children_all: List[Tensor] = []
        parent_all: List[Tensor] = []
        root_ids: List[Tensor] = []
        root_scales: List[Tensor] = []
        for b, t in enumerate(tokens_list):
            off = int(node_ptr[b].item())
            ch = t.tree_children_index.clone()
            pa = t.tree_parent_index.clone()
            ch = torch.where(ch >= 0, ch + off, ch)
            pa = torch.where(pa >= 0, pa + off, pa)
            children_all.append(ch)
            parent_all.append(pa)
            root_ids.append(t.root_id)
            root_scales.append(t.root_scale_s)
        tree_children_index = torch.cat(children_all, dim=0)
        tree_parent_index = torch.cat(parent_all, dim=0)

        def cat_node_field(name: str) -> Tensor:
            return torch.cat([getattr(t, name) for t in tokens_list], dim=0)

        tokens = PackedNodeTokens(
            tree_node_feat_raw=tree_node_feat_raw,
            tree_node_feat_rel=tree_node_feat_rel,
            tree_node_depth=tree_node_depth,
            tree_children_index=tree_children_index,
            tree_parent_index=tree_parent_index,
            is_leaf=is_leaf,
            root_id=torch.stack(root_ids, dim=0),
            root_scale_s=torch.stack(root_scales, dim=0),

            iface_mask=cat_node_field("iface_mask"),
            iface_eid=cat_node_field("iface_eid"),
            iface_feat6=cat_node_field("iface_feat6"),
            iface_boundary_dir=cat_node_field("iface_boundary_dir"),
            iface_inside_endpoint=cat_node_field("iface_inside_endpoint"),
            iface_inter_rel_xy=cat_node_field("iface_inter_rel_xy"),
            iface_inside_quadrant=cat_node_field("iface_inside_quadrant"),

            cross_mask=cat_node_field("cross_mask"),
            cross_eid=cat_node_field("cross_eid"),
            cross_feat6=cat_node_field("cross_feat6"),
            cross_child_pair=cat_node_field("cross_child_pair"),
            cross_is_leaf_internal=cat_node_field("cross_is_leaf_internal"),
        )

        if total_L == 0:
            leaves = PackedLeafPoints(
                leaf_node_id=torch.empty((0,), dtype=torch.long, device=device),
                point_idx=torch.empty((0, self.max_points_per_leaf), dtype=torch.long, device=device),
                point_mask=torch.empty((0, self.max_points_per_leaf), dtype=torch.bool, device=device),
                point_xy=torch.empty((0, self.max_points_per_leaf, 2), dtype=torch.float32, device=device),
            )
        else:
            leaf_node_id_all: List[Tensor] = []
            point_idx_all: List[Tensor] = []
            point_mask_all: List[Tensor] = []
            point_xy_all: List[Tensor] = []
            for b, l in enumerate(leaves_list):
                off = int(node_ptr[b].item())
                leaf_node_id_all.append(torch.where(l.leaf_node_id >= 0, l.leaf_node_id + off, l.leaf_node_id))
                point_idx_all.append(l.point_idx)
                point_mask_all.append(l.point_mask)
                point_xy_all.append(l.point_xy)
            leaves = PackedLeafPoints(
                leaf_node_id=torch.cat(leaf_node_id_all, dim=0),
                point_idx=torch.cat(point_idx_all, dim=0),
                point_mask=torch.cat(point_mask_all, dim=0),
                point_xy=torch.cat(point_xy_all, dim=0),
            )

        return PackedBatch(
            node_ptr=node_ptr,
            graph_id_for_node=graph_id_for_node,
            tokens=tokens,
            leaves=leaves,
            leaf_ptr=leaf_ptr,
        )
