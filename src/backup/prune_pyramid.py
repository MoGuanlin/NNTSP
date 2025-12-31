# src/graph/prune_pyramid.py
import torch
import argparse
import os
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional

try:
    from torch_geometric.data import Data
except ImportError:
    class Data:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


# boundary_dir encoding:
# 0: Left, 1: Right, 2: Bottom, 3: Top
TL, TR, BL, BR = 0, 1, 2, 3


def _safe_torch_load(path: str):
    try:
        return torch.load(path)
    except Exception:
        return torch.load(path, weights_only=False)


def _line_segment_intersection_with_rect(
    x0: float, y0: float, x1: float, y1: float,
    rx: float, ry: float, rw: float, rh: float,
    boundary_dir: int
) -> Tuple[float, float]:
    """
    Compute intersection of segment (x0,y0)->(x1,y1) with the rectangle boundary specified by boundary_dir.
    Returns (xi, yi). Assumes (x0,y0) is inside the rect and (x1,y1) is outside (or vice versa),
    and segment intersects the boundary.
    boundary_dir: 0 left, 1 right, 2 bottom, 3 top
    """
    rx0, ry0, rx1, ry1 = rx, ry, rx + rw, ry + rh
    dx = x1 - x0
    dy = y1 - y0

    if boundary_dir == 0:  # left x=rx0
        t = (rx0 - x0) / (dx + 1e-12)
        yi = y0 + t * dy
        return rx0, yi
    if boundary_dir == 1:  # right x=rx1
        t = (rx1 - x0) / (dx + 1e-12)
        yi = y0 + t * dy
        return rx1, yi
    if boundary_dir == 2:  # bottom y=ry0
        t = (ry0 - y0) / (dy + 1e-12)
        xi = x0 + t * dx
        return xi, ry0
    # top y=ry1
    t = (ry1 - y0) / (dy + 1e-12)
    xi = x0 + t * dx
    return xi, ry1


def _inside_quadrant(cx: float, cy: float, px: float, py: float) -> int:
    """
    Determine which quadrant a point (px,py) lies in relative to (cx,cy).
    Quadrant order: TL=0, TR=1, BL=2, BR=3
    """
    if px < cx:
        return TL if py >= cy else BL
    return TR if py >= cy else BR


def prune_r_light_single(raw: Data, r: int, debug: bool = False) -> Data:
    """
    Enforce per-(node, boundary_dir) interface count <= r by removing edges (global kill) based on edge length.

    NOTE (theory caveat):
      This is an engineering approximation: we are pruning the *candidate edge set* rather than performing
      Rao'98-style patching to prove existence of an r-light tour. Use with care and consider ablations.
    """
    if r <= 0:
        raise ValueError(f"r must be positive, got {r}")

    # raw must have spanner + interface metadata
    if not hasattr(raw, "spanner_edge_attr") or not hasattr(raw, "interface_assign_index"):
        raise ValueError("raw data missing required fields: spanner_edge_attr / interface_assign_index")

    E = raw.spanner_edge_attr.size(0)
    edge_alive_mask = torch.ones((E,), dtype=torch.bool)

    ia = raw.interface_assign_index  # [2, I]
    I = ia.size(1)
    if I == 0:
        # trivial: no interface
        new_data = Data()
        new_data.num_nodes = raw.num_nodes
        new_data.num_points = int(getattr(raw, "num_points", raw.num_nodes))
        new_data.pos = raw.pos.to(torch.float32) if hasattr(raw, "pos") else raw.pos
        if hasattr(raw, "root_bbox"):
            new_data.root_bbox = raw.root_bbox
        if hasattr(raw, "pos_norm"):
            new_data.pos_norm = raw.pos_norm
        if hasattr(raw, "num_tree_nodes"):
            new_data.num_tree_nodes = int(raw.num_tree_nodes)

        new_data.spanner_edge_index = raw.spanner_edge_index
        new_data.spanner_edge_attr = raw.spanner_edge_attr
        new_data.edge_alive_mask = edge_alive_mask
        new_data.alive_edge_id = torch.arange(E, dtype=torch.long)
        new_data.alive_spanner_edge_index = raw.spanner_edge_index
        new_data.alive_spanner_edge_attr = raw.spanner_edge_attr

        # copy tree/leaf if present
        for name in [
            "tree_node_feat", "tree_node_depth", "is_leaf", "tree_edge_index",
            "tree_parent_index", "tree_children_index", "node_quadrant_in_parent",
            "leaf_ids", "leaf_ptr", "leaf_points", "point_to_leaf"
        ]:
            if hasattr(raw, name):
                setattr(new_data, name, getattr(raw, name))

        # carry over crossing/interface (empty)
        new_data.interface_assign_index = raw.interface_assign_index
        new_data.interface_node_index = torch.empty((0,), dtype=torch.long)
        new_data.interface_eid_index = torch.empty((0,), dtype=torch.long)
        new_data.interface_edge_attr = raw.interface_edge_attr
        new_data.interface_boundary_dir = raw.interface_boundary_dir
        new_data.interface_inside_endpoint = raw.interface_inside_endpoint

        new_data.crossing_assign_index = raw.crossing_assign_index
        new_data.crossing_node_index = torch.empty((0,), dtype=torch.long)
        new_data.crossing_eid_index = torch.empty((0,), dtype=torch.long)
        new_data.crossing_edge_attr = raw.crossing_edge_attr
        if hasattr(raw, "crossing_child_pair"):
            new_data.crossing_child_pair = raw.crossing_child_pair
        if hasattr(raw, "crossing_is_leaf_internal"):
            new_data.crossing_is_leaf_internal = raw.crossing_is_leaf_internal

        return new_data

    iface_node = ia[0]  # [I]
    iface_eid = ia[1]   # [I]
    iface_dir = raw.interface_boundary_dir  # [I]

    # group by (node, dir)
    buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    iface_node_cpu = iface_node.detach().cpu()
    iface_dir_cpu = iface_dir.detach().cpu()
    for j in range(I):
        buckets[(int(iface_node_cpu[j].item()), int(iface_dir_cpu[j].item()))].append(j)

    # For each bucket: keep shortest r edges, kill the rest (global kill by eid)
    kill_eids: Set[int] = set()
    edge_len = raw.spanner_edge_attr.view(-1).detach().cpu()
    iface_eid_cpu = iface_eid.detach().cpu()

    for _, idx_list in buckets.items():
        if len(idx_list) <= r:
            continue
        idx_list_sorted = sorted(idx_list, key=lambda j: float(edge_len[int(iface_eid_cpu[j].item())].item()))
        for j in idx_list_sorted[r:]:
            kill_eids.add(int(iface_eid_cpu[j].item()))

    if len(kill_eids) > 0:
        kill_tensor = torch.tensor(sorted(list(kill_eids)), dtype=torch.long)
        edge_alive_mask[kill_tensor] = False

    # 4) filter interface/crossing records by alive mask
    keep_iface = edge_alive_mask[iface_eid]  # [I] bool
    ca = raw.crossing_assign_index
    cross_eid = ca[1]  # [C]
    keep_cross = edge_alive_mask[cross_eid]  # [C] bool

    # 5) build new_data
    new_data = Data()
    # Point-space metadata
    new_data.num_nodes = raw.num_nodes  # legacy: number of TSP points
    new_data.num_points = int(getattr(raw, "num_points", raw.num_nodes))
    new_data.pos = raw.pos.to(torch.float32) if hasattr(raw, "pos") else raw.pos

    # Root-box normalization (if present in raw)
    if hasattr(raw, "root_bbox"):
        new_data.root_bbox = raw.root_bbox
    if hasattr(raw, "pos_norm"):
        new_data.pos_norm = raw.pos_norm

    # Tree-space metadata (may be used by DP/encoders)
    if hasattr(raw, "num_tree_nodes"):
        new_data.num_tree_nodes = int(raw.num_tree_nodes)

    # full spanner + mask
    new_data.spanner_edge_index = raw.spanner_edge_index
    new_data.spanner_edge_attr = raw.spanner_edge_attr
    new_data.edge_alive_mask = edge_alive_mask  # [E]

    # derived alive subset (convenience)
    alive_eids = torch.nonzero(edge_alive_mask, as_tuple=False).view(-1)
    new_data.alive_edge_id = alive_eids  # [E_alive]
    new_data.alive_spanner_edge_index = raw.spanner_edge_index[:, alive_eids]
    new_data.alive_spanner_edge_attr = raw.spanner_edge_attr[alive_eids]

    # copy tree/leaf structures if present
    for name in [
        "tree_node_feat", "tree_node_feat_box_mode", "tree_node_depth", "is_leaf", "tree_edge_index",
        "tree_parent_index", "tree_children_index", "node_quadrant_in_parent",
        "leaf_ids", "leaf_ptr", "leaf_points", "point_to_leaf"
    ]:
        if hasattr(raw, name):
            setattr(new_data, name, getattr(raw, name))

    # pruned interface core fields
    new_data.interface_assign_index = raw.interface_assign_index[:, keep_iface]
    new_data.interface_node_index = new_data.interface_assign_index[0].contiguous()
    new_data.interface_eid_index = new_data.interface_assign_index[1].contiguous()
    new_data.interface_edge_attr = raw.interface_edge_attr[keep_iface]
    new_data.interface_boundary_dir = raw.interface_boundary_dir[keep_iface]
    new_data.interface_inside_endpoint = raw.interface_inside_endpoint[keep_iface]

    # filtered crossing
    new_data.crossing_assign_index = raw.crossing_assign_index[:, keep_cross]
    new_data.crossing_node_index = new_data.crossing_assign_index[0].contiguous()
    new_data.crossing_eid_index = new_data.crossing_assign_index[1].contiguous()
    new_data.crossing_edge_attr = raw.crossing_edge_attr[keep_cross]
    if hasattr(raw, "crossing_child_pair"):
        new_data.crossing_child_pair = raw.crossing_child_pair[keep_cross]
    if hasattr(raw, "crossing_is_leaf_internal"):
        new_data.crossing_is_leaf_internal = raw.crossing_is_leaf_internal[keep_cross]

    # ---------------------------------------------------------
    # NEW: compute interface intersections & inside quadrants
    # ---------------------------------------------------------
    # Note: do it after pruning, so we compute only for kept interface records.
    pr_ia = new_data.interface_assign_index  # [2, I2]
    I2 = pr_ia.size(1)
    pr_nodes = pr_ia[0]  # [I2]
    pr_eids = pr_ia[1]   # [I2]
    pr_dir = new_data.interface_boundary_dir
    pr_inside_ep = new_data.interface_inside_endpoint

    # Allocate
    inter_xy = torch.empty((I2, 2), dtype=torch.float32)
    inter_rel = torch.empty((I2, 2), dtype=torch.float32)
    inside_quad = torch.empty((I2,), dtype=torch.long)

    # fetch endpoints
    u_all = new_data.spanner_edge_index[0]  # [E]
    v_all = new_data.spanner_edge_index[1]  # [E]
    pos = new_data.pos  # [N,2]

    # bring to cpu for geometry loop (I2 is O(r * M), acceptable)
    pr_nodes_cpu = pr_nodes.detach().cpu()
    pr_eids_cpu = pr_eids.detach().cpu()
    pr_dir_cpu = pr_dir.detach().cpu()
    pr_inside_cpu = pr_inside_ep.detach().cpu()

    # tree node bbox
    tree_feat = new_data.tree_node_feat.detach().cpu()  # [M,4]

    for k in range(I2):
        nid = int(pr_nodes_cpu[k].item())
        eid = int(pr_eids_cpu[k].item())
        d = int(pr_dir_cpu[k].item())
        inside_ep = int(pr_inside_cpu[k].item())

        u = int(u_all[eid].item())
        v = int(v_all[eid].item())
        pu = pos[u].tolist()
        pv = pos[v].tolist()

        # pick inside/outside endpoint according to inside_ep
        if inside_ep == 0:
            inside = pu
            outside = pv
        else:
            inside = pv
            outside = pu

        rx, ry, rw, rh = tree_feat[nid].tolist()
        xi, yi = _line_segment_intersection_with_rect(
            x0=float(inside[0]), y0=float(inside[1]),
            x1=float(outside[0]), y1=float(outside[1]),
            rx=float(rx), ry=float(ry), rw=float(rw), rh=float(rh),
            boundary_dir=d
        )
        inter_xy[k, 0] = xi
        inter_xy[k, 1] = yi
        # inter_rel[k, 0] = (xi - float(rx)) / (float(rw) + 1e-12)
        # inter_rel[k, 1] = (yi - float(ry)) / (float(rh) + 1e-12)
        
        # center-relative (scale-invariant, consistent with NodeTokenPacker/_node_rel_xy_from_center_box):
        cx = float(rx) + float(rw) / 2.0
        cy = float(ry) + float(rh) / 2.0
        hw = float(rw) / 2.0 + 1e-12
        hh = float(rh) / 2.0 + 1e-12
        inter_rel[k, 0] = (xi - cx) / hw
        inter_rel[k, 1] = (yi - cy) / hh


        # inside quadrant at this node (relative to node center)
        cx = float(rx) + float(rw) / 2.0
        cy = float(ry) + float(rh) / 2.0
        inside_quad[k] = _inside_quadrant(cx, cy, float(inside[0]), float(inside[1]))

    new_data.interface_intersection_xy = inter_xy
    new_data.interface_intersection_rel_xy = inter_rel
    
    new_data.interface_intersection_rel_xy_mode = "center"  # explicit contract

    new_data.interface_inside_quadrant = inside_quad

    # optional debug: verify per-node per-dir <= r among pruned interface records
    if debug:
        counts = defaultdict(int)
        for j in range(I2):
            nid = int(pr_nodes_cpu[j].item())
            d = int(pr_dir_cpu[j].item())
            counts[(nid, d)] += 1
        viol = [(k, v) for k, v in counts.items() if v > r]
        if len(viol) > 0:
            print("[WARN] r-light violation after pruning:", viol[:10], "...")
        else:
            print("[OK] r-light constraint verified.")

    return new_data


def prune_dataset(input_path: str, output_path: str, r: int, debug: bool = False) -> None:
    raw_list: List[Data] = _safe_torch_load(input_path)
    pruned_list: List[Data] = []

    print(f"Loaded raw pyramid list: {len(raw_list)} samples")
    for raw in tqdm(raw_list):
        pruned_list.append(prune_r_light_single(raw, r=r, debug=debug))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(pruned_list, output_path)
    print(f"Saved pruned dataset: {output_path}  (r={r})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input *_raw_pyramid.pt")
    parser.add_argument("--output", type=str, required=True, help="Output *_r_light_pyramid.pt")
    parser.add_argument("--r", type=int, default=5, help="Per-(node,boundary) interface cap")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    prune_dataset(args.input, args.output, r=args.r, debug=args.debug)
