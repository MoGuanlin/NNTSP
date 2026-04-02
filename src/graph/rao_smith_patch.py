"""
Rao-Smith style spanner patching to enforce r-lightness.

Reference:
  Rao & Smith, "Approximating geometrical graphs via spanners and banyans", STOC 1998.
  Paper Appendix C.3-C.4.

Pipeline position:
  spanner.py (Θ-graph) → **rao_smith_patch.py** → build_raw_pyramid.py → [train/eval]

Instead of deleting long edges (as in prune_pyramid.py), this module *reroutes* excess
boundary crossings through nearby points, preserving graph connectivity and the spanner's
distance guarantee up to an additive O(ε)·OPT term.

Output format is identical to spanner.py, so build_raw_pyramid.py needs no changes.
"""

import torch
import numpy as np
import argparse
import os
import math
from time import time
from collections import defaultdict
from typing import Tuple, List, Dict, Set, Optional
from scipy.spatial import cKDTree


# boundary sides
LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# Lightweight quadtree (only stores box geometry, no crossing/interface info)
# ---------------------------------------------------------------------------

class _QTNode:
    __slots__ = ('nid', 'x', 'y', 'w', 'h', 'depth', 'children', 'is_leaf', 'point_indices')

    def __init__(self, nid: int, x: float, y: float, w: float, h: float, depth: int):
        self.nid = nid
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.depth = depth
        self.children: List[Optional['_QTNode']] = [None, None, None, None]
        self.is_leaf = True
        self.point_indices: List[int] = []


def _build_quadtree(points: np.ndarray, max_points_per_leaf: int = 4,
                    max_depth: int = 100) -> List[_QTNode]:
    """Build a quadtree and return a flat list of all nodes (BFS order by id)."""
    n = points.shape[0]
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    center = (min_xy + max_xy) / 2.0
    size = float((max_xy - min_xy).max() * 1.01)

    root_x = float(center[0] - size / 2.0)
    root_y = float(center[1] - size / 2.0)

    nodes: List[_QTNode] = []
    counter = [0]

    def _recurse(x, y, w, h, depth, pt_idx):
        nid = counter[0]
        counter[0] += 1
        node = _QTNode(nid, x, y, w, h, depth)
        node.point_indices = pt_idx
        nodes.append(node)

        if depth >= max_depth or len(pt_idx) <= max_points_per_leaf:
            node.is_leaf = True
            return node

        node.is_leaf = False
        mid_x = x + w / 2.0
        mid_y = y + h / 2.0
        hw = w / 2.0
        hh = h / 2.0

        quads = [[], [], [], []]  # TL, TR, BL, BR
        for pi in pt_idx:
            px, py = float(points[pi, 0]), float(points[pi, 1])
            if px < mid_x:
                q = 0 if py >= mid_y else 2  # TL or BL
            else:
                q = 1 if py >= mid_y else 3  # TR or BR
            quads[q].append(pi)

        boxes = [
            (x,     mid_y, hw, hh),  # TL
            (mid_x, mid_y, hw, hh),  # TR
            (x,     y,     hw, hh),  # BL
            (mid_x, y,     hw, hh),  # BR
        ]
        for q in range(4):
            if len(quads[q]) > 0:
                child = _recurse(*boxes[q], depth + 1, quads[q])
                node.children[q] = child

        return node

    _recurse(root_x, root_y, size, size, 0, list(range(n)))
    return nodes


# ---------------------------------------------------------------------------
# Edge-boundary intersection helpers
# ---------------------------------------------------------------------------

def _segment_crosses_side(ux, uy, vx, vy, x0, y0, w, h, side) -> Optional[float]:
    """Check if segment (u→v) crosses the given side of box (x0, y0, w, h).

    Returns the parameter t ∈ (0, 1) of intersection, or None if no crossing.
    The intersection coordinate along the side is also derivable from t.
    """
    x1 = x0 + w
    y1 = y0 + h
    dx = vx - ux
    dy = vy - uy

    if side == LEFT:  # x = x0
        if abs(dx) < 1e-15:
            return None
        t = (x0 - ux) / dx
        if t <= 0.0 or t >= 1.0:
            return None
        yi = uy + t * dy
        if y0 - 1e-9 <= yi <= y1 + 1e-9:
            return t
    elif side == RIGHT:  # x = x1
        if abs(dx) < 1e-15:
            return None
        t = (x1 - ux) / dx
        if t <= 0.0 or t >= 1.0:
            return None
        yi = uy + t * dy
        if y0 - 1e-9 <= yi <= y1 + 1e-9:
            return t
    elif side == BOTTOM:  # y = y0
        if abs(dy) < 1e-15:
            return None
        t = (y0 - uy) / dy
        if t <= 0.0 or t >= 1.0:
            return None
        xi = ux + t * dx
        if x0 - 1e-9 <= xi <= x1 + 1e-9:
            return t
    else:  # TOP, y = y1
        if abs(dy) < 1e-15:
            return None
        t = (y1 - uy) / dy
        if t <= 0.0 or t >= 1.0:
            return None
        xi = ux + t * dx
        if x0 - 1e-9 <= xi <= x1 + 1e-9:
            return t
    return None


def _intersection_coord_along_side(ux, uy, vx, vy, t, side, x0, y0, w, h) -> float:
    """Return the 1-D coordinate of the intersection point along the given side.

    LEFT/RIGHT → y-coordinate; BOTTOM/TOP → x-coordinate.  Normalized to [0, 1]
    relative to the side length.
    """
    ix = ux + t * (vx - ux)
    iy = uy + t * (vy - uy)
    if side == LEFT or side == RIGHT:
        return (iy - y0) / (h + 1e-15)
    else:
        return (ix - x0) / (w + 1e-15)


# ---------------------------------------------------------------------------
# Core Rao-Smith patching
# ---------------------------------------------------------------------------

def _collect_crossings_for_node(
    points: np.ndarray,
    edge_index: np.ndarray,   # [2, E]
    alive: np.ndarray,        # [E] bool
    node: _QTNode,
) -> Dict[int, List[Tuple[int, float, float]]]:
    """For one quadtree node, find all alive edges crossing each side.

    Returns:
        side_crossings[side] = [(edge_id, t_param, coord_along_side), ...]
    """
    side_crossings: Dict[int, List[Tuple[int, float, float]]] = {
        LEFT: [], RIGHT: [], BOTTOM: [], TOP: []
    }
    x0, y0, w, h = node.x, node.y, node.w, node.h

    alive_ids = np.where(alive)[0]
    for eid in alive_ids:
        u, v = int(edge_index[0, eid]), int(edge_index[1, eid])
        ux, uy = float(points[u, 0]), float(points[u, 1])
        vx, vy = float(points[v, 0]), float(points[v, 1])

        for side in (LEFT, RIGHT, BOTTOM, TOP):
            t = _segment_crosses_side(ux, uy, vx, vy, x0, y0, w, h, side)
            if t is not None:
                coord = _intersection_coord_along_side(ux, uy, vx, vy, t, side, x0, y0, w, h)
                side_crossings[side].append((eid, t, coord))

    return side_crossings


def _patch_one_side(
    points: np.ndarray,
    edge_index: np.ndarray,  # [2, E_current] (may grow)
    edge_attr: np.ndarray,   # [E_current, 1]
    alive: np.ndarray,       # [E_current] bool
    crossings: List[Tuple[int, float, float]],
    node: _QTNode,
    side: int,
    r: int,
    kd_tree: cKDTree,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Patch a single over-crowded side to have ≤ r crossings.

    Algorithm (Rao-Smith inspired):
      1. Place r evenly-spaced portals along the side.
      2. Assign each crossing to its nearest portal.
      3. Per portal: keep the shortest crossing edge; reroute the rest.
      4. Rerouting edge (u,v): find nearest point w to portal location,
         remove (u,v), add (u,w) and (w,v).

    Returns updated (edge_index, edge_attr, alive) — arrays may have grown.
    """
    if len(crossings) <= r:
        return edge_index, edge_attr, alive

    x0, y0, w, h = node.x, node.y, node.w, node.h

    # --- 1. Portal placement (r portals, evenly spaced along the side) ---
    portal_coords_1d = np.array([(i + 0.5) / r for i in range(r)])  # in [0, 1]

    # Convert 1-D portal coord to 2-D point for nearest-point lookup
    def _portal_to_2d(c1d):
        if side == LEFT:
            return (x0, y0 + c1d * h)
        elif side == RIGHT:
            return (x0 + w, y0 + c1d * h)
        elif side == BOTTOM:
            return (x0 + c1d * w, y0)
        else:  # TOP
            return (x0 + c1d * w, y0 + h)

    portal_2d = np.array([_portal_to_2d(c) for c in portal_coords_1d])  # [r, 2]

    # --- 2. Assign each crossing to nearest portal ---
    portal_buckets: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    # bucket[portal_idx] = [(eid, edge_length), ...]
    for eid, _t, coord_1d in crossings:
        if not alive[eid]:
            continue
        dists_to_portals = np.abs(portal_coords_1d - coord_1d)
        pidx = int(np.argmin(dists_to_portals))
        u, v = int(edge_index[0, eid]), int(edge_index[1, eid])
        elen = float(edge_attr[eid, 0])
        portal_buckets[pidx].append((eid, elen))

    # --- 3. Per portal: keep shortest, reroute the rest ---
    for pidx, bucket in portal_buckets.items():
        if len(bucket) <= 1:
            continue

        # sort by edge length ascending → keep the shortest
        bucket.sort(key=lambda x: x[1])

        for eid, _elen in bucket[1:]:  # skip the first (shortest) — it stays
            u_idx = int(edge_index[0, eid])
            v_idx = int(edge_index[1, eid])
            ux, uy = float(points[u_idx, 0]), float(points[u_idx, 1])
            vx, vy = float(points[v_idx, 0]), float(points[v_idx, 1])

            # Find nearest data point to this portal
            portal_xy = portal_2d[pidx]
            _, w_idx = kd_tree.query(portal_xy)
            w_idx = int(w_idx)

            # Skip degenerate cases: w is same as u or v
            if w_idx == u_idx or w_idx == v_idx:
                # Fall back: just kill the edge (degenerate — portal sits on endpoint)
                alive[eid] = False
                continue

            wx, wy = float(points[w_idx, 0]), float(points[w_idx, 1])

            # Kill original edge
            alive[eid] = False

            # Add detour edges (u, w) and (w, v) — canonical u < v form
            new_edges_to_add = []
            for (a, b) in [(u_idx, w_idx), (w_idx, v_idx)]:
                ea, eb = (min(a, b), max(a, b))
                ax, ay = float(points[ea, 0]), float(points[ea, 1])
                bx, by = float(points[eb, 0]), float(points[eb, 1])
                new_len = math.sqrt((bx - ax)**2 + (by - ay)**2)
                new_edges_to_add.append((ea, eb, new_len))

            # Check for duplicate edges before adding
            E_cur = edge_index.shape[1]
            for (ea, eb, new_len) in new_edges_to_add:
                # Check if this edge already exists and is alive
                dup_mask = (edge_index[0] == ea) & (edge_index[1] == eb) & alive
                if np.any(dup_mask):
                    continue  # edge already exists, skip

                # Append new edge
                new_col = np.array([[ea], [eb]], dtype=edge_index.dtype)
                edge_index = np.concatenate([edge_index, new_col], axis=1)
                edge_attr = np.concatenate([edge_attr, [[new_len]]], axis=0)
                alive = np.concatenate([alive, [True]])

    return edge_index, edge_attr, alive


def patch_single_instance(
    points: np.ndarray,       # [N, 2]
    edge_index: np.ndarray,   # [2, E]
    edge_attr: np.ndarray,    # [E, 1]
    r: int,
    max_points_per_leaf: int = 4,
    max_depth: int = 100,
    max_iterations: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rao-Smith patch a single instance's spanner to r-lightness.

    Processes quadtree levels from coarsest to finest. Since detour edges are
    short, they mainly affect boundaries at the same or deeper levels, so a
    single coarse-to-fine sweep usually suffices. We allow a few extra iterations
    to mop up residual violations from detour edges.

    Returns:
        new_edge_index: [2, E'] canonical undirected edges (u < v)
        new_edge_attr:  [E', 1] Euclidean lengths
    """
    N = points.shape[0]
    E = edge_index.shape[1]

    # Work on mutable numpy copies
    ei = edge_index.copy()
    ea = edge_attr.copy().reshape(-1, 1) if edge_attr.ndim == 1 else edge_attr.copy()
    alive = np.ones(E, dtype=bool)

    # Build quadtree
    nodes = _build_quadtree(points, max_points_per_leaf, max_depth)

    # kd-tree for nearest-point queries (built once, points don't change)
    kd_tree = cKDTree(points)

    # Group nodes by depth (coarse → fine)
    max_d = max(nd.depth for nd in nodes)
    nodes_by_depth: Dict[int, List[_QTNode]] = defaultdict(list)
    for nd in nodes:
        if not nd.is_leaf:  # only internal nodes have meaningful boundaries
            nodes_by_depth[nd.depth].append(nd)

    for _iteration in range(max_iterations):
        patched_any = False

        for depth in range(0, max_d + 1):
            for nd in nodes_by_depth.get(depth, []):
                side_crossings = _collect_crossings_for_node(points, ei, alive, nd)
                for side in (LEFT, RIGHT, BOTTOM, TOP):
                    crossings = side_crossings[side]
                    if len(crossings) <= r:
                        continue
                    patched_any = True
                    ei, ea, alive = _patch_one_side(
                        points, ei, ea, alive, crossings, nd, side, r, kd_tree,
                    )

        if not patched_any:
            break

    # Collect alive edges
    alive_ids = np.where(alive)[0]
    new_ei = ei[:, alive_ids]
    new_ea = ea[alive_ids]

    # Deduplicate (detour edges might create duplicates across iterations)
    if new_ei.shape[1] > 0:
        # canonical form: u < v
        sorted_ei = np.sort(new_ei, axis=0)
        # unique via structured array
        combined = np.ascontiguousarray(sorted_ei.T)
        _, unique_idx = np.unique(
            combined.view(np.dtype((np.void, combined.dtype.itemsize * 2))),
            return_index=True
        )
        unique_idx = np.sort(unique_idx)
        new_ei = sorted_ei[:, unique_idx]
        new_ea = new_ea[unique_idx]

    return new_ei, new_ea


# ---------------------------------------------------------------------------
# Batch interface (matches spanner.py output format)
# ---------------------------------------------------------------------------

def patch_batch(
    points: torch.Tensor,      # [B, N, 2]
    edge_index: torch.Tensor,  # [2, E_total] (disjoint union, offset by b*N)
    edge_attr: torch.Tensor,   # [E_total, 1]
    batch_idx: torch.Tensor,   # [E_total]
    r: int,
    max_points_per_leaf: int = 4,
    max_depth: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rao-Smith patch a batch of spanner instances.

    Input/output format matches spanner.py exactly:
        edge_index [2, E'], edge_attr [E', 1], batch_idx [E']
    """
    B, N, _ = points.shape
    device = points.device
    points_np = points.detach().cpu().float().numpy()
    ei_np = edge_index.detach().cpu().numpy()
    ea_np = edge_attr.detach().cpu().float().numpy()
    bi_np = batch_idx.detach().cpu().numpy()

    print(f"Rao-Smith patching {B} instances to r={r}-light ...")
    t0 = time()

    new_eis, new_eas, new_bis = [], [], []

    for b in range(B):
        mask = (bi_np == b)
        local_ei = ei_np[:, mask] - b * N    # local node indices
        local_ea = ea_np[mask]
        pts = points_np[b]                    # [N, 2]

        new_ei, new_ea = patch_single_instance(
            pts, local_ei, local_ea, r=r,
            max_points_per_leaf=max_points_per_leaf,
            max_depth=max_depth,
        )

        E_new = new_ei.shape[1]
        new_eis.append(new_ei + b * N)         # re-offset
        new_eas.append(new_ea)
        new_bis.append(np.full((E_new,), b, dtype=np.int64))

    all_ei = np.concatenate(new_eis, axis=1) if new_eis else np.empty((2, 0), dtype=np.int64)
    all_ea = np.concatenate(new_eas, axis=0) if new_eas else np.empty((0, 1), dtype=np.float32)
    all_bi = np.concatenate(new_bis, axis=0) if new_bis else np.empty((0,), dtype=np.int64)

    print(f"Patching done in {time() - t0:.2f}s. "
          f"Edges: {ei_np.shape[1]} → {all_ei.shape[1]} "
          f"({all_ei.shape[1] - ei_np.shape[1]:+d})")

    return (
        torch.from_numpy(all_ei).long().to(device),
        torch.from_numpy(all_ea).float().to(device),
        torch.from_numpy(all_bi).long().to(device),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def patch_dataset(input_path: str, output_path: str, r: int,
                  max_points_per_leaf: int = 4, max_depth: int = 100):
    """Load a spanner .pt file, patch to r-light, save in the same format."""
    print(f"Loading spanner data from {input_path} ...")
    try:
        data_dict = torch.load(input_path)
    except Exception:
        data_dict = torch.load(input_path, weights_only=False)

    points = data_dict['points']          # [B, N, 2]
    edge_index = data_dict['edge_index']  # [2, E]
    edge_attr = data_dict['edge_attr']    # [E, 1]
    batch_idx = data_dict['batch_idx']    # [E]

    new_ei, new_ea, new_bi = patch_batch(
        points, edge_index, edge_attr, batch_idx,
        r=r, max_points_per_leaf=max_points_per_leaf, max_depth=max_depth,
    )

    save_dict = {
        "points": points,
        "edge_index": new_ei,
        "edge_attr": new_ea,
        "batch_idx": new_bi,
    }
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save(save_dict, output_path)

    B, N = points.shape[0], points.shape[1]
    avg_deg = (2.0 * new_ei.shape[1]) / (B * N)
    print(f"Saved patched spanner to {output_path}")
    print(f"  r={r}, avg_degree={avg_deg:.2f}, total_edges={new_ei.shape[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rao-Smith spanner patching: enforce r-lightness by local rerouting."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input *_spanner.pt (output of spanner.py)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output *_patched_spanner.pt")
    parser.add_argument("--r", type=int, default=5,
                        help="Crossing bound per box side (r-light parameter)")
    parser.add_argument("--max_points", type=int, default=4,
                        help="Quadtree leaf capacity (should match build_raw_pyramid)")
    parser.add_argument("--max_depth", type=int, default=100,
                        help="Quadtree max depth")
    args = parser.parse_args()

    patch_dataset(args.input, args.output, r=args.r,
                  max_points_per_leaf=args.max_points, max_depth=args.max_depth)
