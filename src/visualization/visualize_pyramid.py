
import torch
import argparse
import os
from typing import Tuple, Optional, List

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _safe_torch_load(path: str):
    try:
        return torch.load(path)
    except Exception:
        return torch.load(path, weights_only=False)


def _get_alive_edge_index(data):
    """
    Return alive_edge_index [2, E_alive] for plotting.
    Contract:
    - spanner_edge_index is FULL (old eid space)
    - edge_alive_mask marks alive old eids
    """
    if hasattr(data, "edge_alive_mask") and hasattr(data, "spanner_edge_index"):
        mask = data.edge_alive_mask
        return data.spanner_edge_index[:, mask]
    return getattr(data, "spanner_edge_index")



def _tree_box_mode(data) -> str:
    """
    Contract v2 writes:
      - data.tree_node_feat_box_mode == 0  => llwh(abs)
    Returns: "ll" or "center"
    """
    if not hasattr(data, "tree_node_feat_box_mode"):
        raise RuntimeError("Missing tree_node_feat_box_mode in data. Regenerate dataset with Contract v2.")
    try:
        v = int(torch.as_tensor(getattr(data, "tree_node_feat_box_mode")).view(-1)[0].item())
    except Exception as exc:  # pragma: no cover - defensive decoding for corrupted files
        raise RuntimeError("Failed to parse tree_node_feat_box_mode from data.") from exc
    if v == 0:
        return "ll"
    if v == 1:
        return "center"
    raise RuntimeError(f"Unsupported tree_node_feat_box_mode={v}.")


def _node_bbox_from_feat(node_feat_row, box_mode: str) -> Tuple[float, float, float, float]:
    """
    box_mode:
      - "ll": node_feat_row = [x0, y0, w, h]
      - "center": node_feat_row = [cx, cy, w, h] -> convert to lower-left.
    """
    a0, a1, w, h = float(node_feat_row[0]), float(node_feat_row[1]), float(node_feat_row[2]), float(node_feat_row[3])
    if box_mode == "ll":
        return a0, a1, w, h
    x0 = a0 - w / 2.0
    y0 = a1 - h / 2.0
    return x0, y0, w, h


def _node_center_from_llwh(x0: float, y0: float, w: float, h: float) -> Tuple[float, float]:
    return (x0 + w / 2.0, y0 + h / 2.0)


def _inter_abs_from_inter_rel(
    x0: float, y0: float, w: float, h: float,
    inter_rel_xy: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Contract v2: inter_rel is center-relative:
      inter_rel = (inter_abs - center) / (w/2, h/2)
    """
    cx, cy = _node_center_from_llwh(x0, y0, w, h)
    ix = cx + float(inter_rel_xy[0]) * (w / 2.0)
    iy = cy + float(inter_rel_xy[1]) * (h / 2.0)
    return (ix, iy)


def _segment_rect_boundary_intersection(
    x0: float, y0: float, w: float, h: float,
    inside: Tuple[float, float],
    outside: Tuple[float, float],
    boundary_dir: int
) -> Optional[Tuple[float, float]]:
    """
    Compute intersection point between segment (inside -> outside) and the chosen rectangle boundary:
      boundary_dir: 0 Left, 1 Right, 2 Bottom, 3 Top
    Returns (ix, iy) if exists; otherwise fallback to minimal-t intersection among all boundaries.
    """
    ix, iy = float(inside[0]), float(inside[1])
    ox, oy = float(outside[0]), float(outside[1])
    dx, dy = ox - ix, oy - iy

    x_left = x0
    x_right = x0 + w
    y_bot = y0
    y_top = y0 + h

    def intersect_dir(d: int) -> Optional[Tuple[float, float, float]]:
        # returns (t, x, y)
        if d == 0:  # left x=x_left
            if abs(dx) < 1e-12:
                return None
            t = (x_left - ix) / dx
            y = iy + t * dy
            if 0.0 <= t <= 1.0 and (y_bot - 1e-9) <= y <= (y_top + 1e-9):
                return (t, x_left, y)
            return None
        if d == 1:  # right x=x_right
            if abs(dx) < 1e-12:
                return None
            t = (x_right - ix) / dx
            y = iy + t * dy
            if 0.0 <= t <= 1.0 and (y_bot - 1e-9) <= y <= (y_top + 1e-9):
                return (t, x_right, y)
            return None
        if d == 2:  # bottom y=y_bot
            if abs(dy) < 1e-12:
                return None
            t = (y_bot - iy) / dy
            x = ix + t * dx
            if 0.0 <= t <= 1.0 and (x_left - 1e-9) <= x <= (x_right + 1e-9):
                return (t, x, y_bot)
            return None
        if d == 3:  # top y=y_top
            if abs(dy) < 1e-12:
                return None
            t = (y_top - iy) / dy
            x = ix + t * dx
            if 0.0 <= t <= 1.0 and (x_left - 1e-9) <= x <= (x_right + 1e-9):
                return (t, x, y_top)
            return None
        return None

    # First try the specified boundary
    hit = intersect_dir(int(boundary_dir))
    if hit is not None:
        _, x, y = hit
        return (x, y)

    # Fallback: try all boundaries and take smallest positive t
    candidates = []
    for d in [0, 1, 2, 3]:
        h2 = intersect_dir(d)
        if h2 is not None:
            candidates.append(h2)

    if not candidates:
        return None

    candidates.sort(key=lambda z: z[0])
    _, x, y = candidates[0]
    return (x, y)


def visualize_sample(
    data,
    output_path: str,
    draw_max_depth: int = 20,
    draw_only_leaves: bool = False,
    draw_points: bool = True,
    point_size: float = 18.0,
    leaf_edge_lw: float = 1.2,
    cross_edge_lw: float = 1.2,
    rect_lw: float = 0.6,
    intersection_size: float = 16.0,
    show: bool = False,
):
    """
    Visualize:
      - quadtree rectangles (black)
      - r-light alive edges:
          * leaf-internal edges: gray
          * cross-boundary edges: blue
      - boundary intersection points for pruned interface records: red
      - points: black
    """
    pos = data.pos.detach().cpu().numpy()
    alive_edge_index = _get_alive_edge_index(data).detach().cpu().numpy()

    if not hasattr(data, "tree_node_feat") or not hasattr(data, "tree_node_depth"):
        raise RuntimeError("Missing tree_node_feat/tree_node_depth in data. Regenerate raw_pyramid.")

    tree_node_feat = data.tree_node_feat.detach().cpu().numpy()  # [M,4]
    tree_depth = data.tree_node_depth.detach().cpu().numpy()     # [M]
    is_leaf = data.is_leaf.detach().cpu().numpy() if hasattr(data, "is_leaf") else None

    box_mode = _tree_box_mode(data)

    # point_to_leaf is used to classify edges as leaf-internal vs cross-boundary
    if not hasattr(data, "point_to_leaf"):
        raise RuntimeError("Missing point_to_leaf in data. Regenerate raw_pyramid.")
    point_to_leaf = data.point_to_leaf.detach().cpu().numpy()  # [N]

    # interface records (pruned)
    need_fields = ["interface_assign_index", "interface_boundary_dir", "interface_inside_endpoint"]
    for f in need_fields:
        if not hasattr(data, f):
            raise RuntimeError(f"Missing {f} in data. Regenerate r_light_pyramid.")
    ia = data.interface_assign_index.detach().cpu().numpy()  # [2,I]
    iface_nodes = ia[0]
    iface_eids = ia[1]
    iface_dir = data.interface_boundary_dir.detach().cpu().numpy()        # [I]
    iface_inside = data.interface_inside_endpoint.detach().cpu().numpy()  # [I]

    # Preferred: precomputed ABS intersections from build/prune
    iface_inter_xy = data.interface_intersection_xy.detach().cpu().numpy() if hasattr(data, "interface_intersection_xy") else None

    # Contract v2 fallback: derive intersection ABS from interface_edge_attr (inter_rel) + node bbox
    iface_edge_attr = data.interface_edge_attr.detach().cpu().numpy() if hasattr(data, "interface_edge_attr") else None

    if not hasattr(data, "spanner_edge_index"):
        raise RuntimeError("Missing spanner_edge_index in data.")
    sp_edge_index = data.spanner_edge_index.detach().cpu().numpy()  # [2,E]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal", adjustable="box")

    # Points: black
    if draw_points:
        ax.scatter(pos[:, 0], pos[:, 1], s=point_size, c="black", zorder=3)

    # Alive edges: classify by whether endpoints belong to same leaf
    for k in range(alive_edge_index.shape[1]):
        u = int(alive_edge_index[0, k])
        v = int(alive_edge_index[1, k])
        same_leaf = (int(point_to_leaf[u]) == int(point_to_leaf[v]))
        if same_leaf:
            ax.plot([pos[u, 0], pos[v, 0]], [pos[u, 1], pos[v, 1]],
                    color="gray", linewidth=leaf_edge_lw, zorder=1)
        else:
            ax.plot([pos[u, 0], pos[v, 0]], [pos[u, 1], pos[v, 1]],
                    color="blue", linewidth=cross_edge_lw, zorder=2)

    # Build node->iface list for local drawing
    M = tree_node_feat.shape[0]
    node_to_iface_indices: List[List[int]] = [[] for _ in range(M)]
    for j in range(iface_nodes.shape[0]):
        nid = int(iface_nodes[j])
        if 0 <= nid < M:
            node_to_iface_indices[nid].append(j)

    for nid in range(M):
        depth = int(tree_depth[nid])
        if depth > draw_max_depth:
            continue
        if draw_only_leaves and is_leaf is not None and (not bool(is_leaf[nid])):
            continue

        x0, y0, w, h = _node_bbox_from_feat(tree_node_feat[nid], box_mode=box_mode)

        rect = Rectangle((x0, y0), w, h, fill=False, linewidth=rect_lw, edgecolor="black", zorder=0)
        ax.add_patch(rect)

        # intersection points for this node: red
        for rec_idx in node_to_iface_indices[nid]:
            eid = int(iface_eids[rec_idx])
            bdir = int(iface_dir[rec_idx])
            inside_ep = int(iface_inside[rec_idx])  # 0 => sp_edge_index[0,eid] is inside; 1 => sp_edge_index[1,eid]

            u = int(sp_edge_index[0, eid])
            v = int(sp_edge_index[1, eid])
            pu = (float(pos[u, 0]), float(pos[u, 1]))
            pv = (float(pos[v, 0]), float(pos[v, 1]))

            inside = pu if inside_ep == 0 else pv
            outside = pv if inside_ep == 0 else pu

            inter: Optional[Tuple[float, float]] = None

            if iface_inter_xy is not None:
                inter = (float(iface_inter_xy[rec_idx, 0]), float(iface_inter_xy[rec_idx, 1]))
            elif iface_edge_attr is not None and iface_edge_attr.shape[1] >= 4:
                # Contract v2: interface_edge_attr = [inside_rel(2), inter_rel(2), ...]
                inter_rel = (float(iface_edge_attr[rec_idx, 2]), float(iface_edge_attr[rec_idx, 3]))
                inter = _inter_abs_from_inter_rel(x0, y0, w, h, inter_rel)
            else:
                inter = _segment_rect_boundary_intersection(x0, y0, w, h, inside, outside, bdir)

            if inter is None:
                continue

            ax.scatter([inter[0]], [inter[1]], s=intersection_size, c="red", zorder=4)

    ax.set_title("Quadtree + r-light alive edges + boundary intersections")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize quadtree, r-light edges, and boundary intersections (contract-v2 aware)")
    parser.add_argument("--input", type=str, required=True, help="Input *_r_light_pyramid.pt (List[Data] or Data)")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index if input is a list")
    parser.add_argument("--output", type=str, default="vis_r_light_intersections.png", help="Output image path")
    parser.add_argument("--max_depth_draw", type=int, default=20, help="Draw nodes up to this depth")
    parser.add_argument("--draw_only_leaves", action="store_true", help="Only draw leaf rectangles")
    parser.add_argument("--no_points", action="store_true", help="Do not draw points")
    parser.add_argument("--show", action="store_true", help="Show interactive window")
    args = parser.parse_args()

    obj = _safe_torch_load(args.input)

    if isinstance(obj, list):
        data_list = obj
        if args.sample_idx < 0 or args.sample_idx >= len(data_list):
            raise ValueError(f"sample_idx out of range: {args.sample_idx} (dataset size={len(data_list)})")
        data = data_list[args.sample_idx]
    else:
        data = obj

    visualize_sample(
        data=data,
        output_path=args.output,
        draw_max_depth=args.max_depth_draw,
        draw_only_leaves=args.draw_only_leaves,
        draw_points=not args.no_points,
        show=args.show,
    )


if __name__ == "__main__":
    main()
