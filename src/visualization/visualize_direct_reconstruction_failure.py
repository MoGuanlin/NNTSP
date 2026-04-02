# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

from src.models.node_token_packer import _stable_sort_interfaces
from src.visualization.visualize_pyramid import (
    _get_alive_edge_index,
    _inter_abs_from_inter_rel,
    _node_bbox_from_feat,
    _segment_rect_boundary_intersection,
    _tree_box_mode,
)


def _cpu_tensor(x: Any, *, dtype=None) -> torch.Tensor:
    t = torch.as_tensor(x)
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t.detach().cpu()


def _sorted_node_slot_points(data: Any) -> Dict[int, List[Dict[str, Any]]]:
    ia = _cpu_tensor(getattr(data, "interface_assign_index"), dtype=torch.long)
    iface_nid = ia[0]
    iface_eid = ia[1]
    iface_feat6 = _cpu_tensor(getattr(data, "interface_edge_attr"), dtype=torch.float32)
    iface_dir = _cpu_tensor(getattr(data, "interface_boundary_dir"), dtype=torch.long)
    iface_inside_ep = _cpu_tensor(getattr(data, "interface_inside_endpoint"), dtype=torch.long)
    iface_inside_quad = _cpu_tensor(getattr(data, "interface_inside_quadrant"), dtype=torch.long)

    iface_nid, iface_eid, iface_feat6, iface_dir, iface_inside_ep, iface_inside_quad = _stable_sort_interfaces(
        iface_nid=iface_nid,
        iface_eid=iface_eid,
        iface_feat6=iface_feat6,
        iface_dir=iface_dir,
        iface_inside_ep=iface_inside_ep,
        iface_inside_quad=iface_inside_quad,
        clockwise=True,
    )

    pos = _cpu_tensor(getattr(data, "pos"), dtype=torch.float32).numpy()
    sp_edge_index = _cpu_tensor(getattr(data, "spanner_edge_index"), dtype=torch.long).numpy()
    tree_node_feat = _cpu_tensor(getattr(data, "tree_node_feat"), dtype=torch.float32).numpy()
    box_mode = _tree_box_mode(data)
    iface_inter_xy = None
    if hasattr(data, "interface_intersection_xy"):
        iface_inter_xy = _cpu_tensor(getattr(data, "interface_intersection_xy"), dtype=torch.float32).numpy()

    node_slots: Dict[int, List[Dict[str, Any]]] = {}
    for idx in range(int(iface_nid.numel())):
        nid = int(iface_nid[idx].item())
        slot = len(node_slots.get(nid, []))
        x0, y0, w, h = _node_bbox_from_feat(tree_node_feat[nid], box_mode=box_mode)
        xy: Optional[Tuple[float, float]]
        if iface_inter_xy is not None:
            xy = (float(iface_inter_xy[idx, 0]), float(iface_inter_xy[idx, 1]))
        else:
            feat6 = iface_feat6[idx]
            inter_rel = (float(feat6[2].item()), float(feat6[3].item()))
            xy = _inter_abs_from_inter_rel(x0, y0, w, h, inter_rel)
            if not all(float(v) == float(v) for v in xy):
                eid = int(iface_eid[idx].item())
                bdir = int(iface_dir[idx].item())
                inside_ep = int(iface_inside_ep[idx].item())
                u = int(sp_edge_index[0, eid])
                v = int(sp_edge_index[1, eid])
                pu = (float(pos[u, 0]), float(pos[u, 1]))
                pv = (float(pos[v, 0]), float(pos[v, 1]))
                inside = pu if inside_ep == 0 else pv
                outside = pv if inside_ep == 0 else pu
                xy = _segment_rect_boundary_intersection(x0, y0, w, h, inside, outside, bdir)

        node_slots.setdefault(nid, []).append(
            {
                "slot": int(slot),
                "xy": xy,
                "eid": int(iface_eid[idx].item()),
                "boundary_dir": int(iface_dir[idx].item()),
            }
        )
    return node_slots


def _slot_xy(node_slots: Dict[int, List[Dict[str, Any]]], nid: int, slot: int) -> Optional[Tuple[float, float]]:
    entries = node_slots.get(int(nid), [])
    if slot < 0 or slot >= len(entries):
        return None
    return entries[slot].get("xy")


def save_direct_reconstruction_failure_plot(
    *,
    data: Any,
    direct_tour_stats: Dict[str, Any],
    sample_idx: int,
    output_path: str | Path,
) -> Optional[str]:
    details = direct_tour_stats.get("error_details")
    if not isinstance(details, dict):
        return None

    node_id = int(details.get("node_id", -1))
    if node_id < 0:
        return None

    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    pos = _cpu_tensor(getattr(data, "pos"), dtype=torch.float32).numpy()
    tree_node_feat = _cpu_tensor(getattr(data, "tree_node_feat"), dtype=torch.float32).numpy()
    tree_parent_index = _cpu_tensor(getattr(data, "tree_parent_index"), dtype=torch.long)
    tree_node_depth = _cpu_tensor(getattr(data, "tree_node_depth"), dtype=torch.long)
    box_mode = _tree_box_mode(data)
    alive_edge_index = _cpu_tensor(_get_alive_edge_index(data), dtype=torch.long).numpy()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal", adjustable="box")

    for k in range(alive_edge_index.shape[1]):
        u = int(alive_edge_index[0, k])
        v = int(alive_edge_index[1, k])
        ax.plot(
            [pos[u, 0], pos[v, 0]],
            [pos[u, 1], pos[v, 1]],
            color="lightgray",
            linewidth=0.5,
            alpha=0.25,
            zorder=1,
        )

    ax.scatter(pos[:, 0], pos[:, 1], s=10, c="black", alpha=0.7, zorder=3)

    node_slots = _sorted_node_slot_points(data)
    child_global_ids = [int(x) for x in details.get("child_global_ids", [])]
    parent_boundary_slots = {int(x) for x in details.get("parent_boundary_slots", [])}
    child_to_parent = {
        (int(item["child"]), int(item["child_slot"])): int(item["parent_slot"])
        for item in details.get("child_to_parent", [])
    }

    focus_nodes = [node_id]
    parent_id = int(tree_parent_index[node_id].item()) if node_id < int(tree_parent_index.numel()) else -1
    if parent_id >= 0:
        focus_nodes.append(parent_id)
    focus_nodes.extend([cid for cid in child_global_ids if cid >= 0])

    for nid in focus_nodes:
        x0, y0, w, h = _node_bbox_from_feat(tree_node_feat[nid], box_mode=box_mode)
        if nid == node_id:
            edgecolor = "red"
            lw = 2.5
            label = f"failed node {nid}"
        elif nid == parent_id:
            edgecolor = "purple"
            lw = 1.8
            label = f"parent {nid}"
        else:
            edgecolor = "darkorange"
            lw = 1.8
            q = child_global_ids.index(nid) if nid in child_global_ids else -1
            label = f"child q={q}, nid={nid}"
        ax.add_patch(Rectangle((x0, y0), w, h, fill=False, linewidth=lw, edgecolor=edgecolor, zorder=2))
        ax.text(x0, y0 + h, label, color=edgecolor, fontsize=9, va="bottom", ha="left")

    for rec in node_slots.get(node_id, []):
        xy = rec.get("xy")
        if xy is None:
            continue
        slot = int(rec["slot"])
        color = "limegreen" if slot in parent_boundary_slots else "gray"
        ax.scatter([xy[0]], [xy[1]], s=40, c=color, zorder=4)
        ax.text(xy[0], xy[1], f"p{slot}", color=color, fontsize=8, ha="left", va="bottom")

    for q, cid in enumerate(child_global_ids):
        if cid < 0:
            continue
        for rec in node_slots.get(cid, []):
            xy = rec.get("xy")
            if xy is None:
                continue
            slot = int(rec["slot"])
            ax.scatter([xy[0]], [xy[1]], s=20, c="steelblue", alpha=0.5, zorder=3)
            ax.text(xy[0], xy[1], f"q{q}s{slot}", color="steelblue", fontsize=7, ha="left", va="top")

    for frag in details.get("open_fragments", []):
        frag_idx = int(frag.get("fragment_idx", -1))
        endpoints: List[Tuple[float, float]] = []
        for end_name in ("start", "end"):
            q = int(frag[f"{end_name}_child"])
            slot = int(frag[f"{end_name}_slot"])
            cid = child_global_ids[q] if 0 <= q < len(child_global_ids) else -1
            xy = _slot_xy(node_slots, cid, slot) if cid >= 0 else None
            mapped = (q, slot) in child_to_parent
            if xy is not None:
                endpoints.append(xy)
                color = "gold" if mapped else "red"
                ax.scatter([xy[0]], [xy[1]], s=90, c=color, marker="x", zorder=5)
                parent_txt = f"->p{child_to_parent[(q, slot)]}" if mapped else "->X"
                ax.text(
                    xy[0],
                    xy[1],
                    f"f{frag_idx}:{end_name[0]} q{q}s{slot}{parent_txt}",
                    color=color,
                    fontsize=8,
                    ha="left",
                    va="bottom",
                )
        if len(endpoints) == 2:
            ax.plot(
                [endpoints[0][0], endpoints[1][0]],
                [endpoints[0][1], endpoints[1][1]],
                color="magenta",
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
                zorder=4,
            )

    ax.set_title(
        f"Direct Traceback Failure - sample {sample_idx}\n"
        f"node={node_id}, depth={int(details.get('node_depth', -1))}, "
        f"reason={details.get('reason', 'unknown')}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


__all__ = ["save_direct_reconstruction_failure_plot"]
