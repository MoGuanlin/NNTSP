import torch
import argparse
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional
from torch.utils.data import Dataset, DataLoader

try:
    from torch_geometric.data import Data
except ImportError:
    class Data:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def to_dict(self):
            return self.__dict__


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

        # contract / feature-system tags (if present)
        for name in ["coord_contract_version", "tree_node_feat_box_mode"]:
            if hasattr(raw, name):
                setattr(new_data, name, getattr(raw, name))

        new_data.spanner_edge_index = raw.spanner_edge_index
        new_data.spanner_edge_attr = raw.spanner_edge_attr
        if hasattr(raw, "spanner_edge_attr_norm"):
            new_data.spanner_edge_attr_norm = raw.spanner_edge_attr_norm
        new_data.edge_alive_mask = edge_alive_mask
        new_data.alive_edge_id = torch.arange(E, dtype=torch.long)
        new_data.edge_alive_mask = edge_alive_mask  # [E] in OLD eid space
        # 可选：保留一个“old eid 列表”便于 debug；语义明确：它不是 new-eid 映射
        new_data.alive_edge_id = torch.arange(E, dtype=torch.long)  # old eid list

        # new_data.alive_spanner_edge_index = raw.spanner_edge_index
        # new_data.alive_spanner_edge_attr = raw.spanner_edge_attr
        # if hasattr(raw, "spanner_edge_attr_norm"):
        #     new_data.alive_spanner_edge_attr_norm = raw.spanner_edge_attr_norm

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
    if hasattr(raw, "spanner_edge_attr_norm"):
        new_data.spanner_edge_attr_norm = raw.spanner_edge_attr_norm
    new_data.edge_alive_mask = edge_alive_mask  # [E]
    
    E = int(raw.spanner_edge_index.size(1))
    assert new_data.edge_alive_mask.numel() == E
    assert new_data.spanner_edge_index.size(1) == E

    
    # convenience: alive_edge_id is a list of OLD eids (same space as spanner_edge_index)
    new_data.alive_edge_id = torch.nonzero(edge_alive_mask, as_tuple=False).view(-1)  # [E_alive], old eid


    # derived alive subset (convenience)
    # alive_eids = torch.nonzero(edge_alive_mask, as_tuple=False).view(-1)
    # new_data.alive_edge_id = alive_eids  # [E_alive]
    # new_data.alive_spanner_edge_index = raw.spanner_edge_index[:, alive_eids]
    # new_data.alive_spanner_edge_attr = raw.spanner_edge_attr[alive_eids]
    # if hasattr(raw, "spanner_edge_attr_norm"):
    #     new_data.alive_spanner_edge_attr_norm = raw.spanner_edge_attr_norm[alive_eids]

    # contract / feature-system tags (avoid downstream auto-detect)
    for name in ["coord_contract_version", "tree_node_feat_box_mode"]:
        if hasattr(raw, name):
            setattr(new_data, name, getattr(raw, name))

    # copy tree/leaf structures if present
    for name in [
        "tree_node_feat", "tree_node_depth", "is_leaf", "tree_edge_index",
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
    
    E = int(raw.spanner_edge_index.size(1))

    if new_data.interface_eid_index.numel() > 0:
        assert int(new_data.interface_eid_index.max()) < E
        assert bool(edge_alive_mask[new_data.interface_eid_index].all().item())

    if new_data.crossing_eid_index.numel() > 0:
        assert int(new_data.crossing_eid_index.max()) < E
        assert bool(edge_alive_mask[new_data.crossing_eid_index].all().item())


    # ---------------------------------------------------------
    # Interface extra metadata (precomputed in build_raw_pyramid.py)
    # ---------------------------------------------------------
    # 为了彻底消除“坐标系混用”以及避免 pruning 时的几何循环，我们要求 raw 数据
    # 已经提供以下字段，并在此处仅做筛选。
    if hasattr(raw, "interface_inside_quadrant"):
        new_data.interface_inside_quadrant = raw.interface_inside_quadrant[keep_iface]
    else:
        raise ValueError(
            "raw data missing interface_inside_quadrant; please regenerate *_raw_pyramid.pt with coord_contract_version=2"
        )

    # optional debug: verify per-node per-dir <= r among pruned interface records
    if debug:
        counts = defaultdict(int)
        I2 = int(new_data.interface_assign_index.size(1))
        pr_nodes_cpu = new_data.interface_assign_index[0].detach().cpu()
        pr_dir_cpu = new_data.interface_boundary_dir.detach().cpu()
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

# --- End of prune_r_light_single body ---


class PruneDataset(Dataset):
    def __init__(self, raw_list_np: List[Dict], r: int, debug: bool):
        self.raw_list_np = raw_list_np
        self.r = r
        self.debug = debug

    def __len__(self):
        return len(self.raw_list_np)

    def __getitem__(self, b: int):
        raw_dict = self.raw_list_np[b]
        
        # Reconstruct Data for the worker locally
        raw = Data()
        for k, v in raw_dict.items():
            setattr(raw, k, torch.from_numpy(v) if isinstance(v, np.ndarray) else v)
            
        data = prune_r_light_single(raw, r=self.r, debug=self.debug)
        
        # Convert back to numpy dict for safe return
        res = {}
        for k, v in (data.to_dict() if hasattr(data, "to_dict") else data.__dict__).items():
            if torch.is_tensor(v):
                res[k] = v.detach().cpu().numpy().copy()
            else:
                res[k] = v
        
        # NUCLEAR OPTION: Pickle to bytes to force single-blob IPC.
        #
        # [Technical Analysis]
        # Previous mmap errors were caused by PyTorch/DataLoader attempting to manage shared memory
        # handles for thousands of small Numpy arrays/Tensors.
        # By pickling to bytes here, we force the IPC to treat the result as a single opaque blob,
        # dramatically reducing the system call overhead and file descriptor usage.
        import pickle
        return pickle.dumps(res)


def prune_dataset(input_path: str, output_path: str, r: int, num_workers: int = 1, debug: bool = False) -> None:
    raw_list = _safe_torch_load(input_path)
    
    print(f"Loaded raw pyramid list: {len(raw_list)} samples. Converting to numpy for stable IPC...")
    raw_list_np = []
    for raw in raw_list:
        d = {k: (v.detach().cpu().numpy().copy() if torch.is_tensor(v) else v) for k, v in (raw.to_dict() if hasattr(raw, "to_dict") else raw.__dict__).items()}
        raw_list_np.append(d)
    del raw_list # Free memory
    
    print(f"Pruning with {num_workers} workers (batch_size=50)...")
    
    dataset = PruneDataset(raw_list_np, r, debug)
    loader = DataLoader(
        dataset,
        batch_size=50, 
        num_workers=num_workers,
        shuffle=False,
        collate_fn=lambda x: x 
    )

    pruned_list = []
    import pickle
    pbar = tqdm(total=len(raw_list_np), desc="Pruning")
    for batch in loader:
        for data_bytes in batch:
            data_dict = pickle.loads(data_bytes)
            data = Data()
            for k, v in data_dict.items():
                if isinstance(v, np.ndarray):
                    setattr(data, k, torch.from_numpy(v))
                else:
                    setattr(data, k, v)
            pruned_list.append(data)
            pbar.update(1)
    pbar.close()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(pruned_list, output_path)
    print(f"Saved pruned dataset: {output_path}  (r={r})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input *_raw_pyramid.pt")
    parser.add_argument("--output", type=str, required=True, help="Output *_r_light_pyramid.pt")
    parser.add_argument("--r", type=int, default=5, help="Per-(node,boundary) interface cap")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    prune_dataset(args.input, args.output, r=args.r, num_workers=args.num_workers, debug=args.debug)
