# src/dataprep/dataset.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Any, List, Dict
import torch
from torch.utils.data import Dataset

import pickle

import os
from pathlib import Path
from types import SimpleNamespace

def smart_load_dataset(path: str) -> FastTSPDataset:
    """
    Smarter dataset loader:
    1. If path ends in .fast.pt, just load it.
    2. If not, check if a .fast.pt version exists in the same directory.
    3. If fast version exists, load it.
    4. If not, load original, consolidate it, save as .fast.pt, and return.
    """
    p = Path(path)
    
    # 1. Direct fast.pt load
    if ".fast.pt" in p.name:
        if not p.exists():
            raise FileNotFoundError(f"Fast dataset {path} not found.")
        print(f"[data] Loading consolidated dataset: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        return FastTSPDataset(data)

    # 2. Check for companion fast.pt
    fast_path = p.parent / (p.stem + ".fast.pt")
    if fast_path.exists():
        print(f"[data] Detected fast version at {fast_path}, using it.")
        data = torch.load(str(fast_path), map_location="cpu", weights_only=False)
        return FastTSPDataset(data)

    # 3. Load original and consolidate
    if not p.exists():
        raise FileNotFoundError(f"Original dataset {path} not found.")
    
    print(f"[data] Loading original dataset: {path}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    
    if isinstance(obj, dict) and "num_samples" in obj:
        # It's already consolidated but didn't have .fast.pt suffix
        return FastTSPDataset(obj)
    
    if not isinstance(obj, list):
        obj = [obj] # Handle single sample files
        
    print(f"[data] Consolidating {path} for faster loading next time...")
    consolidated = consolidate_data_list(obj)
    
    try:
        torch.save(consolidated, fast_path)
        print(f"[data] Saved consolidated version to {fast_path}")
    except Exception as e:
        print(f"[data] Warning: Could not save consolidated version: {e}")
        
    return FastTSPDataset(consolidated)

class TSPDataset(Dataset):
    """
    Simple Dataset wrapper for TSP Data objects.
    """
    def __init__(self, data_list: List[Any], use_pickle: bool = False) -> None:
        self.data_list = data_list
        self.use_pickle = use_pickle

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Any:
        item = self.data_list[idx]
        if self.use_pickle:
            return pickle.dumps(item)
        return item

def consolidate_data_list(data_list: List[Any]) -> Dict[str, Any]:
    """
    Consolidate a list of PyG Data objects into a single dictionary of concatenated tensors.
    This significantly speeds up loading from disk.
    """
    if not data_list:
        return {}

    keys = ["pos", "spanner_edge_index", "spanner_edge_attr", "target_edges", "tour_len",
            "tree_node_feat", "tree_children_index", "tree_parent_index", "tree_node_depth", "is_leaf",
            "interface_assign_index", "interface_edge_attr", "interface_boundary_dir",
            "interface_inside_endpoint", "interface_inside_quadrant",
            "crossing_assign_index", "crossing_edge_attr", "crossing_child_pair", "crossing_is_leaf_internal",
            "leaf_ids", "leaf_ptr", "leaf_points", "coord_contract_version", "tree_node_feat_box_mode"]

    # Filter keys that exist in the first item
    available_keys = [k for k in keys if hasattr(data_list[0], k)]
    
    # Store data
    consolidated = {
        "num_samples": len(data_list),
        "keys": available_keys
    }

    ptrs = {k: [0] for k in available_keys if k not in ["coord_contract_version", "tree_node_feat_box_mode", "tour_len"]}
    
    # Pre-collect all tensors to cat
    all_data = {k: [] for k in available_keys}
    
    # Keys that are typically (2, E) and need to be transposed to (E, 2) for consolidation
    transpose_keys = ["spanner_edge_index", "interface_assign_index", "crossing_assign_index"]

    for d in data_list:
        for k in available_keys:
            v = getattr(d, k)
            if not torch.is_tensor(v):
                v = torch.as_tensor(v)
            
            # If it's a (2, E) index tensor, transpose to (E, 2) so variadic dim is 0
            if k in transpose_keys and v.dim() == 2 and v.shape[0] == 2:
                v = v.t()

            all_data[k].append(v)
            
            if k in ptrs:
                # Most keys are (N, ...) or (E, ...), we track the first dimension
                if v.dim() > 0:
                    ptrs[k].append(ptrs[k][-1] + v.shape[0])
                else:
                    ptrs[k].append(ptrs[k][-1] + 1)

    for k in available_keys:
        if k in ["coord_contract_version", "tree_node_feat_box_mode"]:
            # These are usually scalar/same for all, just take first
            consolidated[k] = all_data[k][0]
        elif k == "tour_len":
            # [B]
            consolidated[k] = torch.stack(all_data[k])
        else:
            consolidated[k] = torch.cat(all_data[k], dim=0)
            consolidated[f"{k}_ptr"] = torch.tensor(ptrs[k], dtype=torch.long)

    return consolidated

class FastTSPDataset(Dataset):
    """
    Loads data from a consolidated dictionary of tensors. 
    Avoids Python object overhead by only materializing SimpleNamespace on __getitem__.
    
    WARNING: We removed 'use_pickle' support. Do not attempt to pickle items here.
    Memory sharing relies on OS-level Copy-On-Write (Linux/Fork).
    """
    def __init__(self, consolidated: Dict[str, Any]) -> None:
        self.c = consolidated
        self.num_samples = consolidated["num_samples"]
        self.available_keys = consolidated["keys"]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Any:
        # Materialize a SimpleNamespace that mimics the Data object
        item = SimpleNamespace()
        for k in self.available_keys:
            if k in ["coord_contract_version", "tree_node_feat_box_mode"]:
                setattr(item, k, self.c[k])
            elif k == "tour_len":
                setattr(item, k, self.c[k][idx])
            else:
                ptr_name = f"{k}_ptr"
                start = self.c[ptr_name][idx]
                end = self.c[ptr_name][idx+1]
                val = self.c[k][start:end]
                # Transpose back (E, 2) -> (2, E) if needed
                transpose_keys = ["spanner_edge_index", "interface_assign_index", "crossing_assign_index"]
                if k in transpose_keys and val.dim() == 2 and val.shape[1] == 2:
                    val = val.t()
                setattr(item, k, val)
        
        return item
