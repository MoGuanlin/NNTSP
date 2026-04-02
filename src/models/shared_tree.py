# src/models/shared_tree.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
from torch import Tensor

from .node_token_packer import PackedNodeTokens


def extract_z(out: Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]) -> Tuple[Tensor, Dict[str, Tensor]]:
    if torch.is_tensor(out):
        return out, {}
    if isinstance(out, tuple) and len(out) >= 1 and torch.is_tensor(out[0]):
        aux = out[1] if len(out) >= 2 and isinstance(out[1], dict) else {}
        return out[0], aux
    raise TypeError("Encoder output must be Tensor or (Tensor, dict).")


def build_leaf_row_for_node(total_nodes: int, leaf_node_id: Tensor) -> Tensor:
    device = leaf_node_id.device
    leaf_row_for_node = torch.full((total_nodes,), -1, dtype=torch.long, device=device)
    if leaf_node_id.numel() == 0:
        return leaf_row_for_node
    uniq = torch.unique(leaf_node_id)
    if uniq.numel() != leaf_node_id.numel():
        raise ValueError("leaves.leaf_node_id contains duplicates; expected unique ids.")
    leaf_row_for_node[leaf_node_id] = torch.arange(leaf_node_id.numel(), device=device, dtype=torch.long)
    return leaf_row_for_node


def gather_node_fields(tokens: PackedNodeTokens, nids: Tensor) -> Dict[str, Tensor]:
    return {
        "node_feat_rel": tokens.tree_node_feat_rel[nids],
        "node_depth": tokens.tree_node_depth[nids],
        "iface_feat6": tokens.iface_feat6[nids],
        "iface_mask": tokens.iface_mask[nids],
        "iface_boundary_dir": tokens.iface_boundary_dir[nids],
        "iface_inside_endpoint": tokens.iface_inside_endpoint[nids],
        "iface_inside_quadrant": tokens.iface_inside_quadrant[nids],
        "cross_feat6": tokens.cross_feat6[nids],
        "cross_mask": tokens.cross_mask[nids],
        "cross_child_pair": tokens.cross_child_pair[nids],
        "cross_is_leaf_internal": tokens.cross_is_leaf_internal[nids],
    }


__all__ = ["build_leaf_row_for_node", "extract_z", "gather_node_fields"]
