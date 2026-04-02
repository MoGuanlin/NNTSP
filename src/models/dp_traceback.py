# -*- coding: utf-8 -*-
"""Traceback and CPU cache helpers for the 1-pass DP runner."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch

from .node_token_packer import PackedNodeTokens

if TYPE_CHECKING:
    from .dp_runner import CostTableEntry


def traceback_leaf_states(
    *,
    root_id: int,
    root_sigma: int,
    tokens: PackedNodeTokens,
    cost_tables: Dict[int, "CostTableEntry"],
) -> Dict[int, int]:
    """Top-down traceback to recover the chosen leaf states."""
    leaf_states: Dict[int, int] = {}

    if root_sigma < 0:
        return leaf_states

    queue: List[Tuple[int, int]] = [(root_id, root_sigma)]

    while queue:
        nid, sigma_idx = queue.pop()

        if tokens.is_leaf[nid].item():
            leaf_states[nid] = sigma_idx
            continue

        ct = cost_tables.get(nid)
        if ct is None:
            continue

        child_indices = ct.backptr.get(sigma_idx)
        if child_indices is None:
            continue

        children = tokens.tree_children_index[nid].long()
        for q in range(4):
            cid = int(children[q].item())
            if cid < 0:
                continue
            c_si = child_indices[q]
            if c_si < 0:
                continue
            queue.append((cid, c_si))

    return leaf_states


def make_cpu_token_cache(tokens: PackedNodeTokens) -> SimpleNamespace:
    """Cache the DP-combinatorial token fields on CPU.

    The neural forward stays on the requested device, but all post-decode
    DP routines are Python-heavy and call `.item()` frequently. Keeping
    those tensors on CPU avoids thousands of tiny CUDA sync points.
    """
    field_names = [
        "tree_parent_index",
        "tree_children_index",
        "tree_node_depth",
        "tree_node_feat_rel",
        "is_leaf",
        "root_id",
        "iface_eid",
        "iface_mask",
        "iface_boundary_dir",
        "iface_feat6",
        "cross_eid",
        "cross_mask",
        "cross_child_pair",
    ]
    payload: Dict[str, Any] = {}
    for name in field_names:
        value = getattr(tokens, name)
        payload[name] = value.detach().cpu() if torch.is_tensor(value) else value
    return SimpleNamespace(**payload)


__all__ = ["make_cpu_token_cache", "traceback_leaf_states"]
