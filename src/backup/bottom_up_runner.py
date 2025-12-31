# src/models/bottom_up_runner.py
# -*- coding: utf-8 -*-
"""
Bottom-up Tree Runner for Neural DP (Rao'98-style TSP).

This module is responsible for *DP-aligned control flow*:
- process quadtree nodes bottom-up (by depth descending)
- call leaf_encoder on leaves
- call merge_encoder on internal nodes using already-computed child latents

Key properties:
- Scale-invariant by construction: it only uses normalized/relative features
  provided by NodeTokenPacker:
    - leaf points: leaves.point_xy (cell-relative: (x-cx)/(w/2),(y-cy)/(h/2))
    - node boxes: tokens.tree_node_feat_rel (root-normalized)
    - edge/event features: iface_feat6/cross_feat6, iface_inter_rel_xy, etc.
- I/O per node is O(r): token pack caps Ti and Tc independent of N.

This module does NOT implement any neural network. It defines clean interfaces
(encoders are injected) and provides robust batching over tree depth.

Expected upstream: src/models/node_token_packer.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

import torch
from torch import Tensor


# Import packed dataclasses (support both absolute and relative imports)
try:
    from src.models.node_token_packer import PackedBatch, PackedLeafPoints, PackedNodeTokens
except Exception:  # pragma: no cover
    from .node_token_packer import PackedBatch, PackedLeafPoints, PackedNodeTokens


# -----------------------------
# Encoder interfaces (injected)
# -----------------------------

@runtime_checkable
class LeafEncoder(Protocol):
    """
    Leaf encoder interface.

    All inputs MUST be scale-invariant:
      node_feat_rel: [B,4]  root-normalized box features
      iface_*: per-node packed tensors with cap Ti
      cross_*: per-node packed tensors with cap Tc
      leaf_points_xy: [B, P, 2] cell-relative normalized coordinates
      leaf_points_mask: [B, P] bool

    Return:
      z_leaf: [B, d] (or (z_leaf, aux_dict))
    """
    def __call__(
        self,
        *,
        node_feat_rel: Tensor,
        node_depth: Tensor,
        iface_feat6: Tensor,
        iface_mask: Tensor,
        iface_boundary_dir: Tensor,
        iface_inside_endpoint: Tensor,
        iface_inter_rel_xy: Tensor,
        iface_inside_quadrant: Tensor,
        cross_feat6: Tensor,
        cross_mask: Tensor,
        cross_child_pair: Tensor,
        cross_is_leaf_internal: Tensor,
        leaf_points_xy: Tensor,
        leaf_points_mask: Tensor,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        ...


@runtime_checkable
class MergeEncoder(Protocol):
    """
    Merge encoder interface for internal nodes.

    Inputs (all scale-invariant except indices/masks):
      node_feat_rel: [B,4]
      child_z:       [B,4,d]
      child_mask:    [B,4] bool
      iface/cross packed tensors

    Return:
      z_parent: [B, d] (or (z_parent, aux_dict))
    """
    def __call__(
        self,
        *,
        node_feat_rel: Tensor,
        node_depth: Tensor,
        iface_feat6: Tensor,
        iface_mask: Tensor,
        iface_boundary_dir: Tensor,
        iface_inside_endpoint: Tensor,
        iface_inter_rel_xy: Tensor,
        iface_inside_quadrant: Tensor,
        cross_feat6: Tensor,
        cross_mask: Tensor,
        cross_child_pair: Tensor,
        cross_is_leaf_internal: Tensor,
        child_z: Tensor,
        child_mask: Tensor,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        ...


# -----------------------------
# Outputs
# -----------------------------

@dataclass(frozen=True)
class BottomUpResult:
    """
    Bottom-up latents for a (possibly flattened) forest of graphs.

    z:          [total_M, d]
    computed:   [total_M] bool (True if z[nid] computed)
    root_ids:   [B] long (root nid for each graph in the flattened batch)
    node_ptr:   [B+1] long (prefix sums; nodes for graph b in [node_ptr[b], node_ptr[b+1]))
    aux:        optional dict for debugging/analysis; may be empty
    """
    z: Tensor
    computed: Tensor
    root_ids: Tensor
    node_ptr: Tensor
    aux: Dict[str, Tensor]


# -----------------------------
# Internal utilities
# -----------------------------

def _extract_z(out: Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]) -> Tuple[Tensor, Dict[str, Tensor]]:
    if torch.is_tensor(out):
        return out, {}
    if isinstance(out, tuple) and len(out) == 2 and torch.is_tensor(out[0]) and isinstance(out[1], dict):
        return out[0], out[1]
    raise TypeError(
        "Encoder output must be either a Tensor z [B,d] or a tuple (z, aux_dict)."
    )


def _build_leaf_row_for_node(total_M: int, leaf_node_id: Tensor) -> Tensor:
    """
    Create a mapping array leaf_row_for_node of length total_M:
      leaf_row_for_node[nid] = row index in leaves arrays, else -1.
    """
    device = leaf_node_id.device
    leaf_row_for_node = torch.full((total_M,), -1, dtype=torch.long, device=device)
    if leaf_node_id.numel() == 0:
        return leaf_row_for_node

    # Check uniqueness (leaf_node_id should be unique)
    # If duplicates exist, later gathering would be ambiguous.
    uniq = torch.unique(leaf_node_id)
    if uniq.numel() != leaf_node_id.numel():
        raise ValueError("leaves.leaf_node_id contains duplicates; expected unique leaf node ids.")

    leaf_row_for_node[leaf_node_id] = torch.arange(leaf_node_id.numel(), device=device, dtype=torch.long)
    return leaf_row_for_node


def _find_roots_per_graph(tree_parent_index: Tensor, node_ptr: Tensor) -> Tensor:
    """
    Find exactly one root (parent < 0) in each graph segment.
    """
    device = tree_parent_index.device
    B = int(node_ptr.numel() - 1)
    root_ids = torch.empty((B,), dtype=torch.long, device=device)
    for b in range(B):
        lo = int(node_ptr[b].item())
        hi = int(node_ptr[b + 1].item())
        seg = tree_parent_index[lo:hi]
        roots = torch.nonzero(seg < 0, as_tuple=False).flatten()
        if roots.numel() != 1:
            raise ValueError(
                f"Expected exactly 1 root in graph {b}, found {int(roots.numel())} roots."
            )
        root_ids[b] = lo + roots[0]
    return root_ids


# -----------------------------
# Runner
# -----------------------------

class BottomUpTreeRunner:
    """
    DP-aligned bottom-up runner.

    It can run on:
      - a single graph: (PackedNodeTokens, PackedLeafPoints)
      - a flattened batch: PackedBatch

    Implementation detail:
      - processes nodes by depth descending, batched per depth (vectorized)
      - guarantees children are computed before parent if depths are consistent
    """

    def __init__(self, *, validate_completeness: bool = True) -> None:
        self.validate_completeness = bool(validate_completeness)

    def run_single(
        self,
        *,
        tokens: PackedNodeTokens,
        leaves: PackedLeafPoints,
        leaf_encoder: LeafEncoder,
        merge_encoder: MergeEncoder,
    ) -> BottomUpResult:
        """
        Run bottom-up on a single graph.
        """
        M = int(tokens.tree_parent_index.numel())
        device = tokens.tree_parent_index.device

        node_ptr = torch.tensor([0, M], dtype=torch.long, device=device)
        root_ids = _find_roots_per_graph(tokens.tree_parent_index, node_ptr)

        return self._run_core(
            tokens=tokens,
            leaves=leaves,
            node_ptr=node_ptr,
            root_ids=root_ids,
            leaf_encoder=leaf_encoder,
            merge_encoder=merge_encoder,
        )

    def run_batch(
        self,
        *,
        batch: PackedBatch,
        leaf_encoder: LeafEncoder,
        merge_encoder: MergeEncoder,
    ) -> BottomUpResult:
        """
        Run bottom-up on a flattened batch produced by NodeTokenPacker.pack_batch.
        """
        tokens = batch.tokens
        leaves = batch.leaves
        node_ptr = batch.node_ptr
        root_ids = _find_roots_per_graph(tokens.tree_parent_index, node_ptr)

        return self._run_core(
            tokens=tokens,
            leaves=leaves,
            node_ptr=node_ptr,
            root_ids=root_ids,
            leaf_encoder=leaf_encoder,
            merge_encoder=merge_encoder,
        )

    def _run_core(
        self,
        *,
        tokens: PackedNodeTokens,
        leaves: PackedLeafPoints,
        node_ptr: Tensor,
        root_ids: Tensor,
        leaf_encoder: LeafEncoder,
        merge_encoder: MergeEncoder,
    ) -> BottomUpResult:
        """
        Shared core for single/batch run.

        Preconditions:
          - depths must be consistent with parent/child (child depth > parent depth)
          - leaves.leaf_node_id must list all leaf nodes (unique) that are is_leaf=True
        """
        device = tokens.tree_parent_index.device
        total_M = int(tokens.tree_parent_index.numel())

        # Allocate output latent tensor; dimension d is inferred from encoder output at runtime.
        # We first compute leaves at max depth, then infer d.
        computed = torch.zeros((total_M,), dtype=torch.bool, device=device)

        # Map leaf node id -> row in leaves arrays
        leaf_row_for_node = _build_leaf_row_for_node(total_M, leaves.leaf_node_id)

        # Depth ordering: process nodes by descending depth
        depth = tokens.tree_node_depth
        max_depth = int(depth.max().item()) if depth.numel() > 0 else 0

        # Helper: gather packed token fields for a set of node ids
        def gather_node_fields(nids: Tensor) -> Dict[str, Tensor]:
            return {
                "node_feat_rel": tokens.tree_node_feat_rel[nids],
                "node_depth": tokens.tree_node_depth[nids],
                "iface_feat6": tokens.iface_feat6[nids],
                "iface_mask": tokens.iface_mask[nids],
                "iface_boundary_dir": tokens.iface_boundary_dir[nids],
                "iface_inside_endpoint": tokens.iface_inside_endpoint[nids],
                "iface_inter_rel_xy": tokens.iface_inter_rel_xy[nids],
                "iface_inside_quadrant": tokens.iface_inside_quadrant[nids],
                "cross_feat6": tokens.cross_feat6[nids],
                "cross_mask": tokens.cross_mask[nids],
                "cross_child_pair": tokens.cross_child_pair[nids],
                "cross_is_leaf_internal": tokens.cross_is_leaf_internal[nids],
            }

        z: Optional[Tensor] = None
        aux: Dict[str, Tensor] = {}

        # We process depth by depth, from deepest to root
        for d in range(max_depth, -1, -1):
            nids_at_d = torch.nonzero(depth == d, as_tuple=False).flatten()
            if nids_at_d.numel() == 0:
                continue

            # Split leaves and internal nodes
            is_leaf_d = tokens.is_leaf[nids_at_d]
            leaf_nids = nids_at_d[is_leaf_d]
            internal_nids = nids_at_d[~is_leaf_d]

            # 1) Leaf encoding
            if leaf_nids.numel() > 0:
                rows = leaf_row_for_node[leaf_nids]
                if (rows < 0).any().item():
                    bad = leaf_nids[rows < 0][:20].tolist()
                    raise ValueError(f"Missing leaf rows for some leaf nodes (example up to 20): {bad}")

                leaf_inputs = gather_node_fields(leaf_nids)
                leaf_inputs.update({
                    "leaf_points_xy": leaves.point_xy[rows],
                    "leaf_points_mask": leaves.point_mask[rows],
                })

                out = leaf_encoder(**leaf_inputs)
                z_leaf, aux_leaf = _extract_z(out)
                if z is None:
                    # initialize z with correct latent dim
                    z = torch.zeros((total_M, z_leaf.shape[1]), dtype=z_leaf.dtype, device=device)
                z[leaf_nids] = z_leaf
                computed[leaf_nids] = True

                # record aux stats (optional)
                if len(aux_leaf) > 0:
                    aux[f"leaf_depth_{d}"] = torch.tensor(1, device=device)

            # 2) Internal merge encoding
            if internal_nids.numel() > 0:
                if z is None:
                    raise RuntimeError("Internal node encountered before any leaf was encoded; check depth consistency.")

                ch = tokens.tree_children_index[internal_nids]  # [B,4]
                child_mask = ch >= 0
                # gather child z; for invalid children, use zeros
                child_z = torch.zeros((internal_nids.numel(), 4, z.shape[1]), dtype=z.dtype, device=device)
                valid_pos = torch.nonzero(child_mask, as_tuple=False)
                if valid_pos.numel() > 0:
                    b_idx = valid_pos[:, 0]
                    q_idx = valid_pos[:, 1]
                    child_ids = ch[b_idx, q_idx]
                    if not computed[child_ids].all().item():
                        missing = child_ids[~computed[child_ids]][:20].tolist()
                        raise RuntimeError(
                            f"Some children not computed before parent at depth {d}. "
                            f"Example missing child nids (up to 20): {missing}"
                        )
                    child_z[b_idx, q_idx] = z[child_ids]

                merge_inputs = gather_node_fields(internal_nids)
                merge_inputs.update({
                    "child_z": child_z,
                    "child_mask": child_mask,
                })

                out = merge_encoder(**merge_inputs)
                z_parent, aux_merge = _extract_z(out)
                z[internal_nids] = z_parent
                computed[internal_nids] = True

                if len(aux_merge) > 0:
                    aux[f"merge_depth_{d}"] = torch.tensor(1, device=device)

        if z is None:
            # Degenerate: empty graph
            z = torch.zeros((total_M, 1), dtype=torch.float32, device=device)

        if self.validate_completeness:
            if not computed.all().item():
                missing = torch.nonzero(~computed, as_tuple=False).flatten()[:50].tolist()
                raise RuntimeError(f"Bottom-up incomplete: missing z for node ids (up to 50): {missing}")

        return BottomUpResult(
            z=z,
            computed=computed,
            root_ids=root_ids,
            node_ptr=node_ptr,
            aux=aux,
        )
