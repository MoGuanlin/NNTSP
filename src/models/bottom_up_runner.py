# src/models/bottom_up_runner.py
# -*- coding: utf-8 -*-
"""
Bottom-up Tree Runner for Neural DP (Rao'98-style TSP).

IMPORTANT (training):
- Do NOT write encoder outputs into a preallocated tensor via in-place assignment like:
    z[nids] = z_leaf
  This breaks autograd connectivity (returned z won't require grad).

- Instead, use *functional* index_copy / scatter so that z has a grad_fn and
  gradients flow back to leaf/merge encoders.

This file is a drop-in replacement of your previous runner, but makes the
write-back differentiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple, Union, runtime_checkable

import torch
from torch import Tensor

try:
    from src.models.node_token_packer import PackedBatch, PackedLeafPoints, PackedNodeTokens
except Exception:  # pragma: no cover
    from .node_token_packer import PackedBatch, PackedLeafPoints, PackedNodeTokens


@runtime_checkable
class LeafEncoder(Protocol):
    def __call__(
        self,
        *,
        node_feat_rel: Tensor,
        node_depth: Tensor,
        iface_feat6: Tensor,
        iface_mask: Tensor,
        iface_boundary_dir: Tensor,
        iface_inside_endpoint: Tensor,
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
    def __call__(
        self,
        *,
        node_feat_rel: Tensor,
        node_depth: Tensor,
        iface_feat6: Tensor,
        iface_mask: Tensor,
        iface_boundary_dir: Tensor,
        iface_inside_endpoint: Tensor,
        iface_inside_quadrant: Tensor,
        cross_feat6: Tensor,
        cross_mask: Tensor,
        cross_child_pair: Tensor,
        cross_is_leaf_internal: Tensor,
        child_z: Tensor,
        child_mask: Tensor,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        ...


@dataclass
class BottomUpResult:
    z: Tensor
    computed: Tensor
    root_ids: Tensor
    node_ptr: Tensor
    aux: Dict[str, Tensor]


def _extract_z(out: Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]) -> Tuple[Tensor, Dict[str, Tensor]]:
    if torch.is_tensor(out):
        return out, {}
    if isinstance(out, tuple) and len(out) == 2 and torch.is_tensor(out[0]) and isinstance(out[1], dict):
        return out[0], out[1]
    raise TypeError("Encoder output must be Tensor or (Tensor, dict).")


def _build_leaf_row_for_node(total_M: int, leaf_node_id: Tensor) -> Tensor:
    device = leaf_node_id.device
    leaf_row_for_node = torch.full((total_M,), -1, dtype=torch.long, device=device)
    if leaf_node_id.numel() == 0:
        return leaf_row_for_node
    uniq = torch.unique(leaf_node_id)
    if uniq.numel() != leaf_node_id.numel():
        raise ValueError("leaves.leaf_node_id contains duplicates; expected unique ids.")
    leaf_row_for_node[leaf_node_id] = torch.arange(leaf_node_id.numel(), device=device, dtype=torch.long)
    return leaf_row_for_node


def _find_roots_per_graph(tree_parent_index: Tensor, node_ptr: Tensor) -> Tensor:
    device = tree_parent_index.device
    B = int(node_ptr.numel() - 1)
    root_ids = torch.empty((B,), dtype=torch.long, device=device)
    for b in range(B):
        lo = int(node_ptr[b].item())
        hi = int(node_ptr[b + 1].item())
        seg = tree_parent_index[lo:hi]
        roots = torch.nonzero(seg < 0, as_tuple=False).flatten()
        if roots.numel() != 1:
            raise ValueError(f"Expected 1 root in graph {b}, found {int(roots.numel())}.")
        root_ids[b] = lo + roots[0]
    return root_ids


class BottomUpTreeRunner:
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
        M = int(tokens.tree_parent_index.numel())
        device = tokens.tree_parent_index.device
        node_ptr = torch.tensor([0, M], dtype=torch.long, device=device)

        # packer 已经算好并校验过 root_id；这里直接使用
        root_ids = tokens.root_id.reshape(-1).to(dtype=torch.long)
        if root_ids.numel() != 1:
            raise ValueError(f"Expected 1 root_id for single graph, got shape {tuple(root_ids.shape)}")
        if tokens.tree_parent_index[root_ids[0]].item() >= 0:
            raise ValueError("tokens.root_id does not point to a root (tree_parent_index[root_id] must be < 0).")

        return self._run_core(tokens=tokens, leaves=leaves, node_ptr=node_ptr, root_ids=root_ids,
                            leaf_encoder=leaf_encoder, merge_encoder=merge_encoder)


    def run_batch(
        self,
        *,
        batch: PackedBatch,
        leaf_encoder: LeafEncoder,
        merge_encoder: MergeEncoder,
    ) -> BottomUpResult:
        tokens = batch.tokens
        leaves = batch.leaves
        node_ptr = batch.node_ptr

        # packer 已经算好并校验过每张图的 root_id（形状应为 [B]）
        root_ids = tokens.root_id.reshape(-1).to(dtype=torch.long)
        B = int(node_ptr.numel() - 1)
        if root_ids.numel() != B:
            raise ValueError(f"Expected root_id shape [B]={B}, got {tuple(root_ids.shape)}")

        # 轻量校验：每个 root 必须落在自己的图的 node 段里
        lo = node_ptr[:-1]
        hi = node_ptr[1:]
        if not (((root_ids >= lo) & (root_ids < hi)).all().item()):
            raise ValueError("tokens.root_id contains ids outside their graph segments.")
        if not ((tokens.tree_parent_index[root_ids] < 0).all().item()):
            raise ValueError("tokens.root_id contains non-root ids (parent must be < 0).")

        return self._run_core(tokens=tokens, leaves=leaves, node_ptr=node_ptr, root_ids=root_ids,
                            leaf_encoder=leaf_encoder, merge_encoder=merge_encoder)

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
        device = tokens.tree_parent_index.device
        total_M = int(tokens.tree_parent_index.numel())
        computed = torch.zeros((total_M,), dtype=torch.bool, device=device)

        leaf_row_for_node = _build_leaf_row_for_node(total_M, leaves.leaf_node_id)

        depth = tokens.tree_node_depth.long()
        max_depth = int(depth.max().item()) if depth.numel() > 0 else 0

        def gather_node_fields(nids: Tensor) -> Dict[str, Tensor]:
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

        z: Optional[Tensor] = None
        aux: Dict[str, Tensor] = {}

        for d in range(max_depth, -1, -1):
            nids_at_d = torch.nonzero(depth == d, as_tuple=False).flatten()
            if nids_at_d.numel() == 0:
                continue

            is_leaf_d = tokens.is_leaf[nids_at_d]
            leaf_nids = nids_at_d[is_leaf_d]
            internal_nids = nids_at_d[~is_leaf_d]

            # 1) Leaves
            if leaf_nids.numel() > 0:
                rows = leaf_row_for_node[leaf_nids]
                if (rows < 0).any().item():
                    bad = leaf_nids[rows < 0][:20].tolist()
                    raise ValueError(f"Missing leaf rows for leaf nodes (up to 20): {bad}")

                leaf_inputs = gather_node_fields(leaf_nids)
                leaf_inputs.update({
                    "leaf_points_xy": leaves.point_xy[rows],
                    "leaf_points_mask": leaves.point_mask[rows],
                })
                out = leaf_encoder(**leaf_inputs)
                z_leaf, aux_leaf = _extract_z(out)

                if z is None:
                    z = torch.zeros((total_M, z_leaf.shape[1]), dtype=z_leaf.dtype, device=device)

                # DIFFERENTIABLE write-back
                z = z.index_copy(0, leaf_nids, z_leaf)
                computed[leaf_nids] = True

                if aux_leaf:
                    aux[f"leaf_depth_{d}"] = torch.tensor(1, device=device)

            # 2) Internal nodes
            if internal_nids.numel() > 0:
                if z is None:
                    raise RuntimeError("Internal node encountered before any leaf was encoded.")

                ch = tokens.tree_children_index[internal_nids].long()  # [B,4]
                child_mask = ch >= 0                                  # [B,4]
                ch_clamped = ch.clamp_min(0)

                # DIFFERENTIABLE gather (no in-place into child_z)
                child_z = z[ch_clamped]  # [B,4,d]
                child_z = child_z * child_mask.unsqueeze(-1).to(dtype=child_z.dtype)

                # Sanity: ensure required children already computed
                if not computed[ch_clamped[child_mask]].all().item():
                    missing = ch_clamped[child_mask][~computed[ch_clamped[child_mask]]][:20].tolist()
                    raise RuntimeError(f"Some children not computed before parent at depth {d}. Example: {missing}")

                merge_inputs = gather_node_fields(internal_nids)
                merge_inputs.update({
                    "child_z": child_z,
                    "child_mask": child_mask,
                })

                out = merge_encoder(**merge_inputs)
                z_parent, aux_merge = _extract_z(out)

                # DIFFERENTIABLE write-back
                z = z.index_copy(0, internal_nids, z_parent)
                computed[internal_nids] = True

                if aux_merge:
                    aux[f"merge_depth_{d}"] = torch.tensor(1, device=device)

        if z is None:
            z = torch.zeros((total_M, 1), dtype=torch.float32, device=device)

        if self.validate_completeness and (not computed.all().item()):
            missing = torch.nonzero(~computed, as_tuple=False).flatten()[:50].tolist()
            raise RuntimeError(f"Bottom-up incomplete: missing nodes (up to 50): {missing}")

        return BottomUpResult(
            z=z,
            computed=computed,
            root_ids=root_ids,
            node_ptr=node_ptr,
            aux=aux,
        )


__all__ = ["BottomUpTreeRunner", "BottomUpResult"]
