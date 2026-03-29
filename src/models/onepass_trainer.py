# src/models/onepass_trainer.py
# -*- coding: utf-8 -*-
"""
Training-time 1-pass bottom-up runner with σ-conditioned MergeDecoder.

Unlike the inference-time OnePassDPRunner (which enumerates σ candidates and builds
cost tables), this module runs a single forward pass per internal node using the
**teacher σ** from the labeler.  The output is differentiable child activation
logits [M, 4, Ti] that can be trained with standard BCE loss against y_child_iface.

Flow (per internal node):
  1. MergeEncoder(children_z) → z_parent          (shared with 2-pass)
  2. MergeDecoder.build_parent_memory(z_parent, children_z, tokens)
  3. MergeDecoder.decode_sigma(teacher_σ)  → child_scores [1, 4, Ti]
  4. Collect into output tensor for loss computation

The bottom-up encoding part (leaf + merge) reuses BottomUpTreeRunner logic with
differentiable index_copy write-back.

IMPORTANT: This is a NEW module for the 1-pass DP training. It does NOT modify
or replace any existing 2-pass training code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from .bc_state_catalog import BoundaryStateCatalog
from .merge_decoder import MergeDecoder, ParentMemory
from .node_token_packer import PackedBatch, PackedLeafPoints, PackedNodeTokens


# ─── Output ──────────────────────────────────────────────────────────────────

@dataclass
class OnePassTrainResult:
    """Output of the 1-pass training forward pass.

    z:                 [M, d_model]  — node embeddings (differentiable)
    child_scores:      [M, 4, Ti]   — predicted child activation logits (differentiable)
    decode_mask:       [M] bool     — which internal nodes had valid teacher σ
    """
    z: Tensor
    child_scores: Tensor
    decode_mask: Tensor


# ─── Helper ──────────────────────────────────────────────────────────────────

def _extract_z(out):
    if torch.is_tensor(out):
        return out
    if isinstance(out, tuple) and len(out) >= 1 and torch.is_tensor(out[0]):
        return out[0]
    raise TypeError("Encoder output must be Tensor or (Tensor, dict).")


# ─── Main Runner ─────────────────────────────────────────────────────────────

class OnePassTrainRunner:
    """Training-time 1-pass runner: bottom-up encoding + σ-conditioned decoding.

    Usage:
        runner = OnePassTrainRunner()
        result = runner.run_batch(
            batch=packed_batch,
            leaf_encoder=leaf_enc,
            merge_encoder=merge_enc,
            merge_decoder=merge_dec,
            catalog=catalog,
            target_state_idx=labels.target_state_idx,   # [M] long
            m_state=labels.m_state,                      # [M] bool
        )
        # result.child_scores: [M, 4, Ti] — use with bc_child_iface_losses
    """

    def run_batch(
        self,
        *,
        batch: PackedBatch,
        leaf_encoder,
        merge_encoder,
        merge_decoder: MergeDecoder,
        catalog: BoundaryStateCatalog,
        target_state_idx: Tensor,   # [M] long — teacher state per node (-1 = skip)
        m_state: Tensor,            # [M] bool — which nodes have valid teacher state
                                    # IMPORTANT: pass m_state_exact here to avoid
                                    # inconsistent σ ↔ y_child_iface pairs (Bug 1 fix)
    ) -> OnePassTrainResult:
        """Run differentiable 1-pass forward on a packed batch."""
        tokens = batch.tokens
        leaves = batch.leaves
        node_ptr = batch.node_ptr

        return self._run_core(
            tokens=tokens,
            leaves=leaves,
            node_ptr=node_ptr,
            leaf_encoder=leaf_encoder,
            merge_encoder=merge_encoder,
            merge_decoder=merge_decoder,
            catalog=catalog,
            target_state_idx=target_state_idx,
            m_state=m_state,
        )

    def run_single(
        self,
        *,
        tokens: PackedNodeTokens,
        leaves: PackedLeafPoints,
        leaf_encoder,
        merge_encoder,
        merge_decoder: MergeDecoder,
        catalog: BoundaryStateCatalog,
        target_state_idx: Tensor,
        m_state: Tensor,
    ) -> OnePassTrainResult:
        """Run on a single graph (no batch packing)."""
        M = int(tokens.tree_parent_index.numel())
        device = tokens.tree_parent_index.device
        node_ptr = torch.tensor([0, M], dtype=torch.long, device=device)

        return self._run_core(
            tokens=tokens,
            leaves=leaves,
            node_ptr=node_ptr,
            leaf_encoder=leaf_encoder,
            merge_encoder=merge_encoder,
            merge_decoder=merge_decoder,
            catalog=catalog,
            target_state_idx=target_state_idx,
            m_state=m_state,
        )

    def _run_core(
        self,
        *,
        tokens: PackedNodeTokens,
        leaves: PackedLeafPoints,
        node_ptr: Tensor,
        leaf_encoder,
        merge_encoder,
        merge_decoder: MergeDecoder,
        catalog: BoundaryStateCatalog,
        target_state_idx: Tensor,
        m_state: Tensor,
    ) -> OnePassTrainResult:
        device = tokens.tree_parent_index.device
        total_M = int(tokens.tree_parent_index.numel())
        Ti = int(tokens.iface_mask.shape[1])

        # Build leaf_row_for_node
        leaf_row_for_node = torch.full((total_M,), -1, dtype=torch.long, device=device)
        if leaves.leaf_node_id.numel() > 0:
            leaf_row_for_node[leaves.leaf_node_id] = torch.arange(
                leaves.leaf_node_id.numel(), device=device, dtype=torch.long,
            )

        depth = tokens.tree_node_depth.long()
        max_depth = int(depth.max().item()) if depth.numel() > 0 else 0

        # Pre-transfer catalog to device (avoid per-node transfers)
        cat_used = catalog.used_iface.float().to(device)   # [S, Ti]
        cat_mate = catalog.mate.to(device)                 # [S, Ti]

        # Output tensors
        z: Optional[Tensor] = None
        # child_scores accumulates per-depth batched (nids, scores) pairs
        child_scores_list = []  # [(nids_tensor, scores_tensor), ...]
        decode_mask = torch.zeros(total_M, dtype=torch.bool, device=device)

        computed = torch.zeros(total_M, dtype=torch.bool, device=device)

        # ─── Bottom-up traversal ─────────────────────────────────────────
        for d in range(max_depth, -1, -1):
            nids_at_d = torch.nonzero(depth == d, as_tuple=False).flatten()
            if nids_at_d.numel() == 0:
                continue

            is_leaf_d = tokens.is_leaf[nids_at_d]
            leaf_nids = nids_at_d[is_leaf_d]
            internal_nids = nids_at_d[~is_leaf_d]

            # ── Leaves: encode ────────────────────────────────────────
            if leaf_nids.numel() > 0:
                rows = leaf_row_for_node[leaf_nids]
                leaf_inputs = self._gather_node_fields(tokens, leaf_nids)
                leaf_inputs["leaf_points_xy"] = leaves.point_xy[rows]
                leaf_inputs["leaf_points_mask"] = leaves.point_mask[rows]

                z_leaf = _extract_z(leaf_encoder(**leaf_inputs))

                if z is None:
                    z = torch.zeros(total_M, z_leaf.shape[1], dtype=z_leaf.dtype, device=device)
                elif z.dtype != z_leaf.dtype:
                    z = z.to(z_leaf.dtype)

                # Differentiable write-back
                z = z.index_copy(0, leaf_nids, z_leaf)
                computed[leaf_nids] = True

            # ── Internal nodes: encode + decode teacher σ ─────────────
            if internal_nids.numel() > 0:
                if z is None:
                    raise RuntimeError("Internal node before any leaf.")

                # Batched merge encoding
                ch = tokens.tree_children_index[internal_nids].long()
                child_mask_batch = ch >= 0
                ch_clamped = ch.clamp_min(0)
                child_z_batch = z[ch_clamped]
                child_z_batch = child_z_batch * child_mask_batch.unsqueeze(-1).float()

                merge_inputs = self._gather_node_fields(tokens, internal_nids)
                merge_inputs["child_z"] = child_z_batch
                merge_inputs["child_mask"] = child_mask_batch

                z_parent = _extract_z(merge_encoder(**merge_inputs))
                z = z.index_copy(0, internal_nids, z_parent)
                computed[internal_nids] = True

                # ── Batched σ-conditioned decoding ──────────────────
                # Filter to nodes with valid teacher state (no Python loop)
                valid_mask = m_state[internal_nids] & (target_state_idx[internal_nids] >= 0)
                valid_nids = internal_nids[valid_mask]

                if valid_nids.numel() > 0:
                    # Gather teacher sigma for all valid nodes at once
                    si = target_state_idx[valid_nids]                    # [K] long
                    sigma_a = cat_used[si]                               # [K, Ti] float
                    sigma_mate = cat_mate[si]                            # [K, Ti] long

                    # Gather children info (batched)
                    ch_v = tokens.tree_children_index[valid_nids].long() # [K, 4]
                    child_exists_v = ch_v >= 0                           # [K, 4]
                    ch_v_clamped = ch_v.clamp_min(0)                     # [K, 4]

                    # Build parent memory (batched, K nodes at once)
                    # child_z: [K, 4, d] — gather from z
                    child_z_v = z[ch_v_clamped.view(-1)].view(valid_nids.numel(), 4, -1)
                    child_z_v = child_z_v * child_exists_v.unsqueeze(-1).float()

                    parent_mem = merge_decoder.build_parent_memory(
                        node_feat_rel=tokens.tree_node_feat_rel[valid_nids],
                        node_depth=tokens.tree_node_depth[valid_nids],
                        z_node=z[valid_nids],
                        iface_feat6=tokens.iface_feat6[valid_nids],
                        iface_mask=tokens.iface_mask[valid_nids],
                        iface_boundary_dir=tokens.iface_boundary_dir[valid_nids],
                        iface_inside_endpoint=tokens.iface_inside_endpoint[valid_nids],
                        iface_inside_quadrant=tokens.iface_inside_quadrant[valid_nids],
                        cross_feat6=tokens.cross_feat6[valid_nids],
                        cross_mask=tokens.cross_mask[valid_nids],
                        cross_child_pair=tokens.cross_child_pair[valid_nids],
                        cross_is_leaf_internal=tokens.cross_is_leaf_internal[valid_nids],
                        child_z=child_z_v,
                        child_exists_mask=child_exists_v,
                    )

                    # Build child iface mask: [K, 4, Ti]
                    # For each valid node, gather iface_mask of its 4 children
                    child_iface_mask = tokens.iface_mask[ch_v_clamped.view(-1)].bool()
                    child_iface_mask = child_iface_mask.view(valid_nids.numel(), 4, Ti)
                    child_iface_mask = child_iface_mask & child_exists_v.unsqueeze(-1)

                    # Decode all sigmas in one batched call
                    out = merge_decoder.decode_sigma(
                        sigma_a=sigma_a,
                        sigma_mate=sigma_mate,
                        sigma_iface_mask=tokens.iface_mask[valid_nids].bool(),
                        parent_memory=parent_mem,
                        child_iface_mask=child_iface_mask,
                    )
                    # out.child_scores: [K, 4, Ti]

                    child_scores_list.append((valid_nids, out.child_scores))
                    decode_mask[valid_nids] = True

        if z is None:
            z = torch.zeros(total_M, 1, dtype=torch.float32, device=device)

        # Assemble child_scores [M, 4, Ti]
        child_scores = torch.zeros(total_M, 4, Ti, dtype=torch.float32, device=device)
        if child_scores_list:
            # Each entry is (nids_batch [K], scores_batch [K, 4, Ti])
            all_nids = torch.cat([n for n, _ in child_scores_list])
            all_scores = torch.cat([s for _, s in child_scores_list]).float()
            child_scores = child_scores.index_copy(0, all_nids, all_scores)

        return OnePassTrainResult(
            z=z,
            child_scores=child_scores,
            decode_mask=decode_mask,
        )

    @staticmethod
    def _gather_node_fields(tokens: PackedNodeTokens, nids: Tensor) -> Dict[str, Tensor]:
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


# ─── Loss helper ─────────────────────────────────────────────────────────────

def onepass_loss(
    *,
    child_scores: Tensor,       # [M, 4, Ti] from OnePassTrainResult
    decode_mask: Tensor,        # [M] bool
    y_child_iface: Tensor,      # [M, 4, Ti] from TokenLabels
    m_child_iface: Tensor,      # [M, 4, Ti] bool from TokenLabels
    pos_weight: Optional[float] = None,
) -> Tensor:
    """Compute BCE loss for 1-pass σ-conditioned child predictions.

    Only nodes with decode_mask=True contribute to the loss.
    This combines the decode_mask with the existing m_child_iface mask.
    """
    # Expand decode_mask to [M, 4, Ti]
    combined_mask = m_child_iface.bool() & decode_mask.unsqueeze(-1).unsqueeze(-1)

    if not combined_mask.any().item():
        return child_scores.new_tensor(0.0)

    import torch.nn.functional as F

    logit = child_scores[combined_mask]
    target = y_child_iface[combined_mask].float()

    if pos_weight is not None:
        pw = child_scores.new_tensor(float(pos_weight))
        loss = F.binary_cross_entropy_with_logits(logit, target, reduction="mean", pos_weight=pw)
    else:
        loss = F.binary_cross_entropy_with_logits(logit, target, reduction="mean")

    return loss


def build_child_iface_targets_from_states(
    *,
    tree_children_index: Tensor,   # [M, 4]
    m_child_iface: Tensor,         # [M, 4, Ti] bool
    target_state_idx: Tensor,      # [M] long
    child_state_mask: Tensor,      # [M] bool
    state_used_iface: Tensor,      # [S, Ti] bool
) -> Tuple[Tensor, Tensor]:
    """Build child-interface activation targets from catalog-aligned child states.

    This supervision is aligned with the actual DP state space: each child target
    is the activation vector of that child's discrete boundary state, rather than
    the raw teacher edge mask that may fall outside the capped catalog.
    """
    device = tree_children_index.device
    ch = tree_children_index.long()
    exists = ch >= 0
    ch0 = ch.clamp_min(0)

    child_state_ok = child_state_mask[ch0] & exists
    child_target_idx = target_state_idx[ch0]
    safe_idx = child_target_idx.clamp_min(0)
    used_table = state_used_iface.to(device=device)
    y_child = used_table[safe_idx].to(dtype=torch.float32)

    valid_child = child_state_ok & (child_target_idx >= 0)
    m_child = m_child_iface.bool() & valid_child.unsqueeze(-1)
    y_child = y_child * m_child.to(dtype=y_child.dtype)
    return y_child, m_child


__all__ = [
    "OnePassTrainRunner",
    "OnePassTrainResult",
    "onepass_loss",
    "build_child_iface_targets_from_states",
]
