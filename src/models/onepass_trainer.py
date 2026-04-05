# src/models/onepass_trainer.py
# -*- coding: utf-8 -*-
"""Training-time 1-pass bottom-up runner with sigma-conditioned MergeDecoder.

Two supervision modes are supported in parallel:

  - `catalog_index`: legacy path using teacher `target_state_idx`
  - `direct_structured`: uncapped path using direct `(used, mate)` parent sigma

Unlike the inference-time OnePassDPRunner (which enumerates sigma candidates and
builds cost tables), this module runs a single forward pass per internal node
using teacher sigma labels from the labeler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from .bc_state_catalog import BoundaryStateCatalog
from .merge_decoder import MergeDecoder
from .node_token_packer import PackedBatch, PackedLeafPoints, PackedNodeTokens
from .shared_tree import extract_z, gather_node_fields


# ─── Output ──────────────────────────────────────────────────────────────────

@dataclass
class OnePassTrainResult:
    """Output of the 1-pass training forward pass.

    z:                 [M, d_model]  — node embeddings (differentiable)
    child_scores:      [M, 4, Ti]   — predicted child activation logits (differentiable)
    child_mate_scores: optional [M, 4, Ti, Ti] — predicted mate logits
    decode_mask:       [M] bool     — which internal nodes had valid teacher σ
    """
    z: Tensor
    child_scores: Tensor
    child_mate_scores: Optional[Tensor]
    decode_mask: Tensor

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
        catalog: Optional[BoundaryStateCatalog] = None,
        target_state_idx: Optional[Tensor] = None,   # [M] long — legacy teacher state per node
        m_state: Optional[Tensor] = None,            # [M] bool — legacy valid teacher-state mask
        parent_sigma_used: Optional[Tensor] = None,  # [M, Ti] bool — direct structured parent sigma
        parent_sigma_mate: Optional[Tensor] = None,  # [M, Ti] long
        supervision_mode: str = "catalog_index",
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
            parent_sigma_used=parent_sigma_used,
            parent_sigma_mate=parent_sigma_mate,
            supervision_mode=supervision_mode,
        )

    def run_single(
        self,
        *,
        tokens: PackedNodeTokens,
        leaves: PackedLeafPoints,
        leaf_encoder,
        merge_encoder,
        merge_decoder: MergeDecoder,
        catalog: Optional[BoundaryStateCatalog] = None,
        target_state_idx: Optional[Tensor] = None,
        m_state: Optional[Tensor] = None,
        parent_sigma_used: Optional[Tensor] = None,
        parent_sigma_mate: Optional[Tensor] = None,
        supervision_mode: str = "catalog_index",
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
            parent_sigma_used=parent_sigma_used,
            parent_sigma_mate=parent_sigma_mate,
            supervision_mode=supervision_mode,
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
        catalog: Optional[BoundaryStateCatalog],
        target_state_idx: Optional[Tensor],
        m_state: Optional[Tensor],
        parent_sigma_used: Optional[Tensor],
        parent_sigma_mate: Optional[Tensor],
        supervision_mode: str,
    ) -> OnePassTrainResult:
        device = tokens.tree_parent_index.device
        total_M = int(tokens.tree_parent_index.numel())
        Ti = int(tokens.iface_mask.shape[1])
        mode = str(supervision_mode)
        if mode not in ("catalog_index", "direct_structured"):
            raise ValueError(
                f"supervision_mode must be 'catalog_index' or 'direct_structured', got {mode!r}"
            )

        # Build leaf_row_for_node
        leaf_row_for_node = torch.full((total_M,), -1, dtype=torch.long, device=device)
        if leaves.leaf_node_id.numel() > 0:
            leaf_row_for_node[leaves.leaf_node_id] = torch.arange(
                leaves.leaf_node_id.numel(), device=device, dtype=torch.long,
            )

        depth = tokens.tree_node_depth.long()
        max_depth = int(depth.max().item()) if depth.numel() > 0 else 0

        cat_used = None
        cat_mate = None
        if mode == "catalog_index":
            if catalog is None or target_state_idx is None or m_state is None:
                raise ValueError("catalog_index supervision requires catalog, target_state_idx and m_state")
            cat_used = catalog.used_iface.float().to(device)   # [S, Ti]
            cat_mate = catalog.mate.to(device)                 # [S, Ti]
        else:
            if parent_sigma_used is None or parent_sigma_mate is None or m_state is None:
                raise ValueError(
                    "direct_structured supervision requires parent_sigma_used, parent_sigma_mate and m_state"
                )

        # Output tensors
        z: Optional[Tensor] = None
        # child_scores accumulates per-depth batched (nids, scores) pairs
        child_scores_list = []  # [(nids_tensor, scores_tensor), ...]
        child_mate_scores_list = []  # [(nids_tensor, mate_scores_tensor), ...]
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
                leaf_inputs = gather_node_fields(tokens, leaf_nids)
                leaf_inputs["leaf_points_xy"] = leaves.point_xy[rows]
                leaf_inputs["leaf_points_mask"] = leaves.point_mask[rows]

                z_leaf = extract_z(leaf_encoder(**leaf_inputs))[0]

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

                merge_inputs = gather_node_fields(tokens, internal_nids)
                merge_inputs["child_z"] = child_z_batch
                merge_inputs["child_mask"] = child_mask_batch

                z_parent = extract_z(merge_encoder(**merge_inputs))[0]
                z = z.index_copy(0, internal_nids, z_parent)
                computed[internal_nids] = True

                # ── Batched σ-conditioned decoding ──────────────────
                if mode == "catalog_index":
                    assert m_state is not None and target_state_idx is not None
                    valid_mask = m_state[internal_nids] & (target_state_idx[internal_nids] >= 0)
                else:
                    assert m_state is not None
                    valid_mask = m_state[internal_nids].bool()
                valid_nids = internal_nids[valid_mask]

                if valid_nids.numel() > 0:
                    # Gather teacher sigma for all valid nodes at once
                    if mode == "catalog_index":
                        assert target_state_idx is not None and cat_used is not None and cat_mate is not None
                        si = target_state_idx[valid_nids]                    # [K] long
                        sigma_a = cat_used[si]                               # [K, Ti] float
                        sigma_mate = cat_mate[si]                            # [K, Ti] long
                    else:
                        assert parent_sigma_used is not None and parent_sigma_mate is not None
                        sigma_a = parent_sigma_used[valid_nids].float()
                        sigma_mate = parent_sigma_mate[valid_nids].long()

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
                    if out.child_mate_scores is not None:
                        child_mate_scores_list.append((valid_nids, out.child_mate_scores))
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

        child_mate_scores: Optional[Tensor] = None
        if child_mate_scores_list:
            child_mate_scores = torch.zeros(
                total_M, 4, Ti, Ti, dtype=torch.float32, device=device
            )
            all_mate_nids = torch.cat([n for n, _ in child_mate_scores_list])
            all_mate_scores = torch.cat([s for _, s in child_mate_scores_list]).float()
            child_mate_scores = child_mate_scores.index_copy(0, all_mate_nids, all_mate_scores)

        return OnePassTrainResult(
            z=z,
            child_scores=child_scores,
            child_mate_scores=child_mate_scores,
            decode_mask=decode_mask,
        )
# ─── Loss helper ─────────────────────────────────────────────────────────────

def onepass_loss(
    *,
    child_scores: Tensor,       # [M, 4, Ti] from OnePassTrainResult
    decode_mask: Tensor,        # [M] bool
    y_child_iface: Tensor,      # [M, 4, Ti] from TokenLabels
    m_child_iface: Tensor,      # [M, 4, Ti] bool from TokenLabels
    pos_weight: Optional[float] = None,
    child_mate_scores: Optional[Tensor] = None,  # [M, 4, Ti, Ti]
    y_child_mate: Optional[Tensor] = None,       # [M, 4, Ti] long
    m_child_mate: Optional[Tensor] = None,       # [M, 4, Ti] bool
    mate_weight: float = 0.0,
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
        iface_loss = F.binary_cross_entropy_with_logits(logit, target, reduction="mean", pos_weight=pw)
    else:
        iface_loss = F.binary_cross_entropy_with_logits(logit, target, reduction="mean")

    loss = iface_loss
    use_mate = (
        child_mate_scores is not None
        and y_child_mate is not None
        and m_child_mate is not None
        and float(mate_weight) > 0.0
    )
    if use_mate:
        mate_mask = m_child_mate.bool() & decode_mask.unsqueeze(-1).unsqueeze(-1)
        if mate_mask.any().item():
            mate_logits = child_mate_scores[mate_mask]
            mate_target = y_child_mate[mate_mask].long()
            mate_loss = F.cross_entropy(mate_logits, mate_target, reduction="mean")
            loss = loss + child_scores.new_tensor(float(mate_weight)) * mate_loss

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


def build_child_mate_targets_from_states(
    *,
    tree_children_index: Tensor,   # [M, 4]
    m_child_iface: Tensor,         # [M, 4, Ti] bool
    target_state_idx: Tensor,      # [M] long
    child_state_mask: Tensor,      # [M] bool
    state_used_iface: Tensor,      # [S, Ti] bool
    state_mate: Tensor,            # [S, Ti] long
) -> Tuple[Tensor, Tensor]:
    """Build child mate supervision from catalog-aligned child states.

    Only active child slots from valid child states are supervised.
    """
    device = tree_children_index.device
    ch = tree_children_index.long()
    exists = ch >= 0
    ch0 = ch.clamp_min(0)

    child_state_ok = child_state_mask[ch0] & exists
    child_target_idx = target_state_idx[ch0]
    safe_idx = child_target_idx.clamp_min(0)
    used_table = state_used_iface.to(device=device).bool()
    mate_table = state_mate.to(device=device).long()

    child_used = used_table[safe_idx]
    y_child_mate = mate_table[safe_idx]
    valid_child = child_state_ok & (child_target_idx >= 0)
    m_child_mate = m_child_iface.bool() & valid_child.unsqueeze(-1) & child_used
    y_child_mate = torch.where(
        m_child_mate,
        y_child_mate,
        torch.zeros_like(y_child_mate),
    )
    return y_child_mate, m_child_mate


def build_child_mate_targets_from_structured_states(
    *,
    tree_children_index: Tensor,        # [M, 4]
    m_child_iface: Tensor,              # [M, 4, Ti] bool
    parent_sigma_used: Tensor,          # [M, Ti] bool
    parent_sigma_mate: Tensor,          # [M, Ti] long
    m_parent_sigma_structured: Tensor,  # [M] bool
) -> Tuple[Tensor, Tensor]:
    """Build child mate supervision by gathering direct structured child sigma."""
    device = tree_children_index.device
    ch = tree_children_index.long()
    exists = ch >= 0
    ch0 = ch.clamp_min(0)

    child_sigma_used = parent_sigma_used.to(device=device).bool()[ch0]
    child_sigma_mate = parent_sigma_mate.to(device=device).long()[ch0]
    child_structured_ok = m_parent_sigma_structured.to(device=device).bool()[ch0] & exists
    m_child_mate = m_child_iface.bool() & child_structured_ok.unsqueeze(-1) & child_sigma_used
    y_child_mate = torch.where(
        m_child_mate,
        child_sigma_mate,
        torch.zeros_like(child_sigma_mate),
    )
    return y_child_mate, m_child_mate


__all__ = [
    "OnePassTrainRunner",
    "OnePassTrainResult",
    "build_child_mate_targets_from_structured_states",
    "build_child_mate_targets_from_states",
    "onepass_loss",
    "build_child_iface_targets_from_states",
]
