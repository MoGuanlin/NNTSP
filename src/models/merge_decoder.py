# src/models/merge_decoder.py
# -*- coding: utf-8 -*-
"""
Sigma-conditioned merge decoder for the 1-pass DP pipeline.

For an internal box B with children (B_1, ..., B_4), this module:
  1. Builds a *parent memory* from parent tokens (CLS + IFACE + CROSS + CHILD latents)
     via self-attention — this is done ONCE per box.
  2. For each parent boundary state sigma in Omega(B), encodes sigma into a query
     embedding and cross-attends to the parent memory to predict continuous
     child-state 4-tuples:  tau_tilde(sigma) = (sigma_tilde_{B_1}, ..., sigma_tilde_{B_4}).

The continuous output is then discretized by PARSE (dp_core.py) and verified by
VERIFYTUPLE externally.  This module only handles the neural forward pass.

Architecture (aligned with paper Section 3.1, Figure 2):
  - Parent memory:  SelfAttn(CLS + IFACE + CROSS + CHILD_LATENT tokens + z_node)
  - Sigma query:    MLP(concat(a_vec, mate_onehot)) -> [d_model]
  - Cross-attn:     sigma_query attends to parent_memory
  - Child head:     linear -> [4, Ti] activation scores (sigmoid)

IMPORTANT: This is a NEW module for the 1-pass DP pipeline. It does NOT modify
or replace any existing 2-pass code (TopDownDecoder, TopDownRunner, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .set_transformer_block import FeedForward, SetTransformerBlock
from .tokenization import NodeTokenizer


@dataclass
class ParentMemory:
    """Cached parent memory for reuse across sigma queries.

    tokens:     [B, T, d_model]  — self-attended parent token sequence
    mask:       [B, T] bool      — valid token mask
    iface_slice: slice            — index range of parent IFACE tokens in tokens
    cross_slice: slice            — index range of parent CROSS tokens in tokens
    """
    tokens: Tensor
    mask: Tensor
    iface_slice: slice
    cross_slice: slice


@dataclass
class MergeDecoderOutput:
    """Output from sigma-conditioned decoding.

    child_scores: [B_sigma, 4, Ti] float — per-child, per-slot activation logits
                  (apply sigmoid to get probabilities)
    child_mate_scores: optional [B_sigma, 4, Ti, Ti] float — per-child mate
                  logits for "slot s prefers mate t"; only populated when the
                  decoder is built in `iface_mate` mode.
    """
    child_scores: Tensor
    child_mate_scores: Optional[Tensor] = None


class SigmaEncoder(nn.Module):
    """Encode a discrete boundary state sigma = (a, mate) into a d_model vector.

    Input:
      a:    [B, Ti] float/bool — activation vector
      mate: [B, Ti] long       — mate map (-1 = inactive)
      mask: [B, Ti] bool       — valid iface slots

    Output:
      sigma_emb: [B, d_model]
    """

    def __init__(self, *, num_iface_slots: int, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        # Per-slot features: activation (1) + mate one-hot (Ti) = 1 + Ti
        # We project per-slot, then pool.
        self.num_slots = num_iface_slots
        self.d_model = d_model

        slot_feat_dim = 1 + num_iface_slots  # activation + mate_onehot
        self.slot_proj = nn.Sequential(
            nn.Linear(slot_feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pool_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, a: Tensor, mate: Tensor, mask: Tensor) -> Tensor:
        """
        a:    [B, Ti] float
        mate: [B, Ti] long
        mask: [B, Ti] bool
        Returns: [B, d_model]
        """
        B, Ti = a.shape
        device = a.device

        # Build per-slot feature: [B, Ti, 1 + Ti]
        a_feat = a.unsqueeze(-1).float()  # [B, Ti, 1]

        # Mate one-hot: for each slot, one-hot encode its partner
        mate_clamped = mate.clamp(min=0)  # replace -1 with 0 for indexing
        mate_oh = torch.zeros(B, Ti, Ti, device=device, dtype=torch.float32)
        mate_oh.scatter_(2, mate_clamped.unsqueeze(-1), 1.0)
        # Zero out inactive slots (mate == -1)
        mate_active = (mate >= 0).unsqueeze(-1).float()
        mate_oh = mate_oh * mate_active

        slot_feat = torch.cat([a_feat, mate_oh], dim=-1)  # [B, Ti, 1+Ti]
        slot_emb = self.slot_proj(slot_feat)  # [B, Ti, d_model]

        # Masked mean pooling
        mask_f = mask.unsqueeze(-1).float()  # [B, Ti, 1]
        pooled = (slot_emb * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        return self.dropout(self.pool_proj(pooled))  # [B, d_model]


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention: query attends to key-value memory.

    q = q + MHA(LN(q), LN(kv), LN(kv))
    q = q + FFN(LN(q))
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        ff_hidden_mult: int = 4,
    ) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drop_attn = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, hidden_mult=ff_hidden_mult, dropout=dropout)
        self.drop_ff = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, q: Tensor, kv: Tensor, kv_mask: Tensor) -> Tensor:
        """
        q:       [B, Nq, d]  — query tokens
        kv:      [B, T, d]   — key-value memory
        kv_mask: [B, T] bool — True = valid
        Returns: [B, Nq, d]
        """
        yq = self.ln_q(q)
        ykv = self.ln_kv(kv)
        attn_out, _ = self.attn(
            query=yq, key=ykv, value=ykv,
            key_padding_mask=~kv_mask,
            need_weights=False,
        )
        q = q + self.drop_attn(attn_out)

        ff_out = self.ff(self.ln_ff(q))
        q = q + self.drop_ff(ff_out)
        return q


class MergeDecoder(nn.Module):
    """Sigma-conditioned decoder for the 1-pass DP merge step.

    Usage pattern:
        decoder = MergeDecoder(...)

        # 1) Build parent memory once per box
        mem = decoder.build_parent_memory(
            node_feat_rel=..., node_depth=..., z_node=...,
            iface_feat6=..., ..., cross_feat6=..., ...,
            child_z=..., child_exists_mask=...,
        )

        # 2) For each sigma, decode child states
        out = decoder.decode_sigma(
            sigma_a=..., sigma_mate=..., sigma_iface_mask=...,
            parent_memory=mem,
            child_iface_mask=...,
        )
        # out.child_scores: [B_sigma, 4, Ti]
    """

    def __init__(
        self,
        *,
        d_model: int = 128,
        n_heads: int = 8,
        num_iface_slots: int = 16,
        decoder_variant: str = "iface",
        parent_num_layers: int = 3,
        cross_num_layers: int = 2,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        ff_mult: int = 4,
        max_depth: int = 64,
        use_angle_sincos: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_iface_slots = num_iface_slots
        self.decoder_variant = str(decoder_variant)
        if self.decoder_variant not in ("iface", "iface_mate"):
            raise ValueError(
                f"decoder_variant must be 'iface' or 'iface_mate', got {self.decoder_variant!r}"
            )

        # --- Parent memory construction ---
        # Tokenizer for CLS + IFACE + CROSS + CHILD
        self.tokenizer = NodeTokenizer(
            d_model=d_model,
            max_depth=max_depth,
            use_angle_sincos=use_angle_sincos,
            node_ctx=True,
            type_embed=True,
        )

        # Inject z_node into CLS token
        self.z_node_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

        # Self-attention blocks for parent memory
        self.parent_blocks = nn.ModuleList([
            SetTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ff_hidden_mult=ff_mult,
            )
            for _ in range(parent_num_layers)
        ])

        # --- Sigma-conditioned decoding ---
        # Encode discrete sigma into query embedding
        self.sigma_encoder = SigmaEncoder(
            num_iface_slots=num_iface_slots,
            d_model=d_model,
            dropout=dropout,
        )

        # Cross-attention: sigma query -> parent memory
        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(
                d_model=d_model,
                n_heads=n_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ff_hidden_mult=ff_mult,
            )
            for _ in range(cross_num_layers)
        ])

        # Output head: project sigma-attended features to child activation scores
        # Output [4 * Ti] logits, reshaped to [4, Ti]
        self.child_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4 * num_iface_slots),
        )
        self.child_mate_head: Optional[nn.Module]
        if self.decoder_variant == "iface_mate":
            self.child_mate_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 4 * num_iface_slots * num_iface_slots),
            )
        else:
            self.child_mate_head = None

    def build_parent_memory(
        self,
        *,
        node_feat_rel: Tensor,              # [B, 4]
        node_depth: Tensor,                 # [B]
        z_node: Tensor,                     # [B, d_model]
        # parent iface tokens
        iface_feat6: Tensor,                # [B, Ti, 6]
        iface_mask: Tensor,                 # [B, Ti] bool
        iface_boundary_dir: Tensor,         # [B, Ti] long
        iface_inside_endpoint: Tensor,      # [B, Ti] long
        iface_inside_quadrant: Tensor,      # [B, Ti] long
        # parent crossing tokens
        cross_feat6: Tensor,                # [B, Tc, 6]
        cross_mask: Tensor,                 # [B, Tc] bool
        cross_child_pair: Tensor,           # [B, Tc, 2] long
        cross_is_leaf_internal: Tensor,     # [B, Tc] bool
        # children latents
        child_z: Tensor,                    # [B, 4, d_model]
        child_exists_mask: Tensor,          # [B, 4] bool
    ) -> ParentMemory:
        """Build self-attended parent memory. Call once per box, reuse for all sigma."""
        B = node_feat_rel.shape[0]

        # Tokenize: CLS + IFACE + CROSS + CHILD
        mem = self.tokenizer(
            node_feat_rel=node_feat_rel,
            node_depth=node_depth,
            iface_feat6=iface_feat6,
            iface_mask=iface_mask,
            iface_boundary_dir=iface_boundary_dir,
            iface_inside_endpoint=iface_inside_endpoint,
            iface_inside_quadrant=iface_inside_quadrant,
            cross_feat6=cross_feat6,
            cross_mask=cross_mask,
            cross_child_pair=cross_child_pair,
            cross_is_leaf_internal=cross_is_leaf_internal,
            child_z=child_z,
            child_mask=child_exists_mask,
        )

        tokens = mem.tokens       # [B, T, d]
        mask = mem.mask.bool()    # [B, T]

        # Inject z_node into CLS token
        cls_idx = mem.cls_index
        tokens[:, cls_idx, :] = tokens[:, cls_idx, :] + self.z_node_proj(z_node)

        # Self-attention
        for blk in self.parent_blocks:
            tokens = blk(tokens, mask)

        return ParentMemory(
            tokens=tokens,
            mask=mask,
            iface_slice=mem.iface_slice,
            cross_slice=mem.cross_slice,
        )

    def decode_sigma(
        self,
        *,
        sigma_a: Tensor,           # [B_sigma, Ti] float/bool — activation vector
        sigma_mate: Tensor,        # [B_sigma, Ti] long — mate map
        sigma_iface_mask: Tensor,  # [B_sigma, Ti] bool — valid parent iface slots
        parent_memory: ParentMemory,
        child_iface_mask: Tensor,  # [B_sigma, 4, Ti] bool — valid child iface slots
    ) -> MergeDecoderOutput:
        """Decode child states conditioned on parent state sigma.

        When processing multiple sigmas for the SAME parent box, the caller should:
          - Expand parent_memory.tokens and parent_memory.mask to match B_sigma
            (via .expand() or by passing multiple identical copies)
          - OR call this in a loop with B_sigma=1 each time

        For efficient batch processing of S sigmas for one parent box:
          sigma_a:          [S, Ti]
          sigma_mate:       [S, Ti]
          sigma_iface_mask: [S, Ti]  (same mask repeated S times)
          parent_memory.tokens: [S, T, d]  (same memory repeated S times)
          parent_memory.mask:   [S, T]     (same mask repeated S times)
          child_iface_mask: [S, 4, Ti]     (same mask repeated S times)
        """
        B_sigma = sigma_a.shape[0]
        Ti = sigma_a.shape[1]

        # Encode sigma into query: [B_sigma, d_model]
        sigma_emb = self.sigma_encoder(
            a=sigma_a.float(),
            mate=sigma_mate,
            mask=sigma_iface_mask.bool(),
        )

        # Reshape to [B_sigma, 1, d_model] for cross-attention
        q = sigma_emb.unsqueeze(1)

        # Cross-attend to parent memory
        kv = parent_memory.tokens
        kv_mask = parent_memory.mask

        for blk in self.cross_blocks:
            q = blk(q=q, kv=kv, kv_mask=kv_mask)

        # q: [B_sigma, 1, d_model] -> [B_sigma, d_model]
        q = q.squeeze(1)

        # Project to child activation scores: [B_sigma, 4 * Ti]
        raw = self.child_head(q)
        child_scores = raw.view(B_sigma, 4, Ti)

        # Mask invalid child iface slots with large negative value
        neg_inf = torch.tensor(-1e9, device=child_scores.device, dtype=child_scores.dtype)
        child_scores = torch.where(child_iface_mask.bool(), child_scores, neg_inf)

        child_mate_scores: Optional[Tensor] = None
        if self.child_mate_head is not None:
            mate_raw = self.child_mate_head(q).view(B_sigma, 4, Ti, Ti)
            src_mask = child_iface_mask.bool().unsqueeze(-1)
            dst_mask = child_iface_mask.bool().unsqueeze(-2)
            mate_mask = src_mask & dst_mask
            child_mate_scores = torch.where(mate_mask, mate_raw, neg_inf)

        return MergeDecoderOutput(
            child_scores=child_scores,
            child_mate_scores=child_mate_scores,
        )

    def decode_sigma_chunked(
        self,
        *,
        sigma_a: Tensor,           # [B_sigma, Ti] float/bool — activation vector
        sigma_mate: Tensor,        # [B_sigma, Ti] long — mate map
        sigma_iface_mask: Tensor,  # [B_sigma, Ti] bool — valid parent iface slots
        parent_memory: ParentMemory,
        child_iface_mask: Tensor,  # [B_sigma, 4, Ti] bool — valid child iface slots
        max_batch_size: Optional[int] = None,
    ) -> MergeDecoderOutput:
        """Decode sigma batches in chunks to avoid oversized attention launches.

        This is functionally equivalent to `decode_sigma(...)`, but splits the
        batch dimension when `max_batch_size` is set and smaller than B_sigma.
        """
        B_sigma = int(sigma_a.shape[0])
        if max_batch_size is None or max_batch_size <= 0 or B_sigma <= int(max_batch_size):
            return self.decode_sigma(
                sigma_a=sigma_a,
                sigma_mate=sigma_mate,
                sigma_iface_mask=sigma_iface_mask,
                parent_memory=parent_memory,
                child_iface_mask=child_iface_mask,
            )

        chunk_outputs = []
        chunk_mate_outputs = []
        chunk_size = int(max_batch_size)
        for start in range(0, B_sigma, chunk_size):
            end = min(start + chunk_size, B_sigma)
            out = self.decode_sigma(
                sigma_a=sigma_a[start:end],
                sigma_mate=sigma_mate[start:end],
                sigma_iface_mask=sigma_iface_mask[start:end],
                parent_memory=ParentMemory(
                    tokens=parent_memory.tokens[start:end],
                    mask=parent_memory.mask[start:end],
                    iface_slice=parent_memory.iface_slice,
                    cross_slice=parent_memory.cross_slice,
                ),
                child_iface_mask=child_iface_mask[start:end],
            )
            chunk_outputs.append(out.child_scores)
            if out.child_mate_scores is not None:
                chunk_mate_outputs.append(out.child_mate_scores)

        return MergeDecoderOutput(
            child_scores=torch.cat(chunk_outputs, dim=0),
            child_mate_scores=(
                torch.cat(chunk_mate_outputs, dim=0)
                if chunk_mate_outputs
                else None
            ),
        )

    def decode_sigma_batch(
        self,
        *,
        sigma_a: Tensor,           # [S, Ti] — all candidate sigmas
        sigma_mate: Tensor,        # [S, Ti]
        sigma_iface_mask: Tensor,  # [Ti] bool — shared across all sigmas (single node)
        parent_memory: ParentMemory,  # tokens [1, T, d], mask [1, T]
        child_iface_mask: Tensor,  # [4, Ti] bool — shared across all sigmas
        max_batch_size: Optional[int] = None,
    ) -> MergeDecoderOutput:
        """Convenience: decode S sigmas for a single parent box.

        Automatically expands parent_memory and masks to batch dimension S.
        More memory-efficient than manually expanding tensors.
        """
        S = sigma_a.shape[0]
        Ti = sigma_a.shape[1]

        if parent_memory.tokens.shape[0] != 1:
            raise ValueError(
                f"decode_sigma_batch expects parent_memory with batch=1, "
                f"got {parent_memory.tokens.shape[0]}"
            )

        # Expand parent memory to [S, T, d]
        expanded_mem = ParentMemory(
            tokens=parent_memory.tokens.expand(S, -1, -1),
            mask=parent_memory.mask.expand(S, -1),
            iface_slice=parent_memory.iface_slice,
            cross_slice=parent_memory.cross_slice,
        )

        # Expand masks to [S, ...]
        sigma_iface_mask_expanded = sigma_iface_mask.unsqueeze(0).expand(S, -1)
        child_iface_mask_expanded = child_iface_mask.unsqueeze(0).expand(S, -1, -1)

        return self.decode_sigma_chunked(
            sigma_a=sigma_a,
            sigma_mate=sigma_mate,
            sigma_iface_mask=sigma_iface_mask_expanded,
            parent_memory=expanded_mem,
            child_iface_mask=child_iface_mask_expanded,
            max_batch_size=max_batch_size,
        )


__all__ = ["MergeDecoder", "MergeDecoderOutput", "ParentMemory"]
