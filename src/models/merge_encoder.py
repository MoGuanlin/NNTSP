# src/models/merge_encoder.py
# -*- coding: utf-8 -*-
"""
Merge encoder for Neural DP (Rao'98-style TSP).

Contract v2 (size/scale invariance):
- This module NEVER consumes absolute coordinates.
- It assumes all continuous geometric features are already relative/normalized:
  - node_feat_rel: [cx, cy, w, h] normalized by root_scale (packer contract)
  - iface_feat6:  [inside_rel(2), inter_rel(2), norm_len, angle]  (node-relative)
  - cross_feat6:  [u_rel(2), v_rel(2), norm_len, angle]          (node-relative)
- Interface intersection rel is NOT a separate channel; it is iface_feat6[...,2:4].

DP-aligned aggregation:
- For each internal quadtree node, merge:
    (i) O(r) interface tokens (Ti cap),
    (ii) O(r) crossing tokens (Tc cap),
    (iii) exactly 4 child latent tokens (TL/TR/BL/BR), masked for missing children,
  into a fixed-dim latent z_parent ∈ R^{d_model}.

Architecture (token-level memory):
1) Tokenization via NodeTokenizer:
     CLS + IFACE(Ti) + CROSS(Tc) + CHILD(4)
2) Add node-context vector (from node_feat_rel + depth embedding) to every token
3) Add token-type embedding (CLS/IFACE/CROSS/CHILD)
4) SetTransformerEncoder (masked self-attention) over token memory
5) CLS pool -> z_parent

This file implements a callable nn.Module matching the MergeEncoder Protocol used by BottomUpTreeRunner.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

try:
    from src.models.tokenization import MLP, NodeTokenizer
    from src.models.set_transformer_block import SetTransformerEncoder, cls_pool
except Exception:  # pragma: no cover
    from .tokenization import MLP, NodeTokenizer
    from .set_transformer_block import SetTransformerEncoder, cls_pool


class MergeEncoderModule(nn.Module):
    """
    Merge encoder producing z_parent: [B, d_model].

    Notes:
    - d_model MUST match leaf encoder output dim.
    - Child slots are always 4 tokens; missing children are masked out.
    """

    # local token type ids (must align with NodeTokenizer.TYPE_*)
    TYPE_CLS = 0
    TYPE_IFACE = 1
    TYPE_CROSS = 2
    TYPE_CHILD = 3

    def __init__(
        self,
        *,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        ff_hidden_mult: int = 4,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        tokenizer_hidden: int = 256,
        tokenizer_mlp_layers: int = 2,
        use_angle_sincos: bool = True,
        max_depth: int = 64,
        return_aux: bool = False,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if n_heads <= 0:
            raise ValueError("n_heads must be positive.")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        if n_layers <= 0:
            raise ValueError("n_layers must be positive.")
        if max_depth <= 0:
            raise ValueError("max_depth must be positive.")

        self.d_model = int(d_model)
        self.return_aux = bool(return_aux)

        # Base tokenizer for CLS/IFACE/CROSS/CHILD (scale-invariant)
        # NOTE: node_ctx/type_embed are added explicitly in this module, mirroring LeafEncoderModule.
        self.tokenizer = NodeTokenizer(
            d_model=d_model,
            iface_hidden=tokenizer_hidden,
            cross_hidden=tokenizer_hidden,
            child_hidden=tokenizer_hidden,
            num_mlp_layers=tokenizer_mlp_layers,
            dropout=dropout,
            node_ctx=False,
            use_angle_sincos=use_angle_sincos,
            max_depth=max_depth,
            type_embed=False,
        )

        # Node context shared across all tokens.
        # depth embedding comes from tokenizer.emb_depth with dim=d_model.
        self.mlp_node_ctx = MLP(
            in_dim=4 + d_model,  # node_feat_rel(4) + depth_emb(d)
            out_dim=d_model,
            hidden_dim=max(d_model, 128),
            num_layers=2,
            dropout=dropout,
        )

        # Token type embedding: CLS / IFACE / CROSS / CHILD
        self.emb_type = nn.Embedding(4, d_model)

        # Token interaction (masked self-attention)
        self.encoder = SetTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            attn_dropout=attn_dropout,
            dropout=dropout,
            ff_hidden_mult=ff_hidden_mult,
            final_norm=True,
        )

    @staticmethod
    def _check_shapes_merge(
        *,
        node_feat_rel: Tensor,            # [B,4]
        node_depth: Tensor,               # [B]
        iface_feat6: Tensor,              # [B,Ti,6]
        iface_mask: Tensor,               # [B,Ti]
        iface_boundary_dir: Tensor,       # [B,Ti]
        iface_inside_endpoint: Tensor,    # [B,Ti]
        iface_inside_quadrant: Tensor,    # [B,Ti]
        cross_feat6: Tensor,              # [B,Tc,6]
        cross_mask: Tensor,               # [B,Tc]
        cross_child_pair: Tensor,         # [B,Tc,2]
        cross_is_leaf_internal: Tensor,   # [B,Tc]
        child_z: Tensor,                  # [B,4,d]
        child_mask: Tensor,               # [B,4]
    ) -> None:
        B = int(node_feat_rel.shape[0])
        if node_feat_rel.shape != (B, 4):
            raise ValueError(f"node_feat_rel must be [B,4], got {tuple(node_feat_rel.shape)}")
        if node_depth.shape != (B,):
            raise ValueError(f"node_depth must be [B], got {tuple(node_depth.shape)}")

        if iface_feat6.dim() != 3 or iface_feat6.shape[0] != B or iface_feat6.shape[2] != 6:
            raise ValueError(f"iface_feat6 must be [B,Ti,6], got {tuple(iface_feat6.shape)}")
        if iface_mask.shape != iface_feat6.shape[:2]:
            raise ValueError(f"iface_mask must match [B,Ti], got {tuple(iface_mask.shape)}")
        if iface_boundary_dir.shape != iface_feat6.shape[:2]:
            raise ValueError(f"iface_boundary_dir must be [B,Ti], got {tuple(iface_boundary_dir.shape)}")
        if iface_inside_endpoint.shape != iface_feat6.shape[:2]:
            raise ValueError(f"iface_inside_endpoint must be [B,Ti], got {tuple(iface_inside_endpoint.shape)}")
        if iface_inside_quadrant.shape != iface_feat6.shape[:2]:
            raise ValueError(f"iface_inside_quadrant must be [B,Ti], got {tuple(iface_inside_quadrant.shape)}")

        if cross_feat6.dim() != 3 or cross_feat6.shape[0] != B or cross_feat6.shape[2] != 6:
            raise ValueError(f"cross_feat6 must be [B,Tc,6], got {tuple(cross_feat6.shape)}")
        if cross_mask.shape != cross_feat6.shape[:2]:
            raise ValueError(f"cross_mask must match [B,Tc], got {tuple(cross_mask.shape)}")
        if cross_child_pair.shape != (B, cross_feat6.shape[1], 2):
            raise ValueError(f"cross_child_pair must be [B,Tc,2], got {tuple(cross_child_pair.shape)}")
        if cross_is_leaf_internal.shape != cross_feat6.shape[:2]:
            raise ValueError(f"cross_is_leaf_internal must be [B,Tc], got {tuple(cross_is_leaf_internal.shape)}")

        if child_z.dim() != 3 or child_z.shape[0] != B or child_z.shape[1] != 4:
            raise ValueError(f"child_z must be [B,4,d], got {tuple(child_z.shape)}")
        if child_mask.shape != (B, 4):
            raise ValueError(f"child_mask must be [B,4], got {tuple(child_mask.shape)}")

    def forward(
        self,
        *,
        node_feat_rel: Tensor,            # [B,4] root-normalized (cx,cy,w,h)
        node_depth: Tensor,               # [B]
        iface_feat6: Tensor,              # [B,Ti,6]
        iface_mask: Tensor,               # [B,Ti] bool
        iface_boundary_dir: Tensor,       # [B,Ti] long
        iface_inside_endpoint: Tensor,    # [B,Ti] long
        iface_inside_quadrant: Tensor,    # [B,Ti] long
        cross_feat6: Tensor,              # [B,Tc,6]
        cross_mask: Tensor,               # [B,Tc] bool
        cross_child_pair: Tensor,         # [B,Tc,2] long
        cross_is_leaf_internal: Tensor,   # [B,Tc] bool
        child_z: Tensor,                  # [B,4,d_model] (TL/TR/BL/BR)
        child_mask: Tensor,               # [B,4] bool
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """
        Return:
          z_parent: [B,d_model]
          optionally (z_parent, aux_dict)
        """
        self._check_shapes_merge(
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
            child_mask=child_mask,
        )

        # ---- 1) Tokenize base memory (CLS + IFACE + CROSS + CHILD) ----
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
            child_mask=child_mask,
        )

        tokens = mem.tokens                # [B,T,d]
        mask = mem.mask.bool()             # [B,T]
        type_id = mem.type_id.long()       # [B,T] in {0,1,2,3}

        B = int(node_feat_rel.shape[0])
        if tokens.shape[0] != B:
            raise ValueError("Tokenizer returned batch size mismatch.")
        if tokens.shape[2] != self.d_model:
            raise ValueError(f"Tokenizer returned d_model={tokens.shape[2]} != self.d_model={self.d_model}")

        # ---- 2) Node context (shared across all tokens) ----
        depth_clipped = torch.clamp(node_depth.long(), 0, self.tokenizer.max_depth)
        depth_emb = self.tokenizer.emb_depth(depth_clipped)  # [B,d]
        node_ctx = self.mlp_node_ctx(torch.cat([node_feat_rel.float(), depth_emb], dim=-1))  # [B,d]

        tokens = tokens + node_ctx.unsqueeze(1)

        # ---- 3) Type embedding ----
        tokens = tokens + self.emb_type(type_id)

        # ---- 4) Token interaction ----
        out_tokens = self.encoder(tokens, mask)  # [B,T,d]

        # ---- 5) Pool (CLS) ----
        z_parent = cls_pool(out_tokens, cls_index=mem.cls_index)  # [B,d]

        if not self.return_aux:
            return z_parent

        aux: Dict[str, Tensor] = {
            "T_total": torch.tensor([tokens.shape[1]], device=node_feat_rel.device),
            "num_iface_valid": iface_mask.sum(dim=1).to(dtype=torch.long),
            "num_cross_valid": cross_mask.sum(dim=1).to(dtype=torch.long),
            "num_child_valid": child_mask.sum(dim=1).to(dtype=torch.long),
        }
        return z_parent, aux


# Backward-compatible public API name expected by runner/tests
MergeEncoder = MergeEncoderModule
__all__ = ["MergeEncoder", "MergeEncoderModule"]
