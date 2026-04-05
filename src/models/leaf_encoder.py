# src/models/leaf_encoder.py
# -*- coding: utf-8 -*-
"""
Leaf encoder for Neural DP (Rao'98-style TSP).

Goal (DP-aligned + size/scale invariant):
- For each *leaf quadtree cell* (node), produce a fixed-dim latent vector z_leaf ∈ R^{d_model}
- Input per node is O(r) interface/crossing tokens + O(P) leaf-point tokens, where r and P are constants
  independent of problem size N.

Scale invariance (critical):
- This module assumes leaf_points_xy is already expressed in a *cell-relative normalized coordinate system*,
  consistent with interface_edge_attr:
    leaf_points_xy[...,0] ≈ (pos.x - cx) / (w/2)
    leaf_points_xy[...,1] ≈ (pos.y - cy) / (h/2)
  i.e. values are typically in a stable range (roughly [-1,1], possibly slightly outside due to numeric issues).

- All other continuous features come from *_feat6 (contract v2), which are defined node-relative.

Architecture:
1) Tokenize interface/crossing events via NodeTokenizer (without node_ctx/type embedding).
2) Tokenize leaf points as point tokens using an MLP on (x_rel, y_rel).
3) Add shared node-context (from node_feat_rel + depth embedding) to *all* tokens.
4) Add token-type embedding (CLS/IFACE/CROSS/POINT).
5) Run a small SetTransformerEncoder (masked self-attention) over the token memory.
6) Pool via CLS token -> z_leaf.

This file implements a callable nn.Module matching the LeafEncoder Protocol used by BottomUpTreeRunner.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

try:
    from src.models.tokenization import MLP, NodeTokenizer
    from src.models.set_transformer_block import SetTransformerEncoder, cls_pool
except Exception:  # pragma: no cover
    from .tokenization import MLP, NodeTokenizer
    from .set_transformer_block import SetTransformerEncoder, cls_pool


class LeafEncoder(nn.Module):
    """
    Leaf encoder producing z_leaf: [B, d_model].

    Notes:
    - d_model is the latent dimension and MUST be shared with merge encoder output.
    - max_points_per_leaf is a constant cap set by your packer; it does not affect parameter shapes.
    """

    # local token type ids (we control this inside leaf encoder)
    TYPE_CLS = 0
    TYPE_IFACE = 1
    TYPE_CROSS = 2
    TYPE_POINT = 3

    def __init__(
        self,
        *,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        ff_hidden_mult: int = 4,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        point_hidden: int = 256,
        point_mlp_layers: int = 2,
        tokenizer_hidden: int = 256,
        tokenizer_mlp_layers: int = 2,
        use_angle_sincos: bool = True,
        max_depth: int = 64,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if n_heads <= 0:
            raise ValueError("n_heads must be positive.")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

        self.d_model = int(d_model)

        # Tokenizer for interface/cross only; we disable node_ctx and type_embed here
        # so that leaf encoder can add ONE consistent node_ctx/type embedding to all token groups.
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

        # Node context (shared across all tokens)
        self.mlp_node_ctx = MLP(
            in_dim=4 + d_model,  # node_feat_rel(4) + depth_emb(d)
            out_dim=d_model,
            hidden_dim=max(d_model, 128),
            num_layers=2,
            dropout=dropout,
        )

        # Token type embedding: CLS / IFACE / CROSS / POINT
        self.emb_type = nn.Embedding(4, d_model)

        # Leaf point tokenization (scale-invariant, uses normalized cell-relative xy)
        self.mlp_point = MLP(
            in_dim=2,
            out_dim=d_model,
            hidden_dim=point_hidden,
            num_layers=point_mlp_layers,
            dropout=dropout,
        )

        # Token interaction
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
    def _check_shapes_leaf(
        *,
        node_feat_rel: Tensor,
        node_depth: Tensor,
        iface_feat6: Tensor,
        iface_mask: Tensor,
        cross_feat6: Tensor,
        cross_mask: Tensor,
        leaf_points_xy: Tensor,
        leaf_points_mask: Tensor,
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

        if cross_feat6.dim() != 3 or cross_feat6.shape[0] != B or cross_feat6.shape[2] != 6:
            raise ValueError(f"cross_feat6 must be [B,Tc,6], got {tuple(cross_feat6.shape)}")
        if cross_mask.shape != cross_feat6.shape[:2]:
            raise ValueError(f"cross_mask must match [B,Tc], got {tuple(cross_mask.shape)}")

        if leaf_points_xy.dim() != 3 or leaf_points_xy.shape[0] != B or leaf_points_xy.shape[2] != 2:
            raise ValueError(f"leaf_points_xy must be [B,P,2], got {tuple(leaf_points_xy.shape)}")
        if leaf_points_mask.shape != leaf_points_xy.shape[:2]:
            raise ValueError(f"leaf_points_mask must be [B,P], got {tuple(leaf_points_mask.shape)}")

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
        leaf_points_xy: Tensor,           # [B,P,2] (cell-relative normalized coords; see header doc)
        leaf_points_mask: Tensor,         # [B,P] bool
    ) -> Tensor:
        """Return `z_leaf` with shape `[B, d_model]`."""
        self._check_shapes_leaf(
            node_feat_rel=node_feat_rel,
            node_depth=node_depth,
            iface_feat6=iface_feat6,
            iface_mask=iface_mask,
            cross_feat6=cross_feat6,
            cross_mask=cross_mask,
            leaf_points_xy=leaf_points_xy,
            leaf_points_mask=leaf_points_mask,
        )

        B = int(node_feat_rel.shape[0])
        device = node_feat_rel.device

        # ---- 1) Tokenize interface/cross (no node_ctx, no type embed yet) ----
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
            child_z=None,
            child_mask=None,
        )
        base_tokens = mem.tokens                # [B, 1+Ti+Tc, d]
        base_mask = mem.mask.bool()             # [B, 1+Ti+Tc]
        base_type = mem.type_id.long()          # [B, 1+Ti+Tc] values in {0,1,2}

        # ---- 2) Node context (shared) ----
        depth_clipped = torch.clamp(node_depth.long(), 0, self.tokenizer.max_depth)
        depth_emb = self.tokenizer.emb_depth(depth_clipped)  # [B,d]
        node_ctx = self.mlp_node_ctx(torch.cat([node_feat_rel.float(), depth_emb], dim=-1))  # [B,d]

        # add node context to base tokens
        base_tokens = base_tokens + node_ctx.unsqueeze(1)

        # add type embedding to base tokens (CLS/IFACE/CROSS)
        base_tokens = base_tokens + self.emb_type(base_type)

        # ---- 3) Leaf point tokens ----
        # (cell-relative normalized coords; no absolute coord used)
        P = int(leaf_points_xy.shape[1])
        point_tokens = self.mlp_point(leaf_points_xy.float())  # [B,P,d]
        point_tokens = point_tokens + node_ctx.unsqueeze(1)

        point_type = torch.full((B, P), self.TYPE_POINT, dtype=torch.long, device=device)
        point_tokens = point_tokens + self.emb_type(point_type)

        # ---- 4) Concatenate token memory ----
        tokens = torch.cat([base_tokens, point_tokens], dim=1)                  # [B, T, d]
        mask = torch.cat([base_mask, leaf_points_mask.bool()], dim=1)           # [B, T]

        # ---- 5) Interaction (masked self-attention) ----
        out_tokens = self.encoder(tokens, mask)  # [B,T,d]

        # ---- 6) Pool (CLS) ----
        z_leaf = cls_pool(out_tokens, cls_index=mem.cls_index)  # [B,d]

        return z_leaf
