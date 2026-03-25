# src/models/tokenization.py
# -*- coding: utf-8 -*-
"""
Tokenization module for Neural DP (Rao'98-style TSP).

Contract v2 note:
- interface_feat6 already includes intersection-relative coords in feat6[2:4].
- We therefore DO NOT accept a separate iface_inter_rel_xy input, which prevents
  accidental coordinate-frame mixing and removes redundant concatenation.

Scale invariance:
- All continuous geometric features fed here must already be relative/normalized:
  - node_feat_rel: root-normalized box
  - iface_feat6/cross_feat6: center-relative endpoints + normalized distance + angle
- No absolute coordinates are used.

New in this refactor:
- Expose `embed_iface_only(...)` to embed interface tokens without constructing a full
  (CLS + IFACE + CROSS + CHILD) token sequence. This is used by the top-down decoder
  when child interfaces query the parent's memory via cross-attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class TokenizedNodeMemory:
    tokens: Tensor
    mask: Tensor
    type_id: Tensor
    cls_index: int
    iface_slice: slice
    cross_slice: slice
    child_slice: Optional[slice]


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers = []
        d = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _safe_cat_emb(emb: nn.Embedding, x: Tensor) -> Tensor:
    """
    Map padding value -1 to emb.padding_idx, then embed.

    We use -1 as padding in packed tensors, while torch Embedding expects indices in [0..V-1].
    """
    if x.dtype != torch.long:
        x = x.long()
    pad = emb.padding_idx
    if pad is None:
        raise ValueError("Embedding must have padding_idx set for safe padding handling.")
    x2 = torch.where(x < 0, torch.full_like(x, pad), x)
    return emb(x2)


class NodeTokenizer(nn.Module):
    """
    Build scale-invariant token embeddings per node.

    Inputs:
      - iface_feat6: [inside_rel_x, inside_rel_y, inter_rel_x, inter_rel_y, norm_len, angle]
      - cross_feat6: [u_rel_x, u_rel_y, v_rel_x, v_rel_y, norm_len, angle]

    When use_angle_sincos=True, we REPLACE the raw angle scalar with (sin, cos),
    so continuous dims become:
      - iface: 5 + 2 = 7
      - cross: 5 + 2 = 7
    """

    TYPE_CLS = 0
    TYPE_IFACE = 1
    TYPE_CROSS = 2
    TYPE_CHILD = 3

    def __init__(
        self,
        *,
        d_model: int,
        iface_hidden: int = 128,
        cross_hidden: int = 128,
        child_hidden: int = 128,
        num_mlp_layers: int = 2,
        dropout: float = 0.0,
        use_angle_sincos: bool = True,
        node_ctx: bool = True,
        max_depth: int = 64,
        type_embed: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.use_angle_sincos = bool(use_angle_sincos)
        self.node_ctx = bool(node_ctx)
        self.max_depth = int(max_depth)
        self.type_embed = bool(type_embed)

        # Categorical embeddings (all with padding_idx)
        self.emb_boundary_dir = nn.Embedding(5, d_model, padding_idx=4)          # 0..3, pad->4
        self.emb_inside_endpoint = nn.Embedding(3, d_model, padding_idx=2)       # 0..1, pad->2
        self.emb_inside_quadrant = nn.Embedding(5, d_model, padding_idx=4)       # 0..3, pad->4
        self.emb_cross_quad = nn.Embedding(5, d_model, padding_idx=4)            # 0..3, pad->4
        self.emb_leaf_internal = nn.Embedding(2, d_model)
        self.emb_child_slot = nn.Embedding(4, d_model)
        self.emb_depth = nn.Embedding(max_depth + 1, d_model)

        if self.type_embed:
            self.emb_type = nn.Embedding(4, d_model)

        # Continuous feature projections
        iface_in = 7 if self.use_angle_sincos else 6
        cross_in = 7 if self.use_angle_sincos else 6
        self.mlp_iface = MLP(iface_in, d_model, iface_hidden, num_mlp_layers, dropout)
        self.mlp_cross = MLP(cross_in, d_model, cross_hidden, num_mlp_layers, dropout)

        self.proj_child = nn.Linear(d_model, d_model)
        self.mlp_child = MLP(d_model, d_model, child_hidden, max(1, num_mlp_layers - 1), dropout)

        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)

        if self.node_ctx:
            self.mlp_node_ctx = MLP(4 + d_model, d_model, max(d_model, 128), 2, dropout)

    # ---------------------------
    # Public lightweight embedding APIs
    # ---------------------------

    def build_node_ctx(self, *, node_feat_rel: Tensor, node_depth: Tensor) -> Tensor:
        """
        Compute a per-node context vector (broadcast to all tokens in that node).

        Returns:
          node_ctx: [B, d_model]
        """
        B = int(node_feat_rel.shape[0])
        device = node_feat_rel.device
        depth_clipped = torch.clamp(node_depth.long(), 0, self.max_depth)
        depth_emb = self.emb_depth(depth_clipped)

        if self.node_ctx:
            node_ctx = self.mlp_node_ctx(torch.cat([node_feat_rel.float(), depth_emb], dim=-1))
        else:
            node_ctx = torch.zeros((B, self.d_model), dtype=torch.float32, device=device)
        return node_ctx

    def embed_iface_only(
        self,
        *,
        node_feat_rel: Tensor,            # [B,4]
        node_depth: Tensor,               # [B]
        iface_feat6: Tensor,              # [B,Ti,6]
        iface_mask: Tensor,               # [B,Ti] bool
        iface_boundary_dir: Tensor,       # [B,Ti] long (0..3, pad=-1)
        iface_inside_endpoint: Tensor,    # [B,Ti] long (0..1, pad=-1)
        iface_inside_quadrant: Tensor,    # [B,Ti] long (0..3, pad=-1)
        zero_padded: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Embed only interface tokens (no CLS/CROSS/CHILD tokens constructed).

        This is intended for top-down cross-attention where child interfaces act as Query
        and parent memory tokens act as Key/Value.

        Returns:
          iface_tok:  [B,Ti,d_model]
          iface_mask: [B,Ti] bool (passed through)
        """
        node_ctx = self.build_node_ctx(node_feat_rel=node_feat_rel, node_depth=node_depth)
        iface_tok = self._embed_iface_tokens(
            node_ctx=node_ctx,
            iface_feat6=iface_feat6,
            iface_boundary_dir=iface_boundary_dir,
            iface_inside_endpoint=iface_inside_endpoint,
            iface_inside_quadrant=iface_inside_quadrant,
        )

        if self.type_embed:
            B, Ti = int(iface_tok.shape[0]), int(iface_tok.shape[1])
            type_id = torch.full((B, Ti), self.TYPE_IFACE, dtype=torch.long, device=iface_tok.device)
            iface_tok = iface_tok + self.emb_type(type_id)

        if zero_padded:
            iface_tok = iface_tok.masked_fill(~iface_mask.bool().unsqueeze(-1), 0.0)

        return iface_tok, iface_mask.bool()

    # ---------------------------
    # Internal helpers (shared by forward and embed_* APIs)
    # ---------------------------

    def _cont_with_angle(self, feat6: Tensor) -> Tensor:
        """
        Convert Contract-v2 feat6 into continuous input for MLP.
        If use_angle_sincos: replace angle with (sin(pi*angle), cos(pi*angle)).
        """
        angle = feat6[..., 5:6]
        if self.use_angle_sincos:
            return torch.cat(
                [feat6[..., 0:5].float(), torch.sin(torch.pi * angle), torch.cos(torch.pi * angle)],
                dim=-1,
            )
        return feat6.float()

    def _embed_iface_tokens(
        self,
        *,
        node_ctx: Tensor,                 # [B,d]
        iface_feat6: Tensor,              # [B,Ti,6]
        iface_boundary_dir: Tensor,       # [B,Ti]
        iface_inside_endpoint: Tensor,    # [B,Ti]
        iface_inside_quadrant: Tensor,    # [B,Ti]
    ) -> Tensor:
        cont_i = self._cont_with_angle(iface_feat6)
        iface_base = self.mlp_iface(cont_i)

        e_bd = _safe_cat_emb(self.emb_boundary_dir, iface_boundary_dir)
        e_ep = _safe_cat_emb(self.emb_inside_endpoint, iface_inside_endpoint)
        e_qd = _safe_cat_emb(self.emb_inside_quadrant, iface_inside_quadrant)

        return iface_base + e_bd + e_ep + e_qd + node_ctx.unsqueeze(1)

    def _embed_cross_tokens(
        self,
        *,
        node_ctx: Tensor,                  # [B,d]
        cross_feat6: Tensor,               # [B,Tc,6]
        cross_child_pair: Tensor,          # [B,Tc,2]
        cross_is_leaf_internal: Tensor,    # [B,Tc] bool
    ) -> Tensor:
        cont_c = self._cont_with_angle(cross_feat6)
        cross_base = self.mlp_cross(cont_c)

        quad_u = cross_child_pair[..., 0]
        quad_v = cross_child_pair[..., 1]
        e_qu = _safe_cat_emb(self.emb_cross_quad, quad_u)
        e_qv = _safe_cat_emb(self.emb_cross_quad, quad_v)

        li = cross_is_leaf_internal.long().clamp(0, 1)
        e_li = self.emb_leaf_internal(li)

        return cross_base + e_qu + e_qv + e_li + node_ctx.unsqueeze(1)

    # ---------------------------
    # Full node tokenization (still used by bottom-up / existing heads)
    # ---------------------------

    def forward(
        self,
        *,
        node_feat_rel: Tensor,            # [B,4]
        node_depth: Tensor,               # [B]
        # interface
        iface_feat6: Tensor,              # [B,Ti,6]
        iface_mask: Tensor,               # [B,Ti] bool
        iface_boundary_dir: Tensor,       # [B,Ti] long (0..3, pad=-1)
        iface_inside_endpoint: Tensor,    # [B,Ti] long (0..1, pad=-1)
        iface_inside_quadrant: Tensor,    # [B,Ti] long (0..3, pad=-1)
        # crossing
        cross_feat6: Tensor,              # [B,Tc,6]
        cross_mask: Tensor,               # [B,Tc] bool
        cross_child_pair: Tensor,         # [B,Tc,2] long (0..3, pad=-1)
        cross_is_leaf_internal: Tensor,   # [B,Tc] bool
        # optional child summaries
        child_z: Optional[Tensor] = None,      # [B,4,d]
        child_mask: Optional[Tensor] = None,   # [B,4] bool
    ) -> TokenizedNodeMemory:
        B = int(node_feat_rel.shape[0])
        device = node_feat_rel.device

        node_ctx = self.build_node_ctx(node_feat_rel=node_feat_rel, node_depth=node_depth)

        # Interface tokens
        iface_tok = self._embed_iface_tokens(
            node_ctx=node_ctx,
            iface_feat6=iface_feat6,
            iface_boundary_dir=iface_boundary_dir,
            iface_inside_endpoint=iface_inside_endpoint,
            iface_inside_quadrant=iface_inside_quadrant,
        )
        Ti = int(iface_feat6.shape[1])
        iface_m = iface_mask.bool()
        iface_type = torch.full((B, Ti), self.TYPE_IFACE, dtype=torch.long, device=device)

        # Crossing tokens
        cross_tok = self._embed_cross_tokens(
            node_ctx=node_ctx,
            cross_feat6=cross_feat6,
            cross_child_pair=cross_child_pair,
            cross_is_leaf_internal=cross_is_leaf_internal,
        )
        Tc = int(cross_feat6.shape[1])
        cross_m = cross_mask.bool()
        cross_type = torch.full((B, Tc), self.TYPE_CROSS, dtype=torch.long, device=device)

        # Optional child summary tokens
        child_slice: Optional[slice] = None
        if child_z is not None:
            if child_mask is None:
                raise ValueError("child_mask must be provided when child_z is provided.")
            if child_z.dim() != 3 or child_z.shape[1] != 4:
                raise ValueError("child_z must have shape [B,4,d].")
            if child_mask.shape != (B, 4):
                raise ValueError("child_mask must have shape [B,4].")
            if child_z.shape[2] != self.d_model:
                raise ValueError(
                    f"child_z latent dim {int(child_z.shape[2])} != d_model {self.d_model}. "
                    "Please ensure leaf/merge encoders use a shared latent dim."
                )

            cz = self.proj_child(child_z)
            cz = self.mlp_child(cz)

            slot_ids = torch.arange(4, device=device).view(1, 4).expand(B, 4)
            e_slot = self.emb_child_slot(slot_ids)
            child_tok = cz + e_slot + node_ctx.unsqueeze(1)

            child_m = child_mask.bool()
            child_type = torch.full((B, 4), self.TYPE_CHILD, dtype=torch.long, device=device)
            child_slice = slice(1 + Ti + Tc, 1 + Ti + Tc + 4)
        else:
            child_tok = None
            child_m = None
            child_type = None

        # CLS
        cls_tok = self.cls.expand(B, 1, self.d_model) + node_ctx.unsqueeze(1)
        cls_m = torch.ones((B, 1), dtype=torch.bool, device=device)
        cls_type = torch.full((B, 1), self.TYPE_CLS, dtype=torch.long, device=device)

        # Concatenate
        parts = [cls_tok, iface_tok, cross_tok]
        masks = [cls_m, iface_m, cross_m]
        types = [cls_type, iface_type, cross_type]

        if child_tok is not None:
            parts.append(child_tok)
            masks.append(child_m)    # type: ignore[arg-type]
            types.append(child_type) # type: ignore[arg-type]

        tokens = torch.cat(parts, dim=1)
        mask = torch.cat(masks, dim=1)
        type_id = torch.cat(types, dim=1)

        if self.type_embed:
            tokens = tokens + self.emb_type(type_id)

        cls_index = 0
        iface_slice = slice(1, 1 + Ti)
        cross_slice = slice(1 + Ti, 1 + Ti + Tc)

        return TokenizedNodeMemory(
            tokens=tokens,
            mask=mask,
            type_id=type_id,
            cls_index=cls_index,
            iface_slice=iface_slice,
            cross_slice=cross_slice,
            child_slice=child_slice,
        )
