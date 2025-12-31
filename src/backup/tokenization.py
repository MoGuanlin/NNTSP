# src/models/tokenization.py
# -*- coding: utf-8 -*-
"""
Tokenization module for Neural DP (Rao'98-style TSP).

This module converts per-node packed tensors (from NodeTokenPacker) into:
- token embeddings: [B, T, d_model]
- token mask:       [B, T] bool
- token type ids:   [B, T] long (for debugging/ablation)
- slice metadata describing where each token group sits in the concatenation

Scale invariance:
- All continuous geometric features fed here must already be relative/normalized:
  - node_feat_rel: root-normalized box
  - iface_feat6/cross_feat6: node-relative endpoints + normalized distance + angle
  - iface_inter_rel_xy: node-relative intersection
- No absolute coordinates are used.

Token groups:
1) CLS token (always present)
2) Interface tokens (cap Ti)
3) Crossing tokens (cap Tc)
4) Child summary tokens (optional; typically 4 tokens for TL/TR/BL/BR), used by merge encoder

This file contains no attention / GNN; it only produces token embeddings ready for
a SetTransformer/Transformer-style interaction layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# -----------------------------
# Output dataclass
# -----------------------------

@dataclass(frozen=True)
class TokenizedNodeMemory:
    """
    tokens:      [B, T, d_model]
    mask:        [B, T] bool
    type_id:     [B, T] long
    cls_index:   int (always 0)
    iface_slice: slice
    cross_slice: slice
    child_slice: Optional[slice]
    """
    tokens: Tensor
    mask: Tensor
    type_id: Tensor
    cls_index: int
    iface_slice: slice
    cross_slice: slice
    child_slice: Optional[slice]


# -----------------------------
# Small building blocks
# -----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers = []
        d = in_dim
        for i in range(num_layers - 1):
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
    """
    if x.dtype != torch.long:
        x = x.long()
    pad = emb.padding_idx
    if pad is None:
        raise ValueError("Embedding must have padding_idx set for safe padding handling.")
    x2 = torch.where(x >= 0, x, torch.full_like(x, pad))
    return emb(x2)


# -----------------------------
# Main tokenizer
# -----------------------------

class NodeTokenizer(nn.Module):
    """
    NodeTokenizer builds scale-invariant token embeddings per node.

    Parameters:
      d_model: token embedding dimension
      iface_hidden / cross_hidden: hidden dims for type-specific MLPs
      node_ctx: whether to add a node-context vector (from node_feat_rel + depth) to every token
      use_angle_sincos: whether to replace/augment angle with sin/cos (recommended for continuity)
      max_depth: max depth for depth embedding (values beyond are clipped)
    """

    TYPE_CLS = 0
    TYPE_IFACE = 1
    TYPE_CROSS = 2
    TYPE_CHILD = 3

    def __init__(
        self,
        *,
        d_model: int = 128,
        iface_hidden: int = 256,
        cross_hidden: int = 256,
        child_hidden: int = 256,
        num_mlp_layers: int = 2,
        dropout: float = 0.0,
        node_ctx: bool = True,
        use_angle_sincos: bool = True,
        max_depth: int = 64,
        type_embed: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if max_depth <= 0:
            raise ValueError("max_depth must be positive.")

        self.d_model = int(d_model)
        self.use_angle_sincos = bool(use_angle_sincos)
        self.node_ctx = bool(node_ctx)
        self.max_depth = int(max_depth)
        self.type_embed = bool(type_embed)

        # --- Categorical embeddings (all with padding_idx) ---
        # boundary_dir: {0,1,2,3}, padding=-1 -> idx=4
        self.emb_boundary_dir = nn.Embedding(5, d_model, padding_idx=4)
        # inside_endpoint: {0,1}, padding=-1 -> idx=2
        self.emb_inside_endpoint = nn.Embedding(3, d_model, padding_idx=2)
        # inside_quadrant: {0,1,2,3}, padding=-1 -> idx=4
        self.emb_inside_quadrant = nn.Embedding(5, d_model, padding_idx=4)
        # crossing child quadrant per endpoint: {0,1,2,3}, padding=-1 -> idx=4
        self.emb_cross_quad = nn.Embedding(5, d_model, padding_idx=4)
        # leaf_internal: {0,1} as bool -> embedding with padding_idx not needed, but we keep it simple
        self.emb_leaf_internal = nn.Embedding(2, d_model)

        # child slot embedding: 0..3 for TL/TR/BL/BR
        self.emb_child_slot = nn.Embedding(4, d_model)

        # depth embedding (clipped): 0..max_depth, padding not needed
        self.emb_depth = nn.Embedding(max_depth + 1, d_model)

        # token type embedding
        if self.type_embed:
            self.emb_type = nn.Embedding(4, d_model)

        # --- Continuous feature projections (scale-invariant inputs only) ---
        # iface continuous dims: feat6 (6) + inter_rel_xy (2) + (optional sincos(2))
        iface_in = 6 + 2 + (2 if self.use_angle_sincos else 0)
        self.mlp_iface = MLP(iface_in, d_model, iface_hidden, num_mlp_layers, dropout)

        # cross continuous dims: feat6 (6) + (optional sincos(2))
        cross_in = 6 + (2 if self.use_angle_sincos else 0)
        self.mlp_cross = MLP(cross_in, d_model, cross_hidden, num_mlp_layers, dropout)

        # child tokens: either use child_z directly (project) or if absent, skip
        self.proj_child = nn.Linear(d_model, d_model)
        # optional MLP over child_z if you want more capacity
        self.mlp_child = MLP(d_model, d_model, child_hidden, max(1, num_mlp_layers - 1), dropout)

        # CLS token (learned)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)

        # Node context projection: from node_feat_rel (4) + depth_emb (d_model) -> d_model
        if self.node_ctx:
            self.mlp_node_ctx = MLP(4 + d_model, d_model, max(d_model, 128), 2, dropout)

    def forward(
        self,
        *,
        node_feat_rel: Tensor,            # [B,4]  root-normalized
        node_depth: Tensor,               # [B]
        # interface
        iface_feat6: Tensor,              # [B,Ti,6] node-relative
        iface_mask: Tensor,               # [B,Ti] bool
        iface_boundary_dir: Tensor,       # [B,Ti] long (0..3, pad=-1)
        iface_inside_endpoint: Tensor,    # [B,Ti] long (0..1, pad=-1)
        iface_inter_rel_xy: Tensor,       # [B,Ti,2] node-relative
        iface_inside_quadrant: Tensor,    # [B,Ti] long (0..3, pad=-1)
        # crossing
        cross_feat6: Tensor,              # [B,Tc,6] node-relative
        cross_mask: Tensor,               # [B,Tc] bool
        cross_child_pair: Tensor,         # [B,Tc,2] long (0..3, pad=-1)
        cross_is_leaf_internal: Tensor,   # [B,Tc] bool
        # optional children (merge encoder)
        child_z: Optional[Tensor] = None,     # [B,4,d_model] or [B,4,d_child] (we project if needed)
        child_mask: Optional[Tensor] = None,  # [B,4] bool
    ) -> TokenizedNodeMemory:
        """
        Build token memory.

        Returns TokenizedNodeMemory with concatenation order:
          [CLS | iface(0..Ti-1) | cross(0..Tc-1) | child(0..3)] (child optional)
        """
        B = int(node_feat_rel.shape[0])
        device = node_feat_rel.device

        # ---------- Node context (scale-invariant) ----------
        depth_clipped = torch.clamp(node_depth.long(), 0, self.max_depth)  # [B]
        depth_emb = self.emb_depth(depth_clipped)  # [B,d]
        if self.node_ctx:
            node_ctx = self.mlp_node_ctx(torch.cat([node_feat_rel.float(), depth_emb], dim=-1))  # [B,d]
        else:
            node_ctx = torch.zeros((B, self.d_model), dtype=torch.float32, device=device)

        # ---------- CLS ----------
        cls_tok = self.cls.expand(B, -1, -1)  # [B,1,d]
        cls_mask = torch.ones((B, 1), dtype=torch.bool, device=device)
        cls_type = torch.full((B, 1), self.TYPE_CLS, dtype=torch.long, device=device)

        # ---------- Interface tokens ----------
        Ti = int(iface_feat6.shape[1])

        iface_angle = iface_feat6[..., 5:6]  # [B,Ti,1]
        iface_cont = [iface_feat6.float(), iface_inter_rel_xy.float()]
        if self.use_angle_sincos:
            iface_cont.append(torch.sin(iface_angle))
            iface_cont.append(torch.cos(iface_angle))
        iface_cont_x = torch.cat(iface_cont, dim=-1)  # [B,Ti, 6+2(+2)]
        iface_base = self.mlp_iface(iface_cont_x)  # [B,Ti,d]

        # categorical additions
        e_dir = _safe_cat_emb(self.emb_boundary_dir, iface_boundary_dir)
        e_ep = _safe_cat_emb(self.emb_inside_endpoint, iface_inside_endpoint)
        e_quad = _safe_cat_emb(self.emb_inside_quadrant, iface_inside_quadrant)

        iface_tok = iface_base + e_dir + e_ep + e_quad
        iface_tok = iface_tok + node_ctx.unsqueeze(1)  # add node context to every token

        iface_type = torch.full((B, Ti), self.TYPE_IFACE, dtype=torch.long, device=device)

        # ---------- Crossing tokens ----------
        Tc = int(cross_feat6.shape[1])
        cross_angle = cross_feat6[..., 5:6]  # [B,Tc,1]
        cross_cont = [cross_feat6.float()]
        if self.use_angle_sincos:
            cross_cont.append(torch.sin(cross_angle))
            cross_cont.append(torch.cos(cross_angle))
        cross_cont_x = torch.cat(cross_cont, dim=-1)  # [B,Tc, 6(+2)]
        cross_base = self.mlp_cross(cross_cont_x)  # [B,Tc,d]

        # child pair embedding (two endpoints)
        quad_u = cross_child_pair[..., 0]
        quad_v = cross_child_pair[..., 1]
        e_qu = _safe_cat_emb(self.emb_cross_quad, quad_u)
        e_qv = _safe_cat_emb(self.emb_cross_quad, quad_v)

        # leaf_internal embedding
        li = cross_is_leaf_internal.long().clamp(0, 1)
        e_li = self.emb_leaf_internal(li)

        cross_tok = cross_base + e_qu + e_qv + e_li
        cross_tok = cross_tok + node_ctx.unsqueeze(1)

        cross_type = torch.full((B, Tc), self.TYPE_CROSS, dtype=torch.long, device=device)

        # ---------- Optional child summary tokens ----------
        child_slice: Optional[slice] = None
        if child_z is not None:
            if child_mask is None:
                raise ValueError("child_mask must be provided when child_z is provided.")
            if child_z.dim() != 3 or child_z.shape[1] != 4:
                raise ValueError("child_z must have shape [B,4,d].")
            if child_mask.shape != (B, 4):
                raise ValueError("child_mask must have shape [B,4].")

            # project child_z to d_model if necessary
            if child_z.shape[2] != self.d_model:
                # simple linear projection from d_child -> d_model
                proj = nn.Linear(child_z.shape[2], self.d_model, bias=False).to(device=device, dtype=child_z.dtype)
                # NOTE: creating modules in forward is not ideal; we instead require child_z already d_model in practice.
                # To keep this module strict/clean, enforce same dimension:
                raise ValueError(
                    f"child_z latent dim {int(child_z.shape[2])} != d_model {self.d_model}. "
                    "Please ensure leaf/merge encoders use a shared latent dim."
                )

            cz = self.proj_child(child_z)  # [B,4,d]
            cz = self.mlp_child(cz)        # [B,4,d]
            slot_ids = torch.arange(4, device=device).view(1, 4).expand(B, 4)  # [B,4]
            e_slot = self.emb_child_slot(slot_ids)  # [B,4,d]
            child_tok = cz + e_slot + node_ctx.unsqueeze(1)  # [B,4,d]
            child_m = child_mask.bool()  # [B,4]
            child_type = torch.full((B, 4), self.TYPE_CHILD, dtype=torch.long, device=device)
            child_slice = slice(1 + Ti + Tc, 1 + Ti + Tc + 4)
        else:
            child_tok = None
            child_m = None
            child_type = None

        # ---------- Concatenate ----------
        tokens_list = [cls_tok, iface_tok, cross_tok]
        mask_list = [cls_mask, iface_mask.bool(), cross_mask.bool()]
        type_list = [cls_type, iface_type, cross_type]

        if child_tok is not None:
            tokens_list.append(child_tok)
            mask_list.append(child_m)
            type_list.append(child_type)

        tokens = torch.cat(tokens_list, dim=1)  # [B,T,d]
        mask = torch.cat(mask_list, dim=1)      # [B,T]
        type_id = torch.cat(type_list, dim=1)   # [B,T]

        # add type embedding if enabled
        if self.type_embed:
            tokens = tokens + self.emb_type(type_id)

        # Slices
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
