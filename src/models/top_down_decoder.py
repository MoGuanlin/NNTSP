# src/models/top_down_decoder.py
# -*- coding: utf-8 -*-
"""
Top-down decoder aligned with Rao'98-style DP, using boundary-condition scoring.

For an internal box v:

Input:
  (A) parent-provided boundary condition on v's interfaces: bc_in_iface_logit [B,Ti]
  (B) v-local known tokens:
      - v interface tokens (geometry + discrete attrs)
      - v crossing tokens (between its 4 children)
      - v bottom-up latent z_v
      - its 4 children bottom-up latents z_child[q]
  (C) v's 4 children local interface tokens (geometry + discrete attrs), as queries

Output:
  - v local iface/cross logits (optional/aux supervision)
  - child boundary conditions: child_iface_logit [B,4,Ti]
    (each child's interface usage logits)

Two modes:
  - mode="two_stage" (default):
      Stage A: build parent memory (self-attention) conditioned on bc_in + z_v (+ child latents)
      Stage B: child iface tokens query parent memory via cross-attention
  - mode="one_stage":
      Put child iface tokens into the same token set and do one self-attention;
      read out child logits from child iface token outputs.

No extra child↔child coupling layer is used (crossing tokens are the explicit constraint channel).
All token lengths are O(1) in N, hence fully scale-invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import torch
import torch.nn as nn
from torch import Tensor

from .set_transformer_block import FeedForward, SetTransformerBlock
from .tokenization import NodeTokenizer

Mode = Literal["two_stage", "one_stage"]


@dataclass(frozen=True)
class TopDownDecoderOutput:
    """
    iface_logit:       local per-node iface logits            [B, Ti]
    cross_logit:       local per-node crossing logits         [B, Tc]
    child_iface_logit: per-child iface logits (boundary cond) [B, 4, Ti]
    aux:               optional debug stats
    """
    iface_logit: Tensor
    cross_logit: Tensor
    child_iface_logit: Tensor
    aux: Dict[str, Tensor]


class CrossAttentionBlock(nn.Module):
    """
    Pre-norm cross-attention block with robust padding.

    q = q + MHA(LN(q), LN(kv), LN(kv), key_padding_mask=~kv_mask)
    q = q + FFN(LN(q))

    Padded query positions are kept at 0.
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
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError("n_heads must be positive and divide d_model.")

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

    @staticmethod
    def _apply_pad_zero(x: Tensor, mask: Tensor) -> Tensor:
        return torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))

    def forward(self, q: Tensor, q_mask: Tensor, kv: Tensor, kv_mask: Tensor) -> Tensor:
        if q.dim() != 3:
            raise ValueError(f"q must be [B,Q,d], got {tuple(q.shape)}")
        if kv.dim() != 3:
            raise ValueError(f"kv must be [B,T,d], got {tuple(kv.shape)}")
        if q_mask.shape != q.shape[:2]:
            raise ValueError(f"q_mask must be [B,Q], got {tuple(q_mask.shape)}")
        if kv_mask.shape != kv.shape[:2]:
            raise ValueError(f"kv_mask must be [B,T], got {tuple(kv_mask.shape)}")

        q = self._apply_pad_zero(q, q_mask)

        yq = self.ln_q(q)
        ykv = self.ln_kv(kv)
        key_padding_mask = ~kv_mask  # True=ignore
        attn_out, _ = self.attn(
            query=yq,
            key=ykv,
            value=ykv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        q = q + self.drop_attn(attn_out)
        q = self._apply_pad_zero(q, q_mask)

        ff_out = self.ff(self.ln_ff(q))
        q = q + self.drop_ff(ff_out)
        q = self._apply_pad_zero(q, q_mask)
        return q


class TopDownDecoderModule(nn.Module):
    """
    Boundary-condition decoder with optional two-stage cross-attn.

    IMPORTANT:
    - This module assumes the packer guarantees a *stable order* of interfaces
      inside each child (so [q, i] indexing is consistent across the tree).
    """

    _NEG_INF: float = -1e9

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int = 8,
        parent_num_layers: int = 3,
        cross_num_layers: int = 1,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        ff_mult: int = 4,
        max_depth: int = 64,
        use_angle_sincos: bool = True,
        mode: Mode = "two_stage",
        bc_clip: float = 20.0,
        return_aux: bool = False,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError("n_heads must be positive and divide d_model.")
        if parent_num_layers <= 0:
            raise ValueError("parent_num_layers must be positive.")
        if cross_num_layers <= 0:
            raise ValueError("cross_num_layers must be positive.")
        if max_depth <= 0:
            raise ValueError("max_depth must be positive.")
        if mode not in ("two_stage", "one_stage"):
            raise ValueError(f"Unknown mode: {mode}")
        if bc_clip <= 0:
            raise ValueError("bc_clip must be positive.")

        self.d_model = int(d_model)
        self.mode: Mode = mode
        self.bc_clip = float(bc_clip)
        self.return_aux = bool(return_aux)

        # Parent tokenizer: CLS + IFACE + CROSS + CHILD(4 latents)
        # (Note: this tokenizer does NOT embed child-iface tokens; those are embedded below.)
        self.tokenizer = NodeTokenizer(
            d_model=d_model,
            max_depth=max_depth,
            use_angle_sincos=use_angle_sincos,
            node_ctx=True,
            type_embed=True,
        )

        # Parent memory blocks (self-attention)
        self.parent_blocks = nn.ModuleList(
            [
                SetTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    ff_hidden_mult=ff_mult,
                )
                for _ in range(parent_num_layers)
            ]
        )

        # Query->memory blocks (cross-attention), used in two_stage
        self.cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    ff_hidden_mult=ff_mult,
                )
                for _ in range(cross_num_layers)
            ]
        )

        # Inject bc_in [B,Ti] -> [B,Ti,d]
        self.bc_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Inject bottom-up latent to parent CLS
        self.z_node_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

        # Local heads (optional aux)
        self.iface_head = nn.Linear(d_model, 1)
        self.cross_head = nn.Linear(d_model, 1)

        # Child boundary-condition head (per child iface token)
        self.child_iface_head = nn.Linear(d_model, 1)

    # -----------------------------
    # Helpers: safe categorical embedding
    # -----------------------------
    @staticmethod
    def _safe_cat_emb(emb: nn.Embedding, x: Tensor) -> Tensor:
        if x.dtype != torch.long:
            x = x.long()
        pad = emb.padding_idx
        if pad is None:
            raise ValueError("Embedding must have padding_idx set.")
        x2 = torch.where(x >= 0, x, torch.full_like(x, pad))
        # keep in-range defensively (pad is the max index we expect)
        x2 = x2.clamp(min=0, max=pad)
        return emb(x2)

    def _embed_child_iface_tokens(
        self,
        *,
        child_node_feat_rel: Tensor,            # [B,4,4]
        child_node_depth: Tensor,               # [B,4]
        child_iface_feat6: Tensor,              # [B,4,Ti,6]
        child_iface_mask: Tensor,               # [B,4,Ti] bool
        child_iface_boundary_dir: Tensor,       # [B,4,Ti] long
        child_iface_inside_endpoint: Tensor,    # [B,4,Ti] long
        child_iface_inside_quadrant: Tensor,    # [B,4,Ti] long
        child_exists_mask: Tensor,              # [B,4] bool
    ) -> tuple[Tensor, Tensor]:
        """
        Return:
          tok:  [B,4,Ti,d]
          mask: [B,4,Ti] bool  (already AND with child_exists_mask)
        """
        if child_node_feat_rel.dim() != 3 or child_node_feat_rel.shape[1:] != (4, 4):
            raise ValueError(f"child_node_feat_rel must be [B,4,4], got {tuple(child_node_feat_rel.shape)}")
        if child_node_depth.dim() != 2 or child_node_depth.shape[1] != 4:
            raise ValueError(f"child_node_depth must be [B,4], got {tuple(child_node_depth.shape)}")
        if child_exists_mask.shape != child_node_depth.shape:
            raise ValueError("child_exists_mask must match child_node_depth shape [B,4].")
        if child_iface_feat6.dim() != 4 or child_iface_feat6.shape[1] != 4 or child_iface_feat6.shape[3] != 6:
            raise ValueError(f"child_iface_feat6 must be [B,4,Ti,6], got {tuple(child_iface_feat6.shape)}")
        if child_iface_mask.shape != child_iface_feat6.shape[:3]:
            raise ValueError("child_iface_mask must be [B,4,Ti].")

        B = int(child_node_feat_rel.shape[0])
        Ti = int(child_iface_feat6.shape[2])
        device = child_node_feat_rel.device

        # flatten (B,4,...) -> (B4,...)
        B4 = B * 4
        feat_rel = child_node_feat_rel.reshape(B4, 4)
        depth = child_node_depth.reshape(B4)
        iface_feat6 = child_iface_feat6.reshape(B4, Ti, 6)
        iface_mask = child_iface_mask.reshape(B4, Ti).bool()
        bdir = child_iface_boundary_dir.reshape(B4, Ti).long()
        iep = child_iface_inside_endpoint.reshape(B4, Ti).long()
        iquad = child_iface_inside_quadrant.reshape(B4, Ti).long()

        # node_ctx (reuse tokenizer weights; tokenizer.node_ctx is expected True)
        depth_clip = torch.clamp(depth.long(), 0, self.tokenizer.max_depth)
        depth_emb = self.tokenizer.emb_depth(depth_clip)  # [B4,d]
        if not getattr(self.tokenizer, "node_ctx", False):
            raise RuntimeError("NodeTokenizer must be constructed with node_ctx=True for top-down child embedding.")
        node_ctx = self.tokenizer.mlp_node_ctx(torch.cat([feat_rel.float(), depth_emb], dim=-1))  # [B4,d]

        # continuous features (match tokenization.py)
        angle = iface_feat6[..., 5:6]
        if self.tokenizer.use_angle_sincos:
            cont = torch.cat(
                [iface_feat6[..., 0:5].float(),
                 torch.sin(torch.pi * angle),
                 torch.cos(torch.pi * angle)],
                dim=-1,
            )
        else:
            cont = iface_feat6.float()

        base = self.tokenizer.mlp_iface(cont)
        e_dir = self._safe_cat_emb(self.tokenizer.emb_boundary_dir, bdir)
        e_ep = self._safe_cat_emb(self.tokenizer.emb_inside_endpoint, iep)
        e_quad = self._safe_cat_emb(self.tokenizer.emb_inside_quadrant, iquad)

        tok = base + e_dir + e_ep + e_quad + node_ctx.unsqueeze(1)  # [B4,Ti,d]

        # add child quadrant embedding so queries know which child they belong to
        child_ids = torch.arange(4, device=device, dtype=torch.long).view(1, 4, 1).expand(B, 4, Ti)
        child_ids = child_ids.reshape(B4, Ti)
        tok = tok + self.tokenizer.emb_child_slot(child_ids)

        # reuse IFACE type embedding if enabled
        if getattr(self.tokenizer, "type_embed", False):
            type_id = torch.full((B4, Ti), self.tokenizer.TYPE_IFACE, device=device, dtype=torch.long)
            tok = tok + self.tokenizer.emb_type(type_id)

        # mask: iface_mask AND child_exists
        exists = child_exists_mask.reshape(B4, 1).expand(B4, Ti).bool()
        mask = iface_mask & exists

        # keep padded tokens at 0
        tok = torch.where(mask.unsqueeze(-1), tok, torch.zeros_like(tok))

        # reshape back
        tok = tok.reshape(B, 4, Ti, self.d_model)
        mask = mask.reshape(B, 4, Ti)
        return tok, mask

    # -----------------------------
    # Forward
    # -----------------------------
    def forward(
        self,
        *,
        node_feat_rel: Tensor,                 # [B,4]
        node_depth: Tensor,                    # [B]
        z_node: Tensor,                        # [B,d]
        # parent local tokens
        iface_feat6: Tensor,                   # [B,Ti,6]
        iface_mask: Tensor,                    # [B,Ti] bool
        iface_boundary_dir: Tensor,            # [B,Ti] long
        iface_inside_endpoint: Tensor,         # [B,Ti] long
        iface_inside_quadrant: Tensor,         # [B,Ti] long
        cross_feat6: Tensor,                   # [B,Tc,6]
        cross_mask: Tensor,                    # [B,Tc] bool
        cross_child_pair: Tensor,              # [B,Tc,2] long
        cross_is_leaf_internal: Tensor,        # [B,Tc] bool
        # parent boundary condition on THIS node
        bc_in_iface_logit: Tensor,             # [B,Ti]
        # children latents
        child_z: Tensor,                       # [B,4,d]
        child_exists_mask: Tensor,             # [B,4] bool
        # child local iface tokens
        child_node_feat_rel: Tensor,           # [B,4,4]
        child_node_depth: Tensor,              # [B,4]
        child_iface_feat6: Tensor,             # [B,4,Ti,6]
        child_iface_mask: Tensor,              # [B,4,Ti] bool
        child_iface_boundary_dir: Tensor,      # [B,4,Ti] long
        child_iface_inside_endpoint: Tensor,   # [B,4,Ti] long
        child_iface_inside_quadrant: Tensor,   # [B,4,Ti] long
    ) -> TopDownDecoderOutput:
        B = int(node_feat_rel.shape[0])
        if node_feat_rel.shape != (B, 4):
            raise ValueError(f"node_feat_rel must be [B,4], got {tuple(node_feat_rel.shape)}")
        if node_depth.shape != (B,):
            raise ValueError(f"node_depth must be [B], got {tuple(node_depth.shape)}")
        if z_node.shape != (B, self.d_model):
            raise ValueError(f"z_node must be [B,{self.d_model}], got {tuple(z_node.shape)}")

        if iface_feat6.dim() != 3 or iface_feat6.shape[0] != B or iface_feat6.shape[2] != 6:
            raise ValueError(f"iface_feat6 must be [B,Ti,6], got {tuple(iface_feat6.shape)}")
        if iface_mask.shape != iface_feat6.shape[:2]:
            raise ValueError("iface_mask must be [B,Ti] and match iface_feat6.")
        Ti = int(iface_feat6.shape[1])

        if cross_feat6.dim() != 3 or cross_feat6.shape[0] != B or cross_feat6.shape[2] != 6:
            raise ValueError(f"cross_feat6 must be [B,Tc,6], got {tuple(cross_feat6.shape)}")
        if cross_mask.shape != cross_feat6.shape[:2]:
            raise ValueError("cross_mask must be [B,Tc] and match cross_feat6.")
        Tc = int(cross_feat6.shape[1])

        if bc_in_iface_logit.shape != (B, Ti):
            raise ValueError(f"bc_in_iface_logit must be [B,Ti]={B,Ti}, got {tuple(bc_in_iface_logit.shape)}")

        if child_z.shape != (B, 4, self.d_model):
            raise ValueError(f"child_z must be [B,4,{self.d_model}], got {tuple(child_z.shape)}")
        if child_exists_mask.shape != (B, 4):
            raise ValueError(f"child_exists_mask must be [B,4], got {tuple(child_exists_mask.shape)}")

        # ---- A) parent memory tokenization (CLS + IFACE + CROSS + CHILD) ----
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
        tokens = mem.tokens          # [B,T,d]
        mask = mem.mask.bool()       # [B,T]
        cls_idx = int(mem.cls_index)

        # inject z_node into CLS
        tokens[:, cls_idx, :] = tokens[:, cls_idx, :] + self.z_node_proj(z_node)

        # inject bc_in into parent IFACE tokens
        bc = bc_in_iface_logit.to(dtype=tokens.dtype)
        bc = torch.where(iface_mask.bool(), bc, torch.zeros_like(bc))
        bc = torch.clamp(bc, -self.bc_clip, self.bc_clip)
        bc_emb = self.bc_proj(bc.unsqueeze(-1))  # [B,Ti,d]
        tokens[:, mem.iface_slice, :] = tokens[:, mem.iface_slice, :] + bc_emb

        # ---- prepare child iface queries (always built; used differently per mode) ----
        child_tok, child_tok_mask = self._embed_child_iface_tokens(
            child_node_feat_rel=child_node_feat_rel,
            child_node_depth=child_node_depth,
            child_iface_feat6=child_iface_feat6,
            child_iface_mask=child_iface_mask,
            child_iface_boundary_dir=child_iface_boundary_dir,
            child_iface_inside_endpoint=child_iface_inside_endpoint,
            child_iface_inside_quadrant=child_iface_inside_quadrant,
            child_exists_mask=child_exists_mask.bool(),
        )  # tok [B,4,Ti,d], mask [B,4,Ti]
        child_q = child_tok.reshape(B, 4 * Ti, self.d_model)
        child_q_mask = child_tok_mask.reshape(B, 4 * Ti).bool()

        # ---- A') build memory with parent blocks (and optionally include queries for one_stage) ----
        if self.mode == "one_stage":
            base_T = int(tokens.shape[1])
            tokens_all = torch.cat([tokens, child_q], dim=1)          # [B, base_T+4Ti, d]
            mask_all = torch.cat([mask, child_q_mask], dim=1)         # [B, base_T+4Ti]
            x = tokens_all
            for blk in self.parent_blocks:
                x = blk(x, mask_all)

            # slices: parent part uses mem slices; child query part is the tail
            parent_x = x[:, :base_T, :]
            child_x = x[:, base_T:, :]  # [B,4Ti,d]
        else:
            parent_x = tokens
            for blk in self.parent_blocks:
                parent_x = blk(parent_x, mask)
            child_x = None  # filled by cross-attn below

        # ---- local logits from parent memory ----
        iface_out = parent_x[:, mem.iface_slice, :]  # [B,Ti,d]
        cross_out = parent_x[:, mem.cross_slice, :]  # [B,Tc,d]
        iface_logit = self.iface_head(iface_out).squeeze(-1)  # [B,Ti]
        cross_logit = self.cross_head(cross_out).squeeze(-1)  # [B,Tc]

        # mask local logits to sentinel
        neg_inf = torch.tensor(self._NEG_INF, device=iface_logit.device, dtype=iface_logit.dtype)
        iface_logit = torch.where(iface_mask.bool(), iface_logit, neg_inf)
        cross_logit = torch.where(cross_mask.bool(), cross_logit, neg_inf)

        # ---- child boundary logits ----
        if self.mode == "one_stage":
            assert child_x is not None
            child_iface_logit_flat = self.child_iface_head(child_x).squeeze(-1)  # [B,4Ti]
        else:
            # two_stage: cross-attend queries to parent memory
            q = child_q
            kv = parent_x
            kv_mask = mask
            for blk in self.cross_blocks:
                q = blk(q=q, q_mask=child_q_mask, kv=kv, kv_mask=kv_mask)
            child_iface_logit_flat = self.child_iface_head(q).squeeze(-1)  # [B,4Ti]

        child_iface_logit = child_iface_logit_flat.reshape(B, 4, Ti)

        # mask child logits
        valid_child_iface = child_tok_mask.bool()  # already AND with child_exists
        child_iface_logit = torch.where(valid_child_iface, child_iface_logit, neg_inf)

        aux: Dict[str, Tensor] = {}
        if self.return_aux:
            aux = {
                "num_parent_iface_valid": iface_mask.sum(dim=1).to(dtype=torch.long),
                "num_parent_cross_valid": cross_mask.sum(dim=1).to(dtype=torch.long),
                "num_child_iface_valid_total": valid_child_iface.sum(dim=(1, 2)).to(dtype=torch.long),
                "has_any_child": child_exists_mask.any(dim=1).to(dtype=torch.long),
            }

        return TopDownDecoderOutput(
            iface_logit=iface_logit,
            cross_logit=cross_logit,
            child_iface_logit=child_iface_logit,
            aux=aux,
        )


# Public alias
TopDownDecoder = TopDownDecoderModule
__all__ = ["TopDownDecoder", "TopDownDecoderModule", "TopDownDecoderOutput"]
