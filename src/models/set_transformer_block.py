# src/models/set_transformer_block.py
# -*- coding: utf-8 -*-
"""
SetTransformer-style interaction blocks for token-level memory in Neural DP.

This module provides masked self-attention blocks that operate on per-node token
memories produced by NodeTokenizer:

  tokens: [B, T, d_model]
  mask:   [B, T] bool   (True = valid token, False = padding)

Design goals:
- Scale-invariant: purely operates on learned embeddings; no absolute coordinates.
- Robust padding: padded tokens are ignored in attention and kept as zeros.
- Production-friendly: pre-norm Transformer blocks, configurable depth/width.

Typical usage (inside Leaf/Merge Encoder):
  mem = tokenizer(...)
  x = encoder(mem.tokens, mem.mask)
  z = x[:, mem.cls_index]    # CLS pooling (recommended)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# -----------------------------
# Pooling helpers
# -----------------------------

def cls_pool(tokens: Tensor, cls_index: int = 0) -> Tensor:
    """
    CLS pooling: return tokens[:, cls_index, :].
    tokens: [B,T,d]
    """
    return tokens[:, cls_index, :]


def masked_mean_pool(tokens: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Mean pool over valid tokens.
    tokens: [B,T,d]
    mask:   [B,T] bool
    """
    m = mask.to(dtype=tokens.dtype).unsqueeze(-1)  # [B,T,1]
    s = (tokens * m).sum(dim=1)                    # [B,d]
    denom = m.sum(dim=1).clamp_min(eps)            # [B,1]
    return s / denom


# -----------------------------
# Core blocks
# -----------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if hidden_mult <= 0:
            raise ValueError("hidden_mult must be positive.")
        hidden = d_model * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, d_model),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.net(x))


class SetTransformerBlock(nn.Module):
    """
    A pre-norm Transformer encoder block with masked self-attention.

    Forward:
      x = x + MHA(LN(x), key_padding_mask=~mask)
      x = x + FFN(LN(x))
      x padded positions are kept at 0.
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
        if n_heads <= 0:
            raise ValueError("n_heads must be positive.")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drop_attn = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model=d_model, hidden_mult=ff_hidden_mult, dropout=dropout)

    @staticmethod
    def _apply_pad_zero(x: Tensor, mask: Tensor) -> Tensor:
        # Ensure padded token embeddings stay exactly zero.
        return torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        x:    [B,T,d_model]
        mask: [B,T] bool (True=valid)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x shape [B,T,d], got {tuple(x.shape)}")
        if mask.dim() != 2:
            raise ValueError(f"Expected mask shape [B,T], got {tuple(mask.shape)}")
        if x.shape[:2] != mask.shape:
            raise ValueError(f"x and mask shape mismatch: x={tuple(x.shape)}, mask={tuple(mask.shape)}")

        # Force padded positions to zero before any computation.
        x = self._apply_pad_zero(x, mask)

        # ---- Self-attention (pre-norm) ----
        y = self.ln1(x)
        # key_padding_mask: True means "ignore"
        key_padding_mask = ~mask  # [B,T]
        attn_out, _ = self.attn(
            query=y,
            key=y,
            value=y,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop_attn(attn_out)
        x = self._apply_pad_zero(x, mask)

        # ---- Feed-forward (pre-norm) ----
        y = self.ln2(x)
        x = x + self.ff(y)
        x = self._apply_pad_zero(x, mask)

        return x


class SetTransformerEncoder(nn.Module):
    """
    Stack of SetTransformerBlocks.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_layers: int = 2,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        ff_hidden_mult: int = 4,
        final_norm: bool = True,
    ) -> None:
        super().__init__()
        if n_layers <= 0:
            raise ValueError("n_layers must be positive.")
        self.blocks = nn.ModuleList(
            [
                SetTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    ff_hidden_mult=ff_hidden_mult,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model) if final_norm else nn.Identity()

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        x = tokens
        for blk in self.blocks:
            x = blk(x, mask)
        return self.final_norm(x)


# -----------------------------
# Optional: a thin wrapper that also returns pooled latent
# -----------------------------

@dataclass(frozen=True)
class EncoderOutput:
    tokens: Tensor  # [B,T,d]
    pooled: Tensor  # [B,d]


class CLSSetEncoder(nn.Module):
    """
    Convenience module: SetTransformerEncoder + CLS pooling.

    Use when your encoder's latent is simply the CLS token after interaction.
    """

    def __init__(self, encoder: SetTransformerEncoder, cls_index: int = 0) -> None:
        super().__init__()
        self.encoder = encoder
        self.cls_index = int(cls_index)

    def forward(self, tokens: Tensor, mask: Tensor) -> EncoderOutput:
        x = self.encoder(tokens, mask)
        z = cls_pool(x, self.cls_index)
        return EncoderOutput(tokens=x, pooled=z)
