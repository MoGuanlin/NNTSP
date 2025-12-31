# src/models/edge_aggregation.py
# -*- coding: utf-8 -*-
"""
Edge aggregation utilities.

Primary use:
- Convert per-node per-crossing logits into a per-edge score vector using cross_eid.
  In this project design, each spanner edge has a unique "home" crossing token,
  so aggregation should be a no-op; nevertheless we implement a safe reduce.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

try:
    from src.models.node_token_packer import PackedNodeTokens
except Exception:  # pragma: no cover
    from .node_token_packer import PackedNodeTokens


@dataclass(frozen=True)
class EdgeScores:
    edge_logit: Tensor   # [E]
    edge_mask: Tensor    # [E] bool (True for edges that appear in tokens)


def infer_num_edges_from_tokens(tokens: PackedNodeTokens) -> int:
    """
    Infer E = max_eid+1 from packed tokens (iface and cross eids).
    Assumes eids are contiguous and -1 is used for padding.
    """
    e1 = tokens.iface_eid[tokens.iface_eid >= 0]
    e2 = tokens.cross_eid[tokens.cross_eid >= 0]
    if e1.numel() == 0 and e2.numel() == 0:
        return 0
    mx = 0
    if e1.numel() > 0:
        mx = max(mx, int(e1.max().item()))
    if e2.numel() > 0:
        mx = max(mx, int(e2.max().item()))
    return mx + 1


def aggregate_cross_logits_to_edges(
    *,
    tokens: PackedNodeTokens,
    cross_logit: Tensor,           # [M,Tc]
    fill_value: float = float("-inf"),
) -> EdgeScores:
    """
    Aggregate cross logits into per-edge logits using cross_eid.

    Uses amax reduction to be robust to accidental duplicates.

    Returns:
      EdgeScores(edge_logit=[E], edge_mask=[E])
    """
    if cross_logit.dim() != 2 or cross_logit.shape != tokens.cross_eid.shape:
        raise ValueError(
            f"cross_logit must match tokens.cross_eid shape {tuple(tokens.cross_eid.shape)}, got {tuple(cross_logit.shape)}"
        )

    E = infer_num_edges_from_tokens(tokens)
    device = cross_logit.device
    edge_logit = torch.full((E,), fill_value=fill_value, device=device, dtype=cross_logit.dtype)
    edge_mask = torch.zeros((E,), device=device, dtype=torch.bool)
    if E == 0:
        return EdgeScores(edge_logit=edge_logit, edge_mask=edge_mask)

    m = tokens.cross_mask.bool() & (tokens.cross_eid >= 0)
    if not m.any().item():
        return EdgeScores(edge_logit=edge_logit, edge_mask=edge_mask)

    eid = tokens.cross_eid[m].long()          # [K]
    val = cross_logit[m]                      # [K]
    edge_mask[eid] = True

    # scatter-reduce amax (PyTorch 2.0+)
    try:
        edge_logit = edge_logit.scatter_reduce(0, eid, val, reduce="amax", include_self=True)
    except Exception:
        # Fallback: sort by eid then take max per group.
        order = torch.argsort(eid)
        eid_s = eid[order]
        val_s = val[order]
        diff = torch.ones_like(eid_s, dtype=torch.bool)
        diff[1:] = eid_s[1:] != eid_s[:-1]
        starts = torch.nonzero(diff, as_tuple=False).view(-1)
        for si, sj in zip(starts.tolist(), starts.tolist()[1:] + [eid_s.numel()]):
            e = int(eid_s[si].item())
            edge_logit[e] = val_s[si:sj].max()

    return EdgeScores(edge_logit=edge_logit, edge_mask=edge_mask)


__all__ = ["EdgeScores", "infer_num_edges_from_tokens", "aggregate_cross_logits_to_edges"]
