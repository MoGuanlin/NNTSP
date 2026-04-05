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


@dataclass
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


def aggregate_logits_to_edges(
    *,
    tokens: PackedNodeTokens,
    cross_logit: Tensor,           # [M,Tc]
    iface_logit: Tensor | None = None, # [M,Ti]
    reduce: str = "amax",
    fill_value: float = float("-inf"),
    num_edges: int | None = None,
) -> EdgeScores:
    """
    Aggregate cross (and optionally iface) logits into per-edge logits.

    Args:
      tokens: PackedNodeTokens containing eid mappings.
      cross_logit: Logits for crossing tokens.
      iface_logit: Optional logits for interface tokens.
      reduce: Reduction method ("amax" or "mean").
      fill_value: Initial value for edge_logit.
      num_edges: Optional explicit total number of edges. If None, inferred from tokens.
                 CRITICAL: When using r-light pruning, some high-index edges might not be covered 
                 by any tokens. In such cases, inferring from tokens will underestimate E, 
                 causing downstream shape mismatches. Always provide this if known.

    Returns:
      EdgeScores(edge_logit=[E], edge_mask=[E])
    """
    if cross_logit.dim() != 2 or cross_logit.shape != tokens.cross_eid.shape:
        raise ValueError(
            f"cross_logit must match tokens.cross_eid shape {tuple(tokens.cross_eid.shape)}, got {tuple(cross_logit.shape)}"
        )

    E_inferred = infer_num_edges_from_tokens(tokens)
    if num_edges is not None:
        E = int(num_edges)
        if E < E_inferred:
            raise ValueError(f"Provided num_edges={E} is less than inferred max eid+1={E_inferred}.")
    else:
        E = E_inferred

    device = cross_logit.device
    
    # Initialization Differentiation:
    # - For "amax": We start with -inf (fill_value) so that any real logit will replace it.
    # - For "mean": We start with 0.0. Note that if we use include_self=False later, 
    #   this 0.0 is just a placeholder and won't affect the average calculation.
    init_val = 0.0 if reduce == "mean" else fill_value
    edge_logit = torch.full((E,), fill_value=init_val, device=device, dtype=cross_logit.dtype)
    edge_mask = torch.zeros((E,), device=device, dtype=torch.bool)
    if E == 0:
        return EdgeScores(edge_logit=edge_logit, edge_mask=edge_mask)

    # 1. Gather crossing evidence
    m_c = tokens.cross_mask.bool() & (tokens.cross_eid >= 0)
    eid_list = [tokens.cross_eid[m_c].long()]
    val_list = [cross_logit[m_c]]
    
    # 2. Gather interface evidence (optional)
    if iface_logit is not None:
        if iface_logit.dim() != 2 or iface_logit.shape != tokens.iface_mask.shape:
            raise ValueError(
                f"iface_logit must match tokens.iface_mask shape {tuple(tokens.iface_mask.shape)}, got {tuple(iface_logit.shape)}"
            )
        m_i = tokens.iface_mask.bool() & (tokens.iface_eid >= 0)
        eid_list.append(tokens.iface_eid[m_i].long())
        val_list.append(iface_logit[m_i])

    if not any(e.numel() > 0 for e in eid_list):
        return EdgeScores(edge_logit=edge_logit, edge_mask=edge_mask)

    eid = torch.cat(eid_list, dim=0)
    val = torch.cat(val_list, dim=0)
    edge_mask[eid] = True

    # scatter-reduce (PyTorch 2.0+)
    # Logic for include_self:
    # - For "amax": inclusive_self=True is used because -inf is the identity element for max.
    # - For "mean": inclusive_self=False is CRITICAL. If True, the initial 0.0 would be 
    #   treated as an additional data point, biasing the average towards zero.
    try:
        edge_logit = edge_logit.scatter_reduce(0, eid, val, reduce=reduce, include_self=(reduce != "mean"))
    except Exception:
        # Fallback: sort by eid then reduce per group.
        order = torch.argsort(eid)
        eid_s = eid[order]
        val_s = val[order]
        diff = torch.ones_like(eid_s, dtype=torch.bool)
        diff[1:] = eid_s[1:] != eid_s[:-1]
        starts = torch.nonzero(diff, as_tuple=False).view(-1)
        for i in range(len(starts)):
            si = int(starts[i].item())
            sj = int(starts[i+1].item()) if i+1 < len(starts) else eid_s.numel()
            e = int(eid_s[si].item())
            if reduce == "amax":
                edge_logit[e] = val_s[si:sj].max()
            elif reduce == "mean":
                edge_logit[e] = val_s[si:sj].mean()

    return EdgeScores(edge_logit=edge_logit, edge_mask=edge_mask)

__all__ = ["EdgeScores", "infer_num_edges_from_tokens", "aggregate_logits_to_edges"]
