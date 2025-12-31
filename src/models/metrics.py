# src/models/metrics.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class EdgeLabelPack:
    """
    Edge-level labels aligned to GLOBAL EID space in a PackedBatch.

    y_edge:   [E_total] float in {0,1}
    m_edge:   [E_total] bool indicating edges that are "covered" by cross tokens
    """
    y_edge: Tensor
    m_edge: Tensor


def build_edge_labels_from_token_labels(
    *,
    total_E: int,
    cross_eid: Tensor,     # [M,Tc] long (GLOBAL eid)
    cross_mask: Tensor,    # [M,Tc] bool
    y_cross: Tensor,       # [M,Tc] float in {0,1}
) -> EdgeLabelPack:
    """
    Convert token-level labels into edge-level labels:
      y_edge[e] = 1  if there exists any cross token with eid==e and y_cross==1.
      m_edge[e] = 1  if there exists any cross token with eid==e (i.e., edge is covered).

    This is non-differentiable (labels only), safe and fast enough for current batch sizes.
    """
    if total_E <= 0:
        raise ValueError("total_E must be positive.")
    if cross_eid.shape != cross_mask.shape or cross_eid.shape != y_cross.shape:
        raise ValueError("cross_eid/cross_mask/y_cross must have same shape.")

    device = cross_eid.device
    y_edge = torch.zeros((total_E,), device=device, dtype=torch.float32)
    m_edge = torch.zeros((total_E,), device=device, dtype=torch.bool)

    m = cross_mask.bool() & (cross_eid >= 0)
    if not m.any().item():
        return EdgeLabelPack(y_edge=y_edge, m_edge=m_edge)

    eids = cross_eid[m].long()
    # covered edges
    m_edge.index_fill_(0, torch.unique(eids), True)

    # positive edges (any token positive)
    pos = m & (y_cross > 0.5)
    if pos.any().item():
        pos_eids = cross_eid[pos].long()
        y_edge.index_fill_(0, torch.unique(pos_eids), 1.0)

    return EdgeLabelPack(y_edge=y_edge, m_edge=m_edge)


def edge_pr_at_k(
    *,
    edge_logit: Tensor,   # [E_total]
    y_edge: Tensor,       # [E_total]
    m_edge: Tensor,       # [E_total] bool
    k: int,
) -> Dict[str, Tensor]:
    """
    Precision/Recall at top-k edges within the masked set.
    """
    if edge_logit.dim() != 1:
        raise ValueError("edge_logit must be 1D [E].")
    if edge_logit.shape != y_edge.shape or edge_logit.shape != m_edge.shape:
        raise ValueError("edge_logit/y_edge/m_edge must have same shape.")

    device = edge_logit.device
    m = m_edge.bool()
    if not m.any().item():
        return {
            "p@k": torch.tensor(0.0, device=device),
            "r@k": torch.tensor(0.0, device=device),
            "k_eff": torch.tensor(0, device=device, dtype=torch.long),
        }

    scores = edge_logit[m]
    labels = y_edge[m]

    k_eff = min(int(k), int(scores.numel()))
    if k_eff <= 0:
        return {
            "p@k": torch.tensor(0.0, device=device),
            "r@k": torch.tensor(0.0, device=device),
            "k_eff": torch.tensor(0, device=device, dtype=torch.long),
        }

    topk = torch.topk(scores, k=k_eff, largest=True).indices
    sel = labels[topk]

    tp = sel.sum()
    total_pos = labels.sum().clamp_min(1.0)

    p = tp / float(k_eff)
    r = tp / total_pos
    return {
        "p@k": p,
        "r@k": r,
        "k_eff": torch.tensor(k_eff, device=device, dtype=torch.long),
    }
