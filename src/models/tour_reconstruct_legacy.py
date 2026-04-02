# src/models/tour_reconstruct_legacy.py
# -*- coding: utf-8 -*-
"""
Legacy 1-pass reconstruction helpers.

These utilities project DP traceback states back into hard iface/cross logits
and then reuse the existing edge-score decode stack. They remain available for
compatibility and baselines, but are not the primary 1-pass reconstruction path.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from .bc_state_catalog import BoundaryStateCatalog
from .dp_runner import CostTableEntry, OnePassDPResult
from .edge_aggregation import EdgeScores, aggregate_logits_to_edges
from .node_token_packer import PackedNodeTokens


_LOGIT_ON = 10.0
_LOGIT_OFF = -10.0


def _collect_internal_states(
    *,
    root_id: int,
    root_sigma: int,
    tokens: PackedNodeTokens,
    cost_tables: Dict[int, CostTableEntry],
    node_states: Dict[int, int],
) -> None:
    """Traverse backpointers to collect states at all internal nodes."""
    queue = [(root_id, root_sigma)]
    while queue:
        nid, sigma_idx = queue.pop()
        node_states[nid] = sigma_idx

        if tokens.is_leaf[nid].item():
            continue

        ct = cost_tables.get(nid)
        if ct is None:
            continue
        child_indices = ct.backptr.get(sigma_idx)
        if child_indices is None:
            continue

        children = tokens.tree_children_index[nid].long()
        for q in range(4):
            cid = int(children[q].item())
            if cid < 0:
                continue
            c_si = child_indices[q]
            if c_si < 0:
                continue
            queue.append((cid, c_si))


def dp_result_to_logits(
    *,
    result: OnePassDPResult,
    tokens: PackedNodeTokens,
    catalog: BoundaryStateCatalog,
) -> Tuple[Tensor, Tensor]:
    """Convert DP result to per-node iface/cross logits."""
    m = int(tokens.iface_mask.shape[0])
    ti = int(tokens.iface_mask.shape[1])
    tc = int(tokens.cross_mask.shape[1])
    device = tokens.iface_mask.device

    iface_logit = torch.full((m, ti), _LOGIT_OFF, dtype=torch.float32, device=device)
    cross_logit = torch.full((m, tc), _LOGIT_OFF, dtype=torch.float32, device=device)

    node_states: Dict[int, int] = {}
    node_states.update(result.leaf_states)

    root_id = int(tokens.root_id[0].item())
    if result.root_sigma >= 0:
        _collect_internal_states(
            root_id=root_id,
            root_sigma=result.root_sigma,
            tokens=tokens,
            cost_tables=result.cost_tables,
            node_states=node_states,
        )

    for nid, si in node_states.items():
        if si < 0:
            continue
        a = catalog.used_iface[si]
        mask = tokens.iface_mask[nid].bool()
        for slot in range(ti):
            if not mask[slot].item():
                continue
            if a[slot].item():
                iface_logit[nid, slot] = _LOGIT_ON

    for nid, si in node_states.items():
        if tokens.is_leaf[nid].item():
            continue
        if si < 0:
            continue

        children = tokens.tree_children_index[nid].long()
        child_exists = children >= 0

        for t in range(tc):
            if not tokens.cross_mask[nid, t].bool().item():
                continue
            cross_eid = int(tokens.cross_eid[nid, t].item())
            if cross_eid < 0:
                continue

            qi = int(tokens.cross_child_pair[nid, t, 0].item())
            qj = int(tokens.cross_child_pair[nid, t, 1].item())
            if qi < 0 or qj < 0:
                continue
            if not child_exists[qi].item() or not child_exists[qj].item():
                continue

            cid_i = int(children[qi].item())
            cid_j = int(children[qj].item())

            si_i = node_states.get(cid_i, -1)
            si_j = node_states.get(cid_j, -1)
            if si_i < 0 or si_j < 0:
                continue

            active_i = False
            for s in range(ti):
                if (tokens.iface_eid[cid_i, s].item() == cross_eid
                        and tokens.iface_mask[cid_i, s].bool().item()):
                    active_i = bool(catalog.used_iface[si_i, s].item())
                    break

            active_j = False
            for s in range(ti):
                if (tokens.iface_eid[cid_j, s].item() == cross_eid
                        and tokens.iface_mask[cid_j, s].bool().item()):
                    active_j = bool(catalog.used_iface[si_j, s].item())
                    break

            if active_i and active_j:
                cross_logit[nid, t] = _LOGIT_ON

    return iface_logit, cross_logit


def dp_result_to_edge_scores(
    *,
    result: OnePassDPResult,
    tokens: PackedNodeTokens,
    catalog: BoundaryStateCatalog,
    num_edges: Optional[int] = None,
) -> EdgeScores:
    """Convert DP result to spanner edge scores for legacy post-decoding baselines."""
    iface_logit, cross_logit = dp_result_to_logits(
        result=result,
        tokens=tokens,
        catalog=catalog,
    )
    return aggregate_logits_to_edges(
        tokens=tokens,
        cross_logit=cross_logit,
        iface_logit=iface_logit,
        reduce="amax",
        num_edges=num_edges,
    )


__all__ = ["dp_result_to_edge_scores", "dp_result_to_logits"]
