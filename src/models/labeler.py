# src/models/labeler.py
# -*- coding: utf-8 -*-

from __future__ import annotations

"""Pseudo label generation for Neural Rao'98 DP.

Outputs:
  - y_cross/m_cross: token-level labels for crossing tokens
  - y_iface/m_iface: token-level labels for interface tokens
  - y_child_iface/m_child_iface: per-node labels supervising the boundary
    conditions passed from each node to its 4 children.

The teacher is built by computing a heuristic TSP tour on the complete Euclidean
graph and projecting each tour edge onto the alive spanner subgraph using
Dijkstra shortest paths.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import heapq
from types import SimpleNamespace

import torch
from torch import Tensor

try:
    from src.models.tour_solver import solve_tsp_heuristic, tour_edges
except Exception:  # pragma: no cover
    from .tour_solver import solve_tsp_heuristic, tour_edges


@dataclass(frozen=True)
class TokenLabels:
    """Token-level pseudo labels aligned to PackedNodeTokens fields.

    Shapes:
      y_cross: [M,Tc] float in {0,1}
      m_cross: [M,Tc] bool (valid tokens)
      y_iface: [M,Ti] float in {0,1}
      m_iface: [M,Ti] bool
      y_child_iface: [M,4,Ti] float in {0,1} (targets for parent->child BC)
      m_child_iface: [M,4,Ti] bool
      stats: dict of diagnostics
    """

    y_cross: Tensor
    m_cross: Tensor
    y_iface: Tensor
    m_iface: Tensor
    y_child_iface: Tensor
    m_child_iface: Tensor
    stats: Dict[str, Tensor]


def _build_spanner_eid_map(edge_index: Tensor) -> Dict[Tuple[int, int], int]:
    """Build (min(u,v), max(u,v)) -> local_eid map for spanner edges."""
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be [2,E], got {tuple(edge_index.shape)}")
    E = int(edge_index.shape[1])
    mp: Dict[Tuple[int, int], int] = {}
    u = edge_index[0].tolist()
    v = edge_index[1].tolist()
    for eid in range(E):
        a, b = int(u[eid]), int(v[eid])
        mp[(a, b) if a < b else (b, a)] = eid
    return mp


def _dijkstra_path_eids(
    N: int,
    adj: List[List[Tuple[int, int, float]]],
    src: int,
    dst: int,
) -> Optional[List[int]]:
    """Shortest path over alive adjacency.

    adj[u] = list of (v, local_eid, w)
    Returns list of local_eid along the shortest path, or None if unreachable.
    """
    INF = 1e100
    dist = [INF] * N
    prev_v = [-1] * N
    prev_e = [-1] * N
    dist[src] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, src)]

    while pq:
        du, u = heapq.heappop(pq)
        if du != dist[u]:
            continue
        if u == dst:
            break
        for v, eid, w in adj[u]:
            nd = du + w
            if nd < dist[v]:
                dist[v] = nd
                prev_v[v] = u
                prev_e[v] = eid
                heapq.heappush(pq, (nd, v))

    if dist[dst] >= INF / 2:
        return None

    path_eids: List[int] = []
    cur = dst
    while cur != src:
        eid = prev_e[cur]
        if eid < 0:
            return None
        path_eids.append(int(eid))
        cur = prev_v[cur]
        if cur < 0:
            return None
    path_eids.reverse()
    return path_eids


def _make_child_iface_targets_local(
    *,
    y_iface: Tensor,            # [M,Ti]
    iface_mask: Tensor,         # [M,Ti] bool
    tree_children_index: Tensor # [M,4] long
) -> Tuple[Tensor, Tensor]:
    """Derive y_child_iface/m_child_iface from per-node y_iface.

    This function assumes children indices are LOCAL within the same tensor,
    i.e., child id in [0, M). If this does not hold, the caller should compute
    child targets at the global packed level (see label_batch).
    """
    if y_iface.dim() != 2:
        raise ValueError("y_iface must be [M,Ti].")
    if iface_mask.shape != y_iface.shape:
        raise ValueError("iface_mask must match y_iface shape.")
    if tree_children_index.dim() != 2 or tree_children_index.shape[1] != 4:
        raise ValueError("tree_children_index must be [M,4].")

    M, Ti = int(y_iface.shape[0]), int(y_iface.shape[1])
    ch = tree_children_index.to(device=y_iface.device, dtype=torch.long)
    exists = ch >= 0

    # If the slice uses global indices, we cannot safely compute local child targets.
    if exists.any().item():
        mx = int(ch[exists].max().item())
        if mx >= M:
            # return empty (caller should compute globally)
            y_child = torch.zeros((M, 4, Ti), device=y_iface.device, dtype=y_iface.dtype)
            m_child = torch.zeros((M, 4, Ti), device=y_iface.device, dtype=torch.bool)
            return y_child, m_child

    ch0 = ch.clamp_min(0)
    y_child = y_iface[ch0]  # [M,4,Ti]
    m_child = iface_mask.bool()[ch0] & exists.unsqueeze(-1)
    y_child = y_child * m_child.to(dtype=y_child.dtype)
    return y_child, m_child


class PseudoLabeler:
    """Pseudo labels for pretraining and top-down BC supervision.

    For each graph, we obtain a teacher tour on the complete Euclidean graph and
    then project those edges onto the alive subgraph of the spanner. A spanner
    edge is considered "alive" if it appears in any crossing token with
    cross_mask=True.

    The resulting selected spanner edges produce:
      - cross token labels: y_cross
      - iface token labels: y_iface
      - child BC labels: y_child_iface (derived from y_iface + tree children)

    Notes on batching:
      - In a PackedBatch, eids and node indices are globally offset.
      - label_batch() handles global offsets correctly and computes child targets
        on the global packed tensors.
    """

    def __init__(
        self,
        *,
        two_opt_passes: int = 50,
        prefer_cpu: bool = True,
    ) -> None:
        self.two_opt_passes = int(two_opt_passes)
        self.prefer_cpu = bool(prefer_cpu)

    def label_one(
        self,
        *,
        data: Any,
        tokens_slice: Any,
        device: torch.device,
        eid_offset: int = 0,
    ) -> TokenLabels:
        """Label a single graph slice.

        tokens_slice is expected to have at least:
          cross_eid, cross_mask, iface_eid, iface_mask

        If tokens_slice additionally has local tree_children_index, child BC
        labels are derived locally; otherwise child labels are returned as all
        zeros and should be computed by label_batch.
        """
        eid_offset = int(eid_offset)

        # ---- gather required fields ----
        pos = getattr(data, "pos")
        sp_edge_index = getattr(data, "spanner_edge_index")
        sp_edge_attr = getattr(data, "spanner_edge_attr")

        if self.prefer_cpu:
            pos = pos.detach().cpu()
            sp_edge_index = sp_edge_index.detach().cpu()
            sp_edge_attr = sp_edge_attr.detach().cpu()

        if sp_edge_attr.dim() == 2 and sp_edge_attr.shape[1] == 1:
            sp_w = sp_edge_attr[:, 0].tolist()
        else:
            sp_w = sp_edge_attr.view(-1).tolist()

        N = int(pos.shape[0])
        E = int(sp_edge_index.shape[1])

        # ---- token eid/mask (potentially global eids) ----
        cross_eid_g = getattr(tokens_slice, "cross_eid")
        cross_mask = getattr(tokens_slice, "cross_mask").bool()
        iface_eid_g = getattr(tokens_slice, "iface_eid")
        iface_mask = getattr(tokens_slice, "iface_mask").bool()

        if self.prefer_cpu:
            cross_eid_g = cross_eid_g.detach().cpu()
            cross_mask = cross_mask.detach().cpu()
            iface_eid_g = iface_eid_g.detach().cpu()
            iface_mask = iface_mask.detach().cpu()

        # global->local eid mapping for this graph
        cross_eid_l = torch.where(cross_eid_g >= 0, cross_eid_g - eid_offset, cross_eid_g)
        iface_eid_l = torch.where(iface_eid_g >= 0, iface_eid_g - eid_offset, iface_eid_g)

        alive_m = cross_mask & (cross_eid_l >= 0) & (cross_eid_l < E)
        if alive_m.any().item():
            alive_eids_local = torch.unique(cross_eid_l[alive_m].long()).tolist()
        else:
            alive_eids_local = []
        alive_set_local = set(int(x) for x in alive_eids_local)

        # ---- build spanner edge map (u,v)->local_eid ----
        eid_map = _build_spanner_eid_map(sp_edge_index)

        # ---- build alive adjacency (local eids) for Dijkstra projection ----
        adj: List[List[Tuple[int, int, float]]] = [[] for _ in range(N)]
        u = sp_edge_index[0].tolist()
        v = sp_edge_index[1].tolist()
        for eid in alive_set_local:
            if not (0 <= eid < E):
                continue
            a, b = int(u[eid]), int(v[eid])
            w = float(sp_w[eid])
            adj[a].append((b, eid, w))
            adj[b].append((a, eid, w))

        # ---- teacher tour on complete graph ----
        tour = solve_tsp_heuristic(pos, start=0, max_2opt_passes=self.two_opt_passes)
        t_edges = tour_edges(tour.order)  # list of (u<v)

        # ---- project teacher edges onto alive spanner graph (local eids) ----
        selected_local_eids: set[int] = set()
        num_direct = 0
        num_projected = 0
        num_unreachable = 0
        num_not_alive_direct = 0

        for a, b in t_edges:
            local_eid = eid_map.get((a, b), None)
            if local_eid is not None and local_eid in alive_set_local:
                selected_local_eids.add(int(local_eid))
                num_direct += 1
                continue
            if local_eid is not None and local_eid not in alive_set_local:
                num_not_alive_direct += 1

            path = _dijkstra_path_eids(N, adj, a, b)
            if path is None:
                num_unreachable += 1
                continue
            for pe in path:
                selected_local_eids.add(int(pe))
            num_projected += 1

        # ---- vectorized label construction in LOCAL eid space ----
        eid_table = torch.zeros((E,), dtype=torch.bool)
        for le in selected_local_eids:
            if 0 <= le < E:
                eid_table[le] = True

        # cross labels
        y_cross = torch.zeros_like(cross_eid_g, dtype=torch.float32)
        valid_cross = cross_mask & (cross_eid_l >= 0) & (cross_eid_l < E)
        if valid_cross.any().item():
            idx = cross_eid_l[valid_cross].long()
            y_cross[valid_cross] = eid_table[idx].to(dtype=torch.float32)

        # iface labels
        y_iface = torch.zeros_like(iface_eid_g, dtype=torch.float32)
        valid_iface = iface_mask & (iface_eid_l >= 0) & (iface_eid_l < E)
        if valid_iface.any().item():
            idx2 = iface_eid_l[valid_iface].long()
            y_iface[valid_iface] = eid_table[idx2].to(dtype=torch.float32)

        # ---- optional child BC labels (only if local child indices are usable) ----
        if hasattr(tokens_slice, "tree_children_index"):
            y_child_iface, m_child_iface = _make_child_iface_targets_local(
                y_iface=y_iface,
                iface_mask=iface_mask,
                tree_children_index=getattr(tokens_slice, "tree_children_index"),
            )
        else:
            M = int(y_iface.shape[0])
            Ti = int(y_iface.shape[1])
            y_child_iface = torch.zeros((M, 4, Ti), dtype=torch.float32)
            m_child_iface = torch.zeros((M, 4, Ti), dtype=torch.bool)

        stats = {
            "tour_len": tour.length.detach().cpu(),
            "alive_edges": torch.tensor([len(alive_set_local)], dtype=torch.long),
            "selected_alive_edges": torch.tensor([len(selected_local_eids)], dtype=torch.long),
            "num_direct": torch.tensor([num_direct], dtype=torch.long),
            "num_projected": torch.tensor([num_projected], dtype=torch.long),
            "num_unreachable": torch.tensor([num_unreachable], dtype=torch.long),
            "num_not_alive_direct": torch.tensor([num_not_alive_direct], dtype=torch.long),
        }

        # move to target device
        return TokenLabels(
            y_cross=y_cross.to(device),
            m_cross=cross_mask.to(device),
            y_iface=y_iface.to(device),
            m_iface=iface_mask.to(device),
            y_child_iface=y_child_iface.to(device),
            m_child_iface=m_child_iface.to(device),
            stats={k: v.to(device) for k, v in stats.items()},
        )

    def label_batch(
        self,
        *,
        datas: List[Any],
        packed: Any,
        device: torch.device,
    ) -> TokenLabels:
        """Label a PackedBatch (global packed tensors).

        Requirements:
          packed.node_ptr: [B+1]
          packed.edge_ptr: [B+1]
          packed.tokens: has (cross_eid,cross_mask,iface_eid,iface_mask,tree_children_index)

        Returns labels on `device` with shapes matching packed.tokens.
        """
        if not hasattr(packed, "node_ptr") or not hasattr(packed, "edge_ptr") or not hasattr(packed, "tokens"):
            raise ValueError("packed must have fields: node_ptr, edge_ptr, tokens")

        node_ptr: Tensor = packed.node_ptr
        edge_ptr: Tensor = packed.edge_ptr
        tokens = packed.tokens

        if node_ptr.dim() != 1 or node_ptr.numel() < 2:
            raise ValueError("packed.node_ptr must be 1D with length B+1.")
        if edge_ptr.dim() != 1 or edge_ptr.numel() != node_ptr.numel():
            raise ValueError("packed.edge_ptr must be 1D with the same length as node_ptr.")

        B = int(node_ptr.numel() - 1)
        if len(datas) != B:
            raise ValueError(f"len(datas)={len(datas)} must equal batch size B={B}.")

        total_M = int(node_ptr[-1].item())
        Ti = int(tokens.iface_mask.shape[1])
        Tc = int(tokens.cross_mask.shape[1])

        y_cross_all = torch.zeros((total_M, Tc), dtype=torch.float32, device=device)
        y_iface_all = torch.zeros((total_M, Ti), dtype=torch.float32, device=device)

        m_cross_all = tokens.cross_mask.bool().to(device)
        m_iface_all = tokens.iface_mask.bool().to(device)

        # aggregate stats
        tour_lens: List[Tensor] = []
        alive_edges_sum = 0
        selected_edges_sum = 0
        direct_sum = 0
        projected_sum = 0
        unreachable_sum = 0
        not_alive_direct_sum = 0

        for b in range(B):
            n0 = int(node_ptr[b].item())
            n1 = int(node_ptr[b + 1].item())
            e0 = int(edge_ptr[b].item())

            ts = SimpleNamespace(
                cross_eid=tokens.cross_eid[n0:n1],
                cross_mask=tokens.cross_mask[n0:n1],
                iface_eid=tokens.iface_eid[n0:n1],
                iface_mask=tokens.iface_mask[n0:n1],
                # include children so label_one can compute local child labels if indices are local;
                # in batch they are global, so label_one will detect and return zeros for child.
                tree_children_index=tokens.tree_children_index[n0:n1],
            )

            lab = self.label_one(data=datas[b], tokens_slice=ts, device=device, eid_offset=e0)

            y_cross_all[n0:n1] = lab.y_cross
            y_iface_all[n0:n1] = lab.y_iface

            # stats
            tour_lens.append(lab.stats["tour_len"].view(-1))
            alive_edges_sum += int(lab.stats["alive_edges"].item())
            selected_edges_sum += int(lab.stats["selected_alive_edges"].item())
            direct_sum += int(lab.stats["num_direct"].item())
            projected_sum += int(lab.stats["num_projected"].item())
            unreachable_sum += int(lab.stats["num_unreachable"].item())
            not_alive_direct_sum += int(lab.stats["num_not_alive_direct"].item())

        # ---- child BC labels (GLOBAL packed indices) ----
        ch = tokens.tree_children_index.long().to(device)  # [total_M,4]
        exists = ch >= 0
        ch0 = ch.clamp_min(0)

        y_child_iface = y_iface_all[ch0]  # [M,4,Ti]
        m_child_iface = m_iface_all[ch0] & exists.unsqueeze(-1)
        y_child_iface = y_child_iface * m_child_iface.to(dtype=y_child_iface.dtype)

        # ---- stats dict ----
        if len(tour_lens) > 0:
            tour_len_mean = torch.cat(tour_lens, dim=0).mean()
        else:
            tour_len_mean = torch.tensor(0.0, device=device)

        stats = {
            "tour_len_mean": tour_len_mean,
            "alive_edges_sum": torch.tensor([alive_edges_sum], dtype=torch.long, device=device),
            "selected_alive_edges_sum": torch.tensor([selected_edges_sum], dtype=torch.long, device=device),
            "num_direct_sum": torch.tensor([direct_sum], dtype=torch.long, device=device),
            "num_projected_sum": torch.tensor([projected_sum], dtype=torch.long, device=device),
            "num_unreachable_sum": torch.tensor([unreachable_sum], dtype=torch.long, device=device),
            "num_not_alive_direct_sum": torch.tensor([not_alive_direct_sum], dtype=torch.long, device=device),
        }

        return TokenLabels(
            y_cross=y_cross_all,
            m_cross=m_cross_all,
            y_iface=y_iface_all,
            m_iface=m_iface_all,
            y_child_iface=y_child_iface,
            m_child_iface=m_child_iface,
            stats=stats,
        )


__all__ = ["PseudoLabeler", "TokenLabels"]
