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
    from src.models.bc_state_catalog import project_iface_usage_to_state_index, project_matching_to_state_index
    from src.models.tour_solver import solve_tsp_heuristic, tour_edges, Tour, tour_length, pairwise_dist
    from src.utils.lkh_solver import solve_tsp_lkh
except Exception:  # pragma: no cover
    from .bc_state_catalog import project_iface_usage_to_state_index, project_matching_to_state_index
    from .tour_solver import solve_tsp_heuristic, tour_edges, Tour, tour_length, pairwise_dist
    from ..utils.lkh_solver import solve_tsp_lkh


@dataclass
class TokenLabels:
    """Token-level pseudo labels aligned to PackedNodeTokens fields.

    Shapes:
      y_cross: [M,Tc] float in {0,1}
      m_cross: [M,Tc] bool (valid tokens)
      y_iface: [M,Ti] float in {0,1}
      m_iface: [M,Ti] bool
      y_child_iface: [M,4,Ti] float in {0,1} (targets for parent->child BC)
      m_child_iface: [M,4,Ti] bool
      target_state_idx: [M] long (matching-state target id, or -1 if unavailable)
      m_state: [M] bool
      stats: dict of diagnostics
    """

    y_cross: Tensor
    m_cross: Tensor
    y_iface: Tensor
    m_iface: Tensor
    y_child_iface: Tensor
    m_child_iface: Tensor
    target_state_idx: Tensor
    m_state: Tensor
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
        use_lkh: bool = False,
        lkh_exe: str = "LKH",
        prefer_cpu: bool = True,
    ) -> None:
        self.two_opt_passes = int(two_opt_passes)
        self.use_lkh = bool(use_lkh)
        self.lkh_exe = str(lkh_exe)
        self.prefer_cpu = bool(prefer_cpu)

    @staticmethod
    def simplify_data_for_ipc(data: Any) -> Dict[str, Any]:
        """Convert torch.Data tensors to numpy for safe/efficient IPC."""
        return {
            "pos": data.pos.detach().cpu().numpy(),
            "spanner_edge_index": data.spanner_edge_index.detach().cpu().numpy(),
            "spanner_edge_attr": data.spanner_edge_attr.detach().cpu().numpy(),
        }

    def extract_target_edges(
        self,
        data: Any,
    ) -> Tuple[Any, float]:
        """Compute the set of spanner edge indices that form the projected teacher tour.
        
        Args:
            data: Either a PyG Data object or a dict of numpy arrays (for IPC).

        Returns:
            Tuple of (numpy.ndarray of edge indices, float of tour length).
        """
        import numpy as np

        if isinstance(data, dict):
            # Input from IPC (numpy simplified)
            pos_np = data["pos"]
            edge_index_np = data["spanner_edge_index"]
            edge_attr_np = data["spanner_edge_attr"]
        else:
            # Standard Data object
            pos_np = data.pos.detach().cpu().numpy()
            edge_index_np = data.spanner_edge_index.detach().cpu().numpy()
            edge_attr_np = data.spanner_edge_attr.detach().cpu().numpy()

        if edge_attr_np.ndim == 2 and edge_attr_np.shape[1] == 1:
            sp_w = edge_attr_np[:, 0].tolist()
        else:
            sp_w = edge_attr_np.reshape(-1).tolist()

        N = int(pos_np.shape[0])
        E = int(edge_index_np.shape[1])

        # build spanner edge map (u,v)->local_eid
        # _build_spanner_eid_map can take a tensor or we can pass a numpy version
        # Let's just use numpy for the logic inside extract_target_edges
        mp: Dict[Tuple[int, int], int] = {}
        for eid in range(E):
            a, b = int(edge_index_np[0, eid]), int(edge_index_np[1, eid])
            mp[(a, b) if a < b else (b, a)] = eid
        eid_map = mp

        # build adjacency for Dijkstra
        adj: List[List[Tuple[int, int, float]]] = [[] for _ in range(N)]
        for eid in range(E):
            a, b = int(edge_index_np[0, eid]), int(edge_index_np[1, eid])
            w = float(sp_w[eid])
            adj[a].append((b, eid, w))
            adj[b].append((a, eid, w))

        # teacher tour on complete graph
        if self.use_lkh:
            order_list = solve_tsp_lkh(pos_np, executable=self.lkh_exe)
            if not order_list:
                # fallback to heuristic if LKH fails
                tour = solve_tsp_heuristic(torch.from_numpy(pos_np), start=0, max_2opt_passes=self.two_opt_passes)
            else:
                order_t = torch.tensor(order_list, dtype=torch.long)
                D = pairwise_dist(torch.from_numpy(pos_np))
                L = tour_length(order_t, D)
                tour = Tour(order=order_t, length=L)
        else:
            pos_t = torch.from_numpy(pos_np)
            tour = solve_tsp_heuristic(pos_t, start=0, max_2opt_passes=self.two_opt_passes)
        
        t_edges = tour_edges(tour.order)

        # project teacher edges onto spanner graph
        selected_local_eids: set[int] = set()
        for a, b in t_edges:
            local_eid = eid_map.get((a, b), None)
            if local_eid is not None:
                selected_local_eids.add(int(local_eid))
                continue

            path = _dijkstra_path_eids(N, adj, a, b)
            if path is not None:
                for pe in path:
                    selected_local_eids.add(int(pe))

        res = np.array(sorted(list(selected_local_eids)), dtype=np.int64)
        return res, float(tour.length.item())

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

        # ---- teacher tour and projection (Skip if target_edges is already in data) ----
        if hasattr(data, "target_edges") and data.target_edges is not None:
            selected_local_eids = set(int(x) for x in data.target_edges.tolist())
            tour_len_val = getattr(data, "tour_len", torch.tensor(0.0)).detach().cpu()
            num_direct = num_projected = num_unreachable = num_not_alive_direct = 0
        else:
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

            # teacher tour on complete graph 
            if self.use_lkh:
                pos_np = pos.detach().cpu().numpy()
                order_list = solve_tsp_lkh(pos_np, executable=self.lkh_exe)
                if not order_list:
                    tour = solve_tsp_heuristic(pos, start=0, max_2opt_passes=self.two_opt_passes)
                else:
                    order_t = torch.tensor(order_list, dtype=torch.long)
                    D = pairwise_dist(pos)
                    L = tour_length(order_t, D)
                    tour = Tour(order=order_t, length=L)
            else:
                tour = solve_tsp_heuristic(pos, start=0, max_2opt_passes=self.two_opt_passes)
            
            t_edges = tour_edges(tour.order)  # list of (u<v)

            # ---- project teacher edges onto alive spanner graph (local eids) ----
            selected_local_eids = set()
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
            
            tour_len_val = tour.length.detach().cpu()

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
            "tour_len": tour_len_val,
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
            target_state_idx=torch.full((int(y_iface.shape[0]),), -1, dtype=torch.long, device=device),
            m_state=torch.zeros((int(y_iface.shape[0]),), dtype=torch.bool, device=device),
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

        # ---- Vectorized Bitmask Construction (REPLACES THE LOOP) ----
        # 1. Gather all global target eids
        all_global_targets = []
        for b in range(B):
            if hasattr(datas[b], "target_edges") and datas[b].target_edges is not None:
                # We can use the pre-computed spanner indices directly
                targets = datas[b].target_edges.to(device)
            else:
                # Fallback: if somehow not pre-computed, do one one-by-one (rare/backup)
                # But to keep things vectorized, we prioritize pre-computed.
                # If missing, we'll just skip this graph's contribution to target bits for this batch
                # or we could call label_one. For "Zero Redundancy", we assume pre-computed.
                targets = torch.tensor([], dtype=torch.long, device=device)
            all_global_targets.append(targets + edge_ptr[b])
        
        targets_g = torch.cat(all_global_targets) if all_global_targets else torch.tensor([], dtype=torch.long, device=device)

        m_cross_all = tokens.cross_mask.bool().to(device)
        m_iface_all = tokens.iface_mask.bool().to(device)

        # 2. Optimized bitmask search for targets_g (much faster than torch.isin on CPU)
        max_eid = int(edge_ptr[-1].item())
        mask_table = torch.zeros(max_eid + 1, dtype=torch.bool, device=device)
        if targets_g.numel() > 0:
            mask_table[targets_g] = True
        
        # Map cross_eid and iface_eid (handling -1 padding)
        def map_labels(eids):
            # [M, T]
            out = torch.zeros_like(eids, dtype=torch.float32)
            valid = eids >= 0
            if valid.any():
                out[valid] = mask_table[eids[valid]].to(dtype=torch.float32)
            return out

        y_cross_all = map_labels(tokens.cross_eid)
        y_iface_all = map_labels(tokens.iface_eid)

        target_state_idx = torch.full((total_M,), -1, dtype=torch.long, device=device)
        m_state = torch.zeros((total_M,), dtype=torch.bool, device=device)
        num_state_exact = 0
        num_state_fallback = 0
        if getattr(tokens, "state_mask", None) is not None and getattr(packed, "state_catalog", None) is not None:
            state_mask_all = tokens.state_mask.bool().to(device)
            state_used_iface = packed.state_catalog.used_iface.to(device)
            state_mate = packed.state_catalog.mate.to(device)
            m_state = state_mask_all.any(dim=1)

            for b in range(B):
                m0 = int(node_ptr[b].item())
                m1 = int(node_ptr[b + 1].item())
                e0 = int(edge_ptr[b].item())
                selected_local_eids = set(int(x) for x in getattr(datas[b], "target_edges", torch.empty((0,), dtype=torch.long)).detach().cpu().tolist())
                node_points = _compute_node_point_sets(datas[b])
                sp_edge_index = getattr(datas[b], "spanner_edge_index").detach().cpu()
                sp_u = sp_edge_index[0].tolist()
                sp_v = sp_edge_index[1].tolist()

                iface_eid_local = tokens.iface_eid[m0:m1].to(device) - e0
                iface_mask_local = m_iface_all[m0:m1]
                iface_inside_ep_local = tokens.iface_inside_endpoint[m0:m1].to(device)
                state_mask_local = state_mask_all[m0:m1]

                for local_nid in range(m1 - m0):
                    mid = m0 + local_nid
                    if not bool(m_state[mid].item()):
                        continue

                    state_idx, exact_used = _build_matching_target_for_node(
                        local_node_id=local_nid,
                        points_in_node=node_points[local_nid],
                        selected_local_eids=selected_local_eids,
                        sp_u=sp_u,
                        sp_v=sp_v,
                        iface_eid_row=iface_eid_local[local_nid],
                        iface_mask_row=iface_mask_local[local_nid],
                        iface_inside_ep_row=iface_inside_ep_local[local_nid],
                        state_mask_row=state_mask_local[local_nid],
                        state_used_iface=state_used_iface,
                        state_mate=state_mate,
                    )
                    target_state_idx[mid] = int(state_idx)
                    if exact_used:
                        num_state_exact += 1
                    else:
                        num_state_fallback += 1

        # 3. Vectorized stats aggregation (using tour_len from datas)
        t_lens = [torch.as_tensor(getattr(d, "tour_len", 0.0), device=device).view(-1) for d in datas]
        tour_len_cat = torch.cat(t_lens, dim=0) if t_lens else torch.tensor([], device=device)
        
        # Combined stats dictionary for returning in TokenLabels
        stats = {
            "tour_len": tour_len_cat,
            "tour_len_mean": tour_len_cat.mean() if tour_len_cat.numel() > 0 else torch.tensor(0.0, device=device),
            # Placeholder for projection sums (skipped in vectorized loop to eliminate redundancy)
            "num_direct_sum": torch.tensor([0], device=device), 
            "num_projected_sum": torch.tensor([0], device=device),
            "num_unreachable_sum": torch.tensor([0], device=device),
            "num_state_exact_sum": torch.tensor([num_state_exact], device=device),
            "num_state_fallback_sum": torch.tensor([num_state_fallback], device=device),
        }

        # ---- child BC labels (GLOBAL packed indices) ----
        ch = tokens.tree_children_index.long().to(device)  # [total_M,4]
        exists = ch >= 0
        ch0 = ch.clamp_min(0)
        y_child_iface = y_iface_all[ch0]  # [M,4,Ti]
        m_child_iface = m_iface_all[ch0] & exists.unsqueeze(-1)
        y_child_iface = y_child_iface * m_child_iface.to(dtype=y_child_iface.dtype)

        return TokenLabels(
            y_cross=y_cross_all,
            m_cross=m_cross_all,
            y_iface=y_iface_all,
            m_iface=m_iface_all,
            y_child_iface=y_child_iface,
            m_child_iface=m_child_iface,
            target_state_idx=target_state_idx,
            m_state=m_state,
            stats=stats,
        )


def _compute_node_point_sets(data: Any) -> List[set[int]]:
    cached = getattr(data, "_cached_node_point_sets", None)
    if cached is not None:
        return cached

    tree_parent_index = getattr(data, "tree_parent_index").detach().cpu().tolist()
    num_nodes = len(tree_parent_index)
    node_points: List[set[int]] = [set() for _ in range(num_nodes)]
    if hasattr(data, "point_to_leaf"):
        point_to_leaf = getattr(data, "point_to_leaf").detach().cpu().tolist()
        for pid, leaf_id in enumerate(point_to_leaf):
            nid = int(leaf_id)
            while nid >= 0:
                node_points[nid].add(int(pid))
                nid = int(tree_parent_index[nid])
    else:
        leaf_ids = getattr(data, "leaf_ids").detach().cpu().tolist()
        leaf_ptr = getattr(data, "leaf_ptr").detach().cpu().tolist()
        leaf_points = getattr(data, "leaf_points").detach().cpu().tolist()
        for li, leaf_nid in enumerate(leaf_ids):
            pts = leaf_points[leaf_ptr[li]: leaf_ptr[li + 1]]
            nid = int(leaf_nid)
            while nid >= 0:
                node_points[nid].update(int(pid) for pid in pts)
                nid = int(tree_parent_index[nid])

    try:
        setattr(data, "_cached_node_point_sets", node_points)
    except Exception:
        pass
    return node_points


def _infer_inside_point_for_interface(
    *,
    a: int,
    b: int,
    inside_ep_attr: int,
    points_in_node: set[int],
) -> Optional[int]:
    if inside_ep_attr == 0 and a in points_in_node:
        return a
    if inside_ep_attr == 1 and b in points_in_node:
        return b

    in_a = a in points_in_node
    in_b = b in points_in_node
    if in_a and not in_b:
        return a
    if in_b and not in_a:
        return b
    if in_a and in_b:
        return a if inside_ep_attr == 0 else b
    return None


def _build_matching_target_for_node(
    *,
    local_node_id: int,
    points_in_node: set[int],
    selected_local_eids: set[int],
    sp_u: List[int],
    sp_v: List[int],
    iface_eid_row: Tensor,               # [Ti] local eid
    iface_mask_row: Tensor,              # [Ti] bool
    iface_inside_ep_row: Tensor,         # [Ti] long
    state_mask_row: Tensor,              # [S] bool
    state_used_iface: Tensor,            # [S,Ti] bool
    state_mate: Tensor,                  # [S,Ti] long
) -> tuple[int, bool]:
    del local_node_id  # reserved for future debugging / stats hooks

    Ti = int(iface_mask_row.numel())
    target_used = torch.zeros((Ti,), dtype=torch.bool, device=iface_mask_row.device)
    target_mate = torch.full((Ti,), -1, dtype=torch.long, device=iface_mask_row.device)

    stub_points: Dict[int, int] = {}
    fallback_needed = False

    for i in range(Ti):
        if not bool(iface_mask_row[i].item()):
            continue
        eid = int(iface_eid_row[i].item())
        if eid < 0 or eid not in selected_local_eids:
            continue

        a = int(sp_u[eid])
        b = int(sp_v[eid])
        inside_point = _infer_inside_point_for_interface(
            a=a,
            b=b,
            inside_ep_attr=int(iface_inside_ep_row[i].item()),
            points_in_node=points_in_node,
        )
        if inside_point is None:
            fallback_needed = True
            continue

        target_used[i] = True
        stub_points[i] = inside_point

    used_slots = [slot for slot, used in enumerate(target_used.tolist()) if used]
    if len(used_slots) == 0:
        return project_matching_to_state_index(
            iface_used=target_used,
            iface_mate=target_mate,
            iface_mask=iface_mask_row,
            state_mask=state_mask_row,
            state_used_iface=state_used_iface,
            state_mate=state_mate,
        ), True

    if len(used_slots) % 2 != 0:
        fallback_needed = True

    touched_points = set(stub_points.values())
    internal_edges: List[Tuple[int, int]] = []
    for eid in selected_local_eids:
        a = int(sp_u[eid])
        b = int(sp_v[eid])
        if a in points_in_node and b in points_in_node:
            internal_edges.append((a, b))
            touched_points.add(a)
            touched_points.add(b)

    parent: Dict[int, int] = {p: p for p in touched_points}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in internal_edges:
        union(a, b)

    comp_to_slots: Dict[int, List[int]] = {}
    for slot, point in stub_points.items():
        root = find(point)
        comp_to_slots.setdefault(root, []).append(slot)

    for slots in comp_to_slots.values():
        if len(slots) != 2:
            fallback_needed = True
            continue
        i, j = int(slots[0]), int(slots[1])
        target_mate[i] = j
        target_mate[j] = i

    exact_usable = (not fallback_needed) and all(int(target_mate[i].item()) >= 0 for i in used_slots)
    if exact_usable:
        return project_matching_to_state_index(
            iface_used=target_used,
            iface_mate=target_mate,
            iface_mask=iface_mask_row,
            state_mask=state_mask_row,
            state_used_iface=state_used_iface,
            state_mate=state_mate,
        ), True

    return project_iface_usage_to_state_index(
        iface_target=target_used.to(dtype=torch.float32),
        iface_mask=iface_mask_row,
        state_mask=state_mask_row,
        state_used_iface=state_used_iface,
    ), False


__all__ = ["PseudoLabeler", "TokenLabels"]
