# src/models/exact_decode.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import time

import numpy as np
import torch
from torch import Tensor

from src.models.edge_decode import TourDecodeResult


def _edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _tour_length(pos: Tensor, order: Sequence[int]) -> float:
    p = pos.detach().cpu()
    total = 0.0
    n = len(order)
    for i in range(n):
        a = int(order[i])
        b = int(order[(i + 1) % n])
        dx = float(p[a, 0].item() - p[b, 0].item())
        dy = float(p[a, 1].item() - p[b, 1].item())
        total += (dx * dx + dy * dy) ** 0.5
    return total


def _extract_cycle_order(adj: List[List[int]], start: int = 0) -> Optional[List[int]]:
    n = len(adj)
    if n == 0:
        return []
    order = [start]
    prev = -1
    cur = start
    for _ in range(n - 1):
        nbrs = adj[cur]
        if len(nbrs) != 2:
            return None
        nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
        order.append(nxt)
        prev, cur = cur, nxt
    if len(adj[cur]) != 2 or start not in adj[cur]:
        return None
    if len(set(order)) != n:
        return None
    return order


def _connected_components(num_nodes: int, chosen_edges: List[Tuple[int, int]]) -> List[List[int]]:
    adj = [[] for _ in range(num_nodes)]
    for a, b in chosen_edges:
        adj[a].append(b)
        adj[b].append(a)

    seen = [False] * num_nodes
    comps: List[List[int]] = []
    for s in range(num_nodes):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps


def _build_sparse_graph(
    *,
    pos: Tensor,
    spanner_edge_index: Tensor,
    edge_logit: Tensor,
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    edge_map: Dict[Tuple[int, int], Tuple[float, float]] = {}
    pos_cpu = pos.detach().cpu()

    for eid in range(int(spanner_edge_index.shape[1])):
        a = int(spanner_edge_index[0, eid].item())
        b = int(spanner_edge_index[1, eid].item())
        if a == b:
            continue

        key = _edge_key(a, b)
        score = float(edge_logit[eid].item())
        dx = float(pos_cpu[a, 0].item() - pos_cpu[b, 0].item())
        dy = float(pos_cpu[a, 1].item() - pos_cpu[b, 1].item())
        length = (dx * dx + dy * dy) ** 0.5

        prev = edge_map.get(key)
        if prev is None or score > prev[0]:
            edge_map[key] = (score, length)

    edges = sorted(edge_map.keys())
    scores = np.array([edge_map[e][0] for e in edges], dtype=np.float64)
    lengths = np.array([edge_map[e][1] for e in edges], dtype=np.float64)
    return edges, scores, lengths


def _sanitize_scores(raw_scores: np.ndarray, invalid_threshold: float = -1e8) -> np.ndarray:
    scores = raw_scores.astype(np.float64, copy=True)
    invalid = scores <= float(invalid_threshold)
    if not np.any(invalid):
        return scores

    valid = scores[~invalid]
    if valid.size == 0:
        scores[:] = -1.0
        return scores

    span = max(float(valid.max() - valid.min()), 1.0)
    scores[invalid] = float(valid.min()) - 5.0 * span - 1.0
    return scores


def decode_tour_exact_from_edge_logits(
    *,
    pos: Tensor,
    spanner_edge_index: Tensor,
    edge_logit: Tensor,
    time_limit: Optional[float] = 30.0,
    length_weight: float = 0.0,
    mip_rel_gap: float = 0.0,
    max_cuts: int = 256,
) -> TourDecodeResult:
    """
    Solve the maximum-score Hamiltonian cycle exactly on the sparse candidate graph.

    Objective:
      maximize sum(score_e * x_e) - length_weight * normalized_length_e * x_e

    Notes:
      - This is exact only on the supplied sparse graph, not on the full Euclidean complete graph.
      - It is intended as a drop-in replacement for greedy decoding when we want to remove
        heuristic postprocessing from the last stage.
    """
    start_t = time.time()

    if pos.dim() != 2 or pos.shape[1] != 2:
        raise ValueError(f"pos must be [N,2], got {tuple(pos.shape)}")
    if spanner_edge_index.dim() != 2 or spanner_edge_index.shape[0] != 2:
        raise ValueError(f"spanner_edge_index must be [2,E], got {tuple(spanner_edge_index.shape)}")
    if edge_logit.dim() != 1 or edge_logit.numel() != spanner_edge_index.shape[1]:
        raise ValueError("edge_logit must be 1D with length E matching spanner_edge_index.")

    try:
        from scipy import sparse
        from scipy.optimize import Bounds, LinearConstraint, milp
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Exact decoding requires scipy.optimize.milp (SciPy >= 1.11)."
        ) from exc

    n = int(pos.shape[0])
    if n < 3:
        return TourDecodeResult(
            order=[],
            length=float("inf"),
            feasible=False,
            num_off_spanner_edges=0,
            num_components_initial=0,
            num_edges_broken=0,
            fallback_used=False,
            num_patching_steps=0,
            duration=time.time() - start_t,
        )

    edges, raw_scores, lengths = _build_sparse_graph(
        pos=pos,
        spanner_edge_index=spanner_edge_index,
        edge_logit=edge_logit,
    )
    m = len(edges)
    if m < n:
        return TourDecodeResult(
            order=[],
            length=float("inf"),
            feasible=False,
            num_off_spanner_edges=0,
            num_components_initial=0,
            num_edges_broken=0,
            fallback_used=False,
            num_patching_steps=0,
            duration=time.time() - start_t,
        )

    degree_counts = [0] * n
    for a, b in edges:
        degree_counts[a] += 1
        degree_counts[b] += 1
    if any(d < 2 for d in degree_counts):
        return TourDecodeResult(
            order=[],
            length=float("inf"),
            feasible=False,
            num_off_spanner_edges=0,
            num_components_initial=0,
            num_edges_broken=0,
            fallback_used=False,
            num_patching_steps=0,
            duration=time.time() - start_t,
        )

    scores = _sanitize_scores(raw_scores)
    if float(length_weight) != 0.0:
        denom = max(float(lengths.mean()), 1e-9)
        objective = -(scores - float(length_weight) * (lengths / denom))
    else:
        objective = -scores

    deg_rows: List[int] = []
    deg_cols: List[int] = []
    deg_data: List[float] = []
    for eid, (a, b) in enumerate(edges):
        deg_rows.extend([a, b])
        deg_cols.extend([eid, eid])
        deg_data.extend([1.0, 1.0])
    A_eq = sparse.csr_matrix((deg_data, (deg_rows, deg_cols)), shape=(n, m))
    eq_constraint = LinearConstraint(A_eq, lb=np.full(n, 2.0), ub=np.full(n, 2.0))

    cuts: List[Tuple[Sequence[int], float]] = []
    cut_signatures: set[frozenset[int]] = set()
    integrality = np.ones(m, dtype=np.int32)
    bounds = Bounds(np.zeros(m), np.ones(m))

    first_num_components: Optional[int] = None

    for _ in range(int(max_cuts) + 1):
        constraints = [eq_constraint]
        if cuts:
            cut_rows: List[int] = []
            cut_cols: List[int] = []
            cut_data: List[float] = []
            cut_ubs = np.empty((len(cuts),), dtype=np.float64)
            for row_id, (edge_ids, ub) in enumerate(cuts):
                cut_ubs[row_id] = float(ub)
                cut_rows.extend([row_id] * len(edge_ids))
                cut_cols.extend(edge_ids)
                cut_data.extend([1.0] * len(edge_ids))
            A_ub = sparse.csr_matrix((cut_data, (cut_rows, cut_cols)), shape=(len(cuts), m))
            constraints.append(
                LinearConstraint(
                    A_ub,
                    lb=np.full(len(cuts), -np.inf, dtype=np.float64),
                    ub=cut_ubs,
                )
            )

        options = {"disp": False}
        if time_limit is not None:
            remaining = float(time_limit) - (time.time() - start_t)
            if remaining <= 1e-6:
                break
            options["time_limit"] = remaining
        if mip_rel_gap is not None:
            options["mip_rel_gap"] = float(mip_rel_gap)

        res = milp(
            c=objective,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options=options,
        )
        if res.x is None:
            break

        chosen_ids = [eid for eid, val in enumerate(res.x.tolist()) if val > 0.5]
        chosen_edges = [edges[eid] for eid in chosen_ids]
        comps = _connected_components(n, chosen_edges)
        if first_num_components is None:
            first_num_components = len(comps)

        if len(comps) == 1:
            adj = [[] for _ in range(n)]
            for a, b in chosen_edges:
                adj[a].append(b)
                adj[b].append(a)
            order = _extract_cycle_order(adj, start=0)
            if order is not None:
                return TourDecodeResult(
                    order=order,
                    length=_tour_length(pos, order),
                    feasible=True,
                    num_off_spanner_edges=0,
                    num_components_initial=int(first_num_components or 1),
                    num_edges_broken=0,
                    fallback_used=False,
                    num_patching_steps=0,
                    duration=time.time() - start_t,
                )

        added_cut = False
        edge_lookup = [(set(edge), idx) for idx, edge in enumerate(edges)]
        for comp in comps:
            if len(comp) >= n:
                continue
            comp_set = frozenset(comp)
            if comp_set in cut_signatures:
                continue
            internal_eids = [
                idx for edge_set, idx in edge_lookup
                if edge_set.issubset(comp_set)
            ]
            if not internal_eids:
                continue
            cuts.append((internal_eids, len(comp) - 1))
            cut_signatures.add(comp_set)
            added_cut = True
        if not added_cut:
            break

    return TourDecodeResult(
        order=[],
        length=float("inf"),
        feasible=False,
        num_off_spanner_edges=0,
        num_components_initial=int(first_num_components or 0),
        num_edges_broken=0,
        fallback_used=False,
        num_patching_steps=0,
        duration=time.time() - start_t,
    )


__all__ = ["decode_tour_exact_from_edge_logits"]
