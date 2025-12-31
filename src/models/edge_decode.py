# src/models/edge_decode.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class TourDecodeResult:
    order: List[int]                 # Hamiltonian cycle order, length N, starts at 0 by convention
    length: float                    # Euclidean tour length on pos
    feasible: bool                   # degree=2 and single cycle
    num_off_spanner_edges: int       # how many edges not in spanner were used in patching
    num_edges_broken: int            # how many edges were removed to break subtours
    num_components_initial: int       # components after greedy selection


class _DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))
        self.sz = [1] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.sz[ra] += self.sz[rb]

    def groups(self) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {}
        for i in range(len(self.p)):
            r = self.find(i)
            out.setdefault(r, []).append(i)
        return out


def _edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _tour_length(pos: Tensor, order: List[int]) -> float:
    p = pos.detach().cpu()
    N = len(order)
    s = 0.0
    for i in range(N):
        a = order[i]
        b = order[(i + 1) % N]
        dx = float(p[a, 0].item() - p[b, 0].item())
        dy = float(p[a, 1].item() - p[b, 1].item())
        s += (dx * dx + dy * dy) ** 0.5
    return s


def _extract_cycle_order(adj: List[List[int]], start: int = 0) -> Optional[List[int]]:
    N = len(adj)
    order = [start]
    prev = -1
    cur = start
    for _ in range(N - 1):
        nbrs = adj[cur]
        if len(nbrs) != 2:
            return None
        nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
        order.append(nxt)
        prev, cur = cur, nxt
    if len(adj[cur]) != 2:
        return None
    if start not in adj[cur]:
        return None
    if len(set(order)) != N:
        return None
    return order


def decode_tour_from_edge_logits(
    *,
    pos: Tensor,                  # [N,2]
    spanner_edge_index: Tensor,   # [2,E] local
    edge_logit: Tensor,           # [E] local, logits/scores (higher=better)
    prefer_spanner_only: bool = True,
    allow_off_spanner_patch: bool = True,
) -> TourDecodeResult:
    """
    Decode a Hamiltonian cycle from edge logits.

    Important semantics (fixed):
      - prefer_spanner_only=True means: "use spanner edges whenever available",
        BUT if allow_off_spanner_patch=True, we MAY add off-spanner edges for patching.
      - If allow_off_spanner_patch=False, decoding is constrained to spanner edges only
        and may fail on some instances (expected).

    This is a postprocessing heuristic (engineering approximation), NOT DP backtracking.
    """
    if pos.dim() != 2 or pos.shape[1] != 2:
        raise ValueError(f"pos must be [N,2], got {tuple(pos.shape)}")
    if spanner_edge_index.dim() != 2 or spanner_edge_index.shape[0] != 2:
        raise ValueError(f"spanner_edge_index must be [2,E], got {tuple(spanner_edge_index.shape)}")
    if edge_logit.dim() != 1 or edge_logit.numel() != spanner_edge_index.shape[1]:
        raise ValueError("edge_logit must be 1D with length E matching spanner_edge_index.")

    N = int(pos.shape[0])
    E = int(spanner_edge_index.shape[1])

    u = spanner_edge_index[0].detach().cpu().tolist()
    v = spanner_edge_index[1].detach().cpu().tolist()
    s = edge_logit.detach().cpu().tolist()

    # spanner membership + spanner logit map
    sp_set: set[Tuple[int, int]] = set()
    sp_logit: Dict[Tuple[int, int], float] = {}
    edges: List[Tuple[float, int, int]] = []  # (score, a, b)

    for eid in range(E):
        a, b = int(u[eid]), int(v[eid])
        if a == b:
            continue
        key = _edge_key(a, b)
        sc = float(s[eid])
        sp_set.add(key)
        sp_logit[key] = sc
        edges.append((sc, a, b))

    # Greedy selection over spanner edges only
    edges.sort(key=lambda x: x[0], reverse=True)

    deg = [0] * N
    adj: List[List[int]] = [[] for _ in range(N)]
    chosen: Dict[Tuple[int, int], float] = {}  # key -> score
    dsu = _DSU(N)

    def can_add(a: int, b: int, chosen_edges: int) -> bool:
        if deg[a] >= 2 or deg[b] >= 2:
            return False
        ra, rb = dsu.find(a), dsu.find(b)
        if ra == rb:
            # allow cycle only if it's the final closing edge
            return chosen_edges == N - 1
        return True

    def add_edge(a: int, b: int, score: float, *, is_off_spanner: bool = False) -> None:
        adj[a].append(b)
        adj[b].append(a)
        deg[a] += 1
        deg[b] += 1
        dsu.union(a, b)
        chosen[_edge_key(a, b)] = score

    chosen_edges = 0
    for score, a, b in edges:
        if chosen_edges >= N:
            break
        if can_add(a, b, chosen_edges):
            add_edge(a, b, score)
            chosen_edges += 1
            if chosen_edges == N:
                break

    def component_nodes() -> Dict[int, List[int]]:
        d = _DSU(N)
        for (a, b) in chosen.keys():
            d.union(a, b)
        return d.groups()

    comps0 = component_nodes()
    num_components_initial = len(comps0)

    # break premature cycles (component is a cycle but not full graph)
    num_edges_broken = 0
    for _root, nodes in list(comps0.items()):
        if len(nodes) == N:
            continue
        if all(deg[x] == 2 for x in nodes):
            node_set = set(nodes)
            best_key = None
            best_score = 1e100
            for (a, b), sc in chosen.items():
                if a in node_set and b in node_set and sc < best_score:
                    best_score = sc
                    best_key = (a, b)
            if best_key is not None:
                a, b = best_key
                chosen.pop(_edge_key(a, b), None)
                adj[a].remove(b)
                adj[b].remove(a)
                deg[a] -= 1
                deg[b] -= 1
                chosen_edges -= 1
                num_edges_broken += 1

    p_cpu = pos.detach().cpu()

    def dist(a: int, b: int) -> float:
        dx = float(p_cpu[a, 0].item() - p_cpu[b, 0].item())
        dy = float(p_cpu[a, 1].item() - p_cpu[b, 1].item())
        return (dx * dx + dy * dy) ** 0.5

    num_off_spanner = 0

    def add_connection(a: int, b: int) -> bool:
        nonlocal chosen_edges, num_off_spanner
        if deg[a] >= 2 or deg[b] >= 2:
            return False
        key = _edge_key(a, b)
        in_sp = key in sp_set
        if (not in_sp) and (not allow_off_spanner_patch):
            return False

        # score for patch edges: use spanner logit if spanner, otherwise negative distance
        sc = sp_logit.get(key, -dist(a, b))
        add_edge(a, b, sc)
        chosen_edges += 1
        if not in_sp:
            num_off_spanner += 1
        return True

    def get_endpoints(nodes: List[int]) -> List[int]:
        # nodes with free degree slots (deg < 2)
        return [x for x in nodes if deg[x] < 2]

    # Patch components until connected
    while True:
        comps = component_nodes()
        if len(comps) == 1:
            break

        comp_list: List[Tuple[List[int], List[int]]] = []  # (nodes, endpoints)
        for _r, nodes in comps.items():
            eps = get_endpoints(nodes)
            # if eps empty, it's a cycle; break its weakest edge and recompute eps
            if len(eps) == 0 and len(nodes) < N:
                node_set = set(nodes)
                best_key = None
                best_score = 1e100
                for (a, b), sc in chosen.items():
                    if a in node_set and b in node_set and sc < best_score:
                        best_score = sc
                        best_key = (a, b)
                if best_key is not None:
                    a, b = best_key
                    chosen.pop(_edge_key(a, b), None)
                    adj[a].remove(b)
                    adj[b].remove(a)
                    deg[a] -= 1
                    deg[b] -= 1
                    chosen_edges -= 1
                    num_edges_broken += 1
                eps = get_endpoints(nodes)
            comp_list.append((nodes, eps))

        # Choose best inter-component connection:
        #   - if prefer_spanner_only: first search for ANY spanner edge between endpoints, choose max logit
        #   - otherwise (or if none exists): choose closest endpoints (off-spanner allowed if enabled)
        best_sp = None  # (logit, a, b)
        best_off = None  # (dist, a, b)

        for i in range(len(comp_list)):
            nodes_i, eps_i = comp_list[i]
            if len(eps_i) == 0:
                continue
            set_i = set(nodes_i)
            for j in range(i + 1, len(comp_list)):
                nodes_j, eps_j = comp_list[j]
                if len(eps_j) == 0:
                    continue
                set_j = set(nodes_j)

                # scan endpoint pairs
                for a in eps_i:
                    if deg[a] >= 2:
                        continue
                    for b in eps_j:
                        if deg[b] >= 2:
                            continue
                        key = _edge_key(a, b)
                        if key in sp_set:
                            sc = sp_logit[key]
                            if best_sp is None or sc > best_sp[0]:
                                best_sp = (sc, a, b)
                        else:
                            d_ab = dist(a, b)
                            if best_off is None or d_ab < best_off[0]:
                                best_off = (d_ab, a, b)

        if best_sp is not None:
            _, a, b = best_sp
            if not add_connection(a, b):
                break
        else:
            if not allow_off_spanner_patch:
                break
            if best_off is None:
                break
            _, a, b = best_off
            if not add_connection(a, b):
                break

    # Close a single path to a cycle if needed
    endpoints = [i for i in range(N) if deg[i] < 2]
    if len(component_nodes()) == 1 and len(endpoints) == 2:
        a, b = endpoints
        add_connection(a, b)

    feasible = all(d == 2 for d in deg) and (len(component_nodes()) == 1) and (len(chosen) == N)
    order = _extract_cycle_order(adj, start=0) if feasible else None

    if order is None:
        # fallback: nearest neighbor order (evaluation-only)
        nn_order = [0]
        used = [False] * N
        used[0] = True
        cur = 0
        for _ in range(N - 1):
            best_j = None
            best_d = 1e100
            for j in range(N):
                if not used[j]:
                    dj = dist(cur, j)
                    if dj < best_d:
                        best_d = dj
                        best_j = j
            assert best_j is not None
            nn_order.append(best_j)
            used[best_j] = True
            cur = best_j
        order = nn_order
        feasible = False

    length = _tour_length(pos, order)

    return TourDecodeResult(
        order=order,
        length=length,
        feasible=feasible,
        num_off_spanner_edges=int(num_off_spanner),
        num_edges_broken=int(num_edges_broken),
        num_components_initial=int(num_components_initial),
    )


__all__ = ["decode_tour_from_edge_logits", "TourDecodeResult"]
