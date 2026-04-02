# src/models/edge_decode.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class TourDecodeResult:
    order: List[int]                 # Hamiltonian cycle order, length N, starts at 0 by convention
    length: float                    # Euclidean tour length on pos
    feasible: bool                   # degree=2 and single cycle
    num_off_spanner_edges: int       # how many edges not in spanner were used in patching
    num_components_initial: int       # components after greedy selection
    num_edges_broken: int            # legacy: how many edges were removed to break subtours (should be 0)
    fallback_used: bool              # whether nearest-neighbor fallback was used
    num_patching_steps: int          # how many edges added in patching phase
    duration: float                  # execution time in seconds


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
    refine_max_n: Optional[int] = None,
    fallback_max_n: Optional[int] = None,
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
    import heapq
    import time
    start_t = time.time()

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

    chosen_edges_count = 0
    for score, a, b in edges:
        if chosen_edges_count >= N:
            break
        # CORRECTED CONDITION: Only allow cycle if it covers all nodes
        ra, rb = dsu.find(a), dsu.find(b)
        if deg[a] < 2 and deg[b] < 2:
            if ra != rb:
                add_edge(a, b, score)
                chosen_edges_count += 1
            elif chosen_edges_count == N - 1 and dsu.sz[ra] == N:
                # This is the final closing edge for the full Hamiltonian cycle
                add_edge(a, b, score)
                chosen_edges_count += 1
                break

    # After initial greedy, we might have multiple components (paths)
    # maintain endpoints incrementally instead of recomputing
    endpoints: set[int] = {i for i in range(N) if deg[i] < 2}
    # component_id -> list of endpoint nodes
    comp_to_eps: Dict[int, List[int]] = {}
    for i in endpoints:
        root = dsu.find(i)
        comp_to_eps.setdefault(root, []).append(i)
    
    num_components_initial = len(comp_to_eps)
    fallback_used = False
    num_patching_steps = 0
    num_off_spanner_patching = 0
    num_edges_broken = 0 # Legacy

    p_cpu = pos.detach().cpu()

    def dist(a: int, b: int) -> float:
        dx = float(p_cpu[a, 0].item() - p_cpu[b, 0].item())
        dy = float(p_cpu[a, 1].item() - p_cpu[b, 1].item())
        return (dx * dx + dy * dy) ** 0.5

    # --- PATCHING PHASE ---
    # We maintain comp_to_eps (component_id -> list of nodes with deg < 2)
    # until only one component remains.
    
    def get_dist(a: int, b: int) -> float:
        dx = float(p_cpu[a, 0].item() - p_cpu[b, 0].item())
        dy = float(p_cpu[a, 1].item() - p_cpu[b, 1].item())
        return (dx * dx + dy * dy) ** 0.5

    # PRE-OPTIMIZATION: The first greedy pass already exhausts useful spanner edges.
    # We only need to merge remaining components using off-spanner edges if allowed.
    
    if len(comp_to_eps) > 1 and allow_off_spanner_patch:
        while len(comp_to_eps) > 1:
            best_off = None # (dist, a, b, target_root)
            min_patch_dist = float('inf')
            
            # Optimization: Instead of O(C^2) to find the absolute best among all pairs,
            # just pick one component and find its nearest neighbor in any other component.
            # This makes the total patching phase O(C^2) instead of O(C^3).
            roots = list(comp_to_eps.keys())
            r_i = roots[0]
            eps_i = comp_to_eps[r_i]
            
            for j in range(1, len(roots)):
                r_j = roots[j]
                for a in eps_i:
                    for b in comp_to_eps[r_j]:
                        d = get_dist(a, b)
                        if d < min_patch_dist:
                            min_patch_dist = d
                            best_off = (d, a, b, r_j)
            
            if best_off:
                d, a, b, r_j = best_off
                add_edge(a, b, -d)
                chosen_edges_count += 1
                num_patching_steps += 1
                num_off_spanner_patching += 1
                
                # Update DSU and component map
                dsu.union(a, b)
                new_root = dsu.find(a)
                eps_merged = comp_to_eps.pop(r_i) + comp_to_eps.pop(r_j)
                comp_to_eps[new_root] = [x for x in eps_merged if deg[x] < 2]
            else:
                break

    # Final closure if only 2 endpoints left in one component
    if len(comp_to_eps) == 1:
        root = next(iter(comp_to_eps))
        eps = comp_to_eps[root]
        if len(eps) == 2:
            a, b = eps
            key = _edge_key(a, b)
            if key not in sp_set:
                num_off_spanner_patching += 1
            add_edge(a, b, sp_logit.get(key, -get_dist(a, b)))
            chosen_edges_count += 1
            num_patching_steps += 1
            comp_to_eps[root] = []

    feasible = (chosen_edges_count == N) and (all(d == 2 for d in deg)) and (len(comp_to_eps) == 1)
    order = _extract_cycle_order(adj, start=0) if feasible else None

    if order is None:
        if fallback_max_n is not None and int(fallback_max_n) >= 0 and N > int(fallback_max_n):
            return TourDecodeResult(
                order=[],
                length=float("inf"),
                feasible=False,
                num_off_spanner_edges=int(num_off_spanner_patching),
                num_edges_broken=int(num_edges_broken),
                num_components_initial=int(num_components_initial),
                fallback_used=False,
                num_patching_steps=int(num_patching_steps),
                duration=time.time() - start_t,
            )
        # fallback: use existing torch-vectorized heuristic (NN + 2-opt)
        from src.models.tour_solver import solve_tsp_heuristic
        tour_h = solve_tsp_heuristic(pos, start=0, max_2opt_passes=50)
        order = tour_h.order.tolist()
        length = float(tour_h.length.item())
        feasible = False
        fallback_used = True
    else:
        do_refine = True
        if refine_max_n is not None and int(refine_max_n) >= 0 and N > int(refine_max_n):
            do_refine = False

        if not do_refine:
            length = _tour_length(pos, order)
        else:
            # OPTIONAL: Candidate-restricted 2-opt refinement
            from src.models.tour_solver import pairwise_dist
            D_full = pairwise_dist(pos).detach().cpu().numpy()
            N = D_full.shape[0]

            # Candidate-restricted 2-opt (Fast O(KN))
            # Increase K for better quality. For large N, we cap this to preserve speed.
            K = 40 if N < 1000 else 20
            _, topk_indices = torch.topk(torch.from_numpy(D_full), k=min(K + 1, N), dim=1, largest=False)
            top_indices = topk_indices[:, 1:].numpy().tolist()  # remove self

            curr_order = list(order)
            pos_in_tour = [0] * N
            for idx, node in enumerate(curr_order):
                pos_in_tour[node] = idx

            improved = True
            passes = 0
            max_passes = 100 if N < 1000 else 30
            while improved and passes < max_passes:
                improved = False
                passes += 1
                for i in range(N):
                    u = curr_order[i]
                    v_idx = (i + 1) % N
                    v = curr_order[v_idx]
                    d_uv = D_full[u][v]

                    # Cache top_indices[u] and other local variables
                    u_nbors = top_indices[u]
                    for w in u_nbors:
                        if w == u or w == v:
                            continue

                        idx_w = pos_in_tour[w]
                        idx_x = (idx_w + 1) % N
                        x = curr_order[idx_x]
                        if x == u or x == v:
                            continue

                        # Cost of current edges: (u,v) and (w,x)
                        # Cost of potential edges: (u,w) and (v,x)
                        if d_uv + D_full[w][x] > D_full[u][w] + D_full[v][x] + 1e-9:
                            # Perform swap: reverse internal segment
                            if i < idx_w:
                                # [A, u][v, ..., w][x, C] -> reverse v...w
                                # In slice notation: i+1 to idx_w+1
                                start, end = i + 1, idx_w + 1
                                curr_order[start:end] = curr_order[start:end][::-1]
                                for k in range(start, end):
                                    pos_in_tour[curr_order[k]] = k
                            else:
                                # [A, x][w, ..., u][v, C] -> reverse x...u
                                # In slice notation: idx_x to i+1
                                start, end = idx_x, i + 1
                                curr_order[start:end] = curr_order[start:end][::-1]
                                for k in range(start, end):
                                    pos_in_tour[curr_order[k]] = k

                            # Update current dist and node for next candidate check
                            v = w
                            d_uv = D_full[u][v]
                            improved = True

            order = curr_order
            # final length
            length = 0.0
            for idx in range(N):
                length += D_full[curr_order[idx]][curr_order[(idx + 1) % N]]

    return TourDecodeResult(
        order=order,
        length=length,
        feasible=feasible,
        num_off_spanner_edges=int(num_off_spanner_patching), 
        num_edges_broken=int(num_edges_broken),
        num_components_initial=int(num_components_initial),
        fallback_used=fallback_used,
        num_patching_steps=int(num_patching_steps),
        duration=time.time() - start_t
    )

__all__ = ["decode_tour_from_edge_logits", "TourDecodeResult"]
