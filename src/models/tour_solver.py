# src/models/tour_solver.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from torch import Tensor


@dataclass
class Tour:
    """
    A Hamiltonian cycle represented by an ordering of vertices.
_:
      order: [N] permutation of {0..N-1}
      length: scalar float (Euclidean length along order cycle)
    """
    order: Tensor
    length: Tensor


def pairwise_dist(pos: Tensor) -> Tensor:
    """
    pos: [N,2] float
    returns D: [N,N] where D[i,j] = ||pos[i]-pos[j]||2
    """
    if pos.dim() != 2 or pos.shape[1] != 2:
        raise ValueError(f"pos must be [N,2], got {tuple(pos.shape)}")
    diff = pos[:, None, :] - pos[None, :, :]
    return torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)


def tour_length(order: Tensor, D: Tensor) -> Tensor:
    """
    order: [N] long
    D: [N,N]
    """
    N = int(order.numel())
    nxt = torch.roll(order, shifts=-1, dims=0)
    return D[order, nxt].sum()


def nearest_neighbor_tour(D: Tensor, start: int = 0) -> Tensor:
    """
    Construct an initial tour via nearest-neighbor heuristic on a complete graph.
    D: [N,N] distance matrix (or any symmetric cost matrix).
    """
    N = int(D.shape[0])
    if D.shape != (N, N):
        raise ValueError("D must be square [N,N].")
    if not (0 <= start < N):
        raise ValueError("start out of range.")

    visited = torch.zeros((N,), dtype=torch.bool)
    order = torch.empty((N,), dtype=torch.long)

    cur = int(start)
    for t in range(N):
        order[t] = cur
        visited[cur] = True
        if t == N - 1:
            break
        # pick nearest unvisited
        row = D[cur].clone()
        row[visited] = float("inf")
        cur = int(torch.argmin(row).item())
    return order


def two_opt_improve(order: Tensor, D: Tensor, max_passes: int = 50) -> Tensor:
    """
    Standard 2-opt local search to improve a tour under cost matrix D.

    This is O(max_passes * N^2). For N=50 it is cheap.
    """
    N = int(order.numel())
    if N < 4:
        return order

    N = int(order.numel())
    if N < 4:
        return order

    order_list = order.tolist()
    D_np = D.detach().cpu().numpy()
    
    # For large N, use candidate-restricted 2-opt to keep it O(KN)
    use_candidate = (N > 500)
    top_indices = None
    if use_candidate:
        K = 20
        _, topk_indices = torch.topk(D, k=min(K+1, N), dim=1, largest=False)
        top_indices = topk_indices[:, 1:].cpu().numpy().tolist()

    pos_in_tour = [0] * N
    for idx, node in enumerate(order_list):
        pos_in_tour[node] = idx

    max_p = max_passes if N < 1000 else min(max_passes, 20)
    
    for _ in range(max_p):
        improved = False
        for i in range(N):
            u = order_list[i]
            v_idx = (i + 1) % N
            v = order_list[v_idx]
            d_uv = D_np[u][v]

            if use_candidate:
                # Optimized KN search
                for w in top_indices[u]:
                    if w == u or w == v: continue
                    idx_w = pos_in_tour[w]
                    idx_x = (idx_w + 1) % N
                    x = order_list[idx_x]
                    if x == u or x == v: continue
                    
                    if d_uv + D_np[w][x] > D_np[u][w] + D_np[v][x] + 1e-9:
                        if i < idx_w:
                            start, end = i + 1, idx_w + 1
                            order_list[start:end] = order_list[start:end][::-1]
                            for k in range(start, end):
                                pos_in_tour[order_list[k]] = k
                        else:
                            start, end = idx_x, i + 1
                            order_list[start:end] = order_list[start:end][::-1]
                            for k in range(start, end):
                                pos_in_tour[order_list[k]] = k
                        v = w
                        d_uv = D_np[u][v]
                        improved = True
            else:
                # Standard N^2 search for small instances
                for j in range(i + 2, N):
                    if i == 0 and j == N - 1: continue
                    w = order_list[j]
                    x_idx = (j + 1) % N
                    x = order_list[x_idx]
                    
                    if d_uv + D_np[w][x] > D_np[u][w] + D_np[v][x] + 1e-9:
                        # reverse segment from v to w
                        # i < j always true here
                        start, end = i + 1, j + 1
                        order_list[start:end] = order_list[start:end][::-1]
                        for k in range(start, end):
                            pos_in_tour[order_list[k]] = k
                        v = w
                        d_uv = D_np[u][v]
                        improved = True
                        
        if not improved:
            break

    return torch.tensor(order_list, dtype=torch.long, device=order.device)


def solve_tsp_heuristic(
    pos: Tensor,
    *,
    start: int = 0,
    max_2opt_passes: int = 50,
    cost_matrix: Optional[Tensor] = None,
) -> Tour:
    """
    Solve (approximately) an Euclidean TSP instance via NN + 2-opt.

    Args:
      pos: [N,2] float
      cost_matrix: optional [N,N] to use instead of Euclidean distances for construction+2opt.
                   If None, use Euclidean.
    """
    pos = pos.detach()
    D = pairwise_dist(pos)
    C = D if cost_matrix is None else cost_matrix
    if C.shape != D.shape:
        raise ValueError("cost_matrix must match [N,N].")

    order0 = nearest_neighbor_tour(C, start=start)
    order1 = two_opt_improve(order0, C, max_passes=max_2opt_passes)

    L = tour_length(order1, D)  # report true Euclidean length
    return Tour(order=order1, length=L)


def tour_edges(order: Tensor) -> List[Tuple[int, int]]:
    """
    Return undirected edges of the cycle as list of (u,v) with u<v.
    """
    N = int(order.numel())
    nxt = torch.roll(order, shifts=-1, dims=0)
    u = order.cpu().tolist()
    v = nxt.cpu().tolist()
    edges = []
    for a, b in zip(u, v):
        if a < b:
            edges.append((a, b))
        else:
            edges.append((b, a))
    return edges


__all__ = [
    "Tour",
    "pairwise_dist",
    "tour_length",
    "solve_tsp_heuristic",
    "tour_edges",
]
