# src/models/tour_solver.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
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

    order = order.clone()
    best_len = float(tour_length(order, D).item())

    for _ in range(max_passes):
        improved = False
        # 2-opt: reverse segment (i..j)
        for i in range(1, N - 2):
            a = int(order[i - 1].item())
            b = int(order[i].item())
            for j in range(i + 1, N - 1):
                c = int(order[j].item())
                d = int(order[j + 1].item())
                # delta = (a-b + c-d) replaced by (a-c + b-d)
                delta = float(D[a, c].item() + D[b, d].item() - D[a, b].item() - D[c, d].item())
                if delta < -1e-12:
                    order[i : j + 1] = torch.flip(order[i : j + 1], dims=[0])
                    best_len += delta
                    improved = True
        if not improved:
            break

    return order


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
