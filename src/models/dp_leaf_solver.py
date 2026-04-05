# src/models/dp_leaf_solver.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor


def _all_permutations(n: int) -> List[List[int]]:
    if n == 0:
        return [[]]
    result: List[List[int]] = []

    def _perm(arr: List[int], start: int) -> None:
        if start == len(arr):
            result.append(arr[:])
            return
        for i in range(start, len(arr)):
            arr[start], arr[i] = arr[i], arr[start]
            _perm(arr, start + 1)
            arr[start], arr[i] = arr[i], arr[start]

    _perm(list(range(n)), 0)
    return result


@dataclass(frozen=True)
class LeafPathWitness:
    start_slot: int
    end_slot: int
    point_indices: Tuple[int, ...]


@dataclass(frozen=True)
class LeafStateWitness:
    open_paths: Tuple[LeafPathWitness, ...] = ()
    closed_cycle: Tuple[int, ...] = ()


def _held_karp_tsp(dist: List[List[float]]) -> float:
    n = len(dist)
    if n <= 1:
        return 0.0
    if n == 2:
        return dist[0][1] + dist[1][0]
    INF = float("inf")
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0.0
    for S in range(1, 1 << n):
        if not (S & 1):
            continue
        for i in range(n):
            if not (S & (1 << i)):
                continue
            if dp[S][i] >= INF:
                continue
            for j in range(n):
                if S & (1 << j):
                    continue
                S2 = S | (1 << j)
                c = dp[S][i] + dist[i][j]
                if c < dp[S2][j]:
                    dp[S2][j] = c
    full = (1 << n) - 1
    return min(dp[full][i] + dist[i][0] for i in range(1, n))


def _held_karp_tsp_with_order(dist: List[List[float]]) -> Tuple[float, List[int]]:
    n = len(dist)
    if n <= 0:
        return 0.0, []
    if n == 1:
        return 0.0, [0]
    if n == 2:
        return dist[0][1] + dist[1][0], [0, 1]

    INF = float("inf")
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    dp[1][0] = 0.0

    for S in range(1, 1 << n):
        if not (S & 1):
            continue
        for i in range(n):
            if not (S & (1 << i)):
                continue
            if dp[S][i] >= INF:
                continue
            for j in range(n):
                if S & (1 << j):
                    continue
                S2 = S | (1 << j)
                c = dp[S][i] + dist[i][j]
                if c < dp[S2][j]:
                    dp[S2][j] = c
                    parent[S2][j] = i

    full = (1 << n) - 1
    best_cost = INF
    best_last = -1
    for i in range(1, n):
        c = dp[full][i] + dist[i][0]
        if c < best_cost:
            best_cost = c
            best_last = i

    if best_last < 0:
        return INF, []

    rev: List[int] = [best_last]
    S = full
    cur = best_last
    while cur != 0:
        prev = parent[S][cur]
        S ^= (1 << cur)
        cur = prev
        if cur < 0:
            return INF, []
        rev.append(cur)

    order = list(reversed(rev))
    return best_cost, order


def _held_karp_path(dist: List[List[float]], start: int, end: int) -> float:
    n = len(dist)
    if n <= 1:
        return 0.0
    if n == 2:
        return dist[start][end]
    INF = float("inf")
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1 << start][start] = 0.0
    for S in range(1, 1 << n):
        if not (S & (1 << start)):
            continue
        for i in range(n):
            if not (S & (1 << i)):
                continue
            if dp[S][i] >= INF:
                continue
            for j in range(n):
                if S & (1 << j):
                    continue
                S2 = S | (1 << j)
                c = dp[S][i] + dist[i][j]
                if c < dp[S2][j]:
                    dp[S2][j] = c
    full = (1 << n) - 1
    return dp[full][end]


def _nn_tsp(points: Tensor) -> float:
    n = points.shape[0]
    if n <= 1:
        return 0.0
    visited = [False] * n
    tour_len = 0.0
    cur = 0
    visited[0] = True
    for _ in range(n - 1):
        best_d, best_j = float("inf"), -1
        for j in range(n):
            if visited[j]:
                continue
            d = (points[cur] - points[j]).norm().item()
            if d < best_d:
                best_d, best_j = d, j
        tour_len += best_d
        visited[best_j] = True
        cur = best_j
    tour_len += (points[cur] - points[0]).norm().item()
    return tour_len


def _nn_tsp_with_order(points: Tensor) -> Tuple[float, List[int]]:
    n = points.shape[0]
    if n <= 0:
        return 0.0, []
    if n == 1:
        return 0.0, [0]

    visited = [False] * n
    order = [0]
    tour_len = 0.0
    cur = 0
    visited[0] = True
    for _ in range(n - 1):
        best_d, best_j = float("inf"), -1
        for j in range(n):
            if visited[j]:
                continue
            d = (points[cur] - points[j]).norm().item()
            if d < best_d:
                best_d, best_j = d, j
        if best_j < 0:
            return float("inf"), []
        order.append(best_j)
        tour_len += best_d
        visited[best_j] = True
        cur = best_j
    tour_len += (points[cur] - points[0]).norm().item()
    return tour_len, order


def _nn_path(all_points: Tensor, start_pos: Tensor, end_pos: Tensor, interior_indices: List[int]) -> float:
    if not interior_indices:
        return (start_pos - end_pos).norm().item()
    path_len = 0.0
    cur = start_pos
    remaining = set(interior_indices)
    for _ in range(len(interior_indices)):
        best_d, best_j = float("inf"), -1
        for j in remaining:
            d = (cur - all_points[j]).norm().item()
            if d < best_d:
                best_d, best_j = d, j
        path_len += best_d
        cur = all_points[best_j]
        remaining.remove(best_j)
    path_len += (cur - end_pos).norm().item()
    return path_len


def _nn_path_with_order(
    all_points: Tensor,
    start_pos: Tensor,
    end_pos: Tensor,
    interior_indices: List[int],
) -> Tuple[float, List[int]]:
    if not interior_indices:
        return (start_pos - end_pos).norm().item(), []

    path_len = 0.0
    order: List[int] = []
    cur = start_pos
    remaining = set(int(i) for i in interior_indices)
    for _ in range(len(interior_indices)):
        best_d, best_j = float("inf"), -1
        for j in remaining:
            d = (cur - all_points[j]).norm().item()
            if d < best_d:
                best_d, best_j = d, j
        if best_j < 0:
            return float("inf"), []
        path_len += best_d
        cur = all_points[best_j]
        remaining.remove(best_j)
        order.append(best_j)
    path_len += (cur - end_pos).norm().item()
    return path_len, order


def _prepare_leaf_geometry(
    *,
    points_xy: Tensor,
    point_mask: Tensor,
    iface_mask: Tensor,
    iface_feat6: Tensor,
    box_xy: Tensor,
) -> Tuple[Tensor, Tensor]:
    device = points_xy.device
    valid_points_rel = points_xy[point_mask]
    num_points = int(point_mask.sum().item())

    cx = float(box_xy[0].item())
    cy = float(box_xy[1].item())
    hw = float(box_xy[2].item()) * 0.5
    hh = float(box_xy[3].item()) * 0.5

    valid_points = torch.zeros_like(valid_points_rel)
    if num_points > 0:
        valid_points[:, 0] = cx + valid_points_rel[:, 0] * hw
        valid_points[:, 1] = cy + valid_points_rel[:, 1] * hh

    Ti = int(iface_mask.shape[0])
    iface_pos = torch.zeros(Ti, 2, device=device)
    for s in range(Ti):
        if not iface_mask[s].item():
            continue
        inter_rx = float(iface_feat6[s, 2].item())
        inter_ry = float(iface_feat6[s, 3].item())
        iface_pos[s, 0] = cx + inter_rx * hw
        iface_pos[s, 1] = cy + inter_ry * hh

    return valid_points, iface_pos


def _solve_leaf_state(
    *,
    valid_points: Tensor,
    iface_pos: Tensor,
    iface_mask: Tensor,
    used: Tensor,
    mate: Tensor,
    is_root: bool,
    return_witness: bool = False,
) -> Tuple[float, Optional[LeafStateWitness]]:
    INF = float("inf")
    num_points = int(valid_points.shape[0])
    Ti = int(iface_mask.shape[0])

    active_slots = [s for s in range(Ti) if used[s].item() and iface_mask[s].item()]
    num_active = len(active_slots)
    if num_active % 2 != 0:
        return INF, None

    pairs: List[Tuple[int, int]] = []
    seen_slots: set[int] = set()
    for p in active_slots:
        if p in seen_slots:
            continue
        m = int(mate[p].item())
        if m < 0 or m in seen_slots or not used[m].item() or not iface_mask[m].item():
            return INF, None
        pairs.append((p, m))
        seen_slots.add(p)
        seen_slots.add(m)

    if len(seen_slots) != num_active:
        return INF, None

    num_paths = len(pairs)

    if num_paths == 0:
        if num_points == 0:
            witness = LeafStateWitness() if return_witness else None
            return 0.0, witness
        if not is_root:
            return INF, None

        if num_points == 1:
            witness = LeafStateWitness(closed_cycle=(0,)) if return_witness else None
            return 0.0, witness

        dist = [[0.0] * num_points for _ in range(num_points)]
        for a in range(num_points):
            for b in range(a + 1, num_points):
                d = (valid_points[a] - valid_points[b]).norm().item()
                dist[a][b] = dist[b][a] = d
        if return_witness:
            cost, order = _held_karp_tsp_with_order(dist)
            return cost, LeafStateWitness(closed_cycle=tuple(order))
        return _held_karp_tsp(dist), None

    if num_paths == 1 and num_points == 0:
        cost = (iface_pos[pairs[0][0]] - iface_pos[pairs[0][1]]).norm().item()
        witness = None
        if return_witness:
            witness = LeafStateWitness(open_paths=(LeafPathWitness(pairs[0][0], pairs[0][1], ()),))
        return cost, witness

    all_pts = list(range(num_points))
    best_cost = INF
    best_orders: Optional[List[List[int]]] = None

    def _enumerate_distributions(remaining: List[int], assignment: List[List[int]], depth: int) -> None:
        nonlocal best_cost, best_orders
        if depth == len(remaining):
            total = 0.0
            chosen_orders: List[List[int]] = []
            for path_idx, (pa, pb) in enumerate(pairs):
                pts_in_path = assignment[path_idx]
                start = iface_pos[pa]
                end = iface_pos[pb]
                if len(pts_in_path) == 0:
                    total += (start - end).norm().item()
                    chosen_orders.append([])
                    continue

                best_path = INF
                best_order: List[int] = []
                for perm in _all_permutations(len(pts_in_path)):
                    path_len = 0.0
                    prev = start
                    order_local: List[int] = []
                    for pi in perm:
                        cur_idx = pts_in_path[pi]
                        cur = valid_points[cur_idx]
                        path_len += (prev - cur).norm().item()
                        prev = cur
                        order_local.append(cur_idx)
                    path_len += (prev - end).norm().item()
                    if path_len < best_path:
                        best_path = path_len
                        best_order = order_local
                total += best_path
                chosen_orders.append(best_order)

            if total < best_cost:
                best_cost = total
                if return_witness:
                    best_orders = [o[:] for o in chosen_orders]
            return

        pt = remaining[depth]
        for path_idx in range(num_paths):
            assignment[path_idx].append(pt)
            _enumerate_distributions(remaining, assignment, depth + 1)
            assignment[path_idx].pop()

    init_assignment: List[List[int]] = [[] for _ in range(num_paths)]
    _enumerate_distributions(all_pts, init_assignment, 0)

    if not return_witness:
        return best_cost, None
    if best_orders is None:
        return best_cost, None

    open_paths = tuple(
        LeafPathWitness(pa, pb, tuple(best_orders[path_idx]))
        for path_idx, (pa, pb) in enumerate(pairs)
    )
    return best_cost, LeafStateWitness(open_paths=open_paths)


def leaf_solve_state(
    *,
    points_xy: Tensor,
    point_mask: Tensor,
    iface_mask: Tensor,
    iface_feat6: Tensor,
    state_used: Tensor,
    state_mate: Tensor,
    box_xy: Tensor,
    is_root: bool = False,
) -> Tuple[float, Optional[LeafStateWitness]]:
    valid_points, iface_pos = _prepare_leaf_geometry(
        points_xy=points_xy,
        point_mask=point_mask,
        iface_mask=iface_mask,
        iface_feat6=iface_feat6,
        box_xy=box_xy,
    )
    return _solve_leaf_state(
        valid_points=valid_points,
        iface_pos=iface_pos,
        iface_mask=iface_mask,
        used=state_used,
        mate=state_mate,
        is_root=is_root,
        return_witness=True,
    )


def leaf_exact_solve(
    *,
    points_xy: Tensor,
    point_mask: Tensor,
    iface_eid: Tensor,
    iface_mask: Tensor,
    iface_boundary_dir: Tensor,
    iface_feat6: Tensor,
    state_used_iface: Tensor,
    state_mate: Tensor,
    state_mask: Tensor,
    box_xy: Tensor,
    is_root: bool = False,
) -> Tensor:
    S = int(state_used_iface.shape[0])
    device = points_xy.device
    costs = torch.full((S,), float("inf"), device=device)
    valid_points, iface_pos = _prepare_leaf_geometry(
        points_xy=points_xy,
        point_mask=point_mask,
        iface_mask=iface_mask,
        iface_feat6=iface_feat6,
        box_xy=box_xy,
    )

    for si in range(S):
        if not state_mask[si].item():
            continue
        cost, _ = _solve_leaf_state(
            valid_points=valid_points,
            iface_pos=iface_pos,
            iface_mask=iface_mask,
            used=state_used_iface[si],
            mate=state_mate[si],
            is_root=is_root,
            return_witness=False,
        )
        costs[si] = cost

    return costs


__all__ = [
    "LeafPathWitness",
    "LeafStateWitness",
    "leaf_exact_solve",
    "leaf_solve_state",
]
