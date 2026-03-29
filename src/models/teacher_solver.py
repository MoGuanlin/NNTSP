# src/models/teacher_solver.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.lkh_solver import (
    parse_tour,
    run_lkh,
    write_candidate_file,
    write_par,
    write_tsp_explicit,
)


TEACHER_LABEL_VERSION = "spanner_lkh_v3_alive"


def _edge_key(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


@dataclass
class TeacherTour:
    order: List[int]
    edge_ids: List[int]
    length: float
    duration: float


def _edge_lengths_from_attr_or_pos(
    *,
    pos: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: Optional[np.ndarray],
) -> Dict[Tuple[int, int], float]:
    lengths: Dict[Tuple[int, int], float] = {}
    E = int(edge_index.shape[1])
    attr_flat: Optional[np.ndarray] = None
    if edge_attr is not None:
        attr_flat = np.asarray(edge_attr).reshape(-1)
        if attr_flat.shape[0] != E:
            attr_flat = None

    for eid in range(E):
        a = int(edge_index[0, eid])
        b = int(edge_index[1, eid])
        key = _edge_key(a, b)
        if attr_flat is not None:
            lengths[key] = float(attr_flat[eid])
        else:
            dx = float(pos[a, 0] - pos[b, 0])
            dy = float(pos[a, 1] - pos[b, 1])
            lengths[key] = (dx * dx + dy * dy) ** 0.5
    return lengths


def _compute_full_euclidean_matrix(pos: np.ndarray) -> np.ndarray:
    diff = pos[:, None, :] - pos[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def _validate_undirected_spanner(edge_index: np.ndarray, num_nodes: int) -> None:
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"spanner_edge_index must be [2,E], got {tuple(edge_index.shape)}")
    deg = [0] * num_nodes
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    E = int(edge_index.shape[1])
    for eid in range(E):
        a = int(edge_index[0, eid])
        b = int(edge_index[1, eid])
        if a == b:
            continue
        deg[a] += 1
        deg[b] += 1
        adj[a].append(b)
        adj[b].append(a)

    if any(d < 2 for d in deg):
        bad = [i for i, d in enumerate(deg) if d < 2][:10]
        raise RuntimeError(
            "Spanner cannot support a Hamiltonian cycle because some vertices "
            f"have degree < 2: {bad}"
        )

    seen = [False] * num_nodes
    stack = [0]
    seen[0] = True
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                stack.append(v)
    if not all(seen):
        raise RuntimeError("Spanner graph is disconnected; cannot build a Hamiltonian teacher tour.")


def _build_sparse_lkh_matrix(
    *,
    pos: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: Optional[np.ndarray],
    cost_scale: int,
    penalty_margin: int,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], int], Dict[Tuple[int, int], float]]:
    n = int(pos.shape[0])
    euc = _compute_full_euclidean_matrix(pos)
    base = np.rint(euc * float(cost_scale)).astype(np.int64)

    eid_map: Dict[Tuple[int, int], int] = {}
    edge_lengths = _edge_lengths_from_attr_or_pos(pos=pos, edge_index=edge_index, edge_attr=edge_attr)

    matrix = base + int(penalty_margin)
    np.fill_diagonal(matrix, 0)
    for eid in range(int(edge_index.shape[1])):
        a = int(edge_index[0, eid])
        b = int(edge_index[1, eid])
        key = _edge_key(a, b)
        eid_map[key] = eid
        cost = int(round(edge_lengths[key] * float(cost_scale)))
        matrix[a, b] = cost
        matrix[b, a] = cost

    return matrix, eid_map, edge_lengths


def _build_spanner_candidates(
    *,
    edge_index: np.ndarray,
    edge_lengths: Dict[Tuple[int, int], float],
    cost_scale: int,
    num_nodes: int,
) -> List[List[Tuple[int, int]]]:
    candidates: List[List[Tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for eid in range(int(edge_index.shape[1])):
        a = int(edge_index[0, eid])
        b = int(edge_index[1, eid])
        alpha = int(round(edge_lengths[_edge_key(a, b)] * float(cost_scale)))
        candidates[a].append((b, alpha))
        candidates[b].append((a, alpha))
    return [sorted(row, key=lambda x: (x[1], x[0])) for row in candidates]


def _validate_tour_on_spanner(
    *,
    order: List[int],
    num_nodes: int,
    eid_map: Dict[Tuple[int, int], int],
    edge_lengths: Dict[Tuple[int, int], float],
) -> Tuple[List[int], float]:
    if len(order) != num_nodes or len(set(order)) != num_nodes:
        raise RuntimeError(
            f"LKH returned an invalid node ordering: len={len(order)}, unique={len(set(order))}, N={num_nodes}."
        )

    edge_ids: List[int] = []
    total = 0.0
    for i in range(num_nodes):
        a = int(order[i])
        b = int(order[(i + 1) % num_nodes])
        key = _edge_key(a, b)
        if key not in eid_map:
            raise RuntimeError(
                "LKH returned a tour containing off-spanner edges; refusing to store an invalid teacher."
            )
        edge_ids.append(int(eid_map[key]))
        total += float(edge_lengths[key])
    return edge_ids, total


def solve_spanner_tour_lkh(
    *,
    pos: np.ndarray,
    spanner_edge_index: np.ndarray,
    spanner_edge_attr: Optional[np.ndarray] = None,
    executable: str = "LKH",
    runs: int = 1,
    timeout: Optional[float] = None,
    seed: int = 1234,
    cost_scale: int = 1000,
    penalty_factor: int = 20,
) -> TeacherTour:
    """Solve a sparse spanner-constrained TSP teacher tour with LKH.

    The written TSPLIB instance is a full EXPLICIT matrix whose non-spanner edges
    receive a very large additive penalty. We then reject any returned tour that
    still uses an off-spanner edge, so only a true spanner Hamiltonian cycle is
    accepted as teacher supervision.
    """
    pos_np = np.asarray(pos, dtype=np.float64)
    edge_index_np = np.asarray(spanner_edge_index, dtype=np.int64)
    edge_attr_np = None if spanner_edge_attr is None else np.asarray(spanner_edge_attr, dtype=np.float64)

    if pos_np.ndim != 2 or pos_np.shape[1] != 2:
        raise ValueError(f"pos must be [N,2], got {tuple(pos_np.shape)}")

    n = int(pos_np.shape[0])
    _validate_undirected_spanner(edge_index_np, n)

    euc = _compute_full_euclidean_matrix(pos_np)
    max_euc = float(np.max(euc)) if euc.size > 0 else 1.0
    max_scaled = max(int(round(max_euc * float(cost_scale))), 1)
    penalty_margin = int(max_scaled * max(1, penalty_factor) * max(1, n))

    matrix, eid_map, edge_lengths = _build_sparse_lkh_matrix(
        pos=pos_np,
        edge_index=edge_index_np,
        edge_attr=edge_attr_np,
        cost_scale=cost_scale,
        penalty_margin=penalty_margin,
    )
    candidates = _build_spanner_candidates(
        edge_index=edge_index_np,
        edge_lengths=edge_lengths,
        cost_scale=cost_scale,
        num_nodes=n,
    )

    tmpdir = tempfile.mkdtemp(prefix="teacher_spanner_lkh_")
    tsp_path = str(Path(tmpdir) / "problem.tsp")
    par_path = str(Path(tmpdir) / "problem.par")
    tour_path = str(Path(tmpdir) / "problem.tour")
    cand_path = str(Path(tmpdir) / "problem.cand")

    start_t = time.time()
    try:
        write_tsp_explicit(tsp_path, "spanner_teacher", matrix)
        write_candidate_file(cand_path, n, candidates)
        write_par(
            par_path,
            tsp_path,
            tour_path,
            runs=max(1, int(runs)),
            seed=int(seed),
            precision=1,
            candidate_path=cand_path,
            subgradient=True,
            # Force LKH to use exactly the spanner edges we provided in the
            # candidate file. Leaving MAX_CANDIDATES at its default lets LKH
            # rebuild its own candidate sets, which can crash inside
            # Minimum1TreeCost on these sparse constrained instances.
            max_candidates=0,
            max_trials=None,
        )
        ok = run_lkh(executable, par_path, timeout=timeout)
        if not ok:
            raise RuntimeError(f"LKH failed while generating a spanner teacher tour (exe={executable}).")

        order = parse_tour(tour_path)
        edge_ids, length = _validate_tour_on_spanner(
            order=order,
            num_nodes=n,
            eid_map=eid_map,
            edge_lengths=edge_lengths,
        )
        return TeacherTour(
            order=order,
            edge_ids=edge_ids,
            length=length,
            duration=time.time() - start_t,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def solve_spanner_tour_exact(
    *,
    pos: np.ndarray,
    spanner_edge_index: np.ndarray,
    spanner_edge_attr: Optional[np.ndarray] = None,
    time_limit: Optional[float] = 30.0,
    length_weight: float = 1.0,
) -> TeacherTour:
    """Exact sparse fallback for small constrained teacher instances."""
    import torch

    from src.models.exact_decode import decode_tour_exact_from_edge_logits

    pos_np = np.asarray(pos, dtype=np.float64)
    edge_index_np = np.asarray(spanner_edge_index, dtype=np.int64)
    edge_attr_np = None if spanner_edge_attr is None else np.asarray(spanner_edge_attr, dtype=np.float64)

    if pos_np.ndim != 2 or pos_np.shape[1] != 2:
        raise ValueError(f"pos must be [N,2], got {tuple(pos_np.shape)}")
    n = int(pos_np.shape[0])
    _validate_undirected_spanner(edge_index_np, n)

    start_t = time.time()
    res = decode_tour_exact_from_edge_logits(
        pos=torch.as_tensor(pos_np, dtype=torch.float32),
        spanner_edge_index=torch.as_tensor(edge_index_np, dtype=torch.long),
        edge_logit=torch.zeros((int(edge_index_np.shape[1]),), dtype=torch.float32),
        time_limit=time_limit,
        length_weight=float(length_weight),
    )
    if not bool(res.feasible) or not res.order or len(res.order) != n:
        raise RuntimeError("Exact sparse fallback failed to find a legal Hamiltonian cycle on the spanner.")

    edge_lengths = _edge_lengths_from_attr_or_pos(
        pos=pos_np,
        edge_index=edge_index_np,
        edge_attr=edge_attr_np,
    )
    eid_map: Dict[Tuple[int, int], int] = {}
    for eid in range(int(edge_index_np.shape[1])):
        a = int(edge_index_np[0, eid])
        b = int(edge_index_np[1, eid])
        eid_map[_edge_key(a, b)] = eid
    edge_ids, length = _validate_tour_on_spanner(
        order=[int(x) for x in res.order],
        num_nodes=n,
        eid_map=eid_map,
        edge_lengths=edge_lengths,
    )
    return TeacherTour(
        order=[int(x) for x in res.order],
        edge_ids=edge_ids,
        length=length,
        duration=time.time() - start_t,
    )


__all__ = [
    "TEACHER_LABEL_VERSION",
    "TeacherTour",
    "solve_spanner_tour_exact",
    "solve_spanner_tour_lkh",
]
