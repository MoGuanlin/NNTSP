# src/models/lkh_decode.py
# -*- coding: utf-8 -*-

from concurrent.futures import ThreadPoolExecutor
import os
import shutil
import tempfile
import time
import numpy as np
import torch
from torch import Tensor
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from dataclasses import dataclass

from src.utils.lkh_solver import (
    parse_tour,
    resolve_lkh_executable,
    run_lkh,
    run_lkh_with_status,
    write_candidate_file,
    write_par,
    write_tour_file,
    write_tsp_euc2d,
)

# Results reported in the paper Table 3: Comparative results on 10 largest instances of TSPLIB.
# Time is converted to seconds: 9.9m -> 594s, 12.4m -> 744s, etc.
TSPLIB_PAPER_RESULTS = {
    "rl5915":   {"obj": 572085,    "time": 594},
    "rl5934":   {"obj": 559712,    "time": 594},
    "pla7397":  {"obj": 23382264,  "time": 744},
    "rl11849":  {"obj": 929001,    "time": 1134},
    "usa13509": {"obj": 20133724,  "time": 1356},
    "brd14051": {"obj": 474149,    "time": 1410},
    "d15112":   {"obj": 1588550,   "time": 1512},
    "d18512":   {"obj": 652911,    "time": 1860},
    "pla33810": {"obj": 67084217,  "time": 3384},
    "pla85900": {"obj": 144004484, "time": 23400}, # 6.5h = 23400s
}

@dataclass
class LKHDecodeResult:
    order: List[int]
    length: float
    feasible: bool
    mode: str  # 'pure', 'guided', or 'spanner_uniform'
    duration: float  # execution time in seconds


@dataclass(frozen=True)
class CandidateLKHConfig:
    subgradient: bool = False
    max_candidates: Optional[int] = 5
    max_trials: Optional[int] = 1
    use_initial_tour: bool = True

    def __post_init__(self) -> None:
        if self.max_candidates is not None and int(self.max_candidates) < 0:
            raise ValueError(f"max_candidates must be >= 0 or None, got {self.max_candidates}")
        if self.max_trials is not None and int(self.max_trials) < 0:
            raise ValueError(f"max_trials must be >= 0 or None, got {self.max_trials}")


@dataclass(frozen=True)
class GuidedLKHConfig(CandidateLKHConfig):
    top_k: int = 20
    logit_scale: float = 1e3

    def __post_init__(self) -> None:
        super().__post_init__()
        if int(self.top_k) <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        if float(self.logit_scale) <= 0.0:
            raise ValueError(f"logit_scale must be positive, got {self.logit_scale}")


DEFAULT_GUIDED_LKH_CONFIG = GuidedLKHConfig()
DEFAULT_SPANNER_UNIFORM_LKH_CONFIG = CandidateLKHConfig(
    subgradient=False,
    max_candidates=0,
    max_trials=1,
    use_initial_tour=True,
)


def build_guided_candidates(
    *,
    num_nodes: int,
    edge_index: Tensor,
    edge_logit: Tensor,
    logit_scale: float,
    top_k: int,
) -> List[List[Tuple[int, int]]]:
    """Build Top-K per-node LKH candidates from predicted edge logits."""
    candidates: List[List[Tuple[int, int]]] = [[] for _ in range(int(num_nodes))]
    if edge_logit is None or edge_index is None:
        return candidates

    logits_np = edge_logit.detach().cpu().numpy()
    valid_mask = logits_np > -1e8
    if not np.any(valid_mask):
        return candidates

    u = edge_index[0].detach().cpu().numpy()
    v = edge_index[1].detach().cpu().numpy()
    max_v = np.max(logits_np[valid_mask])

    for i in range(len(logits_np)):
        if not valid_mask[i]:
            continue
        alpha = int(round(float(logit_scale) * (float(max_v) - float(logits_np[i]))))
        a = int(u[i])
        b = int(v[i])
        candidates[a].append((b, alpha))
        candidates[b].append((a, alpha))

    final_candidates: List[List[Tuple[int, int]]] = []
    for row in candidates:
        row_sorted = sorted(row, key=lambda x: (x[1], x[0]))
        final_candidates.append(row_sorted[: int(top_k)])
    return final_candidates


def build_uniform_spanner_candidates(
    *,
    num_nodes: int,
    edge_index: Tensor,
    uniform_alpha: int = 0,
) -> List[List[Tuple[int, int]]]:
    """Build all spanner candidates with a uniform alpha value."""
    candidates: List[List[Tuple[int, int]]] = [[] for _ in range(int(num_nodes))]
    if edge_index is None:
        return candidates

    u = edge_index[0].detach().cpu().numpy()
    v = edge_index[1].detach().cpu().numpy()
    alpha = int(uniform_alpha)

    for i in range(int(edge_index.shape[1])):
        a = int(u[i])
        b = int(v[i])
        if a == b:
            continue
        candidates[a].append((b, alpha))
        candidates[b].append((a, alpha))

    return [sorted(row, key=lambda x: (x[0], x[1])) for row in candidates]


def write_candidate_file_from_arrays(path: str, neighbors: np.ndarray, alphas: np.ndarray) -> None:
    """Write an LKH candidate file from dense neighbor / alpha arrays."""
    n_nodes, _ = neighbors.shape
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{n_nodes}\n")
        for i in range(n_nodes):
            row_neighbors = neighbors[i]
            row_alphas = alphas[i]
            valid = row_neighbors >= 0
            row_neighbors = row_neighbors[valid]
            row_alphas = row_alphas[valid]
            order = np.lexsort((row_neighbors, row_alphas))
            row_neighbors = row_neighbors[order]
            row_alphas = row_alphas[order]

            line = [str(i + 1), "0", str(int(row_neighbors.shape[0]))]
            for nb, alpha in zip(row_neighbors.tolist(), row_alphas.tolist()):
                line.extend([str(int(nb) + 1), str(int(alpha))])
            f.write(" ".join(line) + "\n")
        f.write("-1\n")


def _compute_euclidean_tour_length(pos_np: np.ndarray, order: Sequence[int]) -> float:
    total = 0.0
    n = len(order)
    for i in range(n):
        u = int(order[i])
        v = int(order[(i + 1) % n])
        total += float(np.linalg.norm(pos_np[u] - pos_np[v]))
    return total


def _run_candidate_lkh_timed(
    *,
    pos: Tensor,
    mode: str,
    initial_tour: Optional[Sequence[int]],
    lkh_executable: str,
    num_runs: int,
    seed: int,
    timeout: Optional[float],
    candidate_config: CandidateLKHConfig,
    write_candidates_fn: Callable[[str, int], None],
    heartbeat_sec: float = 0.0,
    heartbeat_emit: Optional[Callable[[str], None]] = None,
) -> Tuple[LKHDecodeResult, Dict[str, object]]:
    pos_cpu = pos.detach().cpu()
    pos_np = pos_cpu.numpy()
    num_nodes = int(pos_cpu.shape[0])

    tmp_dir = tempfile.mkdtemp(prefix=f"timed_{mode}_lkh_")
    tsp_path = os.path.join(tmp_dir, "problem.tsp")
    par_path = os.path.join(tmp_dir, "config.par")
    tour_path = os.path.join(tmp_dir, "result.tour")
    cand_path = os.path.join(tmp_dir, "candidates.cand")
    init_tour_path = os.path.join(tmp_dir, "initial.itour")

    setup_t0 = time.perf_counter()
    try:
        write_tsp_euc2d(tsp_path, f"{mode.capitalize()}TSP", pos_np)
        write_candidates_fn(cand_path, num_nodes)

        init_p = None
        if candidate_config.use_initial_tour and initial_tour is not None:
            write_tour_file(init_tour_path, [int(x) for x in initial_tour])
            init_p = init_tour_path

        write_par(
            par_path,
            tsp_path,
            tour_path,
            runs=max(1, int(num_runs)),
            seed=int(seed),
            precision=1,
            candidate_path=cand_path,
            initial_tour_path=init_p,
            subgradient=bool(candidate_config.subgradient),
            max_candidates=candidate_config.max_candidates,
            max_trials=candidate_config.max_trials,
        )
        setup_sec = time.perf_counter() - setup_t0

        status = run_lkh_with_status(
            lkh_executable,
            par_path,
            timeout=timeout,
            heartbeat_sec=heartbeat_sec,
            heartbeat_emit=heartbeat_emit,
        )
        search_sec = float(status.elapsed_sec)

        parse_t0 = time.perf_counter()
        order = parse_tour(tour_path) if status.ok else []
        feasible = len(order) == num_nodes and len(set(order)) == num_nodes
        length = _compute_euclidean_tour_length(pos_np, order) if feasible else float("inf")
        parse_sec = time.perf_counter() - parse_t0

        result = LKHDecodeResult(
            order=order,
            length=length,
            feasible=feasible,
            mode=str(mode),
            duration=setup_sec + search_sec + parse_sec,
        )
        return result, {
            "lkh_setup_io_sec": float(setup_sec),
            "lkh_search_sec": float(search_sec),
            "lkh_parse_sec": float(parse_sec),
            "timeout_hit": bool(status.timeout_hit),
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_candidate_lkh_timed(
    *,
    pos: Tensor,
    candidates: List[List[Tuple[int, int]]],
    initial_tour: Optional[Sequence[int]],
    lkh_executable: str,
    num_runs: int,
    seed: int,
    timeout: Optional[float],
    mode: str = "guided",
    candidate_config: Optional[CandidateLKHConfig] = None,
    heartbeat_sec: float = 0.0,
    heartbeat_emit: Optional[Callable[[str], None]] = None,
) -> Tuple[LKHDecodeResult, Dict[str, object]]:
    config = candidate_config or CandidateLKHConfig()
    return _run_candidate_lkh_timed(
        pos=pos,
        mode=mode,
        initial_tour=initial_tour,
        lkh_executable=lkh_executable,
        num_runs=num_runs,
        seed=seed,
        timeout=timeout,
        candidate_config=config,
        write_candidates_fn=lambda cand_path, num_nodes: write_candidate_file(
            cand_path,
            num_nodes,
            list(candidates),
        ),
        heartbeat_sec=heartbeat_sec,
        heartbeat_emit=heartbeat_emit,
    )


def run_candidate_lkh_timed_from_arrays(
    *,
    pos: Tensor,
    neighbors: np.ndarray,
    alphas: np.ndarray,
    initial_tour: Optional[Sequence[int]],
    lkh_executable: str,
    num_runs: int,
    seed: int,
    timeout: Optional[float],
    mode: str = "guided",
    candidate_config: Optional[CandidateLKHConfig] = None,
    heartbeat_sec: float = 0.0,
    heartbeat_emit: Optional[Callable[[str], None]] = None,
) -> Tuple[LKHDecodeResult, Dict[str, object]]:
    config = candidate_config or CandidateLKHConfig()
    return _run_candidate_lkh_timed(
        pos=pos,
        mode=mode,
        initial_tour=initial_tour,
        lkh_executable=lkh_executable,
        num_runs=num_runs,
        seed=seed,
        timeout=timeout,
        candidate_config=config,
        write_candidates_fn=lambda cand_path, _num_nodes: write_candidate_file_from_arrays(
            cand_path,
            neighbors,
            alphas,
        ),
        heartbeat_sec=heartbeat_sec,
        heartbeat_emit=heartbeat_emit,
    )

def _run_pure_lkh_timed(
    *,
    pos: Tensor,
    lkh_executable: str,
    num_runs: int,
    seed: int,
    timeout: Optional[float],
) -> LKHDecodeResult:
    pos_cpu = pos.detach().cpu()
    pos_np = pos_cpu.numpy()
    num_nodes = int(pos_cpu.shape[0])

    tmp_dir = tempfile.mkdtemp(prefix="timed_pure_lkh_")
    tsp_path = os.path.join(tmp_dir, "problem.tsp")
    par_path = os.path.join(tmp_dir, "config.par")
    tour_path = os.path.join(tmp_dir, "result.tour")

    start_t = time.perf_counter()
    try:
        write_tsp_euc2d(tsp_path, "PureTSP", pos_np)
        write_par(
            par_path,
            tsp_path,
            tour_path,
            runs=max(1, int(num_runs)),
            seed=int(seed),
            precision=1,
        )
        ok = run_lkh(lkh_executable, par_path, timeout=timeout)
        order = parse_tour(tour_path) if ok else []
        feasible = len(order) == num_nodes and len(set(order)) == num_nodes
        length = _compute_euclidean_tour_length(pos_np, order) if feasible else float("inf")
        return LKHDecodeResult(
            order=order,
            length=length,
            feasible=feasible,
            mode="pure",
            duration=time.perf_counter() - start_t,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _normalize_lkh_task(task: dict) -> dict:
    cpu_task: Dict[str, object] = {}
    for key, value in task.items():
        if isinstance(value, torch.Tensor):
            cpu_task[key] = value.detach().cpu()
        else:
            cpu_task[key] = value
    return cpu_task


def _solve_lkh_task(
    task: dict,
    *,
    lkh_executable: str,
    num_runs: int,
    seed: int,
    timeout: Optional[float],
    guided_config: GuidedLKHConfig,
) -> Tuple[LKHDecodeResult, float]:
    pos = task["pos"]
    mode = str(task["mode"]).lower()
    initial_tour = task.get("initial_tour")

    if mode == "guided":
        candidates = build_guided_candidates(
            num_nodes=int(pos.shape[0]),
            edge_index=task.get("edge_index"),
            edge_logit=task.get("edge_logit"),
            logit_scale=float(guided_config.logit_scale),
            top_k=int(guided_config.top_k),
        )
        result, _ = run_candidate_lkh_timed(
            pos=pos,
            candidates=candidates,
            initial_tour=initial_tour,
            lkh_executable=lkh_executable,
            num_runs=num_runs,
            seed=seed,
            timeout=timeout,
            mode="guided",
            candidate_config=guided_config,
        )
    elif mode == "spanner_uniform":
        candidates = build_uniform_spanner_candidates(
            num_nodes=int(pos.shape[0]),
            edge_index=task.get("edge_index"),
            uniform_alpha=int(task.get("uniform_alpha", 0)),
        )
        result, _ = run_candidate_lkh_timed(
            pos=pos,
            candidates=candidates,
            initial_tour=initial_tour,
            lkh_executable=lkh_executable,
            num_runs=num_runs,
            seed=seed,
            timeout=timeout,
            mode="spanner_uniform",
            candidate_config=DEFAULT_SPANNER_UNIFORM_LKH_CONFIG,
        )
    elif mode == "pure":
        result = _run_pure_lkh_timed(
            pos=pos,
            lkh_executable=lkh_executable,
            num_runs=num_runs,
            seed=seed,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unsupported LKH decode mode: {mode}")

    return result, float(task.get("teacher_len", 0.0))


def solve_with_lkh_parallel(
    tasks: List[dict],
    *,
    lkh_executable: str = "LKH",
    num_workers: int = 4,
    num_runs: int = 1,
    seed: int = 1234,
    logit_scale: float = DEFAULT_GUIDED_LKH_CONFIG.logit_scale,
    top_k: int = DEFAULT_GUIDED_LKH_CONFIG.top_k,
    guided_config: Optional[GuidedLKHConfig] = None,
    timeout: Optional[float] = None,
) -> List[Tuple[LKHDecodeResult, float]]:
    """Run LKH tasks through an explicit task pool instead of a DataLoader shim."""
    if not tasks:
        return []

    resolved_exe = resolve_lkh_executable(lkh_executable)
    config = guided_config or GuidedLKHConfig(
        top_k=int(top_k),
        logit_scale=float(logit_scale),
    )
    cpu_tasks = [_normalize_lkh_task(task) for task in tasks]
    max_workers = max(1, min(int(num_workers), len(cpu_tasks)))

    if max_workers == 1:
        return [
            _solve_lkh_task(
                task,
                lkh_executable=resolved_exe,
                num_runs=int(num_runs),
                seed=int(seed),
                timeout=timeout,
                guided_config=config,
            )
            for task in cpu_tasks
        ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _solve_lkh_task,
                task,
                lkh_executable=resolved_exe,
                num_runs=int(num_runs),
                seed=int(seed),
                timeout=timeout,
                guided_config=config,
            )
            for task in cpu_tasks
        ]
        return [future.result() for future in futures]
