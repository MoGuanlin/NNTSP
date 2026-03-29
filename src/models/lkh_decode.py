# src/models/lkh_decode.py
# -*- coding: utf-8 -*-

import os
import shutil
import tempfile
import uuid
import time
import numpy as np
import torch
from torch import Tensor
from typing import List, Tuple, Optional
from dataclasses import dataclass

from src.utils.lkh_solver import (
    parse_tour,
    resolve_lkh_executable,
    run_lkh,
    write_candidate_file,
    write_par,
    write_tour_file,
    write_tsp_euc2d,
    write_tsp_explicit,
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
    mode: str  # 'pure' or 'guided'
    duration: float  # execution time in seconds

class LKHDecodingDataset(torch.utils.data.Dataset):
    """Dataset wrapper for parallel LKH decoding via DataLoader."""

    def __init__(
        self, 
        tasks: List[dict],
        lkh_executable: str = "LKH",  # Placeholder path
        num_runs: int = 1,
        seed: int = 1234,
        logit_scale: float = 1e3,
        penalty_cost: int = 10**7,
        top_k: int = 20
    ) -> None:
        """
        tasks: List of dicts containing:
            - pos: [N, 2]
            - mode: 'pure' or 'guided'
            - edge_index: [2, E] (for guided)
            - edge_logit: [E] (for guided)
            - teacher_len: float
            - initial_tour: List[int] (optional, for warm start)
        """
        self.tasks = tasks
        self.lkh_exe = resolve_lkh_executable(lkh_executable)
        self.num_runs = num_runs
        self.seed = seed
        self.logit_scale = logit_scale
        self.penalty_cost = penalty_cost
        self.top_k = top_k

    def __len__(self) -> int:
        return len(self.tasks)

    def _logit_to_cost_matrix(self, N: int, edge_index: Tensor, edge_logit: Tensor, pos: Optional[np.ndarray] = None) -> np.ndarray:
        """Convert edge logits and Euclidean distances to an explicit integer cost matrix."""
        if pos is not None:
            # 1. Base cost = Euclidean distance * 100 (for precision)
            diff = pos[:, None, :] - pos[None, :, :]
            dist_matrix = np.linalg.norm(diff, axis=-1)
            # We use a multiplier to give room for logit penalties
            matrix = (dist_matrix * 100).astype(np.int64)
        else:
            # Fallback to pure logit-based cost if pos is missing
            matrix = np.full((N, N), self.penalty_cost, dtype=np.int64)
            np.fill_diagonal(matrix, 0)
        
        if edge_logit is None or edge_index is None:
            return matrix

        # Scale logits: lower cost for higher logit
        logits_np = edge_logit.detach().cpu().numpy()
        # Filter out masked logits (-1e9)
        valid_mask = logits_np > -1e8
        if not np.any(valid_mask):
            return matrix

        max_v = np.max(logits_np[valid_mask])
        u = edge_index[0].detach().cpu().numpy()
        v = edge_index[1].detach().cpu().numpy()
        
        for i in range(len(logits_np)):
            if valid_mask[i]:
                # Penalty = logit_scale * (max_v - logit)
                penalty = int(round(self.logit_scale * (max_v - logits_np[i])))
                matrix[u[i], v[i]] += penalty
                matrix[v[i], u[i]] += penalty
            else:
                # Masked edge in spanner: excessive penalty
                matrix[u[i], v[i]] += self.penalty_cost
                matrix[v[i], u[i]] += self.penalty_cost
        
        return matrix

    def _logits_to_candidates(self, N: int, edge_index: Tensor, edge_logit: Tensor) -> List[List[Tuple[int, int]]]:
        """Pick Top-K edges per node based on NN logits to serve as LKH candidates."""
        adj = [[] for _ in range(N)]
        if edge_logit is None or edge_index is None:
            return adj

        logits_np = edge_logit.detach().cpu().numpy()
        u = edge_index[0].detach().cpu().numpy()
        v = edge_index[1].detach().cpu().numpy()

        valid_mask = logits_np > -1e8
        if not np.any(valid_mask):
            return adj
            
        max_v = np.max(logits_np[valid_mask])

        for i in range(len(logits_np)):
            if not valid_mask[i]:
                continue
            # Alpha = Scale * (MaxLogit - Logit)
            alpha = int(round(self.logit_scale * (max_v - logits_np[i])))
            adj[u[i]].append((int(v[i]), alpha))
            adj[v[i]].append((int(u[i]), alpha))
        
        # Keep only Top-K for each node
        final_candidates = []
        for i in range(N):
            # Sort by alpha (lower is better/higher logit)
            node_cands = sorted(adj[i], key=lambda x: x[1])
            final_candidates.append(node_cands[:self.top_k])
            
        return final_candidates

    def __getitem__(self, idx: int) -> Tuple[LKHDecodeResult, float]:
        task = self.tasks[idx]
        pos = task['pos']
        pos_np = pos.detach().cpu().numpy()
        mode = task['mode']
        N = pos.shape[0]
        
        # Create temp environment
        tmp_dir = tempfile.mkdtemp(prefix=f"lkh_{uuid.uuid4().hex[:8]}_")
        tsp_path = os.path.join(tmp_dir, "problem.tsp")
        par_path = os.path.join(tmp_dir, "config.par")
        tour_path = os.path.join(tmp_dir, "result.tour")
        cand_path = os.path.join(tmp_dir, "candidates.cand")
        init_tour_path = os.path.join(tmp_dir, "initial.itour")
        
        try:
            start_t = time.time()
            # Always use EUC_2D for problem type to enable geometric optimizations
            write_tsp_euc2d(tsp_path, f"{mode.capitalize()}TSP", pos_np)
            
            cand_p = None
            init_p = None
            subgrad = True
            m_cand = None
            m_trials = None

            if mode == 'guided':
                # Use fewer candidates for guided mode to emphasize NN-guidance speed
                candidates = self._logits_to_candidates(N, task.get('edge_index'), task.get('edge_logit'))
                write_candidate_file(cand_path, N, list(candidates)) # candidates is already a list of lists
                cand_p = cand_path
                
                # Warm start with initial tour
                initial_tour = task.get('initial_tour')
                if initial_tour is not None:
                    write_tour_file(init_tour_path, initial_tour)
                    init_p = init_tour_path
                
                # Guidance is strong, we can skip subgradient optimization
                subgrad = False
                # Restrict LKH search space to our NN candidates for speed
                m_cand = 5 
                m_trials = 1
            
            write_par(par_path, tsp_path, tour_path, runs=self.num_runs, seed=self.seed, 
                      precision=1, candidate_path=cand_p, initial_tour_path=init_p, 
                      subgradient=subgrad, max_candidates=m_cand, max_trials=m_trials)
            
            success = run_lkh(self.lkh_exe, par_path)
            duration = time.time() - start_t
            order = parse_tour(tour_path) if success else []
            
            # Calculate length on original Euclidean coordinates
            length = 0.0
            feasible = len(order) == N and len(set(order)) == N
            if feasible:
                p = pos_np
                for i in range(N):
                    u, v = order[i], order[(i+1)%N]
                    length += np.linalg.norm(p[u] - p[v])
            else:
                length = float('inf')
            
            res = LKHDecodeResult(order=order, length=length, feasible=feasible, mode=mode, duration=duration)
            return res, task.get('teacher_len', 0.0)
            
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

def solve_with_lkh_parallel(tasks: List[dict], num_workers: int = 4, **lkh_args) -> List[Tuple[LKHDecodeResult, float]]:
    """Helper to run LKH on a collection of tasks using DataLoader parallelism."""
    # Ensure all tensors are on CPU before passing to DataLoader workers
    # to avoid CUDA initialization errors in subprocesses.
    cpu_tasks = []
    for task in tasks:
        cpu_task = {}
        for k, v in task.items():
            if isinstance(v, torch.Tensor):
                cpu_task[k] = v.detach().cpu()
            else:
                cpu_task[k] = v
        cpu_tasks.append(cpu_task)

    dataset = LKHDecodingDataset(cpu_tasks, **lkh_args)
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        num_workers=num_workers, 
        shuffle=False, 
        collate_fn=lambda x: x[0]
    )
    
    results = []
    for item in loader:
        results.append(item)
    return results
