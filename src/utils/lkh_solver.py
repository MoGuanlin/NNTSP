# src/utils/lkh_solver.py
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import shutil
import subprocess
from typing import List, Optional, Tuple
import numpy as np


def bundled_lkh_executable() -> Optional[str]:
    """Return the repo-bundled LKH executable when present."""
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "data" / "lkh" / "LKH-3.0.13" / "LKH"
    if candidate.is_file():
        return str(candidate)
    return None


def resolve_lkh_executable(executable: Optional[str] = None) -> str:
    """Resolve an LKH executable path with a repo-local fallback.

    Resolution order:
      1. Existing explicit filesystem path
      2. `LKH_EXE` environment variable when set and valid
      3. PATH lookup for the requested executable name
      4. Repo-bundled `data/lkh/LKH-3.0.13/LKH`

    If a custom non-default executable is requested and cannot be found, we
    raise immediately instead of silently swapping in a different binary.
    """
    requested = str(executable or "").strip()
    default_names = {"", "LKH", "lkh"}

    lookup_name = requested or "LKH"
    if requested:
        requested_path = Path(requested).expanduser()
        if requested_path.is_file():
            return str(requested_path.resolve())

    if lookup_name in default_names:
        env_lkh = os.environ.get("LKH_EXE", "").strip()
        if env_lkh:
            env_path = Path(env_lkh).expanduser()
            if env_path.is_file():
                return str(env_path.resolve())

    which_hit = shutil.which(lookup_name)
    if which_hit:
        return str(Path(which_hit).resolve())

    bundled = bundled_lkh_executable()
    if bundled is not None and lookup_name in default_names:
        return bundled

    if lookup_name not in default_names:
        raise FileNotFoundError(f"LKH executable not found: {lookup_name}")

    if bundled is not None:
        return bundled
    return lookup_name


def default_lkh_executable() -> str:
    """Best-effort default LKH path for CLI defaults."""
    return resolve_lkh_executable("LKH")

def write_tsp_euc2d(path: str, name: str, pos: np.ndarray):
    """Write a TSPLIB file in EUC_2D format."""
    num_nodes = pos.shape[0]
    with open(path, "w") as f:
        f.write(f"NAME : {name}\n")
        f.write("TYPE : TSP\n")
        f.write(f"DIMENSION : {num_nodes}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i in range(num_nodes):
            # LKH expects 1-indexed nodes
            f.write(f"{i+1} {pos[i, 0]} {pos[i, 1]}\n")
        f.write("EOF\n")

def write_tsp_explicit(path: str, name: str, matrix: np.ndarray):
    """Write a TSPLIB file with EXPLICIT weight matrix."""
    num_nodes = matrix.shape[0]
    with open(path, "w") as f:
        f.write(f"NAME : {name}\n")
        f.write("TYPE : TSP\n")
        f.write(f"DIMENSION : {num_nodes}\n")
        f.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        for i in range(num_nodes):
            row = matrix[i].astype(int)
            f.write(" ".join(map(str, row)) + "\n")
        f.write("EOF\n")

def write_par(path: str, tsp_path: str, tour_path: str, runs: int = 1, seed: int = 1234, precision: int = 1, 
              candidate_path: Optional[str] = None, initial_tour_path: Optional[str] = None, 
              subgradient: bool = True, max_candidates: Optional[int] = None, max_trials: Optional[int] = None):
    """Write an LKH parameter file."""
    with open(path, "w") as f:
        f.write(f"PROBLEM_FILE = {tsp_path}\n")
        f.write(f"OUTPUT_TOUR_FILE = {tour_path}\n")
        f.write(f"RUNS = {runs}\n")
        f.write(f"SEED = {seed}\n")
        f.write(f"PRECISION = {precision}\n")
        if candidate_path:
            f.write(f"CANDIDATE_FILE = {candidate_path}\n")
        if initial_tour_path:
            f.write(f"INPUT_TOUR_FILE = {initial_tour_path}\n")
        if not subgradient:
            f.write("SUBGRADIENT = NO\n")
        if max_candidates is not None:
            f.write(f"MAX_CANDIDATES = {max_candidates}\n")
        if max_trials is not None:
            f.write(f"MAX_TRIALS = {max_trials}\n")

def write_candidate_file(path: str, num_nodes: int, candidates: List[List[Tuple[int, int]]]):
    """
    Write an LKH candidate file.
    candidates: List of neighbors for each node (1-indexed node lists).
    Format:
    DIMENSION
    node_idx MST_parent num_candidates [neighbor alpha_value] ...
    """
    with open(path, "w") as f:
        f.write(f"{num_nodes}\n")
        for i in range(num_nodes):
            node_idx = i + 1
            neighbors = candidates[i]
            line = [str(node_idx), "0", str(len(neighbors))]
            for nb, alpha in neighbors:
                line.extend([str(nb + 1), str(alpha)])
            f.write(" ".join(line) + "\n")
        f.write("-1\n")

def write_tour_file(path: str, order: List[int]):
    """Write an LKH initial tour file."""
    with open(path, "w") as f:
        f.write("TOUR_SECTION\n")
        for node in order:
            f.write(f"{node + 1}\n")
        f.write("-1\n")
        f.write("EOF\n")

def run_lkh(executable: str, par_path: str, timeout: Optional[float] = None):
    """Execute LKH-3 subprocess."""
    try:
        resolved_exe = resolve_lkh_executable(executable)
        result = subprocess.run(
            [resolved_exe, par_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
        if result.returncode != 0:
            print(f"[LKH Error] {result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[LKH Timeout] Exceeded {timeout}s")
        return False
    except Exception as e:
        print(f"[LKH System Error] {e}")
        return False

def parse_tour(path: str) -> List[int]:
    """Parse LKH .tour file and return 0-indexed node order."""
    if not os.path.exists(path):
        return []
    
    order = []
    started = False
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "TOUR_SECTION":
                started = True
                continue
            if line == "-1" or line == "EOF":
                break
            if started:
                # LKH nodes are 1-indexed
                node_idx = int(line) - 1
                order.append(node_idx)
    return order

def solve_tsp_lkh(
    pos: np.ndarray,
    executable: str = "LKH",
    runs: int = 1,
    timeout: Optional[float] = None,
    prefix: str = "tsp_lkh"
) -> List[int]:
    """
    Solve TSP using LKH-3.
    Returns 0-indexed node order.
    """
    import tempfile
    import shutil
    from pathlib import Path

    tmpdir = tempfile.mkdtemp(prefix=prefix)
    try:
        tsp_path = str(Path(tmpdir) / "problem.tsp")
        par_path = str(Path(tmpdir) / "problem.par")
        tour_path = str(Path(tmpdir) / "problem.tour")
        
        write_tsp_euc2d(tsp_path, prefix, pos)
        write_par(par_path, tsp_path, tour_path, runs=runs)
        
        success = run_lkh(executable, par_path, timeout=timeout)
        if success:
            return parse_tour(tour_path)
        return []
    finally:
        shutil.rmtree(tmpdir)
