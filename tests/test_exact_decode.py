# tests/test_exact_decode.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


def _add_repo_root_to_syspath() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    sys.path.insert(0, str(repo_root))


def main() -> None:
    _add_repo_root_to_syspath()

    from src.models.decode_backend import decode_tour

    pos = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    spanner_edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0, 1],
            [1, 2, 3, 0, 2, 3],
        ],
        dtype=torch.long,
    )
    edge_logit = torch.tensor([5.0, 5.0, 5.0, 5.0, -2.0, -2.0], dtype=torch.float32)

    res = decode_tour(
        pos=pos,
        spanner_edge_index=spanner_edge_index,
        edge_logit=edge_logit,
        backend="exact",
        exact_time_limit=5.0,
    )

    assert res.feasible, "Exact sparse decoder should find a Hamiltonian cycle on the square."
    assert len(res.order) == 4, f"Expected a 4-node cycle, got {res.order}"
    assert math.isclose(float(res.length), 4.0, rel_tol=1e-6, abs_tol=1e-6), res.length

    print(
        f"[exact-decode] feasible={res.feasible} len={res.length:.6f} "
        f"order={list(map(int, res.order))} duration={res.duration:.3f}s"
    )


if __name__ == "__main__":
    main()
