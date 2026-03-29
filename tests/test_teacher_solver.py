# tests/test_teacher_solver.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.models.teacher_solver as teacher_solver
from src.utils.lkh_solver import default_lkh_executable, resolve_lkh_executable


def _square_instance():
    pos = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    edge_index = np.array(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
        ],
        dtype=np.int64,
    )
    edge_attr = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    return pos, edge_index, edge_attr


def test_solve_spanner_tour_lkh_accepts_spanner_cycle(monkeypatch):
    pos, edge_index, edge_attr = _square_instance()

    monkeypatch.setattr(teacher_solver, "run_lkh", lambda executable, par_path, timeout=None: True)
    monkeypatch.setattr(teacher_solver, "parse_tour", lambda path: [0, 1, 2, 3])

    tour = teacher_solver.solve_spanner_tour_lkh(
        pos=pos,
        spanner_edge_index=edge_index,
        spanner_edge_attr=edge_attr,
        executable="fake_lkh",
    )

    assert tour.order == [0, 1, 2, 3]
    assert tour.edge_ids == [0, 1, 2, 3]
    assert abs(tour.length - 4.0) < 1e-9


def test_solve_spanner_tour_lkh_rejects_off_spanner_cycle(monkeypatch):
    pos, edge_index, edge_attr = _square_instance()

    monkeypatch.setattr(teacher_solver, "run_lkh", lambda executable, par_path, timeout=None: True)
    monkeypatch.setattr(teacher_solver, "parse_tour", lambda path: [0, 2, 1, 3])

    with pytest.raises(RuntimeError, match="off-spanner"):
        teacher_solver.solve_spanner_tour_lkh(
            pos=pos,
            spanner_edge_index=edge_index,
            spanner_edge_attr=edge_attr,
            executable="fake_lkh",
        )


def test_default_lkh_executable_uses_repo_bundled_binary():
    exe = default_lkh_executable()
    assert Path(exe).is_file()
    assert Path(exe).name == "LKH"
    assert "data/lkh/LKH-3.0.13/LKH" in exe.replace("\\", "/")


def test_resolve_lkh_executable_prefers_existing_repo_binary_for_default_name():
    exe = resolve_lkh_executable("LKH")
    assert Path(exe).is_file()
    assert Path(exe).name == "LKH"
