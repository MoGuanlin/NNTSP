# tests/test_teacher_max_used_stats.py
# -*- coding: utf-8 -*-
"""
Regression tests for teacher max_used statistics shown at 1-pass train startup.

Usage:
  python tests/test_teacher_max_used_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.train_onepass import PrecomputedBatch, _collect_teacher_max_used_stats


def test_collect_teacher_max_used_stats_basic():
    packed = SimpleNamespace(
        tokens=SimpleNamespace(
            is_leaf=torch.tensor([True, False, False, True], dtype=torch.bool),
        )
    )
    labels = SimpleNamespace(
        y_iface=torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],  # 0
                [1.0, 1.0, 0.0, 0.0],  # 2
                [1.0, 1.0, 1.0, 0.0],  # 3
                [1.0, 1.0, 1.0, 1.0],  # 4
            ],
            dtype=torch.float32,
        ),
        m_iface=torch.ones((4, 4), dtype=torch.bool),
        m_state_exact=torch.tensor([True, True, False, False], dtype=torch.bool),
    )
    batch = PrecomputedBatch(packed=packed, labels=labels)

    summary = _collect_teacher_max_used_stats(
        [batch],
        configured_max_used=2,
        show_progress=False,
    )

    assert summary["num_nodes_all"] == 4
    assert summary["num_nodes_internal"] == 2
    assert summary["overflow_all"] == 2
    assert summary["overflow_internal"] == 1
    assert summary["raw_hist_all"] == {0: 1, 2: 1, 3: 1, 4: 1}
    assert summary["raw_hist_internal"] == {2: 1, 3: 1}
    assert abs(summary["current_coverage_all"] - 0.5) < 1e-9
    assert abs(summary["current_coverage_internal"] - 0.5) < 1e-9
    assert abs(summary["exact_rate_all"] - 0.5) < 1e-9
    assert abs(summary["exact_rate_internal"] - 0.5) < 1e-9
    assert summary["suggested_max_used_90"] == 3
    assert summary["suggested_max_used_95"] == 3
    assert summary["suggested_max_used_99"] == 3
    print("teacher max_used summary =", summary)


def test_collect_teacher_max_used_stats_uses_exact_mask():
    packed = SimpleNamespace(
        tokens=SimpleNamespace(
            is_leaf=torch.tensor([False, False], dtype=torch.bool),
        )
    )
    labels = SimpleNamespace(
        y_iface=torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        m_iface=torch.ones((2, 4), dtype=torch.bool),
        m_state_exact=torch.tensor([True, False], dtype=torch.bool),
    )
    batch = PrecomputedBatch(packed=packed, labels=labels)

    summary = _collect_teacher_max_used_stats(
        [batch],
        configured_max_used=2,
        show_progress=False,
    )

    assert summary["num_nodes_all"] == 2
    assert summary["num_exact_all"] == 1
    assert abs(summary["exact_rate_all"] - 0.5) < 1e-9


if __name__ == "__main__":
    test_collect_teacher_max_used_stats_basic()
    test_collect_teacher_max_used_stats_uses_exact_mask()
    print("test_teacher_max_used_stats.py: PASS")
