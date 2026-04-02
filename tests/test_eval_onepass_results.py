from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.eval_onepass import _result_record_entry


def test_result_record_entry_keeps_length_without_teacher():
    res = SimpleNamespace(feasible=True, length=123.4, duration=5.0)

    entry = _result_record_entry(res, float("nan"))

    assert entry["feasible"] is True
    assert entry["length"] == 123.4
    assert entry["gap"] is None


def test_result_record_entry_computes_gap_with_teacher():
    res = SimpleNamespace(feasible=True, length=105.0, duration=2.0)

    entry = _result_record_entry(res, 100.0)

    assert entry["feasible"] is True
    assert entry["length"] == 105.0
    assert abs(entry["gap"] - 0.05) < 1e-9


def test_result_record_entry_drops_infeasible_length():
    res = SimpleNamespace(feasible=False, length=float("inf"), duration=1.0)

    entry = _result_record_entry(res, 100.0)

    assert entry["feasible"] is False
    assert entry["length"] is None
    assert entry["gap"] is None
