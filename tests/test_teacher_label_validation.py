# tests/test_teacher_label_validation.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.train import ensure_dataset_labels
from src.cli.train import precompute_labels_for_dataset
from src.dataprep.dataset import FastTSPDataset, consolidate_data_list
from src.models.labeler import InfeasibleTeacherGraphError, PseudoLabeler


def _square_sample() -> SimpleNamespace:
    return SimpleNamespace(
        pos=torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        spanner_edge_index=torch.tensor(
            [
                [0, 1, 2, 3],
                [1, 2, 3, 0],
            ],
            dtype=torch.long,
        ),
        spanner_edge_attr=torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
    )


def _alive_square_with_dead_chords() -> SimpleNamespace:
    return SimpleNamespace(
        pos=torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        spanner_edge_index=torch.tensor(
            [
                [0, 1, 1, 0, 2, 3],
                [1, 3, 2, 2, 3, 0],
            ],
            dtype=torch.long,
        ),
        spanner_edge_attr=torch.tensor([1.0, 2.0, 1.0, 2.0, 1.0, 1.0], dtype=torch.float32),
        edge_alive_mask=torch.tensor([True, False, True, False, True, True], dtype=torch.bool),
        alive_edge_id=torch.tensor([0, 2, 4, 5], dtype=torch.long),
    )


def _infeasible_alive_sample() -> SimpleNamespace:
    return SimpleNamespace(
        pos=torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        spanner_edge_index=torch.tensor(
            [
                [0, 1, 2],
                [1, 2, 3],
            ],
            dtype=torch.long,
        ),
        spanner_edge_attr=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        edge_alive_mask=torch.tensor([True, True, True], dtype=torch.bool),
        alive_edge_id=torch.tensor([0, 1, 2], dtype=torch.long),
    )


def test_validate_teacher_labels_accepts_consistent_spanner_cycle():
    labeler = PseudoLabeler(teacher_mode="spanner_lkh")
    sample = _square_sample()
    labeler.attach_teacher_labels(
        data=sample,
        target_edges=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        tour_len=4.0,
        teacher_order=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        teacher_stats={"num_direct": 4},
    )

    ok, reason = labeler.validate_teacher_labels(sample)
    assert ok
    assert reason == "ok"


def test_validate_teacher_labels_rejects_mismatched_target_edges():
    labeler = PseudoLabeler(teacher_mode="spanner_lkh")
    sample = _square_sample()
    labeler.attach_teacher_labels(
        data=sample,
        target_edges=torch.tensor([0, 1, 1, 3], dtype=torch.long),
        tour_len=4.0,
        teacher_order=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        teacher_stats={"num_direct": 4},
    )

    ok, reason = labeler.validate_teacher_labels(sample)
    assert not ok
    assert reason in {"teacher_cycle_reuses_edge", "target_edges_mismatch_teacher_order"}


def test_validate_teacher_labels_rejects_dead_edge_teacher():
    labeler = PseudoLabeler(teacher_mode="spanner_lkh")
    sample = _alive_square_with_dead_chords()
    labeler.attach_teacher_labels(
        data=sample,
        target_edges=torch.tensor([0, 1, 4, 5], dtype=torch.long),
        tour_len=4.0,
        teacher_order=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        teacher_stats={"num_direct": 4},
    )

    ok, reason = labeler.validate_teacher_labels(sample)
    assert not ok
    assert reason == "target_edge_not_alive=1"


def test_extract_teacher_supervision_uses_alive_subgraph_and_remaps_eids():
    labeler = PseudoLabeler(teacher_mode="spanner_lkh")
    sample = _alive_square_with_dead_chords()

    def fake_solver(*, pos, spanner_edge_index, spanner_edge_attr, executable, runs, timeout):
        expected_edges = torch.tensor(
            [
                [0, 1, 2, 3],
                [1, 2, 3, 0],
            ],
            dtype=torch.long,
        )
        expected_attr = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        assert torch.equal(torch.as_tensor(spanner_edge_index), expected_edges)
        assert torch.equal(torch.as_tensor(spanner_edge_attr), expected_attr)
        assert pos.shape == (4, 2)
        return SimpleNamespace(order=[0, 1, 2, 3], edge_ids=[0, 1, 2, 3], length=4.0)

    with patch("src.models.labeler.solve_spanner_tour_lkh", side_effect=fake_solver):
        target_edges, tour_len, teacher_order, teacher_stats = labeler.extract_teacher_supervision(sample)

    assert torch.equal(torch.as_tensor(target_edges), torch.tensor([0, 2, 4, 5], dtype=torch.long))
    assert torch.equal(torch.as_tensor(teacher_order), torch.tensor([0, 1, 2, 3], dtype=torch.long))
    assert abs(float(tour_len) - 4.0) < 1e-6
    assert teacher_stats["num_direct"] == 4
    assert teacher_stats["num_not_alive_direct"] == 0


def test_extract_teacher_supervision_rejects_infeasible_alive_graph():
    labeler = PseudoLabeler(teacher_mode="spanner_lkh")
    sample = _infeasible_alive_sample()

    try:
        labeler.extract_teacher_supervision(sample)
    except InfeasibleTeacherGraphError as exc:
        assert str(exc).startswith("alive_teacher_graph_infeasible:degree_lt_2:")
        return

    raise AssertionError("Expected infeasible alive graph to raise InfeasibleTeacherGraphError")


def test_ensure_dataset_labels_reloads_original_dataset_when_fast_missing_alive_metadata():
    labeler = PseudoLabeler(teacher_mode="spanner_lkh")
    sample = _alive_square_with_dead_chords()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        source_path = tmp_path / "toy.pt"
        torch.save([sample], source_path)

        legacy = _alive_square_with_dead_chords()
        delattr(legacy, "edge_alive_mask")
        delattr(legacy, "alive_edge_id")
        fast_dataset = FastTSPDataset(consolidate_data_list([legacy]))

        def fake_extract(data):
            assert hasattr(data, "edge_alive_mask")
            assert hasattr(data, "alive_edge_id")
            return (
                torch.tensor([0, 2, 4, 5], dtype=torch.long).numpy(),
                4.0,
                torch.tensor([0, 1, 2, 3], dtype=torch.long).numpy(),
                {"num_direct": 4, "num_projected": 0, "num_unreachable": 0, "num_not_alive_direct": 0},
            )

        with patch.object(labeler, "extract_teacher_supervision", side_effect=fake_extract):
            out = ensure_dataset_labels(
                fast_dataset,
                source_path=str(source_path),
                labeler=labeler,
                num_workers=0,
                desc="test",
            )

        item = out[0]
        assert hasattr(item, "edge_alive_mask")
        assert hasattr(item, "alive_edge_id")
        assert torch.equal(item.edge_alive_mask, torch.tensor([True, False, True, False, True, True], dtype=torch.bool))
        assert torch.equal(item.alive_edge_id, torch.tensor([0, 2, 4, 5], dtype=torch.long))
        assert torch.equal(item.target_edges, torch.tensor([0, 2, 4, 5], dtype=torch.long))
        assert torch.equal(item.teacher_order, torch.tensor([0, 1, 2, 3], dtype=torch.long))
        assert abs(float(item.tour_len.item()) - 4.0) < 1e-6


def test_precompute_labels_drops_infeasible_samples():
    labeler = PseudoLabeler(teacher_mode="spanner_lkh")
    valid = _alive_square_with_dead_chords()
    invalid = _infeasible_alive_sample()

    def fake_extract(data):
        if hasattr(data, "edge_alive_mask") and int(torch.as_tensor(data.edge_alive_mask).numel()) == 3:
            raise InfeasibleTeacherGraphError("alive_teacher_graph_infeasible:degree_lt_2:0,3")
        return (
            torch.tensor([0, 2, 4, 5], dtype=torch.long).numpy(),
            4.0,
            torch.tensor([0, 1, 2, 3], dtype=torch.long).numpy(),
            {"num_direct": 4, "num_projected": 0, "num_unreachable": 0, "num_not_alive_direct": 0},
        )

    with patch.object(labeler, "extract_teacher_supervision", side_effect=fake_extract):
        out = precompute_labels_for_dataset([valid, invalid], labeler, num_workers=0, desc="test", force=True)

    assert len(out) == 1
    assert hasattr(out[0], "target_edges")
    assert torch.equal(out[0].target_edges, torch.tensor([0, 2, 4, 5], dtype=torch.long))


if __name__ == "__main__":
    tests = [
        ("validate accepts consistent spanner cycle", test_validate_teacher_labels_accepts_consistent_spanner_cycle),
        ("validate rejects mismatched target edges", test_validate_teacher_labels_rejects_mismatched_target_edges),
        ("validate rejects dead-edge teacher", test_validate_teacher_labels_rejects_dead_edge_teacher),
        ("extract_teacher_supervision uses alive subgraph and remaps eids", test_extract_teacher_supervision_uses_alive_subgraph_and_remaps_eids),
        ("extract_teacher_supervision rejects infeasible alive graph", test_extract_teacher_supervision_rejects_infeasible_alive_graph),
        ("ensure_dataset_labels reloads original dataset when fast metadata is missing", test_ensure_dataset_labels_reloads_original_dataset_when_fast_missing_alive_metadata),
        ("precompute_labels drops infeasible samples", test_precompute_labels_drops_infeasible_samples),
    ]

    for name, fn in tests:
        print(f"Test: {name}...")
        fn()
        print("  PASS\n")

    print("All teacher_label_validation tests passed!")
