from __future__ import annotations

from argparse import ArgumentParser
from types import SimpleNamespace

import torch

from src.cli.guided_lkh_args import add_guided_lkh_args, guided_lkh_config_from_args
from src.cli.teacher_lkh_args import (
    TeacherLKHConfig,
    add_teacher_lkh_args,
    build_spanner_teacher_labeler,
    teacher_lkh_config_from_args,
)
from src.cli.graph_pipeline import select_effective_edge_index
from src.cli.runtime_batch_io import deserialize_torch_payload, serialize_torch_payload
from src.cli.eval_task_factory import build_lkh_task, mask_edge_logits_with_coverage
from src.cli.train_onepass import _precompute_cache_path
from src.models import lkh_decode
from src.models.lkh_decode import GuidedLKHConfig, build_guided_candidates, solve_with_lkh_parallel
from src.utils.lkh_solver import LKHRunStatus


def test_select_effective_edge_index_prefers_alive_edge_id() -> None:
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)
    data = SimpleNamespace(
        spanner_edge_index=edge_index,
        alive_edge_id=torch.tensor([2, 0], dtype=torch.long),
        edge_alive_mask=torch.tensor([True, False, False], dtype=torch.bool),
    )

    out = select_effective_edge_index(data)

    assert torch.equal(out, edge_index[:, torch.tensor([2, 0], dtype=torch.long)])


def test_select_effective_edge_index_falls_back_to_alive_mask() -> None:
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)
    data = SimpleNamespace(
        spanner_edge_index=edge_index,
        edge_alive_mask=torch.tensor([False, True, True], dtype=torch.bool),
    )

    out = select_effective_edge_index(data)

    assert torch.equal(out, edge_index[:, torch.tensor([1, 2], dtype=torch.long)])


def test_build_guided_candidates_ignores_masked_edges_and_caps_topk() -> None:
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)
    edge_logit = torch.tensor([0.9, -1.0e9, 0.5], dtype=torch.float32)

    candidates = build_guided_candidates(
        num_nodes=3,
        edge_index=edge_index,
        edge_logit=edge_logit,
        logit_scale=1000.0,
        top_k=1,
    )

    assert candidates[0] == [(1, 0)]
    assert candidates[1] == [(0, 0)]
    assert candidates[2] == [(1, 400)]


def test_precompute_cache_path_invalidates_on_teacher_signature(tmp_path) -> None:
    train_pt = tmp_path / "train.pt"
    train_pt.write_bytes(b"dummy")

    path_a = _precompute_cache_path(
        str(train_pt),
        batch_size=8,
        r=4,
        max_used=4,
        teacher_signature="teacher_a",
    )
    path_b = _precompute_cache_path(
        str(train_pt),
        batch_size=8,
        r=4,
        max_used=4,
        teacher_signature="teacher_b",
    )
    path_a_again = _precompute_cache_path(
        str(train_pt),
        batch_size=8,
        r=4,
        max_used=4,
        teacher_signature="teacher_a",
    )

    assert path_a.parent == tmp_path
    assert path_a != path_b
    assert path_a == path_a_again


def test_runtime_batch_io_roundtrip_preserves_payload() -> None:
    payload = {"x": torch.tensor([1, 2, 3], dtype=torch.long)}

    restored = deserialize_torch_payload(serialize_torch_payload(payload))

    assert torch.equal(restored["x"], payload["x"])


def test_build_lkh_task_normalizes_cpu_payload_and_tour() -> None:
    task = build_lkh_task(
        pos=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        mode="guided",
        teacher_len=12.5,
        edge_index=torch.tensor([[0], [0]], dtype=torch.long),
        edge_logit=torch.tensor([0.3], dtype=torch.float32),
        initial_tour=(3, 1, 2),
    )

    assert task["pos"].device.type == "cpu"
    assert task["edge_index"].device.type == "cpu"
    assert task["edge_logit"].device.type == "cpu"
    assert task["initial_tour"] == [3, 1, 2]


def test_mask_edge_logits_with_coverage_pushes_uncovered_edges_down() -> None:
    edge_logit = torch.tensor([0.7, 0.2, -0.4], dtype=torch.float32)
    edge_mask = torch.tensor([True, False, True], dtype=torch.bool)

    masked = mask_edge_logits_with_coverage(edge_logit, edge_mask)

    assert torch.equal(masked[[0, 2]], edge_logit[[0, 2]])
    assert masked[1].item() == -1.0e9


def test_guided_lkh_args_keep_legacy_defaults_but_allow_explicit_overrides() -> None:
    parser = ArgumentParser()
    add_guided_lkh_args(parser)

    defaults = guided_lkh_config_from_args(parser.parse_args([]))
    assert defaults == GuidedLKHConfig()

    args = parser.parse_args(
        [
            "--guided_top_k",
            "11",
            "--guided_logit_scale",
            "250.5",
            "--guided_subgradient",
            "true",
            "--guided_max_candidates",
            "-1",
            "--guided_max_trials",
            "7",
            "--guided_use_initial_tour",
            "false",
        ]
    )
    config = guided_lkh_config_from_args(args)

    assert config == GuidedLKHConfig(
        top_k=11,
        logit_scale=250.5,
        subgradient=True,
        max_candidates=None,
        max_trials=7,
        use_initial_tour=False,
    )


def test_teacher_lkh_args_support_cli_defaults_and_checkpoint_fallbacks() -> None:
    parser = ArgumentParser()
    add_teacher_lkh_args(parser, runs_default=None, timeout_default=None)

    args = parser.parse_args([])
    config = teacher_lkh_config_from_args(args, runs_default=3, timeout_default=12.5)
    assert config == TeacherLKHConfig(runs=3, timeout=12.5)

    args = parser.parse_args(["--teacher_lkh_runs", "5", "--teacher_lkh_timeout", "0"])
    config = teacher_lkh_config_from_args(args, runs_default=3, timeout_default=12.5)
    assert config == TeacherLKHConfig(runs=5, timeout=None)


def test_build_spanner_teacher_labeler_preserves_resolved_timeout(monkeypatch) -> None:
    from src.models import labeler as labeler_module

    monkeypatch.setattr(labeler_module, "resolve_lkh_executable", lambda executable: f"/resolved/{executable}")

    labeler = build_spanner_teacher_labeler(
        lkh_exe="LKH",
        config=TeacherLKHConfig(runs=7, timeout=None),
        prefer_cpu=True,
    )

    assert labeler.lkh_exe == "/resolved/LKH"
    assert labeler.teacher_lkh_runs == 7
    assert labeler.teacher_lkh_timeout is None
    assert labeler.label_signature().startswith(labeler.teacher_label_version)


def test_solve_with_lkh_parallel_honors_explicit_guided_config(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_build_guided_candidates(*, num_nodes, edge_index, edge_logit, logit_scale, top_k):
        captured["num_nodes"] = int(num_nodes)
        captured["logit_scale"] = float(logit_scale)
        captured["top_k"] = int(top_k)
        return [[(1, 0)], [(0, 0), (2, 0)], [(1, 0)]]

    def fake_write_par(path, tsp_path, tour_path, **kwargs):
        captured["subgradient"] = kwargs["subgradient"]
        captured["max_candidates"] = kwargs["max_candidates"]
        captured["max_trials"] = kwargs["max_trials"]
        captured["initial_tour_path"] = kwargs["initial_tour_path"]

    monkeypatch.setattr(lkh_decode, "build_guided_candidates", fake_build_guided_candidates)
    monkeypatch.setattr(lkh_decode, "write_par", fake_write_par)
    monkeypatch.setattr(
        lkh_decode,
        "run_lkh_with_status",
        lambda *args, **kwargs: LKHRunStatus(ok=True, timeout_hit=False, elapsed_sec=0.01),
    )
    monkeypatch.setattr(lkh_decode, "parse_tour", lambda path: [0, 1, 2])

    task = build_lkh_task(
        pos=torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
        mode="guided",
        teacher_len=4.0,
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_logit=torch.tensor([0.8, 0.4], dtype=torch.float32),
        initial_tour=[0, 2, 1],
    )
    config = GuidedLKHConfig(
        top_k=7,
        logit_scale=321.0,
        subgradient=True,
        max_candidates=None,
        max_trials=9,
        use_initial_tour=False,
    )
    result, teacher_len = solve_with_lkh_parallel(
        [task],
        guided_config=config,
        num_workers=1,
    )[0]

    assert teacher_len == 4.0
    assert result.feasible
    assert captured["num_nodes"] == 3
    assert captured["top_k"] == 7
    assert captured["logit_scale"] == 321.0
    assert captured["subgradient"] is True
    assert captured["max_candidates"] is None
    assert captured["max_trials"] == 9
    assert captured["initial_tour_path"] is None
