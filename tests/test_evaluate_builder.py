from __future__ import annotations

from src.cli import evaluate
from src.cli.benchmark_config_builder import (
    ListSettingsRequest,
    SyntheticEvalConfig,
    TrainingCostConfig,
    TwoPassTimingConfig,
    build_evaluate_request,
)
from src.cli.eval_profiles import AVAILABLE_SETTINGS, SETTING_GROUPS


def _parse_args(argv: list[str]):
    return evaluate.build_parser().parse_args(argv)


def test_build_synthetic_request_resolves_processed_dataset_path() -> None:
    args = _parse_args(
        [
            "--benchmark",
            "synthetic",
            "--ckpt",
            "model.pt",
            "--synthetic_n",
            "500",
            "--synthetic_data_root",
            "data/std",
        ]
    )

    request = build_evaluate_request(args)

    assert isinstance(request, SyntheticEvalConfig)
    assert request.dataset.data_pt == "data/std/N500/test_r_light_pyramid.pt"
    argv = request.to_argv()
    assert "--guided_max_candidates" in argv
    assert "--guided_max_trials" in argv
    assert "--output_dir" in argv


def test_build_twopass_timing_request_resolves_raw_dataset_path() -> None:
    args = _parse_args(
        [
            "--benchmark",
            "twopass_timing",
            "--ckpt",
            "model.pt",
            "--synthetic_n",
            "2000",
            "--synthetic_data_root",
            "data/std",
        ]
    )

    request = build_evaluate_request(args)

    assert isinstance(request, TwoPassTimingConfig)
    assert request.dataset.data_pt == "data/std/N2000/test.pt"


def test_build_onepass_request_requires_dataset_or_synthetic_n() -> None:
    args = _parse_args(
        [
            "--benchmark",
            "onepass",
            "--ckpt",
            "model.pt",
        ]
    )

    try:
        build_evaluate_request(args)
    except ValueError as exc:
        assert "--synthetic_n or --data_pt is required when --benchmark onepass" == str(exc)
    else:
        raise AssertionError("expected build_evaluate_request to reject missing onepass dataset")


def test_build_training_cost_request_keeps_teacher_overrides_optional() -> None:
    args = _parse_args(
        [
            "--benchmark",
            "training_cost",
            "--ckpt",
            "model.pt",
            "--data_pt",
            "train.pt",
        ]
    )

    request = build_evaluate_request(args)

    assert isinstance(request, TrainingCostConfig)
    assert request.teacher_data_pt == "train.pt"
    assert request.teacher_lkh_runs_override is None
    assert request.teacher_lkh_timeout_override is None


def test_dispatch_list_settings_prints_onepass_settings(capsys) -> None:
    evaluate.dispatch_evaluate_request(ListSettingsRequest(benchmark="onepass"))

    out = capsys.readouterr().out.strip()
    assert out == evaluate.onepass_eval.describe_settings(
        available=AVAILABLE_SETTINGS,
        groups=SETTING_GROUPS,
    )


def test_dispatch_list_settings_reports_no_presets_for_twopass_timing(capsys) -> None:
    evaluate.dispatch_evaluate_request(ListSettingsRequest(benchmark="twopass_timing"))

    out = capsys.readouterr().out.strip()
    assert out == "twopass_timing benchmark has no preset settings."
