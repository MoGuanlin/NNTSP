# src/cli/benchmark_config_builder.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

from src.cli.guided_lkh_args import GUIDED_LKH_UNSET, guided_lkh_config_from_args
from src.models.lkh_decode import GuidedLKHConfig


BenchmarkName = Literal["synthetic", "tsplib", "onepass", "twopass_timing", "training_cost"]


def resolve_synthetic_data_path(*, synthetic_n: int, synthetic_data_root: str) -> str:
    return str(Path(synthetic_data_root) / f"N{int(synthetic_n)}" / "test_r_light_pyramid.pt")


def resolve_synthetic_raw_data_path(*, synthetic_n: int, synthetic_data_root: str) -> str:
    return str(Path(synthetic_data_root) / f"N{int(synthetic_n)}" / "test.pt")


@dataclass(frozen=True)
class SharedEvalConfig:
    ckpt: str
    r: int
    device: str
    lkh_exe: str
    num_workers: int


@dataclass(frozen=True)
class GuidedLKHCLIConfig:
    config: GuidedLKHConfig

    def to_argv(self) -> list[str]:
        return [
            "--guided_top_k",
            str(int(self.config.top_k)),
            "--guided_logit_scale",
            str(float(self.config.logit_scale)),
            "--guided_subgradient",
            str(bool(self.config.subgradient)),
            "--guided_max_candidates",
            str(GUIDED_LKH_UNSET if self.config.max_candidates is None else int(self.config.max_candidates)),
            "--guided_max_trials",
            str(GUIDED_LKH_UNSET if self.config.max_trials is None else int(self.config.max_trials)),
            "--guided_use_initial_tour",
            str(bool(self.config.use_initial_tour)),
        ]


@dataclass(frozen=True)
class ExactDecodeConfig:
    time_limit: float
    length_weight: float

    def to_argv(self) -> list[str]:
        return [
            "--exact_time_limit",
            str(float(self.time_limit)),
            "--exact_length_weight",
            str(float(self.length_weight)),
        ]


@dataclass(frozen=True)
class DatasetSliceConfig:
    data_pt: str
    sample_idx: int
    sample_idx_end: int | None

    def to_argv(self) -> list[str]:
        argv = [
            "--data_pt",
            str(self.data_pt),
            "--sample_idx",
            str(int(self.sample_idx)),
        ]
        if self.sample_idx_end is not None:
            argv += ["--sample_idx_end", str(int(self.sample_idx_end))]
        return argv


@dataclass(frozen=True)
class SyntheticEvalConfig:
    common: SharedEvalConfig
    dataset: DatasetSliceConfig
    guided_lkh: GuidedLKHCLIConfig
    exact_decode: ExactDecodeConfig
    output_dir: str
    pomo_ckpt: str | None
    neurolkh_ckpt: str | None
    settings: str | None
    no_vis: bool
    use_iface_in_decode: str | None

    def to_argv(self) -> list[str]:
        argv = [
            "--ckpt",
            str(self.common.ckpt),
            *self.dataset.to_argv(),
            "--r",
            str(int(self.common.r)),
            "--device",
            str(self.common.device),
            "--lkh_exe",
            str(self.common.lkh_exe),
            *self.guided_lkh.to_argv(),
            "--num_workers",
            str(int(self.common.num_workers)),
            *self.exact_decode.to_argv(),
            "--output_dir",
            str(self.output_dir),
            "--pomo_ckpt",
            str(self.pomo_ckpt or ""),
            "--neurolkh_ckpt",
            str(self.neurolkh_ckpt or ""),
        ]
        if self.settings:
            argv += ["--settings", str(self.settings)]
        if self.no_vis:
            argv.append("--no_vis")
        if self.use_iface_in_decode is not None:
            argv += ["--use_iface_in_decode", str(self.use_iface_in_decode)]
        return argv


@dataclass(frozen=True)
class OnePassDPConfig:
    max_sigma: int
    child_catalog_cap: int
    child_catalog_widening: str
    parse_mode: str
    catalog_mate_lambda: float
    fallback_exact: bool
    leaf_workers: int
    parse_workers: int

    def to_argv(self) -> list[str]:
        return [
            "--dp_max_sigma",
            str(int(self.max_sigma)),
            "--dp_child_catalog_cap",
            str(int(self.child_catalog_cap)),
            "--dp_child_catalog_widening",
            str(self.child_catalog_widening),
            "--dp_parse_mode",
            str(self.parse_mode),
            "--dp_catalog_mate_lambda",
            str(float(self.catalog_mate_lambda)),
            "--dp_fallback_exact",
            str(bool(self.fallback_exact)),
            "--dp_leaf_workers",
            str(int(self.leaf_workers)),
            "--dp_parse_workers",
            str(int(self.parse_workers)),
        ]


@dataclass(frozen=True)
class OnePassEvalConfig:
    common: SharedEvalConfig
    dataset: DatasetSliceConfig
    guided_lkh: GuidedLKHCLIConfig
    exact_decode: ExactDecodeConfig
    dp: OnePassDPConfig
    output_dir: str
    settings: str | None

    def to_argv(self) -> list[str]:
        argv = [
            "--ckpt",
            str(self.common.ckpt),
            *self.dataset.to_argv(),
            "--r",
            str(int(self.common.r)),
            "--device",
            str(self.common.device),
            "--lkh_exe",
            str(self.common.lkh_exe),
            *self.guided_lkh.to_argv(),
            "--num_workers",
            str(int(self.common.num_workers)),
            *self.exact_decode.to_argv(),
            "--output_dir",
            str(self.output_dir),
            *self.dp.to_argv(),
        ]
        if self.settings:
            argv += ["--settings", str(self.settings)]
        return argv


@dataclass(frozen=True)
class SpannerPipelineConfig:
    spanner_mode: str
    theta_k: int
    patching_mode: str

    def to_argv(self) -> list[str]:
        return [
            "--spanner_mode",
            str(self.spanner_mode),
            "--theta_k",
            str(int(self.theta_k)),
            "--patching_mode",
            str(self.patching_mode),
        ]


@dataclass(frozen=True)
class TwoPassTimingConfig:
    common: SharedEvalConfig
    dataset: DatasetSliceConfig
    guided_lkh: GuidedLKHCLIConfig
    spanner: SpannerPipelineConfig
    output_dir: str
    run_tag: str | None
    use_iface_in_decode: str | None

    def to_argv(self) -> list[str]:
        argv = [
            "--ckpt",
            str(self.common.ckpt),
            *self.dataset.to_argv(),
            "--r",
            str(int(self.common.r)),
            "--device",
            str(self.common.device),
            "--lkh_exe",
            str(self.common.lkh_exe),
            *self.guided_lkh.to_argv(),
            "--num_workers",
            str(int(self.common.num_workers)),
            *self.spanner.to_argv(),
            "--output_dir",
            str(self.output_dir),
        ]
        if self.run_tag:
            argv += ["--run_tag", str(self.run_tag)]
        if self.use_iface_in_decode is not None:
            argv += ["--use_iface_in_decode", str(self.use_iface_in_decode)]
        return argv


@dataclass(frozen=True)
class TrainingCostConfig:
    ckpt: str
    lkh_exe: str
    output_dir: str
    log_dir: str = "checkpoints"
    teacher_data_pt: str | None = None
    run_tag: str | None = None
    teacher_sample_idx: int = 0
    teacher_sample_idx_end: int | None = None
    teacher_num_workers: int | None = None
    teacher_lkh_runs_override: int | None = None
    teacher_lkh_timeout_override: float | None = None
    num_gpus_override: int | None = None
    skip_teacher_timing: bool = False
    skip_curve_plot: bool = False

    def to_argv(self) -> list[str]:
        argv = [
            "--ckpt",
            str(self.ckpt),
            "--log_dir",
            str(self.log_dir),
            "--lkh_exe",
            str(self.lkh_exe),
            "--output_dir",
            str(self.output_dir),
            "--teacher_sample_idx",
            str(int(self.teacher_sample_idx)),
        ]
        if self.teacher_data_pt:
            argv += ["--teacher_data_pt", str(self.teacher_data_pt)]
        if self.run_tag:
            argv += ["--run_tag", str(self.run_tag)]
        if self.teacher_sample_idx_end is not None:
            argv += ["--teacher_sample_idx_end", str(int(self.teacher_sample_idx_end))]
        if self.teacher_num_workers is not None:
            argv += ["--teacher_num_workers", str(int(self.teacher_num_workers))]
        if self.teacher_lkh_runs_override is not None:
            argv += ["--teacher_lkh_runs", str(int(self.teacher_lkh_runs_override))]
        if self.teacher_lkh_timeout_override is not None:
            argv += ["--teacher_lkh_timeout", str(float(self.teacher_lkh_timeout_override))]
        if self.num_gpus_override is not None:
            argv += ["--num_gpus_override", str(int(self.num_gpus_override))]
        if self.skip_teacher_timing:
            argv.append("--skip_teacher_timing")
        if self.skip_curve_plot:
            argv.append("--skip_curve_plot")
        return argv


@dataclass(frozen=True)
class TSPLIBSelectionConfig:
    tsplib_dir: str
    num_instances: int
    instance_preset: str | None
    instances: str | None

    def to_argv(self) -> list[str]:
        argv = [
            "--tsplib_dir",
            str(self.tsplib_dir),
            "--num_instances",
            str(int(self.num_instances)),
        ]
        if self.instance_preset:
            argv += ["--instance_preset", str(self.instance_preset)]
        if self.instances:
            argv += ["--instances", str(self.instances)]
        return argv


@dataclass(frozen=True)
class TSPLIBEvalConfig:
    common: SharedEvalConfig
    guided_lkh: GuidedLKHCLIConfig
    exact_decode: ExactDecodeConfig
    spanner: SpannerPipelineConfig
    selection: TSPLIBSelectionConfig
    save_dir: str
    pomo_ckpt: str | None
    neurolkh_ckpt: str | None
    settings: str | None
    run_tag: str | None

    def to_argv(self) -> list[str]:
        argv = [
            "--ckpt",
            str(self.common.ckpt),
            *self.selection.to_argv(),
            "--device",
            str(self.common.device),
            "--r",
            str(int(self.common.r)),
            "--lkh_exe",
            str(self.common.lkh_exe),
            *self.guided_lkh.to_argv(),
            "--num_workers",
            str(int(self.common.num_workers)),
            *self.spanner.to_argv(),
            *self.exact_decode.to_argv(),
            "--save_dir",
            str(self.save_dir),
            "--pomo_ckpt",
            str(self.pomo_ckpt or ""),
            "--neurolkh_ckpt",
            str(self.neurolkh_ckpt or ""),
        ]
        if self.settings:
            argv += ["--settings", str(self.settings)]
        if self.run_tag:
            argv += ["--run_tag", str(self.run_tag)]
        return argv


@dataclass(frozen=True)
class ListSettingsRequest:
    benchmark: BenchmarkName


@dataclass(frozen=True)
class ListTSPLIBPresetsRequest:
    pass


BenchmarkConfig = Union[
    SyntheticEvalConfig,
    OnePassEvalConfig,
    TwoPassTimingConfig,
    TrainingCostConfig,
    TSPLIBEvalConfig,
]
EvaluateRequest = Union[ListSettingsRequest, ListTSPLIBPresetsRequest, BenchmarkConfig]


def build_evaluate_request(args: Namespace) -> EvaluateRequest:
    benchmark = str(args.benchmark)

    if bool(args.list_tsplib_presets):
        return ListTSPLIBPresetsRequest()

    if bool(args.list_settings):
        return ListSettingsRequest(benchmark=benchmark)  # type: ignore[arg-type]

    ckpt = _require_ckpt(args)
    if benchmark == "synthetic":
        return _build_synthetic_config(args, ckpt)
    if benchmark == "onepass":
        return _build_onepass_config(args, ckpt)
    if benchmark == "twopass_timing":
        return _build_twopass_timing_config(args, ckpt)
    if benchmark == "training_cost":
        return _build_training_cost_config(args, ckpt)
    if benchmark == "tsplib":
        return _build_tsplib_config(args, ckpt)
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def _require_ckpt(args: Namespace) -> str:
    ckpt = getattr(args, "ckpt", None)
    if not ckpt:
        raise ValueError("--ckpt is required unless a list command is used")
    return str(ckpt)


def _shared_eval_config(args: Namespace, *, ckpt: str) -> SharedEvalConfig:
    return SharedEvalConfig(
        ckpt=str(ckpt),
        r=int(args.r),
        device=str(args.device),
        lkh_exe=str(args.lkh_exe),
        num_workers=int(args.num_workers),
    )


def _guided_lkh_cli_config(args: Namespace) -> GuidedLKHCLIConfig:
    return GuidedLKHCLIConfig(config=guided_lkh_config_from_args(args))


def _exact_decode_config(args: Namespace) -> ExactDecodeConfig:
    return ExactDecodeConfig(
        time_limit=float(args.exact_time_limit),
        length_weight=float(args.exact_length_weight),
    )


def _spanner_pipeline_config(args: Namespace) -> SpannerPipelineConfig:
    return SpannerPipelineConfig(
        spanner_mode=str(args.spanner_mode),
        theta_k=int(args.theta_k),
        patching_mode=str(args.patching_mode),
    )


def _resolve_required_dataset_path(
    *,
    benchmark: str,
    data_pt: str | None,
    synthetic_n: int | None,
    synthetic_data_root: str,
    raw: bool,
) -> str:
    if data_pt:
        return str(data_pt)
    if synthetic_n is None:
        raise ValueError(f"--synthetic_n or --data_pt is required when --benchmark {benchmark}")
    if raw:
        return resolve_synthetic_raw_data_path(
            synthetic_n=int(synthetic_n),
            synthetic_data_root=str(synthetic_data_root),
        )
    return resolve_synthetic_data_path(
        synthetic_n=int(synthetic_n),
        synthetic_data_root=str(synthetic_data_root),
    )


def _dataset_slice_config(args: Namespace, *, benchmark: str, raw: bool = False) -> DatasetSliceConfig:
    return DatasetSliceConfig(
        data_pt=_resolve_required_dataset_path(
            benchmark=benchmark,
            data_pt=args.data_pt,
            synthetic_n=args.synthetic_n,
            synthetic_data_root=str(args.synthetic_data_root),
            raw=raw,
        ),
        sample_idx=int(args.sample_idx),
        sample_idx_end=None if args.sample_idx_end is None else int(args.sample_idx_end),
    )


def _build_synthetic_config(args: Namespace, ckpt: str) -> SyntheticEvalConfig:
    return SyntheticEvalConfig(
        common=_shared_eval_config(args, ckpt=ckpt),
        dataset=_dataset_slice_config(args, benchmark="synthetic"),
        guided_lkh=_guided_lkh_cli_config(args),
        exact_decode=_exact_decode_config(args),
        output_dir=str(args.output_dir or "outputs/eval_lkh"),
        pomo_ckpt=None if args.pomo_ckpt is None else str(args.pomo_ckpt),
        neurolkh_ckpt=None if args.neurolkh_ckpt is None else str(args.neurolkh_ckpt),
        settings=None if args.settings is None else str(args.settings),
        no_vis=bool(args.no_vis),
        use_iface_in_decode=None if args.use_iface_in_decode is None else str(args.use_iface_in_decode),
    )


def _build_onepass_config(args: Namespace, ckpt: str) -> OnePassEvalConfig:
    return OnePassEvalConfig(
        common=_shared_eval_config(args, ckpt=ckpt),
        dataset=_dataset_slice_config(args, benchmark="onepass"),
        guided_lkh=_guided_lkh_cli_config(args),
        exact_decode=_exact_decode_config(args),
        dp=OnePassDPConfig(
            max_sigma=int(args.dp_max_sigma),
            child_catalog_cap=int(args.dp_child_catalog_cap),
            child_catalog_widening=str(args.dp_child_catalog_widening),
            parse_mode=str(args.dp_parse_mode),
            catalog_mate_lambda=float(args.dp_catalog_mate_lambda),
            fallback_exact=bool(args.dp_fallback_exact),
            leaf_workers=int(args.dp_leaf_workers),
            parse_workers=int(args.dp_parse_workers),
        ),
        output_dir=str(args.output_dir or "outputs/eval_onepass"),
        settings=None if args.settings is None else str(args.settings),
    )


def _build_twopass_timing_config(args: Namespace, ckpt: str) -> TwoPassTimingConfig:
    return TwoPassTimingConfig(
        common=_shared_eval_config(args, ckpt=ckpt),
        dataset=_dataset_slice_config(args, benchmark="twopass_timing", raw=True),
        guided_lkh=_guided_lkh_cli_config(args),
        spanner=_spanner_pipeline_config(args),
        output_dir=str(args.output_dir or "outputs/eval_twopass_timing"),
        run_tag=None if args.run_tag is None else str(args.run_tag),
        use_iface_in_decode=None if args.use_iface_in_decode is None else str(args.use_iface_in_decode),
    )


def _build_training_cost_config(args: Namespace, ckpt: str) -> TrainingCostConfig:
    teacher_data_pt = args.teacher_data_pt or args.data_pt
    return TrainingCostConfig(
        ckpt=str(ckpt),
        lkh_exe=str(args.lkh_exe),
        output_dir=str(args.output_dir or "outputs/eval_training_cost"),
        teacher_data_pt=None if teacher_data_pt is None else str(teacher_data_pt),
        run_tag=None if args.run_tag is None else str(args.run_tag),
        teacher_sample_idx=int(args.teacher_sample_idx),
        teacher_sample_idx_end=None if args.teacher_sample_idx_end is None else int(args.teacher_sample_idx_end),
        teacher_num_workers=None if args.teacher_num_workers is None else int(args.teacher_num_workers),
        teacher_lkh_runs_override=None if args.teacher_lkh_runs is None else int(args.teacher_lkh_runs),
        teacher_lkh_timeout_override=(
            None if args.teacher_lkh_timeout is None else float(args.teacher_lkh_timeout)
        ),
        num_gpus_override=None if args.num_gpus_override is None else int(args.num_gpus_override),
        skip_teacher_timing=bool(args.skip_teacher_timing),
        skip_curve_plot=bool(args.skip_curve_plot),
    )


def _build_tsplib_config(args: Namespace, ckpt: str) -> TSPLIBEvalConfig:
    return TSPLIBEvalConfig(
        common=_shared_eval_config(args, ckpt=ckpt),
        guided_lkh=_guided_lkh_cli_config(args),
        exact_decode=_exact_decode_config(args),
        spanner=_spanner_pipeline_config(args),
        selection=TSPLIBSelectionConfig(
            tsplib_dir=str(args.tsplib_dir),
            num_instances=int(args.num_instances),
            instance_preset=None if args.tsplib_set is None else str(args.tsplib_set),
            instances=None if args.tsplib_instances is None else str(args.tsplib_instances),
        ),
        save_dir=str(args.output_dir or "outputs/eval_tsplib"),
        pomo_ckpt=None if args.pomo_ckpt is None else str(args.pomo_ckpt),
        neurolkh_ckpt=None if args.neurolkh_ckpt is None else str(args.neurolkh_ckpt),
        settings=None if args.settings is None else str(args.settings),
        run_tag=None if args.run_tag is None else str(args.run_tag),
    )


__all__ = [
    "BenchmarkConfig",
    "DatasetSliceConfig",
    "EvaluateRequest",
    "ExactDecodeConfig",
    "GuidedLKHCLIConfig",
    "ListSettingsRequest",
    "ListTSPLIBPresetsRequest",
    "OnePassDPConfig",
    "OnePassEvalConfig",
    "SharedEvalConfig",
    "SpannerPipelineConfig",
    "SyntheticEvalConfig",
    "TSPLIBEvalConfig",
    "TSPLIBSelectionConfig",
    "TrainingCostConfig",
    "TwoPassTimingConfig",
    "build_evaluate_request",
    "resolve_synthetic_data_path",
    "resolve_synthetic_raw_data_path",
]
