# src/cli/evaluate_tsplib.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from src.cli.common import log_progress, move_data_tensors_to_device as to_device, resolve_device
from src.cli.guided_lkh_args import add_guided_lkh_args, guided_lkh_config_from_args
from src.cli.eval_profiles import (
    TSPLIB_AVAILABLE_SETTINGS as AVAILABLE_SETTINGS,
    TSPLIB_DEFAULT_SETTINGS as DEFAULT_SETTINGS,
    TSPLIB_SETTING_ALIASES as SETTING_ALIASES,
    TSPLIB_SETTING_GROUPS as SETTING_GROUPS,
)
from src.cli.eval_task_factory import (
    build_lkh_task,
    mask_edge_logits_with_coverage,
    prepare_decode_inputs,
)
from src.cli.eval_settings import describe_settings, resolve_eval_settings
from src.cli.graph_pipeline import (
    build_spanner_builder,
    preprocess_points_to_hierarchy,
    select_effective_edge_index,
)
from src.cli.model_factory import load_twopass_eval_models
from src.graph.build_raw_pyramid import RawPyramidBuilder
from src.models.bottom_up_runner import BottomUpTreeRunner
from src.models.decode_backend import decode_tour
from src.models.edge_aggregation import aggregate_logits_to_edges
from src.models.lkh_decode import TSPLIB_PAPER_RESULTS, solve_with_lkh_parallel
from src.models.node_token_packer import NodeTokenPacker
from src.models.top_down_runner import TopDownTreeRunner

TSPLIB_INSTANCE_PRESETS = {
    "largest5": {"mode": "largest", "count": 5},
    "largest10": {"mode": "largest", "count": 10},
    "largest20": {"mode": "largest", "count": 20},
    "n100_500": {"mode": "range", "min_n": 100, "max_n": 500},
    "small100_500": {"mode": "range", "min_n": 100, "max_n": 500},
    "paper": {"mode": "names", "names": tuple(TSPLIB_PAPER_RESULTS.keys())},
    "all": {"mode": "all"},
}

SETTING_COLUMNS: Dict[str, List[tuple[str, str, int]]] = {
    "greedy": [("obj", "Greedy Obj", 12), ("gap", "Greedy Gap", 10), ("time", "Time", 7)],
    "exact": [("obj", "Exact Obj", 12), ("gap", "Exact Gap", 10), ("time", "Time", 7)],
    "spanner_uniform_lkh": [("obj", "Spanner Obj", 12), ("gap", "Spanner Gap", 11), ("time", "Time", 7)],
    "guided_lkh": [("obj", "Guided Obj", 12), ("gap", "Guided Gap", 10), ("time", "Time", 7)],
    "pomo": [("obj", "POMO Obj", 10), ("gap", "POMO Gap", 10), ("time", "Time", 7)],
    "neurolkh": [("obj", "NeuroLKH Obj", 12), ("gap", "NeuroLKH Gap", 12), ("time", "Time", 7)],
    "paper_lkh": [("obj", "Paper Obj", 10), ("time", "Paper Time", 10)],
}

def parse_tsp_file(path: str) -> np.ndarray:
    """Parse 2D coordinates from TSPLIB node/display sections.

    Supports both:
      - NODE_COORD_SECTION
      - DISPLAY_DATA_SECTION

    Some TSPLIB instances use EXPLICIT edge weights but still provide
    DISPLAY_DATA_SECTION for visualization; those coordinates are sufficient for
    this Euclidean/spanner-based pipeline.
    """
    header_re = re.compile(r"^([A-Z_]+)\s*:\s*(.*)$")
    coords = []
    edge_weight_type = None
    display_data_type = None
    with open(path, "r") as f:
        in_section = False
        for line in f:
            line = line.strip()
            if not line:
                continue

            upper = line.upper()
            header_match = header_re.match(upper)
            if (not in_section) and header_match:
                key = header_match.group(1).strip()
                value = header_match.group(2).strip()
                if key == "EDGE_WEIGHT_TYPE":
                    edge_weight_type = value
                elif key == "DISPLAY_DATA_TYPE":
                    display_data_type = value

            if upper == "NODE_COORD_SECTION" or upper == "DISPLAY_DATA_SECTION":
                in_section = True
                continue
            if upper == "EOF" or (in_section and line[0].isalpha()):
                break
            if in_section:
                parts = line.split()
                if len(parts) >= 3:
                    coords.append([float(parts[1]), float(parts[2])])

    coords_np = np.array(coords, dtype=np.float32)
    if coords_np.ndim != 2 or coords_np.shape[0] == 0 or coords_np.shape[1] != 2:
        raise ValueError(
            f"TSPLIB instance has no usable 2D coordinates: {path} "
            f"(EDGE_WEIGHT_TYPE={edge_weight_type}, DISPLAY_DATA_TYPE={display_data_type}). "
            "Expected NODE_COORD_SECTION or DISPLAY_DATA_SECTION."
        )
    return coords_np

def make_empty_metrics(setting: str) -> Dict[str, str]:
    return {field: "N/A" for field, _, _ in SETTING_COLUMNS[setting]}


def format_result_triplet(*, length: float, duration: float, teacher_len: float | None) -> Dict[str, str]:
    out = {
        "obj": f"{length:.0f}",
        "gap": "N/A",
        "time": f"{duration:.1f}s",
    }
    if teacher_len:
        out["gap"] = f"{(length / teacher_len - 1.0) * 100:.2f}%"
    return out


def build_table_header(selected_settings: List[str]) -> str:
    parts = [f"{'Instance':<10} | {'N':<6}"]
    for setting in selected_settings:
        cols = [f"{label:<{width}}" for _, label, width in SETTING_COLUMNS[setting]]
        parts.append(" | ".join(cols))
    return " | ".join(parts)


def build_table_row(*, name: str, n: int, selected_settings: List[str], metrics: Dict[str, Dict[str, str]]) -> str:
    parts = [f"{name:<10} | {n:<6}"]
    for setting in selected_settings:
        cols = []
        for field, _, width in SETTING_COLUMNS[setting]:
            cols.append(f"{metrics[setting][field]:<{width}}")
        parts.append(" | ".join(cols))
    return " | ".join(parts)


def sanitize_tag(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-_.")
    return cleaned or "run"


def normalize_tsplib_instance_name(name: str) -> str:
    text = str(name).strip()
    if text.lower().endswith(".tsp"):
        text = text[:-4]
    return text


def parse_instance_name_list(text: str | None) -> List[str]:
    if not text:
        return []
    names = []
    for part in re.split(r"[,\s]+", str(text).strip()):
        name = normalize_tsplib_instance_name(part)
        if name:
            names.append(name)
    return names


def describe_instance_presets() -> str:
    return "instance_presets=all,largest5,largest10,largest20,n100_500,small100_500,paper"


def build_partial_run_tag(*, ckpt_path: str, selection_tag: str, r: int, selected_settings: List[str]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ckpt_obj = Path(ckpt_path)
    ckpt_tag = ckpt_obj.parent.name if ckpt_obj.parent.name else ckpt_obj.stem
    settings_tag = "-".join(selected_settings)
    return sanitize_tag(f"{timestamp}_{ckpt_tag}_{selection_tag}_r{r}_{settings_tag}")


def collect_tsplib_files(tsplib_path: Path) -> List[tuple[int, Path]]:
    all_tsp_files = []
    for f in tsplib_path.glob("*.tsp"):
        match = re.search(r"(\d+)\.tsp$", f.name)
        if match:
            all_tsp_files.append((int(match.group(1)), f))
    all_tsp_files.sort(key=lambda x: (x[0], x[1].stem))
    return all_tsp_files


def select_tsplib_files(
    *,
    all_tsp_files: List[tuple[int, Path]],
    instance_preset: str | None,
    instance_names: str | None,
    num_instances: int,
) -> tuple[List[tuple[int, Path]], str, str]:
    by_name = {f_path.stem: (n, f_path) for n, f_path in all_tsp_files}
    explicit_names = parse_instance_name_list(instance_names)
    if explicit_names:
        missing = [name for name in explicit_names if name not in by_name]
        if missing:
            available_preview = ", ".join(sorted(by_name.keys())[:10])
            raise ValueError(
                f"Unknown TSPLIB instances: {', '.join(missing)}. "
                f"Available examples: {available_preview}"
            )
        selected = [by_name[name] for name in explicit_names]
        return selected, f"explicit instances ({len(selected)}): {', '.join(explicit_names)}", f"names{len(selected)}"

    preset = str(instance_preset or "").strip().lower()
    if not preset:
        largest_count = int(num_instances)
        selected = sorted(all_tsp_files, key=lambda x: x[0], reverse=True)[:largest_count]
        selected.sort(key=lambda x: (x[0], x[1].stem))
        return selected, f"top {len(selected)} largest instances", f"largest{largest_count}"

    preset_match = re.fullmatch(r"largest[:_-]?(\d+)", preset)
    if preset == "largest":
        preset_match = re.fullmatch(r"largest", preset)
    if preset_match:
        largest_count = int(preset_match.group(1)) if preset_match.lastindex else int(num_instances)
        selected = sorted(all_tsp_files, key=lambda x: x[0], reverse=True)[:largest_count]
        selected.sort(key=lambda x: (x[0], x[1].stem))
        return selected, f"preset {preset} -> top {len(selected)} largest instances", f"largest{largest_count}"

    preset_cfg = TSPLIB_INSTANCE_PRESETS.get(preset)
    if preset_cfg is None:
        raise ValueError(
            f"Unknown TSPLIB instance preset: {instance_preset}. "
            f"Available presets: {describe_instance_presets()}"
        )

    mode = preset_cfg["mode"]
    if mode == "all":
        return list(all_tsp_files), f"preset {preset} -> all {len(all_tsp_files)} instances", preset
    if mode == "largest":
        largest_count = int(preset_cfg["count"])
        selected = sorted(all_tsp_files, key=lambda x: x[0], reverse=True)[:largest_count]
        selected.sort(key=lambda x: (x[0], x[1].stem))
        return selected, f"preset {preset} -> top {len(selected)} largest instances", preset
    if mode == "range":
        min_n = int(preset_cfg["min_n"])
        max_n = int(preset_cfg["max_n"])
        selected = [item for item in all_tsp_files if min_n <= int(item[0]) <= max_n]
        return selected, f"preset {preset} -> {min_n} <= N <= {max_n} ({len(selected)} instances)", preset
    if mode == "names":
        names = [normalize_tsplib_instance_name(name) for name in preset_cfg["names"]]
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ValueError(
                f"TSPLIB preset {preset} references missing instances: {', '.join(missing)}"
            )
        selected = [by_name[name] for name in names]
        return selected, f"preset {preset} -> {', '.join(names)}", preset
    raise ValueError(f"Unsupported TSPLIB preset mode: {mode}")


def build_result_fieldnames(selected_settings: List[str]) -> List[str]:
    fieldnames = ["instance_index", "instance", "n", "elapsed_sec", "completed_at_utc"]
    for setting in selected_settings:
        for field, _, _ in SETTING_COLUMNS[setting]:
            fieldnames.append(f"{setting}_{field}")
    return fieldnames


def flatten_result_row(
    *,
    inst_idx: int,
    name: str,
    n: int,
    elapsed_sec: float,
    completed_at_utc: str,
    selected_settings: List[str],
    metrics: Dict[str, Dict[str, str]],
) -> Dict[str, str]:
    row = {
        "instance_index": str(inst_idx),
        "instance": name,
        "n": str(n),
        "elapsed_sec": f"{elapsed_sec:.3f}",
        "completed_at_utc": completed_at_utc,
    }
    for setting in selected_settings:
        for field, _, _ in SETTING_COLUMNS[setting]:
            row[f"{setting}_{field}"] = metrics[setting][field]
    return row


def write_summary_snapshot(*, summary_path: Path, metadata: Dict[str, Any], all_results: List[Dict[str, Any]]) -> None:
    payload = {
        "meta": metadata,
        "completed_instances": len(all_results),
        "results": all_results,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def initialize_partial_result_files(
    *,
    save_dir: Path,
    run_tag: str,
    fieldnames: List[str],
    metadata: Dict[str, Any],
) -> Dict[str, Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / f"{run_tag}.csv"
    jsonl_path = save_dir / f"{run_tag}.jsonl"
    summary_path = save_dir / f"{run_tag}_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    jsonl_path.write_text("", encoding="utf-8")
    write_summary_snapshot(summary_path=summary_path, metadata=metadata, all_results=[])
    return {
        "csv": csv_path,
        "jsonl": jsonl_path,
        "summary": summary_path,
    }


def append_partial_result(
    *,
    paths: Dict[str, Path],
    fieldnames: List[str],
    inst_idx: int,
    name: str,
    n: int,
    elapsed_sec: float,
    completed_at_utc: str,
    selected_settings: List[str],
    metrics: Dict[str, Dict[str, str]],
    metadata: Dict[str, Any],
    all_results: List[Dict[str, Any]],
) -> None:
    row = flatten_result_row(
        inst_idx=inst_idx,
        name=name,
        n=n,
        elapsed_sec=elapsed_sec,
        completed_at_utc=completed_at_utc,
        selected_settings=selected_settings,
        metrics=metrics,
    )
    with paths["csv"].open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)
    record = {
        "instance_index": inst_idx,
        "instance": name,
        "n": n,
        "elapsed_sec": round(elapsed_sec, 3),
        "completed_at_utc": completed_at_utc,
        "metrics": metrics,
    }
    with paths["jsonl"].open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_summary_snapshot(summary_path=paths["summary"], metadata=metadata, all_results=all_results)


def load_pomo(device: torch.device, ckpt_path: str):
    from src_baselines.pomo.POMO.TSPEnv import TSPEnv
    from src_baselines.pomo.POMO.TSPModel import TSPModel

    class CustomTSPEnv(TSPEnv):
        def load_problems(self, batch_size, aug_factor=1):
            pass

        def set_problems(self, problems):
            self.batch_size = problems.size(0)
            self.problem_size = problems.size(1)
            self.problems = problems
            self.device = problems.device
            self.BATCH_IDX = torch.arange(self.batch_size, device=self.device)[:, None].expand(self.batch_size, self.pomo_size)
            self.POMO_IDX = torch.arange(self.pomo_size, device=self.device)[None, :].expand(self.batch_size, self.pomo_size)

    model_params = {
        "embedding_dim": 128,
        "sqrt_embedding_dim": 128 ** (1 / 2),
        "encoder_layer_num": 6,
        "qkv_dim": 16,
        "head_num": 8,
        "logit_clipping": 10,
        "ff_hidden_dim": 512,
        "eval_type": "argmax",
    }
    model = TSPModel(**model_params).to(device)
    pomo_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(pomo_ckpt["model_state_dict"])
    model.eval()
    env = CustomTSPEnv(problem_size=100, pomo_size=100)
    env.device = device
    return model, env


def load_neurolkh(device: torch.device, ckpt_path: str):
    from src_baselines.neurolkh.net.sgcn_model import SparseGCNModel

    model = SparseGCNModel(problem="tsp").to(device)
    nlkh_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(nlkh_ckpt["model"])
    model.eval()
    return model


def main(argv: List[str] | None = None):
    from src.utils.lkh_solver import default_lkh_executable

    parser = argparse.ArgumentParser()
    default_lkh = default_lkh_executable()
    parser.add_argument("--ckpt", type=str, required=False, help="path to model checkpoint")
    parser.add_argument("--tsplib_dir", type=str, default="benchmarks/tsplib", help="directory with .tsp files")
    parser.add_argument("--num_instances", type=int, default=10, help="number of largest instances to test")
    parser.add_argument(
        "--instance_preset",
        type=str,
        default=None,
        help="TSPLIB instance preset, for example largest10, paper, all, or largest:25; "
             + describe_instance_presets(),
    )
    parser.add_argument(
        "--instances",
        type=str,
        default=None,
        help="comma-separated explicit TSPLIB instance names, for example rl5915,rl5934,pla7397",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--lkh_exe", type=str, default=default_lkh, help="path to LKH executable")
    add_guided_lkh_args(parser)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--spanner_mode", type=str, default="delaunay", choices=("delaunay", "theta"))
    parser.add_argument("--theta_k", type=int, default=14, help="theta spanner cone count when --spanner_mode theta")
    parser.add_argument("--patching_mode", type=str, default="prune", choices=("prune", "arora"))
    parser.add_argument("--pomo_ckpt", type=str, default=None, help="path to POMO checkpoint")
    parser.add_argument("--neurolkh_ckpt", type=str, default=None, help="path to NeuroLKH checkpoint")
    parser.add_argument("--exact_time_limit", type=float, default=30.0, help="time limit in seconds for exact sparse decoding")
    parser.add_argument("--exact_length_weight", type=float, default=0.0, help="optional Euclidean tie-break weight for exact sparse decoding")
    parser.add_argument("--save_dir", type=str, default="outputs/eval_tsplib", help="directory for per-instance appended TSPLIB results")
    parser.add_argument("--run_tag", type=str, default=None, help="optional run tag for saved TSPLIB result files")
    parser.add_argument(
        "--settings",
        type=str,
        default=None,
        help="comma-separated settings/groups to evaluate; "
             + describe_settings(available=AVAILABLE_SETTINGS, groups=SETTING_GROUPS),
    )
    parser.add_argument("--list_settings", action="store_true", help="print available settings/groups and exit")
    parser.add_argument("--list_instance_presets", action="store_true", help="print TSPLIB instance presets and exit")
    args = parser.parse_args(argv)

    if args.list_settings:
        print(describe_settings(available=AVAILABLE_SETTINGS, groups=SETTING_GROUPS))
        return
    if args.list_instance_presets:
        print(describe_instance_presets())
        return
    if not args.ckpt:
        parser.error("--ckpt is required unless --list_settings or --list_instance_presets is used")

    selected_settings = resolve_eval_settings(
        requested=args.settings,
        available=AVAILABLE_SETTINGS,
        default=DEFAULT_SETTINGS,
        aliases=SETTING_ALIASES,
        groups=SETTING_GROUPS,
    )
    print(f"[eval] selected settings: {', '.join(selected_settings)}")

    device = resolve_device(str(args.device))
    guided_config = guided_lkh_config_from_args(args)
    print(f"[env] device={device}")

    print(f"[ckpt] loading from {args.ckpt}")
    model_bundle = load_twopass_eval_models(
        ckpt_path=args.ckpt,
        device=device,
        r=int(args.r),
    )
    d_model = model_bundle.d_model
    state_mode = model_bundle.state_mode
    matching_max_used = model_bundle.matching_max_used
    leaf_encoder = model_bundle.leaf_encoder
    merge_encoder = model_bundle.merge_encoder
    decoder = model_bundle.decoder
    print(f"[model] detected d_model={d_model}")

    pomo_model = None
    pomo_env = None
    if "pomo" in selected_settings:
        if args.pomo_ckpt and os.path.exists(args.pomo_ckpt):
            print(f"[baseline] loading POMO from {args.pomo_ckpt}")
            pomo_model, pomo_env = load_pomo(device, args.pomo_ckpt)
            print("[baseline] POMO loaded.")
        else:
            print("[warn] POMO selected but --pomo_ckpt is missing or not found; POMO results will be N/A.")

    nlkh_model = None
    if "neurolkh" in selected_settings:
        if args.neurolkh_ckpt and os.path.exists(args.neurolkh_ckpt):
            print(f"[baseline] loading NeuroLKH from {args.neurolkh_ckpt}")
            nlkh_model = load_neurolkh(device, args.neurolkh_ckpt)
            print("[baseline] NeuroLKH loaded.")
        else:
            print("[warn] NeuroLKH selected but --neurolkh_ckpt is missing or not found; NeuroLKH results will be N/A.")

    packer = NodeTokenPacker(
        r=int(args.r),
        state_mode=state_mode,
        matching_max_used=matching_max_used,
    )
    bu_runner = BottomUpTreeRunner()
    td_runner = TopDownTreeRunner()
    spanner_builder = build_spanner_builder(
        spanner_mode=str(args.spanner_mode),
        theta_k=int(args.theta_k),
    )
    raw_builder = RawPyramidBuilder(max_points_per_leaf=4, max_depth=20)

    tsplib_path = Path(args.tsplib_dir)
    all_tsp_files = collect_tsplib_files(tsplib_path)
    tsp_files, selection_desc, selection_tag = select_tsplib_files(
        all_tsp_files=all_tsp_files,
        instance_preset=args.instance_preset,
        instance_names=args.instances,
        num_instances=int(args.num_instances),
    )
    print(f"[data] Selected {selection_desc} from {len(all_tsp_files)} total.")

    if args.run_tag:
        run_tag = sanitize_tag(str(args.run_tag))
    else:
        run_tag = build_partial_run_tag(
            ckpt_path=args.ckpt,
            selection_tag=selection_tag,
            r=int(args.r),
            selected_settings=selected_settings,
        )
    fieldnames = build_result_fieldnames(selected_settings)
    skipped_instances: List[Dict[str, Any]] = []
    save_metadata = {
        "run_tag": run_tag,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ckpt": str(Path(args.ckpt).resolve()),
        "tsplib_dir": str(Path(args.tsplib_dir).resolve()),
        "num_instances_requested": int(args.num_instances),
        "num_instances_selected": len(tsp_files),
        "num_instances_skipped": 0,
        "skipped_instances": skipped_instances,
        "instance_preset": args.instance_preset,
        "instances": parse_instance_name_list(args.instances),
        "selection_desc": selection_desc,
        "r": int(args.r),
        "device": str(device),
        "spanner_mode": str(args.spanner_mode),
        "theta_k": int(args.theta_k),
        "patching_mode": str(args.patching_mode),
        "guided_lkh": {
            "top_k": int(guided_config.top_k),
            "logit_scale": float(guided_config.logit_scale),
            "subgradient": bool(guided_config.subgradient),
            "max_candidates": guided_config.max_candidates,
            "max_trials": guided_config.max_trials,
            "use_initial_tour": bool(guided_config.use_initial_tour),
        },
        "selected_settings": selected_settings,
        "state_mode": state_mode,
        "matching_max_used": matching_max_used,
        "exact_time_limit": float(args.exact_time_limit),
        "exact_length_weight": float(args.exact_length_weight),
    }
    partial_paths = initialize_partial_result_files(
        save_dir=Path(args.save_dir),
        run_tag=run_tag,
        fieldnames=fieldnames,
        metadata=save_metadata,
    )
    print(f"[save] partial CSV   -> {partial_paths['csv']}")
    print(f"[save] partial JSONL -> {partial_paths['jsonl']}")
    print(f"[save] summary JSON  -> {partial_paths['summary']}")

    header = build_table_header(selected_settings)
    divider = "-" * len(header)
    print("\n" + "=" * len(header))
    print("TSPLIB Real-time Evaluation Progress")
    print(header)
    print(divider)

    all_results = []
    need_greedy = ("greedy" in selected_settings) or ("guided_lkh" in selected_settings)

    total_instances = len(tsp_files)
    for inst_idx, (n, f_path) in enumerate(tsp_files, start=1):
        name = f_path.stem
        prefix = f"[tsplib {inst_idx}/{total_instances}] {name} (N={n})"
        metrics = {setting: make_empty_metrics(setting) for setting in selected_settings}

        t_inst0 = time.time()
        log_progress(prefix, "parse TSPLIB coordinates...")
        try:
            coords = parse_tsp_file(str(f_path))
        except ValueError as exc:
            skipped_instances.append(
                {
                    "instance_index": inst_idx,
                    "instance": name,
                    "n": int(n),
                    "reason": str(exc),
                }
            )
            save_metadata["num_instances_skipped"] = len(skipped_instances)
            elapsed_sec = time.time() - t_inst0
            completed_at_utc = datetime.now(timezone.utc).isoformat()
            result_entry = {
                "name": name,
                "n": n,
                "instance_index": inst_idx,
                "elapsed_sec": round(elapsed_sec, 3),
                "completed_at_utc": completed_at_utc,
                "metrics": metrics,
                "skipped_reason": str(exc),
            }
            all_results.append(result_entry)
            append_partial_result(
                paths=partial_paths,
                fieldnames=fieldnames,
                inst_idx=inst_idx,
                name=name,
                n=n,
                elapsed_sec=elapsed_sec,
                completed_at_utc=completed_at_utc,
                selected_settings=selected_settings,
                metrics=metrics,
                metadata=save_metadata,
                all_results=all_results,
            )
            log_progress(prefix, f"skip unsupported instance: {exc}")
            print(build_table_row(name=name, n=n, selected_settings=selected_settings, metrics=metrics))
            continue
        points_cpu = torch.from_numpy(coords).float()
        log_progress(
            prefix,
            f"preprocess hierarchy (spanner={args.spanner_mode}, patching={args.patching_mode}, r={args.r})...",
        )
        prep = preprocess_points_to_hierarchy(
            points_cpu,
            r=int(args.r),
            num_workers=int(args.num_workers),
            raw_builder=raw_builder,
            spanner_builder=spanner_builder,
            patching_mode=str(args.patching_mode),
        )
        raw_data = prep["raw_data"]
        data = prep["data_cpu"]
        shared_timing = prep["shared_timing"]
        spanner_edge_index = prep["spanner_edge_index"]
        log_progress(
            prefix,
            (
                f"preprocess done: E={int(spanner_edge_index.shape[1])}, "
                f"spanner={shared_timing['spanner_construction_sec']:.2f}s, "
                f"quadtree={shared_timing['quadtree_building_sec']:.2f}s, "
                f"patch={shared_timing['patching_sec']:.2f}s"
            ),
        )
        num_tree_nodes = int(getattr(raw_data, "num_tree_nodes", -1))
        num_interfaces = int(getattr(raw_data, "interface_assign_index", torch.empty((2, 0))).shape[1])
        num_crossings = int(getattr(raw_data, "crossing_assign_index", torch.empty((2, 0))).shape[1])
        log_progress(
            prefix,
            f"hierarchy stats: nodes={num_tree_nodes}, interfaces={num_interfaces}, crossings={num_crossings}",
        )
        alive_edges = int(getattr(data, "edge_alive_mask", torch.empty((0,), dtype=torch.bool)).sum().item()) if hasattr(data, "edge_alive_mask") else -1
        if alive_edges >= 0:
            log_progress(prefix, f"effective alive edges={alive_edges}")
        log_progress(prefix, "move tensors to device...")
        data_cuda = to_device(data, device)

        log_progress(prefix, "run neural inference (pack -> bottom-up -> top-down -> edge aggregation)...")
        t_pack0 = time.time()
        with torch.no_grad():
            packed = packer.pack_batch([data_cuda])
            t_pack1 = time.time()
            out_bu = bu_runner.run_batch(batch=packed, leaf_encoder=leaf_encoder, merge_encoder=merge_encoder)
            t_pack2 = time.time()
            out_td = td_runner.run_batch(packed=packed, z=out_bu.z, decoder=decoder)
            t_pack3 = time.time()
            edge_scores = aggregate_logits_to_edges(
                tokens=packed.tokens,
                cross_logit=out_td.cross_logit,
                iface_logit=out_td.iface_logit,
                reduce="mean",
                num_edges=data_cuda.spanner_edge_index.shape[1],
            )
            el = mask_edge_logits_with_coverage(
                edge_scores.edge_logit,
                edge_scores.edge_mask,
            )
        log_progress(
            prefix,
            "inference done: "
            f"pack={t_pack1 - t_pack0:.2f}s, "
            f"bottom_up={t_pack2 - t_pack1:.2f}s, "
            f"top_down={t_pack3 - t_pack2:.2f}s, "
            f"aggregate={time.time() - t_pack3:.2f}s",
        )

        paper_res = TSPLIB_PAPER_RESULTS.get(name)
        teacher_len = paper_res["obj"] if paper_res else None
        pos_cpu, sp_edge_cpu, el_cpu = prepare_decode_inputs(
            pos=data_cuda.pos,
            spanner_edge_index=data_cuda.spanner_edge_index,
            edge_logit=el,
        )

        greedy_res = None
        if need_greedy:
            log_progress(prefix, "start greedy decode...")
            greedy_res = decode_tour(
                pos=pos_cpu,
                spanner_edge_index=sp_edge_cpu,
                edge_logit=el_cpu,
                backend="greedy",
                allow_off_spanner_patch=True,
            )
            if "greedy" in selected_settings and greedy_res.feasible:
                metrics["greedy"] = format_result_triplet(
                    length=greedy_res.length,
                    duration=greedy_res.duration,
                    teacher_len=teacher_len,
                )
            log_progress(prefix, f"greedy done in {greedy_res.duration:.2f}s (feasible={greedy_res.feasible})")

        if "exact" in selected_settings:
            log_progress(prefix, f"start exact sparse decode (time_limit={float(args.exact_time_limit):.1f}s)...")
            exact_res = decode_tour(
                pos=pos_cpu,
                spanner_edge_index=sp_edge_cpu,
                edge_logit=el_cpu,
                backend="exact",
                exact_time_limit=float(args.exact_time_limit),
                exact_length_weight=float(args.exact_length_weight),
            )
            if exact_res.feasible:
                metrics["exact"] = format_result_triplet(
                    length=exact_res.length,
                    duration=exact_res.duration,
                    teacher_len=teacher_len,
                )
            log_progress(prefix, f"exact done in {exact_res.duration:.2f}s (feasible={exact_res.feasible})")

        if "spanner_uniform_lkh" in selected_settings:
            log_progress(prefix, "start spanner-only LKH...")
            spanner_lkh_task = build_lkh_task(
                pos=data_cuda.pos,
                mode="spanner_uniform",
                edge_index=select_effective_edge_index(data_cuda),
                teacher_len=teacher_len if teacher_len else 0.0,
            )
            spanner_res, _ = solve_with_lkh_parallel(
                [spanner_lkh_task],
                lkh_executable=args.lkh_exe,
                num_workers=args.num_workers,
                guided_config=guided_config,
            )[0]
            if spanner_res.feasible:
                metrics["spanner_uniform_lkh"] = format_result_triplet(
                    length=spanner_res.length,
                    duration=spanner_res.duration,
                    teacher_len=teacher_len,
                )
            log_progress(
                prefix,
                f"spanner-only LKH done in {spanner_res.duration:.2f}s (feasible={spanner_res.feasible})",
            )

        if "guided_lkh" in selected_settings:
            log_progress(prefix, "start guided LKH...")
            lkh_task = build_lkh_task(
                pos=data_cuda.pos,
                mode="guided",
                edge_index=data_cuda.spanner_edge_index,
                edge_logit=el,
                teacher_len=teacher_len if teacher_len else 0.0,
                initial_tour=(
                    greedy_res.order
                    if greedy_res is not None and greedy_res.feasible
                    else None
                ),
            )
            guided_res, _ = solve_with_lkh_parallel(
                [lkh_task],
                lkh_executable=args.lkh_exe,
                num_workers=args.num_workers,
                guided_config=guided_config,
            )[0]
            if guided_res.feasible:
                metrics["guided_lkh"] = format_result_triplet(
                    length=guided_res.length,
                    duration=guided_res.duration,
                    teacher_len=teacher_len,
                )
            log_progress(prefix, f"guided LKH done in {guided_res.duration:.2f}s (feasible={guided_res.feasible})")

        if "pomo" in selected_settings and pomo_model is not None and pomo_env is not None:
            log_progress(prefix, "start POMO...")
            t_pomo = time.time()
            with torch.no_grad():
                n_ps = data_cuda.pos.shape[0]
                pomo_env.problem_size = n_ps
                pomo_env.pomo_size = n_ps
                pomo_env.set_problems(data_cuda.pos.unsqueeze(0).to(device) / 10000.0)
                reset_state, _, _ = pomo_env.reset()
                pomo_model.pre_forward(reset_state)
                state, reward, done = pomo_env.pre_step()
                while not done:
                    selected, _ = pomo_model(state)
                    state, reward, done = pomo_env.step(selected)
                best_reward, _ = reward.max(dim=1)
                pomo_len = -best_reward.item() * 10000.0
                if device.type == "cuda":
                    torch.cuda.synchronize()
                metrics["pomo"] = format_result_triplet(
                    length=pomo_len,
                    duration=time.time() - t_pomo,
                    teacher_len=teacher_len,
                )
            log_progress(prefix, f"POMO done in {metrics['pomo']['time']}") 

        if "neurolkh" in selected_settings and nlkh_model is not None:
            log_progress(prefix, "start NeuroLKH-guided LKH...")
            with torch.no_grad():
                n_nodes = data_cuda.pos.shape[0]
                n_edges = min(20, max(int(n_nodes) - 1, 1))
                node_feat = data_cuda.pos.unsqueeze(0).to(device) / 10000.0
                dist = torch.cdist(node_feat, node_feat)
                dist.diagonal(dim1=-2, dim2=-1).fill_(float("inf"))
                topk_values, topk_indices = torch.topk(dist, n_edges, dim=2, largest=False)
                e_idx_nlkh = topk_indices.view(1, -1).to(device)
                e_feat_nlkh = topk_values.view(1, -1, 1).to(device)

                target_nodes_neighbors = torch.gather(
                    topk_indices,
                    1,
                    topk_indices.view(1, n_nodes * n_edges).unsqueeze(-1).expand(-1, -1, n_edges),
                ).view(1, n_nodes, n_edges, n_edges)
                current_nodes_expanded = torch.arange(n_nodes, device=device).view(1, n_nodes, 1, 1).expand(1, -1, n_edges, n_edges)
                matches_mask = target_nodes_neighbors == current_nodes_expanded
                k_indices = matches_mask.max(dim=3)[1]
                inv_e_idx = (topk_indices * n_edges + k_indices).view(1, -1)
                no_match_mask = ~matches_mask.max(dim=3)[0]
                inv_e_idx.view(1, n_nodes, n_edges)[no_match_mask] = n_nodes * n_edges

                y_pred_edges_log, _, _ = nlkh_model(node_feat, e_feat_nlkh, e_idx_nlkh, inv_e_idx, None, None, n_edges)
                nlkh_edge_logit = y_pred_edges_log[0, :, 1].view(n_nodes, n_edges)
                nlkh_sp_index = torch.stack(
                    [
                        torch.arange(n_nodes, device=device).unsqueeze(1).expand(-1, n_edges).reshape(-1),
                        topk_indices.view(-1),
                    ],
                    dim=0,
                )

                nlkh_lkh_task = {
                    **build_lkh_task(
                        pos=data_cuda.pos,
                        mode="guided",
                        edge_index=nlkh_sp_index,
                        edge_logit=nlkh_edge_logit.view(-1),
                        teacher_len=teacher_len or 0.0,
                    )
                }
                nlkh_res, _ = solve_with_lkh_parallel(
                    [nlkh_lkh_task],
                    lkh_executable=args.lkh_exe,
                    num_workers=args.num_workers,
                    guided_config=guided_config,
                )[0]
                if nlkh_res.feasible:
                    metrics["neurolkh"] = format_result_triplet(
                        length=nlkh_res.length,
                        duration=nlkh_res.duration,
                        teacher_len=teacher_len,
                    )
            log_progress(prefix, f"NeuroLKH-guided LKH done in {metrics['neurolkh']['time']}")

        if "paper_lkh" in selected_settings:
            metrics["paper_lkh"]["obj"] = f"{paper_res['obj']}" if paper_res else "N/A"
            metrics["paper_lkh"]["time"] = f"{paper_res['time']}s" if paper_res else "N/A"

        elapsed_sec = time.time() - t_inst0
        completed_at_utc = datetime.now(timezone.utc).isoformat()
        result_entry = {
            "name": name,
            "n": n,
            "instance_index": inst_idx,
            "elapsed_sec": round(elapsed_sec, 3),
            "completed_at_utc": completed_at_utc,
            "metrics": metrics,
        }
        all_results.append(result_entry)
        append_partial_result(
            paths=partial_paths,
            fieldnames=fieldnames,
            inst_idx=inst_idx,
            name=name,
            n=n,
            elapsed_sec=elapsed_sec,
            completed_at_utc=completed_at_utc,
            selected_settings=selected_settings,
            metrics=metrics,
            metadata=save_metadata,
            all_results=all_results,
        )
        log_progress(
            prefix,
            f"instance complete in {elapsed_sec:.2f}s; appended results to {partial_paths['csv'].name}",
        )
        print(build_table_row(name=name, n=n, selected_settings=selected_settings, metrics=metrics))

    print("\n\n" + "=" * len(header))
    print("FINAL TSPLIB EVALUATION SUMMARY")
    print("=" * len(header))
    print(header)
    print(divider)
    for result in all_results:
        print(
            build_table_row(
                name=result["name"],
                n=result["n"],
                selected_settings=selected_settings,
                metrics=result["metrics"],
            )
        )
    print("=" * len(header) + "\n")
    print(f"[save] final CSV   -> {partial_paths['csv']}")
    print(f"[save] final JSONL -> {partial_paths['jsonl']}")
    print(f"[save] final JSON  -> {partial_paths['summary']}")


if __name__ == "__main__":
    main()
