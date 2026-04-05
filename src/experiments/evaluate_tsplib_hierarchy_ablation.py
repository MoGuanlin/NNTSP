# src/experiments/evaluate_tsplib_hierarchy_ablation.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from src.cli.common import log_progress, resolve_device
from src.cli.guided_lkh_args import add_guided_lkh_args, guided_lkh_config_from_args
from src.cli.graph_pipeline import (
    build_spanner_builder,
    preprocess_points_to_hierarchy,
    select_effective_edge_index,
)
from src.cli.evaluate_tsplib import (
    collect_tsplib_files,
    describe_instance_presets,
    parse_instance_name_list,
    parse_tsp_file,
    sanitize_tag,
    select_tsplib_files,
)
from src.experiments.evaluate_tsplib_compare import (
    SHARED_TIMING_KEYS,
    compute_gap_pct,
    metric_entry,
    run_twopass_guided_instance,
    tsplib_euc2d_length,
)
from src.cli.model_factory import load_twopass_eval_models
from src.graph.build_raw_pyramid import RawPyramidBuilder
from src.models.bottom_up_runner import BottomUpTreeRunner
from src.models.lkh_decode import (
    TSPLIB_PAPER_RESULTS,
    CandidateLKHConfig,
    build_uniform_spanner_candidates,
    run_candidate_lkh_timed,
)
from src.models.node_token_packer import NodeTokenPacker
from src.models.top_down_runner import TopDownTreeRunner


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_default_instance_list() -> List[str]:
    return list(TSPLIB_PAPER_RESULTS.keys())[:5]


def build_run_tag(*, ckpt_path: str, selection_tag: str, r: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ckpt_obj = Path(ckpt_path)
    ckpt_tag = ckpt_obj.parent.name if ckpt_obj.parent.name else ckpt_obj.stem
    return sanitize_tag(f"{timestamp}_{ckpt_tag}_{selection_tag}_r{r}_hierarchy_ablation")


def run_spanner_uniform_instance(
    *,
    points_cpu: torch.Tensor,
    spanner_edge_index: torch.Tensor,
    coords_np: np.ndarray,
    lkh_exe: str,
    lkh_runs: int,
    lkh_timeout: float | None,
    paper_obj: float | None,
    initial_tour: List[int] | None = None,
) -> Dict[str, Any]:
    pos_cpu = points_cpu.detach().cpu()
    edge_index_cpu = spanner_edge_index.detach().cpu()

    start_t = time.perf_counter()
    candidates = build_uniform_spanner_candidates(
        num_nodes=int(pos_cpu.shape[0]),
        edge_index=edge_index_cpu,
        uniform_alpha=0,
    )
    result, _ = run_candidate_lkh_timed(
        pos=pos_cpu,
        candidates=candidates,
        initial_tour=initial_tour,
        lkh_executable=lkh_exe,
        num_runs=max(1, int(lkh_runs)),
        seed=1234,
        timeout=lkh_timeout,
        mode="spanner_uniform",
        candidate_config=CandidateLKHConfig(
            subgradient=True,
            max_candidates=0,
            max_trials=None,
            use_initial_tour=True,
        ),
    )
    duration = time.perf_counter() - start_t
    order = list(result.order)
    obj = tsplib_euc2d_length(coords_np, order) if result.feasible else None
    euclidean_length = float(result.length) if result.feasible else None

    return metric_entry(
        feasible=bool(result.feasible),
        obj=obj,
        euclidean_length=(None if euclidean_length is None else float(euclidean_length)),
        order=(list(order) if obj is not None else None),
        time_sec=float(duration),
        optimum=paper_obj,
        extra={
            "used_initial_tour": bool(initial_tour is not None),
            "subgradient": True,
            "max_candidates": 0,
            "max_trials": None,
        },
    )

def build_fieldnames() -> List[str]:
    return [
        "instance_index",
        "instance",
        "n",
        "paper_obj",
        "paper_time_sec",
        "shared_parse_sec",
        "shared_spanner_construction_sec",
        "shared_quadtree_building_sec",
        "shared_patching_sec",
        "shared_total_sec",
        "guided_obj",
        "guided_gap_pct",
        "guided_method_time_sec",
        "guided_end_to_end_sec",
        "spanner_uniform_obj",
        "spanner_uniform_gap_pct",
        "spanner_uniform_method_time_sec",
        "spanner_uniform_end_to_end_sec",
        "completed_at_utc",
    ]


def flatten_record(record: Dict[str, Any]) -> Dict[str, str]:
    shared = record["shared_timing"]
    guided = record["guided_lkh"]
    spanner = record["spanner_uniform_lkh"]
    return {
        "instance_index": str(record["instance_index"]),
        "instance": str(record["instance"]),
        "n": str(record["n"]),
        "paper_obj": "" if record["paper_obj"] is None else f"{float(record['paper_obj']):.0f}",
        "paper_time_sec": "" if record["paper_time_sec"] is None else f"{float(record['paper_time_sec']):.3f}",
        "shared_parse_sec": f"{float(shared['parse_sec']):.6f}",
        "shared_spanner_construction_sec": f"{float(shared['spanner_construction_sec']):.6f}",
        "shared_quadtree_building_sec": f"{float(shared['quadtree_building_sec']):.6f}",
        "shared_patching_sec": f"{float(shared['patching_sec']):.6f}",
        "shared_total_sec": f"{float(record['shared_total_sec']):.6f}",
        "guided_obj": "" if guided["obj"] is None else f"{float(guided['obj']):.0f}",
        "guided_gap_pct": "" if guided["gap_pct"] is None else f"{float(guided['gap_pct']):.6f}",
        "guided_method_time_sec": f"{float(guided['time_sec']):.6f}",
        "guided_end_to_end_sec": f"{float(record['guided_end_to_end_sec']):.6f}",
        "spanner_uniform_obj": "" if spanner["obj"] is None else f"{float(spanner['obj']):.0f}",
        "spanner_uniform_gap_pct": "" if spanner["gap_pct"] is None else f"{float(spanner['gap_pct']):.6f}",
        "spanner_uniform_method_time_sec": f"{float(spanner['time_sec']):.6f}",
        "spanner_uniform_end_to_end_sec": f"{float(record['spanner_uniform_end_to_end_sec']):.6f}",
        "completed_at_utc": str(record["completed_at_utc"]),
    }


def summarize(records: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    def _avg(items: List[float]) -> float | None:
        return None if not items else float(np.mean(np.array(items, dtype=np.float64)))

    guided_feasible = [rec["guided_lkh"] for rec in records if rec["guided_lkh"]["obj"] is not None]
    spanner_feasible = [rec["spanner_uniform_lkh"] for rec in records if rec["spanner_uniform_lkh"]["obj"] is not None]

    return {
        "metadata": metadata,
        "num_records": len(records),
        "records": records,
        "summary": {
            "guided_lkh": {
                "num_feasible": len(guided_feasible),
                "avg_obj": _avg([float(x["obj"]) for x in guided_feasible]),
                "avg_gap_pct": _avg([float(x["gap_pct"]) for x in guided_feasible if x.get("gap_pct") is not None]),
                "avg_method_time_sec": _avg([float(x["time_sec"]) for x in guided_feasible]),
                "avg_end_to_end_sec": _avg(
                    [
                        float(rec["guided_end_to_end_sec"])
                        for rec in records
                        if rec["guided_lkh"]["obj"] is not None
                    ]
                ),
            },
            "spanner_uniform_lkh": {
                "num_feasible": len(spanner_feasible),
                "avg_obj": _avg([float(x["obj"]) for x in spanner_feasible]),
                "avg_gap_pct": _avg([float(x["gap_pct"]) for x in spanner_feasible if x.get("gap_pct") is not None]),
                "avg_method_time_sec": _avg([float(x["time_sec"]) for x in spanner_feasible]),
                "avg_end_to_end_sec": _avg(
                    [
                        float(rec["spanner_uniform_end_to_end_sec"])
                        for rec in records
                        if rec["spanner_uniform_lkh"]["obj"] is not None
                    ]
                ),
            },
        },
    }


def main(argv: List[str] | None = None) -> None:
    from src.utils.lkh_solver import default_lkh_executable, resolve_lkh_executable

    parser = argparse.ArgumentParser(
        description="Hierarchy ablation on TSPLIB: compare 2-pass guided LKH with spanner+uniform LKH."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="path to the 2-pass checkpoint")
    parser.add_argument("--tsplib_dir", type=str, default="benchmarks/tsplib")
    parser.add_argument("--instance_preset", type=str, default=None, help="TSPLIB instance preset; " + describe_instance_presets())
    parser.add_argument("--instances", type=str, default=None, help="comma-separated explicit TSPLIB instance names")
    parser.add_argument("--num_instances", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--r", type=int, default=4)
    add_guided_lkh_args(parser)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--spanner_mode", type=str, default="delaunay", choices=("delaunay", "theta"))
    parser.add_argument("--theta_k", type=int, default=14, help="theta spanner cone count when --spanner_mode theta")
    parser.add_argument("--patching_mode", type=str, default="prune", choices=("prune", "arora"))
    parser.add_argument("--lkh_exe", type=str, default=default_lkh_executable())
    parser.add_argument("--lkh_runs", type=int, default=1)
    parser.add_argument("--lkh_timeout", type=float, default=None)
    parser.add_argument("--use_iface_in_decode", type=str, default="true")
    parser.add_argument("--save_dir", type=str, default="outputs/eval_tsplib_hierarchy_ablation")
    parser.add_argument("--run_tag", type=str, default=None)
    args = parser.parse_args(argv)

    device = resolve_device(str(args.device))
    guided_config = guided_lkh_config_from_args(args)
    lkh_exe = resolve_lkh_executable(str(args.lkh_exe))
    print(f"[env] device={device}")
    print(f"[env] lkh_exe={lkh_exe}")

    explicit_instances = parse_instance_name_list(args.instances)
    if explicit_instances:
        instance_names = explicit_instances
        instance_preset = None
    elif args.instance_preset:
        instance_names = None
        instance_preset = str(args.instance_preset)
    else:
        instance_names = build_default_instance_list()
        instance_preset = None
        print(f"[data] defaulting to first 5 paper instances: {', '.join(instance_names)}")

    print(f"[ckpt] loading from {args.ckpt}")
    model_bundle = load_twopass_eval_models(
        ckpt_path=str(args.ckpt),
        device=device,
        r=int(args.r),
    )
    packer = NodeTokenPacker(
        r=int(args.r),
        state_mode=model_bundle.state_mode,
        matching_max_used=model_bundle.matching_max_used,
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
        instance_preset=instance_preset,
        instance_names=(",".join(instance_names) if instance_names else None),
        num_instances=int(args.num_instances),
    )
    print(f"[data] Selected {selection_desc} from {len(all_tsp_files)} total.")

    run_tag = sanitize_tag(str(args.run_tag)) if args.run_tag else build_run_tag(
        ckpt_path=str(args.ckpt),
        selection_tag=selection_tag,
        r=int(args.r),
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / f"{run_tag}.csv"
    jsonl_path = save_dir / f"{run_tag}.jsonl"
    summary_path = save_dir / f"{run_tag}_summary.json"
    fieldnames = build_fieldnames()

    metadata = {
        "run_tag": run_tag,
        "created_at_utc": utc_now_iso(),
        "ckpt": str(Path(args.ckpt).resolve()),
        "tsplib_dir": str(tsplib_path.resolve()),
        "selection_desc": selection_desc,
        "instance_preset": instance_preset,
        "instances": instance_names,
        "r": int(args.r),
        "device": str(device),
        "guided_lkh": {
            "top_k": int(guided_config.top_k),
            "logit_scale": float(guided_config.logit_scale),
            "subgradient": bool(guided_config.subgradient),
            "max_candidates": guided_config.max_candidates,
            "max_trials": guided_config.max_trials,
            "use_initial_tour": bool(guided_config.use_initial_tour),
        },
        "num_workers": int(args.num_workers),
        "spanner_mode": str(args.spanner_mode),
        "theta_k": int(args.theta_k),
        "patching_mode": str(args.patching_mode),
        "lkh_exe": str(lkh_exe),
        "lkh_runs": int(args.lkh_runs),
        "lkh_timeout": args.lkh_timeout,
        "state_mode": model_bundle.state_mode,
        "matching_max_used": int(model_bundle.matching_max_used),
        "note": "guided_lkh reuses the main 2-pass inference path; spanner_uniform_lkh skips neural inference and uses uniform spanner candidates.",
    }

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    records: List[Dict[str, Any]] = []
    total_instances = len(tsp_files)
    for inst_idx, (n, tsp_file) in enumerate(tsp_files, start=1):
        name = tsp_file.stem
        prefix = f"[hierarchy {inst_idx}/{total_instances}] {name} (N={n})"
        paper_ref = TSPLIB_PAPER_RESULTS.get(name, {})
        paper_obj = paper_ref.get("obj")
        paper_time_sec = paper_ref.get("time")
        shared_timing = {key: 0.0 for key in SHARED_TIMING_KEYS}

        parse_t0 = time.perf_counter()
        coords = parse_tsp_file(str(tsp_file))
        shared_timing["parse_sec"] = time.perf_counter() - parse_t0

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
        data_cpu = prep["data_cpu"]
        shared_timing.update(prep["shared_timing"])
        shared_total_sec = float(sum(shared_timing.values()))

        log_progress(prefix, "run 2-pass guided LKH...")
        guided_res = run_twopass_guided_instance(
            data_cpu=data_cpu,
            coords_np=coords,
            device=device,
            packer=packer,
            model_bundle=model_bundle,
            bu_runner=bu_runner,
            td_runner=td_runner,
            use_iface_in_decode=str(args.use_iface_in_decode).strip().lower() not in {"0", "false", "no"},
            guided_config=guided_config,
            lkh_exe=str(lkh_exe),
            lkh_runs=int(args.lkh_runs),
            lkh_timeout=args.lkh_timeout,
            optimum=paper_obj,
        )

        log_progress(prefix, "run spanner+uniform LKH...")
        patched_edge_index = select_effective_edge_index(data_cpu)
        spanner_res = run_spanner_uniform_instance(
            points_cpu=points_cpu,
            spanner_edge_index=patched_edge_index,
            coords_np=coords,
            lkh_exe=str(lkh_exe),
            lkh_runs=int(args.lkh_runs),
            lkh_timeout=args.lkh_timeout,
            paper_obj=paper_obj,
        )

        record = {
            "instance_index": inst_idx,
            "instance": name,
            "n": int(n),
            "paper_obj": paper_obj,
            "paper_time_sec": paper_time_sec,
            "completed_at_utc": utc_now_iso(),
            "shared_timing": shared_timing,
            "shared_total_sec": shared_total_sec,
            "guided_lkh": guided_res,
            "guided_end_to_end_sec": shared_total_sec + float(guided_res["time_sec"]),
            "spanner_uniform_lkh": spanner_res,
            "spanner_uniform_end_to_end_sec": shared_total_sec + float(spanner_res["time_sec"]),
            "spanner_uniform_num_edges": int(patched_edge_index.shape[1]),
        }
        records.append(record)

        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(flatten_record(record))
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summarize(records, metadata), f, ensure_ascii=False, indent=2)

        print(
            f"[done] {name}: "
            f"guided={guided_res['obj']}/{guided_res['gap_pct']}/{guided_res['time_sec']:.2f}s "
            f"(e2e={record['guided_end_to_end_sec']:.2f}s), "
            f"spanner={spanner_res['obj']}/{spanner_res['gap_pct']}/{spanner_res['time_sec']:.2f}s "
            f"(e2e={record['spanner_uniform_end_to_end_sec']:.2f}s)"
        )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summarize(records, metadata), f, ensure_ascii=False, indent=2)
    print(f"[save] csv={csv_path}")
    print(f"[save] jsonl={jsonl_path}")
    print(f"[save] summary={summary_path}")


if __name__ == "__main__":
    main()
