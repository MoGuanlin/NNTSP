# src/experiments/evaluate_tsplib_pure_full_spanner.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.cli.common import log_progress
from src.cli.evaluate_tsplib import (
    collect_tsplib_files,
    describe_instance_presets,
    parse_instance_name_list,
    parse_tsp_file,
    sanitize_tag,
    select_tsplib_files,
)
from src.experiments.evaluate_tsplib_compare import compute_gap_pct, tsplib_euc2d_length
from src.cli.graph_pipeline import build_spanner_builder, preprocess_points_to_hierarchy
from src.graph.build_raw_pyramid import RawPyramidBuilder
from src.models.lkh_decode import (
    TSPLIB_PAPER_RESULTS,
    CandidateLKHConfig,
    build_uniform_spanner_candidates,
    run_candidate_lkh_timed,
)
from src.utils.lkh_solver import default_lkh_executable, resolve_lkh_executable


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_default_instance_list() -> List[str]:
    return list(TSPLIB_PAPER_RESULTS.keys())[:5]


def build_run_tag(*, selection_tag: str, r: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return sanitize_tag(f"{timestamp}_pure_full_spanner_{selection_tag}_r{r}")


def build_fieldnames() -> List[str]:
    return [
        "instance_index",
        "instance",
        "n",
        "paper_obj",
        "gap_pct",
        "feasible",
        "timeout_hit",
        "obj",
        "method_time_sec",
        "end_to_end_sec",
        "shared_parse_sec",
        "shared_spanner_construction_sec",
        "shared_quadtree_building_sec",
        "shared_patching_sec",
        "shared_total_sec",
        "full_spanner_edges",
        "effective_edges",
        "completed_at_utc",
    ]


def flatten_record(record: Dict[str, Any]) -> Dict[str, str]:
    shared = record["shared_timing"]
    result = record["result"]
    return {
        "instance_index": str(record["instance_index"]),
        "instance": str(record["instance"]),
        "n": str(record["n"]),
        "paper_obj": "" if record["paper_obj"] is None else f"{float(record['paper_obj']):.0f}",
        "gap_pct": "" if result["gap_pct"] is None else f"{float(result['gap_pct']):.6f}",
        "feasible": str(bool(result["feasible"])),
        "timeout_hit": str(bool(result["timeout_hit"])),
        "obj": "" if result["obj"] is None else f"{float(result['obj']):.0f}",
        "method_time_sec": f"{float(result['time_sec']):.6f}",
        "end_to_end_sec": f"{float(record['end_to_end_sec']):.6f}",
        "shared_parse_sec": f"{float(shared['parse_sec']):.6f}",
        "shared_spanner_construction_sec": f"{float(shared['spanner_construction_sec']):.6f}",
        "shared_quadtree_building_sec": f"{float(shared['quadtree_building_sec']):.6f}",
        "shared_patching_sec": f"{float(shared['patching_sec']):.6f}",
        "shared_total_sec": f"{float(record['shared_total_sec']):.6f}",
        "full_spanner_edges": str(int(record["full_spanner_edges"])),
        "effective_edges": str(int(record["effective_edges"])),
        "completed_at_utc": str(record["completed_at_utc"]),
    }


def summarize(records: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    feasible = [rec["result"] for rec in records if rec["result"]["obj"] is not None]

    def _avg(items: List[float]) -> float | None:
        return None if not items else float(np.mean(np.array(items, dtype=np.float64)))

    return {
        "metadata": metadata,
        "num_records": len(records),
        "records": records,
        "summary": {
            "num_feasible": len(feasible),
            "avg_obj": _avg([float(x["obj"]) for x in feasible]),
            "avg_gap_pct": _avg([float(x["gap_pct"]) for x in feasible if x.get("gap_pct") is not None]),
            "avg_method_time_sec": _avg([float(x["time_sec"]) for x in feasible]),
            "avg_end_to_end_sec": _avg(
                [float(rec["end_to_end_sec"]) for rec in records if rec["result"]["obj"] is not None]
            ),
            "num_timeouts": int(sum(1 for rec in records if bool(rec["result"].get("timeout_hit", False)))),
        },
    }


def run_full_spanner_instance(
    *,
    points_cpu: torch.Tensor,
    spanner_edge_index: torch.Tensor,
    coords_np: np.ndarray,
    lkh_exe: str,
    lkh_runs: int,
    lkh_timeout: float | None,
    paper_obj: float | None,
    heartbeat_sec: float,
    prefix: str,
) -> Dict[str, Any]:
    pos_cpu = points_cpu.detach().cpu()
    pos_np = pos_cpu.numpy()
    edge_index_cpu = spanner_edge_index.detach().cpu()
    candidates = build_uniform_spanner_candidates(
        num_nodes=int(pos_cpu.shape[0]),
        edge_index=edge_index_cpu,
        uniform_alpha=0,
    )
    result, lkh_break = run_candidate_lkh_timed(
        pos=pos_cpu,
        candidates=candidates,
        initial_tour=None,
        lkh_executable=lkh_exe,
        num_runs=max(1, int(lkh_runs)),
        seed=1234,
        timeout=lkh_timeout,
        mode="spanner_uniform",
        candidate_config=CandidateLKHConfig(
            subgradient=True,
            max_candidates=0,
            max_trials=None,
            use_initial_tour=False,
        ),
        heartbeat_sec=float(heartbeat_sec),
        heartbeat_emit=lambda msg: log_progress(prefix, msg),
    )
    if result.feasible:
        euclidean_length = float(result.length)
        obj = tsplib_euc2d_length(coords_np, list(result.order))
    else:
        euclidean_length = None
        obj = None
    timeout_hit = bool(lkh_break.get("timeout_hit", False))

    return {
        "feasible": bool(obj is not None),
        "obj": (None if obj is None else float(obj)),
        "gap_pct": compute_gap_pct(obj, paper_obj),
        "time_sec": float(result.duration),
        "euclidean_length": (None if euclidean_length is None else float(euclidean_length)),
        "order": ([] if obj is None else [int(x) for x in result.order]),
        "timeout_hit": bool(timeout_hit),
        "used_initial_tour": False,
        "subgradient": True,
        "max_candidates": 0,
        "max_trials": None,
    }


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run only the Pure+Full-Spanner TSPLIB baseline with progress heartbeats."
    )
    parser.add_argument("--tsplib_dir", type=str, default="benchmarks/tsplib")
    parser.add_argument("--instance_preset", type=str, default=None, help="TSPLIB instance preset; " + describe_instance_presets())
    parser.add_argument("--instances", type=str, default=None, help="comma-separated explicit TSPLIB instance names")
    parser.add_argument("--num_instances", type=int, default=5)
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--spanner_mode", type=str, default="delaunay", choices=("delaunay", "theta"))
    parser.add_argument("--theta_k", type=int, default=14)
    parser.add_argument("--patching_mode", type=str, default="prune", choices=("prune", "arora"))
    parser.add_argument("--lkh_exe", type=str, default=default_lkh_executable())
    parser.add_argument("--lkh_runs", type=int, default=1)
    parser.add_argument("--lkh_timeout", type=float, default=0.0, help="seconds; <=0 means no timeout")
    parser.add_argument("--heartbeat_sec", type=float, default=30.0, help="print a progress heartbeat every N seconds during LKH")
    parser.add_argument("--save_dir", type=str, default="outputs/eval_tsplib_hierarchy_ablation")
    parser.add_argument("--run_tag", type=str, default=None)
    args = parser.parse_args(argv)

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

    tsplib_path = Path(args.tsplib_dir)
    all_tsp_files = collect_tsplib_files(tsplib_path)
    tsp_files, selection_desc, selection_tag = select_tsplib_files(
        all_tsp_files=all_tsp_files,
        instance_preset=instance_preset,
        instance_names=(",".join(instance_names) if instance_names else None),
        num_instances=int(args.num_instances),
    )

    lkh_timeout = None if float(args.lkh_timeout) <= 0 else float(args.lkh_timeout)
    lkh_exe = resolve_lkh_executable(str(args.lkh_exe))
    print(f"[env] lkh_exe={lkh_exe}")
    print(f"[env] lkh_timeout={lkh_timeout}")
    print(f"[data] Selected {selection_desc} from {len(all_tsp_files)} total.")

    raw_builder = RawPyramidBuilder(max_points_per_leaf=4, max_depth=20)
    spanner_builder = build_spanner_builder(
        spanner_mode=str(args.spanner_mode),
        theta_k=int(args.theta_k),
    )

    run_tag = sanitize_tag(str(args.run_tag)) if args.run_tag else build_run_tag(
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
        "tsplib_dir": str(tsplib_path.resolve()),
        "selection_desc": selection_desc,
        "instance_preset": instance_preset,
        "instances": instance_names,
        "r": int(args.r),
        "num_workers": int(args.num_workers),
        "spanner_mode": str(args.spanner_mode),
        "theta_k": int(args.theta_k),
        "patching_mode": str(args.patching_mode),
        "lkh_exe": str(lkh_exe),
        "lkh_runs": int(args.lkh_runs),
        "lkh_timeout": lkh_timeout,
        "heartbeat_sec": float(args.heartbeat_sec),
        "note": "Only runs the Pure+Full-Spanner baseline (no neural guidance, no warm start).",
    }

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    records: List[Dict[str, Any]] = []
    total_instances = len(tsp_files)
    for inst_idx, (n, tsp_file) in enumerate(tsp_files, start=1):
        name = tsp_file.stem
        prefix = f"[pure+full {inst_idx}/{total_instances}] {name} (N={n})"
        paper_ref = TSPLIB_PAPER_RESULTS.get(name, {})
        paper_obj = paper_ref.get("obj")
        shared_timing = {
            "parse_sec": 0.0,
            "spanner_construction_sec": 0.0,
            "quadtree_building_sec": 0.0,
            "patching_sec": 0.0,
        }

        log_progress(prefix, "parse TSPLIB coordinates...")
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
        data_cpu = prep["data_cpu"]
        shared_timing.update(prep["shared_timing"])
        shared_total_sec = float(sum(shared_timing.values()))
        full_edges = int(data_cpu.spanner_edge_index.shape[1])
        effective_edges = int(getattr(data_cpu, "edge_alive_mask").sum().item()) if hasattr(data_cpu, "edge_alive_mask") else -1
        log_progress(
            prefix,
            (
                f"preprocess done: parse={shared_timing['parse_sec']:.2f}s, "
                f"spanner={shared_timing['spanner_construction_sec']:.2f}s, "
                f"quadtree={shared_timing['quadtree_building_sec']:.2f}s, "
                f"patch={shared_timing['patching_sec']:.2f}s, "
                f"full_edges={full_edges}, effective_edges={effective_edges}"
            ),
        )

        log_progress(
            prefix,
            f"start Pure+Full-Spanner LKH (runs={args.lkh_runs}, timeout={lkh_timeout}, heartbeat={args.heartbeat_sec:.0f}s)...",
        )
        result = run_full_spanner_instance(
            points_cpu=points_cpu,
            spanner_edge_index=data_cpu.spanner_edge_index.cpu(),
            coords_np=coords,
            lkh_exe=str(lkh_exe),
            lkh_runs=int(args.lkh_runs),
            lkh_timeout=lkh_timeout,
            paper_obj=paper_obj,
            heartbeat_sec=float(args.heartbeat_sec),
            prefix=prefix,
        )

        record = {
            "instance_index": inst_idx,
            "instance": name,
            "n": int(n),
            "paper_obj": paper_obj,
            "completed_at_utc": utc_now_iso(),
            "shared_timing": shared_timing,
            "shared_total_sec": shared_total_sec,
            "full_spanner_edges": full_edges,
            "effective_edges": effective_edges,
            "result": result,
            "end_to_end_sec": shared_total_sec + float(result["time_sec"]),
        }
        records.append(record)

        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(flatten_record(record))
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summarize(records, metadata), f, ensure_ascii=False, indent=2)

        status = "timeout" if bool(result["timeout_hit"]) else ("ok" if bool(result["feasible"]) else "no-tour")
        log_progress(
            prefix,
            (
                f"done: status={status}, obj={result['obj']}, "
                f"method={result['time_sec']:.2f}s, e2e={record['end_to_end_sec']:.2f}s"
            ),
        )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summarize(records, metadata), f, ensure_ascii=False, indent=2)
    print(f"[save] csv={csv_path}")
    print(f"[save] jsonl={jsonl_path}")
    print(f"[save] summary={summary_path}")


if __name__ == "__main__":
    main()
