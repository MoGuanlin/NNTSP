# src/experiments/evaluate_vlsi_experiment.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import gzip
import json
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import torch

from src.cli.common import parse_bool_arg, resolve_device
from src.cli.guided_lkh_args import add_guided_lkh_args, guided_lkh_config_from_args
from src.cli.evaluate_tsplib import load_neurolkh, parse_tsp_file
from src.experiments.evaluate_tsplib_compare import run_neurolkh_instance, run_twopass_guided_instance
from src.cli.graph_pipeline import (
    build_spanner_builder,
    preprocess_points_to_hierarchy,
    select_effective_edge_index,
)
from src.cli.model_factory import load_twopass_eval_models
from src.graph.build_raw_pyramid import RawPyramidBuilder
from src.models.bottom_up_runner import BottomUpTreeRunner
from src.models.lkh_decode import GuidedLKHConfig
from src.models.node_token_packer import NodeTokenPacker
from src.models.top_down_runner import TopDownTreeRunner
from src.utils.lkh_solver import default_lkh_executable, resolve_lkh_executable


VLSI_PAGE11_INSTANCE_ORDER = (
    "sra104815",
    "ara238025",
    "lra498378",
    "lrb744710",
)

# Source: University of Waterloo VLSI page summary
# https://www.math.uwaterloo.ca/tsp/vlsi/summary.html
VLSI_PAGE11_GROUNDTRUTH = {
    "sra104815": 251342,
    "ara238025": 578761,
    "lra498378": 2168039,
    "lrb744710": 1611232,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_instance_list(text: str | None) -> List[str]:
    if not text:
        return list(VLSI_PAGE11_INSTANCE_ORDER)
    out: List[str] = []
    for part in str(text).replace(",", " ").split():
        name = part.strip()
        if name:
            out.append(name)
    return out


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def should_skip_existing_result(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = load_json(path)
    except Exception:
        return False
    return str(payload.get("status", "")).lower() == "ok"


class TeeLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("w", encoding="utf-8")
        self._lock = threading.Lock()

    def log(self, message: str) -> None:
        line = f"[{utc_now_iso()}] {message}"
        with self._lock:
            print(line, flush=True)
            self._fp.write(line + "\n")
            self._fp.flush()

    def close(self) -> None:
        with self._lock:
            self._fp.close()


def run_with_heartbeat(
    *,
    label: str,
    logger: TeeLogger,
    fn: Callable[[], Any],
    heartbeat_sec: float,
) -> Any:
    t0 = time.perf_counter()
    stop_event = threading.Event()

    def _heartbeat() -> None:
        while not stop_event.wait(timeout=max(1.0, float(heartbeat_sec))):
            elapsed = time.perf_counter() - t0
            logger.log(f"{label} still running... elapsed={elapsed:.1f}s")

    heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
        result = fn()
    except Exception:
        logger.log(f"{label} failed after {time.perf_counter() - t0:.1f}s")
        raise
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=1.0)

    elapsed = time.perf_counter() - t0
    logger.log(f"{label} finished in {elapsed:.1f}s")
    return result


def safe_cuda_empty_cache(device: torch.device, logger: TeeLogger | None = None) -> None:
    if device.type != "cuda":
        return
    try:
        torch.cuda.empty_cache()
    except RuntimeError as exc:
        if logger is not None:
            logger.log(f"[warn] torch.cuda.empty_cache() failed: {exc}")
        else:
            print(f"[warn] torch.cuda.empty_cache() failed: {exc}", flush=True)


def ensure_plain_vlsi_files(
    *,
    src_dir: Path,
    plain_dir: Path,
    instances: Sequence[str],
    groundtruth: Dict[str, int],
) -> Dict[str, Any]:
    plain_dir.mkdir(parents=True, exist_ok=True)
    status: Dict[str, Any] = {
        "plain_dir": str(plain_dir.resolve()),
        "instances": list(instances),
        "files": {},
    }
    for name in instances:
        src_gz = src_dir / f"{name}.tsp.gz"
        dst_tsp = plain_dir / f"{name}.tsp"
        if not src_gz.exists() and not dst_tsp.exists():
            raise FileNotFoundError(f"Missing both {src_gz} and {dst_tsp}")
        if dst_tsp.exists():
            status["files"][name] = {
                "source_gz": (str(src_gz.resolve()) if src_gz.exists() else None),
                "plain_tsp": str(dst_tsp.resolve()),
                "action": "reused_existing_plain",
            }
            continue
        with gzip.open(src_gz, "rb") as f_in:
            with dst_tsp.open("wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        status["files"][name] = {
            "source_gz": str(src_gz.resolve()),
            "plain_tsp": str(dst_tsp.resolve()),
            "action": "decompressed",
        }

    gt_path = plain_dir / "groundtruth_vlsi_page11.json"
    save_json(gt_path, {name: int(groundtruth[name]) for name in instances})
    status["groundtruth_json"] = str(gt_path.resolve())

    solutions_path = plain_dir / "solutions"
    with solutions_path.open("w", encoding="utf-8") as f:
        for name in instances:
            f.write(f"{name}: {int(groundtruth[name])}\n")
    status["solutions_file"] = str(solutions_path.resolve())
    return status


def build_guided_key(*, spanner_mode: str, patching_mode: str) -> str:
    return f"guided_lkh_{str(spanner_mode)}_{str(patching_mode)}"


def build_guided_dirname(*, spanner_mode: str, patching_mode: str) -> str:
    return f"guided_{str(spanner_mode)}_{str(patching_mode)}"


def build_twopass_components(
    *,
    ckpt_path: str,
    device: torch.device,
    r: int,
    spanner_mode: str,
    theta_k: int,
):
    model_bundle = load_twopass_eval_models(
        ckpt_path=str(ckpt_path),
        device=device,
        r=int(r),
    )
    components = {
        "model_bundle": model_bundle,
        "packer": NodeTokenPacker(
            r=int(r),
            state_mode=model_bundle.state_mode,
            matching_max_used=model_bundle.matching_max_used,
        ),
        "spanner_builder": build_spanner_builder(spanner_mode=str(spanner_mode), theta_k=int(theta_k)),
        "raw_builder": RawPyramidBuilder(max_points_per_leaf=4, max_depth=20),
    }
    return components


def build_chunk_schedule(
    *,
    bottom_up_chunk: int,
    top_down_chunk: int,
    min_bottom_up_chunk: int = 64,
    min_top_down_chunk: int = 32,
) -> List[tuple[int, int]]:
    schedule: List[tuple[int, int]] = []
    bu = max(1, int(bottom_up_chunk))
    td = max(1, int(top_down_chunk))
    min_bu = max(1, int(min_bottom_up_chunk))
    min_td = max(1, int(min_top_down_chunk))

    while True:
        pair = (bu, td)
        if pair not in schedule:
            schedule.append(pair)
        if bu <= min_bu and td <= min_td:
            break
        next_bu = max(min_bu, bu // 2)
        next_td = max(min_td, td // 2)
        if next_bu == bu and next_td == td:
            break
        bu, td = next_bu, next_td
    return schedule


def is_cuda_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return ("out of memory" in text) or ("cuda error: out of memory" in text)


def guided_result_payload(
    *,
    instance: str,
    optimum: int,
    coords,
    prep: Dict[str, Any],
    shared_timing: Dict[str, float],
    guided_res: Dict[str, Any],
    guided_key: str,
    spanner_mode: str,
    patching_mode: str,
    used_bottom_up_chunk: int,
    used_top_down_chunk: int,
    guided_attempt_count: int,
) -> Dict[str, Any]:
    data_cpu = prep["data_cpu"]
    raw_data = prep["raw_data"]
    shared_total_sec = float(sum(shared_timing.values()))
    return {
        "instance": instance,
        "method": guided_key,
        "status": "ok",
        "groundtruth": int(optimum),
        "num_points": int(coords.shape[0]),
        "obj": guided_res["obj"],
        "gap_pct": guided_res["gap_pct"],
        "time_sec": guided_res["time_sec"],
        "end_to_end_sec": shared_total_sec + float(guided_res["time_sec"]),
        "shared_timing": shared_timing,
        "shared_total_sec": shared_total_sec,
        "timing": guided_res.get("timing", {}),
        "warm_start_feasible": bool(guided_res.get("warm_start_feasible", False)),
        "warm_start_length": guided_res.get("warm_start_length"),
        "spanner_mode": str(spanner_mode),
        "patching_mode": str(patching_mode),
        "bottom_up_chunk": int(used_bottom_up_chunk),
        "top_down_chunk": int(used_top_down_chunk),
        "guided_attempt_count": int(guided_attempt_count),
        "num_spanner_edges": int(prep["spanner_edge_index"].shape[1]),
        "num_effective_edges": int(select_effective_edge_index(data_cpu).shape[1]),
        "num_tree_nodes": int(getattr(raw_data, "num_tree_nodes", -1)),
        "num_interfaces": int(getattr(raw_data, "interface_assign_index", torch.empty((2, 0))).shape[1]),
        "num_crossings": int(getattr(raw_data, "crossing_assign_index", torch.empty((2, 0))).shape[1]),
        "created_at_utc": utc_now_iso(),
    }


def neurolkh_result_payload(
    *,
    instance: str,
    optimum: int,
    coords,
    result: Dict[str, Any],
    parse_sec: float,
) -> Dict[str, Any]:
    return {
        "instance": instance,
        "method": "neurolkh",
        "status": "ok",
        "groundtruth": int(optimum),
        "num_points": int(coords.shape[0]),
        "obj": result["obj"],
        "gap_pct": result["gap_pct"],
        "time_sec": result["time_sec"],
        "parse_sec": float(parse_sec),
        "end_to_end_sec": float(parse_sec) + float(result["time_sec"]),
        "feasible": bool(result["feasible"]),
        "timing": result.get("timing", {}),
        "created_at_utc": utc_now_iso(),
    }


def failure_payload(*, instance: str, method: str, optimum: int | None, exc_text: str) -> Dict[str, Any]:
    return {
        "instance": instance,
        "method": method,
        "status": "error",
        "groundtruth": (None if optimum is None else int(optimum)),
        "error": exc_text,
        "created_at_utc": utc_now_iso(),
    }


def update_combined_result(
    *,
    combined_dir: Path,
    instance: str,
    method_key: str,
    payload: Dict[str, Any],
) -> None:
    combined_path = combined_dir / f"{instance}.json"
    if combined_path.exists():
        data = load_json(combined_path)
    else:
        data = {"instance": instance}
    data[method_key] = payload
    data["updated_at_utc"] = utc_now_iso()
    save_json(combined_path, data)


def update_summary(
    *,
    summary_path: Path,
    meta: Dict[str, Any],
    guided_dir: Path,
    neurolkh_dir: Path,
    instances: Sequence[str],
    guided_key: str,
) -> None:
    payload: Dict[str, Any] = {
        "meta": meta,
        "updated_at_utc": utc_now_iso(),
        guided_key: {},
        "neurolkh": {},
    }
    for name in instances:
        guided_path = guided_dir / f"{name}.json"
        nlkh_path = neurolkh_dir / f"{name}.json"
        if guided_path.exists():
            payload[guided_key][name] = load_json(guided_path)
        if nlkh_path.exists():
            payload["neurolkh"][name] = load_json(nlkh_path)
    save_json(summary_path, payload)


def run_guided_for_instance(
    *,
    instance: str,
    tsp_path: Path,
    optimum: int,
    device: torch.device,
    logger: TeeLogger,
    heartbeat_sec: float,
    output_path: Path,
    components: Dict[str, Any],
    guided_config: GuidedLKHConfig,
    num_workers: int,
    spanner_mode: str,
    patching_mode: str,
    guided_key: str,
    lkh_exe: str,
    lkh_runs: int,
    lkh_timeout: float | None,
    use_iface_in_decode: bool,
    r: int,
    bottom_up_chunk: int,
    top_down_chunk: int,
    warm_start_max_n: int | None,
    warm_start_refine_max_n: int | None,
    warm_start_fallback_max_n: int | None,
) -> Dict[str, Any]:
    parse_t0 = time.perf_counter()
    logger.log(f"[{instance}] parse coordinates: {tsp_path}")
    coords = parse_tsp_file(str(tsp_path))
    parse_sec = time.perf_counter() - parse_t0
    points_cpu = torch.from_numpy(coords).float()

    logger.log(f"[{instance}] preprocess hierarchy: spanner={spanner_mode}, patching={patching_mode}, r={r}")
    prep = run_with_heartbeat(
        label=f"{instance} preprocess",
        logger=logger,
        heartbeat_sec=heartbeat_sec,
        fn=lambda: preprocess_points_to_hierarchy(
            points_cpu,
            r=int(r),
            num_workers=int(num_workers),
            raw_builder=components["raw_builder"],
            spanner_builder=components["spanner_builder"],
            patching_mode=str(patching_mode),
        ),
    )
    shared_timing = {"parse_sec": float(parse_sec), **prep["shared_timing"]}
    raw_data = prep["raw_data"]
    logger.log(
        (
            f"[{instance}] preprocess stats: "
            f"spanner_edges={int(prep['spanner_edge_index'].shape[1])}, "
            f"tree_nodes={int(getattr(raw_data, 'num_tree_nodes', -1))}, "
            f"interfaces={int(getattr(raw_data, 'interface_assign_index', torch.empty((2, 0))).shape[1])}, "
            f"crossings={int(getattr(raw_data, 'crossing_assign_index', torch.empty((2, 0))).shape[1])}, "
            f"parse={shared_timing['parse_sec']:.2f}s, "
            f"spanner={shared_timing['spanner_construction_sec']:.2f}s, "
            f"quadtree={shared_timing['quadtree_building_sec']:.2f}s, "
            f"patch={shared_timing['patching_sec']:.2f}s"
        )
    )

    logger.log(
        f"[{instance}] run 2-pass guided LKH with initial chunks: "
        f"bottom_up={int(bottom_up_chunk)}, top_down={int(top_down_chunk)}"
    )
    chunk_schedule = build_chunk_schedule(
        bottom_up_chunk=int(bottom_up_chunk),
        top_down_chunk=int(top_down_chunk),
    )
    guided_res = None
    used_bottom_up_chunk = int(bottom_up_chunk)
    used_top_down_chunk = int(top_down_chunk)
    guided_attempt_count = 0
    last_exc: BaseException | None = None
    for guided_attempt_count, (used_bottom_up_chunk, used_top_down_chunk) in enumerate(chunk_schedule, start=1):
        if guided_attempt_count > 1:
            logger.log(
                f"[{instance}] retry guided with smaller chunks: "
                f"bottom_up={used_bottom_up_chunk}, top_down={used_top_down_chunk}"
            )
        bu_runner = BottomUpTreeRunner(
            max_leaf_batch=int(used_bottom_up_chunk),
            max_internal_batch=int(used_bottom_up_chunk),
        )
        td_runner = TopDownTreeRunner(
            max_nodes_per_chunk=int(used_top_down_chunk),
        )
        try:
            guided_res = run_with_heartbeat(
                label=f"{instance} {guided_key}",
                logger=logger,
                heartbeat_sec=heartbeat_sec,
                fn=lambda: run_twopass_guided_instance(
                    data_cpu=prep["data_cpu"],
                    coords_np=coords,
                    device=device,
                    packer=components["packer"],
                    model_bundle=components["model_bundle"],
                    bu_runner=bu_runner,
                    td_runner=td_runner,
                    use_iface_in_decode=bool(use_iface_in_decode),
                    guided_config=guided_config,
                    lkh_exe=str(lkh_exe),
                    lkh_runs=int(lkh_runs),
                    lkh_timeout=lkh_timeout,
                    optimum=float(optimum),
                    warm_start_max_n=warm_start_max_n,
                    warm_start_refine_max_n=warm_start_refine_max_n,
                    warm_start_fallback_max_n=warm_start_fallback_max_n,
                    progress_log_fn=lambda msg: logger.log(f"[{instance}] {msg}"),
                ),
            )
            break
        except RuntimeError as exc:
            last_exc = exc
            if (
                device.type == "cuda"
                and is_cuda_oom_error(exc)
                and guided_attempt_count < len(chunk_schedule)
            ):
                logger.log(
                    f"[{instance}] CUDA OOM at chunks "
                    f"(bottom_up={used_bottom_up_chunk}, top_down={used_top_down_chunk}); "
                    "clear cache and retry"
                )
                gc.collect()
                safe_cuda_empty_cache(device, logger)
                continue
            raise
    if guided_res is None:
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("guided_res is unexpectedly None")

    payload = guided_result_payload(
        instance=instance,
        optimum=optimum,
        coords=coords,
        prep=prep,
        shared_timing=shared_timing,
        guided_res=guided_res,
        guided_key=guided_key,
        spanner_mode=spanner_mode,
        patching_mode=patching_mode,
        used_bottom_up_chunk=used_bottom_up_chunk,
        used_top_down_chunk=used_top_down_chunk,
        guided_attempt_count=guided_attempt_count,
    )
    save_json(output_path, payload)
    logger.log(
        (
            f"[{instance}] result: obj={payload['obj']}, gap={payload['gap_pct']}, "
            f"method_time={payload['time_sec']:.2f}s, end_to_end={payload['end_to_end_sec']:.2f}s"
        )
    )

    del prep, raw_data, points_cpu, coords
    gc.collect()
    safe_cuda_empty_cache(device, logger)
    return payload


def run_neurolkh_for_instance(
    *,
    instance: str,
    tsp_path: Path,
    optimum: int,
    device: torch.device,
    logger: TeeLogger,
    heartbeat_sec: float,
    output_path: Path,
    nlkh_model,
    lkh_exe: str,
    num_workers: int,
    lkh_timeout: float | None,
) -> Dict[str, Any]:
    parse_t0 = time.perf_counter()
    logger.log(f"[{instance}] parse coordinates: {tsp_path}")
    coords = parse_tsp_file(str(tsp_path))
    parse_sec = time.perf_counter() - parse_t0
    points_cpu = torch.from_numpy(coords).float()

    logger.log(f"[{instance}] run NeuroLKH...")
    result = run_with_heartbeat(
        label=f"{instance} neurolkh",
        logger=logger,
        heartbeat_sec=heartbeat_sec,
        fn=lambda: run_neurolkh_instance(
            points_cpu=points_cpu,
            device=device,
            nlkh_model=nlkh_model,
            lkh_exe=str(lkh_exe),
            num_workers=int(num_workers),
            optimum=float(optimum),
            lkh_timeout=lkh_timeout,
            progress_log_fn=lambda msg: logger.log(f"[{instance}] {msg}"),
        ),
    )
    payload = neurolkh_result_payload(
        instance=instance,
        optimum=optimum,
        coords=coords,
        result=result,
        parse_sec=float(parse_sec),
    )
    save_json(output_path, payload)
    logger.log(
        (
            f"[{instance}] result: obj={payload['obj']}, gap={payload['gap_pct']}, "
            f"time={payload['time_sec']:.2f}s, end_to_end={payload['end_to_end_sec']:.2f}s"
        )
    )

    del points_cpu, coords
    gc.collect()
    safe_cuda_empty_cache(device, logger)
    return payload


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the VLSI page-11 experiment with 2-pass+LKH and NeuroLKH. "
            "Produces per-instance logs and per-instance result JSON files."
        )
    )
    parser.add_argument("--twopass_ckpt", type=str, default="checkpoints/run_r4_20260325_034547/ckpt_best.pt")
    parser.add_argument("--neurolkh_ckpt", type=str, default="checkpoints/neurolkh_baselines/epoch_200.pt")
    parser.add_argument("--vlsi_dir", type=str, default="benchmarks/VLSI", help="directory containing *.tsp.gz files")
    parser.add_argument("--plain_dir", type=str, default="benchmarks/VLSI_plain", help="directory for decompressed *.tsp files")
    parser.add_argument("--output_dir", type=str, default="outputs/vlsi_eval", help="output root for logs and per-instance results")
    parser.add_argument("--instances", type=str, default=None, help="comma-separated subset of page11 instances")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--spanner_mode", type=str, default="delaunay", choices=("delaunay", "theta"))
    parser.add_argument("--patching_mode", type=str, default="prune", choices=("prune", "arora"))
    parser.add_argument("--theta_k", type=int, default=14)
    add_guided_lkh_args(parser)
    parser.add_argument("--num_workers", type=int, default=4, help="workers used by spanner build and NeuroLKH/LKH solve")
    parser.add_argument("--bottom_up_chunk", type=int, default=64, help="max nodes per bottom-up chunk for large-instance stability")
    parser.add_argument("--top_down_chunk", type=int, default=32, help="max nodes per top-down chunk for large-instance stability")
    parser.add_argument(
        "--warm_start_max_n",
        type=int,
        default=100000,
        help="skip greedy warm start when N exceeds this value; default disables the current warm start on page11-scale VLSI instances",
    )
    parser.add_argument(
        "--warm_start_refine_max_n",
        type=int,
        default=0,
        help="skip greedy warm-start 2-opt refinement when N exceeds this value; 0 disables refinement entirely",
    )
    parser.add_argument(
        "--warm_start_fallback_max_n",
        type=int,
        default=0,
        help="skip NN+2opt fallback warm start when N exceeds this value; 0 disables fallback entirely",
    )
    parser.add_argument("--lkh_exe", type=str, default=default_lkh_executable())
    parser.add_argument("--lkh_runs", type=int, default=1)
    parser.add_argument("--lkh_timeout", type=float, default=900.0)
    parser.add_argument("--use_iface_in_decode", type=parse_bool_arg, default=True)
    parser.add_argument("--skip_guided", action="store_true")
    parser.add_argument("--skip_neurolkh", action="store_true")
    parser.add_argument("--skip_existing_results", action="store_true", help="skip a method/instance if its JSON result already exists")
    parser.add_argument("--heartbeat_sec", type=float, default=30.0, help="heartbeat interval for long-running stages")
    args = parser.parse_args(argv)

    if args.skip_guided and args.skip_neurolkh:
        raise ValueError("Nothing to run: both --skip_guided and --skip_neurolkh were set.")

    instances = parse_instance_list(args.instances)
    unknown = [name for name in instances if name not in VLSI_PAGE11_GROUNDTRUTH]
    if unknown:
        raise ValueError(
            f"Unsupported VLSI instances for this script: {', '.join(unknown)}. "
            f"Supported page11 instances: {', '.join(VLSI_PAGE11_INSTANCE_ORDER)}"
        )

    device = resolve_device(str(args.device))
    guided_config = guided_lkh_config_from_args(args)
    lkh_exe = resolve_lkh_executable(str(args.lkh_exe))
    lkh_timeout = None if float(args.lkh_timeout) <= 0 else float(args.lkh_timeout)
    guided_key = build_guided_key(
        spanner_mode=str(args.spanner_mode),
        patching_mode=str(args.patching_mode),
    )
    guided_dirname = build_guided_dirname(
        spanner_mode=str(args.spanner_mode),
        patching_mode=str(args.patching_mode),
    )

    src_dir = Path(args.vlsi_dir)
    plain_dir = Path(args.plain_dir)
    output_dir = Path(args.output_dir)
    logs_dir = output_dir / "logs"
    guided_dir = output_dir / guided_dirname
    neurolkh_dir = output_dir / "neurolkh"
    combined_dir = output_dir / "combined"
    summary_path = output_dir / "summary.json"
    prep_meta_path = output_dir / "preprocess_status.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    guided_dir.mkdir(parents=True, exist_ok=True)
    neurolkh_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)

    prep_status = ensure_plain_vlsi_files(
        src_dir=src_dir,
        plain_dir=plain_dir,
        instances=instances,
        groundtruth=VLSI_PAGE11_GROUNDTRUTH,
    )
    save_json(prep_meta_path, prep_status)

    meta = {
        "created_at_utc": utc_now_iso(),
        "instances": instances,
        "device": str(device),
        "twopass_ckpt": str(Path(args.twopass_ckpt).resolve()),
        "neurolkh_ckpt": str(Path(args.neurolkh_ckpt).resolve()),
        "vlsi_dir": str(src_dir.resolve()),
        "plain_dir": str(plain_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "lkh_exe": str(Path(lkh_exe).resolve()),
        "r": int(args.r),
        "spanner_mode": str(args.spanner_mode),
        "patching_mode": str(args.patching_mode),
        "theta_k": int(args.theta_k),
        "guided_lkh": {
            "top_k": int(guided_config.top_k),
            "logit_scale": float(guided_config.logit_scale),
            "subgradient": bool(guided_config.subgradient),
            "max_candidates": guided_config.max_candidates,
            "max_trials": guided_config.max_trials,
            "use_initial_tour": bool(guided_config.use_initial_tour),
        },
        "num_workers": int(args.num_workers),
        "bottom_up_chunk": int(args.bottom_up_chunk),
        "top_down_chunk": int(args.top_down_chunk),
        "warm_start_max_n": int(args.warm_start_max_n),
        "warm_start_refine_max_n": int(args.warm_start_refine_max_n),
        "warm_start_fallback_max_n": int(args.warm_start_fallback_max_n),
        "lkh_runs": int(args.lkh_runs),
        "lkh_timeout": (None if lkh_timeout is None else float(lkh_timeout)),
        "heartbeat_sec": float(args.heartbeat_sec),
        "groundtruth_source": "https://www.math.uwaterloo.ca/tsp/vlsi/page11.html",
        "preprocess_status_json": str(prep_meta_path.resolve()),
    }
    update_summary(
        summary_path=summary_path,
        meta=meta,
        guided_dir=guided_dir,
        neurolkh_dir=neurolkh_dir,
        instances=instances,
        guided_key=guided_key,
    )

    if not args.skip_guided:
        print(f"[guided] loading 2-pass checkpoint on {device}: {args.twopass_ckpt}", flush=True)
        guided_components = build_twopass_components(
            ckpt_path=str(args.twopass_ckpt),
            device=device,
            r=int(args.r),
            spanner_mode=str(args.spanner_mode),
            theta_k=int(args.theta_k),
        )
        for idx, instance in enumerate(instances, start=1):
            optimum = int(VLSI_PAGE11_GROUNDTRUTH[instance])
            tsp_path = plain_dir / f"{instance}.tsp"
            result_path = guided_dir / f"{instance}.json"
            log_path = logs_dir / f"{instance}_{guided_dirname}.log"

            if args.skip_existing_results and should_skip_existing_result(result_path):
                print(f"[guided {idx}/{len(instances)}] {instance}: skip existing result {result_path}", flush=True)
                update_summary(
                    summary_path=summary_path,
                    meta=meta,
                    guided_dir=guided_dir,
                    neurolkh_dir=neurolkh_dir,
                    instances=instances,
                    guided_key=guided_key,
                )
                continue

            logger = TeeLogger(log_path)
            try:
                logger.log(f"start {guided_key} for {instance}")
                payload = run_guided_for_instance(
                    instance=instance,
                    tsp_path=tsp_path,
                    optimum=optimum,
                    device=device,
                    logger=logger,
                    heartbeat_sec=float(args.heartbeat_sec),
                    output_path=result_path,
                    components=guided_components,
                    guided_config=guided_config,
                    num_workers=int(args.num_workers),
                    spanner_mode=str(args.spanner_mode),
                    patching_mode=str(args.patching_mode),
                    guided_key=guided_key,
                    lkh_exe=str(lkh_exe),
                    lkh_runs=int(args.lkh_runs),
                    lkh_timeout=lkh_timeout,
                    use_iface_in_decode=bool(args.use_iface_in_decode),
                    r=int(args.r),
                    bottom_up_chunk=int(args.bottom_up_chunk),
                    top_down_chunk=int(args.top_down_chunk),
                    warm_start_max_n=int(args.warm_start_max_n),
                    warm_start_refine_max_n=int(args.warm_start_refine_max_n),
                    warm_start_fallback_max_n=int(args.warm_start_fallback_max_n),
                )
            except Exception:
                payload = failure_payload(
                    instance=instance,
                    method=guided_key,
                    optimum=optimum,
                    exc_text=traceback.format_exc(),
                )
                save_json(result_path, payload)
                logger.log(f"[{instance}] failed:\n{payload['error']}")
            finally:
                logger.close()

            update_combined_result(
                combined_dir=combined_dir,
                instance=instance,
                method_key=guided_key,
                payload=payload,
            )
            update_summary(
                summary_path=summary_path,
                meta=meta,
                guided_dir=guided_dir,
                neurolkh_dir=neurolkh_dir,
                instances=instances,
                guided_key=guided_key,
            )

        del guided_components
        gc.collect()
        safe_cuda_empty_cache(device)

    if not args.skip_neurolkh:
        print(f"[neurolkh] loading checkpoint on {device}: {args.neurolkh_ckpt}", flush=True)
        nlkh_model = load_neurolkh(device, str(args.neurolkh_ckpt))
        for idx, instance in enumerate(instances, start=1):
            optimum = int(VLSI_PAGE11_GROUNDTRUTH[instance])
            tsp_path = plain_dir / f"{instance}.tsp"
            result_path = neurolkh_dir / f"{instance}.json"
            log_path = logs_dir / f"{instance}_neurolkh.log"

            if args.skip_existing_results and should_skip_existing_result(result_path):
                print(f"[neurolkh {idx}/{len(instances)}] {instance}: skip existing result {result_path}", flush=True)
                update_summary(
                    summary_path=summary_path,
                    meta=meta,
                    guided_dir=guided_dir,
                    neurolkh_dir=neurolkh_dir,
                    instances=instances,
                    guided_key=guided_key,
                )
                continue

            logger = TeeLogger(log_path)
            try:
                logger.log(f"start neurolkh for {instance}")
                payload = run_neurolkh_for_instance(
                    instance=instance,
                    tsp_path=tsp_path,
                    optimum=optimum,
                    device=device,
                    logger=logger,
                    heartbeat_sec=float(args.heartbeat_sec),
                    output_path=result_path,
                    nlkh_model=nlkh_model,
                    lkh_exe=str(lkh_exe),
                    num_workers=int(args.num_workers),
                    lkh_timeout=lkh_timeout,
                )
            except Exception:
                payload = failure_payload(
                    instance=instance,
                    method="neurolkh",
                    optimum=optimum,
                    exc_text=traceback.format_exc(),
                )
                save_json(result_path, payload)
                logger.log(f"[{instance}] failed:\n{payload['error']}")
            finally:
                logger.close()

            update_combined_result(
                combined_dir=combined_dir,
                instance=instance,
                method_key="neurolkh",
                payload=payload,
            )
            update_summary(
                summary_path=summary_path,
                meta=meta,
                guided_dir=guided_dir,
                neurolkh_dir=neurolkh_dir,
                instances=instances,
                guided_key=guided_key,
            )

        del nlkh_model
        gc.collect()
        safe_cuda_empty_cache(device)

    print(f"[done] preprocess status -> {prep_meta_path}")
    print(f"[done] guided results   -> {guided_dir}")
    print(f"[done] neurolkh results -> {neurolkh_dir}")
    print(f"[done] combined        -> {combined_dir}")
    print(f"[done] summary         -> {summary_path}")


if __name__ == "__main__":
    main()
