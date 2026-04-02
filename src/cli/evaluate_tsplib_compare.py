# src/cli/evaluate_tsplib_compare.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.cli.common import (
    log_progress,
    move_data_tensors_to_device as to_device,
    parse_bool_arg,
    resolve_device,
)
from src.cli.graph_pipeline import build_spanner_builder, preprocess_points_to_hierarchy
from src.cli.eval_twopass_timing import run_guided_lkh_timed, sync_device
from src.cli.evaluate_tsplib import (
    collect_tsplib_files,
    describe_instance_presets,
    load_neurolkh,
    load_pomo,
    parse_tsp_file,
    sanitize_tag,
    select_tsplib_files,
)
from src.cli.model_factory import load_onepass_eval_models, load_twopass_eval_models
from src.graph.build_raw_pyramid import RawPyramidBuilder
from src.models.bc_state_catalog import build_boundary_state_catalog
from src.models.bottom_up_runner import BottomUpTreeRunner
from src.models.decode_backend import decode_tour, TourDecodeResult
from src.models.dp_runner import OnePassDPRunner
from src.models.edge_aggregation import aggregate_logits_to_edges
from src.models.lkh_decode import build_guided_candidates, solve_with_lkh_parallel
from src.models.node_token_packer import NodeTokenPacker
from src.models.top_down_runner import TopDownTreeRunner
from src.utils.lkh_solver import parse_tour, run_lkh, write_par, write_tour_file, write_tsp_euc2d


METHODS = (
    "onepass",
    "twopass_guided_lkh",
    "pure_lkh",
    "neurolkh",
    "pomo",
)

SHARED_TIMING_KEYS = (
    "parse_sec",
    "spanner_construction_sec",
    "quadtree_building_sec",
    "patching_sec",
)

ONEPASS_TIMING_KEYS = (
    "device_transfer_sec",
    "token_packing_sec",
    "leaf_encode_sec",
    "leaf_exact_sec",
    "internal_encode_sec",
    "parent_memory_sec",
    "decode_sec",
    "merge_dp_sec",
    "bottom_up_total_sec",
    "traceback_state_sec",
    "direct_reconstruct_sec",
    "other_overhead_sec",
    "total_sec",
)

TWOPASS_TIMING_KEYS = (
    "device_transfer_sec",
    "token_packing_sec",
    "bottom_up_sec",
    "top_down_sec",
    "score_aggregation_sec",
    "warm_start_sec",
    "candidate_construction_sec",
    "lkh_setup_io_sec",
    "lkh_search_sec",
    "lkh_parse_sec",
    "other_overhead_sec",
    "total_sec",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_run_tag(*, onepass_ckpt: str, twopass_ckpt: str, preset_tag: str, r: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    op_tag = Path(onepass_ckpt).parent.name or Path(onepass_ckpt).stem
    tp_tag = Path(twopass_ckpt).parent.name or Path(twopass_ckpt).stem
    return sanitize_tag(f"{ts}_{preset_tag}_r{r}_{op_tag}_vs_{tp_tag}")


def load_tsplib_optima(tsplib_dir: Path) -> Dict[str, float]:
    solutions_path = tsplib_dir / "solutions"
    if not solutions_path.exists():
        return {}
    out: Dict[str, float] = {}
    pattern = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*:\s*([0-9]+(?:\.[0-9]+)?)")
    with solutions_path.open("r", encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line)
            if not match:
                continue
            out[match.group(1)] = float(match.group(2))
    return out


def tsplib_euc2d_length(pos_np: np.ndarray, order: List[int]) -> Optional[float]:
    n = int(pos_np.shape[0])
    if len(order) != n or len(set(int(x) for x in order)) != n:
        return None
    total = 0.0
    for i in range(n):
        u = int(order[i])
        v = int(order[(i + 1) % n])
        dx = float(pos_np[u, 0] - pos_np[v, 0])
        dy = float(pos_np[u, 1] - pos_np[v, 1])
        total += math.floor(math.sqrt(dx * dx + dy * dy) + 0.5)
    return total


def compute_gap_pct(obj: Optional[float], optimum: Optional[float]) -> Optional[float]:
    if obj is None or optimum is None or optimum <= 0:
        return None
    return (float(obj) / float(optimum) - 1.0) * 100.0


def format_obj(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.0f}"


def format_gap(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.2f}%"


def format_time(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.2f}s"


def metric_entry(
    *,
    feasible: bool,
    obj: Optional[float],
    euclidean_length: Optional[float],
    order: Optional[List[int]],
    time_sec: float,
    optimum: Optional[float],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "feasible": bool(feasible),
        "obj": (None if obj is None else float(obj)),
        "gap_pct": compute_gap_pct(obj, optimum),
        "time_sec": float(time_sec),
        "euclidean_length": (None if euclidean_length is None else float(euclidean_length)),
        "order": ([] if order is None else [int(x) for x in order]),
    }
    if extra:
        payload.update(extra)
    return payload


def build_fieldnames() -> List[str]:
    fields = [
        "instance_index",
        "instance",
        "n",
        "opt_obj",
        "elapsed_sec",
        "completed_at_utc",
    ]
    for key in SHARED_TIMING_KEYS:
        fields.append(f"shared_{key}")
    for method in METHODS:
        fields.extend(
            [
                f"{method}_feasible",
                f"{method}_obj",
                f"{method}_gap_pct",
                f"{method}_time_sec",
            ]
        )
    fields.append("onepass_dp_cost")
    fields.append("onepass_num_sigma_total")
    for key in ONEPASS_TIMING_KEYS:
        fields.append(f"onepass_{key}")
    for key in TWOPASS_TIMING_KEYS:
        fields.append(f"twopass_{key}")
    return fields


def flatten_record(record: Dict[str, Any]) -> Dict[str, str]:
    row: Dict[str, str] = {
        "instance_index": str(record["instance_index"]),
        "instance": str(record["instance"]),
        "n": str(record["n"]),
        "opt_obj": "" if record["opt_obj"] is None else f"{float(record['opt_obj']):.0f}",
        "elapsed_sec": f"{float(record['elapsed_sec']):.3f}",
        "completed_at_utc": str(record["completed_at_utc"]),
    }
    shared = record["shared_timing"]
    for key in SHARED_TIMING_KEYS:
        row[f"shared_{key}"] = f"{float(shared.get(key, 0.0)):.6f}"
    for method in METHODS:
        payload = record.get(method, {})
        row[f"{method}_feasible"] = str(bool(payload.get("feasible", False)))
        obj = payload.get("obj")
        gap = payload.get("gap_pct")
        row[f"{method}_obj"] = "" if obj is None else f"{float(obj):.0f}"
        row[f"{method}_gap_pct"] = "" if gap is None else f"{float(gap):.6f}"
        row[f"{method}_time_sec"] = f"{float(payload.get('time_sec', 0.0)):.6f}"
    row["onepass_dp_cost"] = "" if record["onepass"].get("dp_cost") is None else f"{float(record['onepass']['dp_cost']):.6f}"
    row["onepass_num_sigma_total"] = str(int(record["onepass"].get("num_sigma_total", 0)))
    for key in ONEPASS_TIMING_KEYS:
        row[f"onepass_{key}"] = f"{float(record['onepass']['timing'].get(key, 0.0)):.6f}"
    for key in TWOPASS_TIMING_KEYS:
        row[f"twopass_{key}"] = f"{float(record['twopass_guided_lkh']['timing'].get(key, 0.0)):.6f}"
    return row


def write_summary(path: Path, metadata: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
    summary: Dict[str, Any] = {
        "meta": metadata,
        "num_instances": len(records),
        "methods": {},
        "shared_timing_avg_sec": {},
        "onepass_timing_avg_sec": {},
        "twopass_timing_avg_sec": {},
        "records": records,
    }

    if records:
        for key in SHARED_TIMING_KEYS:
            summary["shared_timing_avg_sec"][key] = float(np.mean([rec["shared_timing"].get(key, 0.0) for rec in records]))
        for key in ONEPASS_TIMING_KEYS:
            summary["onepass_timing_avg_sec"][key] = float(np.mean([rec["onepass"]["timing"].get(key, 0.0) for rec in records]))
        for key in TWOPASS_TIMING_KEYS:
            summary["twopass_timing_avg_sec"][key] = float(np.mean([rec["twopass_guided_lkh"]["timing"].get(key, 0.0) for rec in records]))

    for method in METHODS:
        feasible = [rec[method] for rec in records if rec.get(method, {}).get("obj") is not None]
        summary["methods"][method] = {
            "num_feasible": len(feasible),
            "avg_obj": (None if not feasible else float(np.mean([item["obj"] for item in feasible]))),
            "avg_gap_pct": (
                None
                if not feasible or not any(item.get("gap_pct") is not None for item in feasible)
                else float(np.mean([item["gap_pct"] for item in feasible if item.get("gap_pct") is not None]))
            ),
            "avg_time_sec": (None if not feasible else float(np.mean([item["time_sec"] for item in feasible]))),
        }

    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def run_pure_lkh(points_cpu: torch.Tensor, *, optimum: Optional[float], lkh_exe: str) -> Dict[str, Any]:
    res, _ = solve_with_lkh_parallel(
        [{"pos": points_cpu, "mode": "pure", "teacher_len": optimum or 0.0}],
        lkh_executable=lkh_exe,
        num_workers=1,
    )[0]
    pos_np = points_cpu.detach().cpu().numpy()
    obj = tsplib_euc2d_length(pos_np, list(res.order)) if res.feasible else None
    return metric_entry(
        feasible=bool(res.feasible),
        obj=obj,
        euclidean_length=(None if not res.feasible else float(res.length)),
        order=(list(res.order) if res.feasible else None),
        time_sec=float(res.duration),
        optimum=optimum,
    )


def _build_exact_knn_neighbors(
    *,
    points_cpu: torch.Tensor,
    k: int,
    num_workers: int,
    progress_log_fn=None,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.neighbors import NearestNeighbors

    pos_np = points_cpu.detach().cpu().numpy()
    n_nodes = int(pos_np.shape[0])
    if n_nodes <= 1:
        raise ValueError("NeuroLKH requires at least 2 nodes.")

    target_k = min(max(1, int(k)), n_nodes - 1)
    query_k = min(n_nodes, target_k + 1)
    if progress_log_fn is not None:
        progress_log_fn(f"neurolkh stage: exact {target_k}-NN graph")

    nn = NearestNeighbors(
        n_neighbors=query_k,
        algorithm=("kd_tree" if pos_np.shape[1] <= 3 else "auto"),
        n_jobs=max(1, int(num_workers)),
    )
    nn.fit(pos_np)
    distances, indices = nn.kneighbors(pos_np, return_distance=True)

    if query_k == target_k + 1 and np.all(indices[:, 0] == np.arange(n_nodes)):
        neighbors = indices[:, 1 : target_k + 1]
        dists = distances[:, 1 : target_k + 1]
    else:
        neighbors = np.full((n_nodes, target_k), -1, dtype=np.int64)
        dists = np.full((n_nodes, target_k), np.inf, dtype=np.float32)
        for i in range(n_nodes):
            keep = indices[i] != i
            row_idx = indices[i][keep][:target_k]
            row_dist = distances[i][keep][:target_k]
            take = int(row_idx.shape[0])
            neighbors[i, :take] = row_idx
            dists[i, :take] = row_dist
        if (neighbors < 0).any():
            raise ValueError("Failed to build a full fixed-width KNN graph for NeuroLKH inputs.")

    return neighbors.astype(np.int64, copy=False), dists.astype(np.float32, copy=False)


def _build_inverse_edge_index(neighbors: np.ndarray, *, progress_log_fn=None) -> np.ndarray:
    n_nodes, n_edges = neighbors.shape
    sentinel = n_nodes * n_edges
    inverse = np.full((n_nodes, n_edges), sentinel, dtype=np.int64)
    row_ids = np.arange(n_nodes, dtype=np.int64)

    if progress_log_fn is not None:
        progress_log_fn("neurolkh stage: inverse edge index")

    for pos in range(n_edges):
        targets = neighbors[:, pos]
        target_rows = neighbors[targets]
        matches = target_rows == row_ids[:, None]
        found = matches.any(axis=1)
        if np.any(found):
            inverse[found, pos] = (
                targets[found].astype(np.int64) * n_edges
                + matches[found].argmax(axis=1).astype(np.int64)
            )
    return inverse


def _write_candidate_file_from_arrays(path: str, neighbors: np.ndarray, alphas: np.ndarray) -> None:
    n_nodes, _ = neighbors.shape
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{n_nodes}\n")
        for i in range(n_nodes):
            row_neighbors = neighbors[i]
            row_alphas = alphas[i]
            valid = row_neighbors >= 0
            row_neighbors = row_neighbors[valid]
            row_alphas = row_alphas[valid]
            order = np.lexsort((row_neighbors, row_alphas))
            row_neighbors = row_neighbors[order]
            row_alphas = row_alphas[order]

            line = [str(i + 1), "0", str(int(row_neighbors.shape[0]))]
            for nb, alpha in zip(row_neighbors.tolist(), row_alphas.tolist()):
                line.extend([str(int(nb) + 1), str(int(alpha))])
            f.write(" ".join(line) + "\n")
        f.write("-1\n")


def _run_candidate_lkh_timed_from_arrays(
    *,
    pos: torch.Tensor,
    neighbors: np.ndarray,
    alphas: np.ndarray,
    initial_tour: Optional[List[int]],
    lkh_executable: str,
    num_runs: int,
    seed: int,
    timeout: Optional[float],
) -> tuple[Any, Dict[str, float]]:
    from src.models.lkh_decode import LKHDecodeResult

    pos_cpu = pos.detach().cpu()
    pos_np = pos_cpu.numpy()

    tmp_dir = tempfile.mkdtemp(prefix="timed_candidate_lkh_")
    tsp_path = str(Path(tmp_dir) / "problem.tsp")
    par_path = str(Path(tmp_dir) / "config.par")
    tour_path = str(Path(tmp_dir) / "result.tour")
    cand_path = str(Path(tmp_dir) / "candidates.cand")
    init_tour_path = str(Path(tmp_dir) / "initial.itour")

    setup_t0 = time.perf_counter()
    try:
        write_tsp_euc2d(tsp_path, "GuidedTSP", pos_np)
        _write_candidate_file_from_arrays(cand_path, neighbors, alphas)

        init_p = None
        if initial_tour is not None:
            write_tour_file(init_tour_path, [int(x) for x in initial_tour])
            init_p = init_tour_path

        write_par(
            par_path,
            tsp_path,
            tour_path,
            runs=max(1, int(num_runs)),
            seed=int(seed),
            precision=1,
            candidate_path=cand_path,
            initial_tour_path=init_p,
            subgradient=False,
            max_candidates=5,
            max_trials=1,
        )
        setup_sec = time.perf_counter() - setup_t0

        search_t0 = time.perf_counter()
        ok = run_lkh(lkh_executable, par_path, timeout=timeout)
        search_sec = time.perf_counter() - search_t0

        parse_t0 = time.perf_counter()
        order = parse_tour(tour_path) if ok else []
        feasible = len(order) == int(pos_cpu.shape[0]) and len(set(order)) == int(pos_cpu.shape[0])
        length = float("inf")
        if feasible:
            total = 0.0
            n = len(order)
            for i in range(n):
                u = order[i]
                v = order[(i + 1) % n]
                total += float(np.linalg.norm(pos_np[u] - pos_np[v]))
            length = total
        parse_sec = time.perf_counter() - parse_t0

        result = LKHDecodeResult(
            order=order,
            length=length,
            feasible=feasible,
            mode="guided",
            duration=setup_sec + search_sec + parse_sec,
        )
        return result, {
            "lkh_setup_io_sec": float(setup_sec),
            "lkh_search_sec": float(search_sec),
            "lkh_parse_sec": float(parse_sec),
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_pomo_instance(
    *,
    points_cpu: torch.Tensor,
    device: torch.device,
    pomo_model,
    pomo_env,
    optimum: Optional[float],
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    with torch.no_grad():
        n = int(points_cpu.shape[0])
        pomo_env.problem_size = n
        pomo_env.pomo_size = n
        pomo_env.set_problems(points_cpu.unsqueeze(0).to(device) / 10000.0)
        reset_state, _, _ = pomo_env.reset()
        pomo_model.pre_forward(reset_state)
        state, reward, done = pomo_env.pre_step()
        while not done:
            selected, _ = pomo_model(state)
            state, reward, done = pomo_env.step(selected)
        sync_device(device)
        best_reward, best_idx = reward.max(dim=1)
        best_pomo_idx = int(best_idx.item())
        best_order = pomo_env.selected_node_list[0, best_pomo_idx].detach().cpu().tolist()
        euclidean_length = float(-best_reward.item() * 10000.0)
    duration = time.perf_counter() - t0
    pos_np = points_cpu.detach().cpu().numpy()
    obj = tsplib_euc2d_length(pos_np, best_order)
    feasible = obj is not None
    return metric_entry(
        feasible=feasible,
        obj=obj,
        euclidean_length=(euclidean_length if feasible else None),
        order=(best_order if feasible else None),
        time_sec=duration,
        optimum=optimum,
    )


def run_neurolkh_instance(
    *,
    points_cpu: torch.Tensor,
    device: torch.device,
    nlkh_model,
    lkh_exe: str,
    num_workers: int,
    optimum: Optional[float],
    lkh_timeout: Optional[float] = None,
    progress_log_fn=None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    timing: Dict[str, float] = {
        "knn_sec": 0.0,
        "inverse_edge_index_sec": 0.0,
        "model_inference_sec": 0.0,
        "candidate_prep_sec": 0.0,
        "lkh_setup_io_sec": 0.0,
        "lkh_search_sec": 0.0,
        "lkh_parse_sec": 0.0,
        "total_sec": 0.0,
    }

    knn_t0 = time.perf_counter()
    n_nodes = int(points_cpu.shape[0])
    n_edges = min(20, max(n_nodes - 1, 1))
    neighbor_index, neighbor_dist = _build_exact_knn_neighbors(
        points_cpu=points_cpu,
        k=n_edges,
        num_workers=num_workers,
        progress_log_fn=progress_log_fn,
    )
    timing["knn_sec"] = time.perf_counter() - knn_t0

    inv_t0 = time.perf_counter()
    inverse_edge_index = _build_inverse_edge_index(
        neighbor_index,
        progress_log_fn=progress_log_fn,
    )
    timing["inverse_edge_index_sec"] = time.perf_counter() - inv_t0

    model_t0 = time.perf_counter()
    with torch.no_grad():
        node_feat = points_cpu.unsqueeze(0).to(device) / 10000.0
        e_idx_nlkh = torch.from_numpy(neighbor_index.reshape(1, -1)).to(device=device, dtype=torch.long)
        e_feat_nlkh = torch.from_numpy(neighbor_dist.reshape(1, -1, 1)).to(device=device, dtype=torch.float32)
        inv_e_idx = torch.from_numpy(inverse_edge_index.reshape(1, -1)).to(device=device, dtype=torch.long)

        y_pred_edges_log, _, _ = nlkh_model(node_feat, e_feat_nlkh, e_idx_nlkh, inv_e_idx, None, None, n_edges)
    sync_device(device)
    timing["model_inference_sec"] = time.perf_counter() - model_t0

    cand_t0 = time.perf_counter()
    if progress_log_fn is not None:
        progress_log_fn("neurolkh stage: candidate alpha preparation")
    nlkh_edge_logit = y_pred_edges_log[0, :, 1].view(n_nodes, n_edges).detach().cpu().numpy()
    finite_mask = np.isfinite(nlkh_edge_logit)
    max_v = float(np.max(nlkh_edge_logit[finite_mask])) if np.any(finite_mask) else 0.0
    alpha_array = np.full((n_nodes, n_edges), 10**9, dtype=np.int64)
    if np.any(finite_mask):
        alpha_array[finite_mask] = np.rint(1e3 * (max_v - nlkh_edge_logit[finite_mask])).astype(np.int64)
    timing["candidate_prep_sec"] = time.perf_counter() - cand_t0

    if progress_log_fn is not None:
        progress_log_fn("neurolkh stage: LKH search")
    nlkh_res, lkh_break = _run_candidate_lkh_timed_from_arrays(
        pos=points_cpu,
        neighbors=neighbor_index,
        alphas=alpha_array,
        initial_tour=None,
        lkh_executable=lkh_exe,
        num_runs=1,
        seed=1234,
        timeout=lkh_timeout,
    )
    timing["lkh_setup_io_sec"] = float(lkh_break.get("lkh_setup_io_sec", 0.0))
    timing["lkh_search_sec"] = float(lkh_break.get("lkh_search_sec", 0.0))
    timing["lkh_parse_sec"] = float(lkh_break.get("lkh_parse_sec", 0.0))

    duration = time.perf_counter() - t0
    timing["total_sec"] = float(duration)
    pos_np = points_cpu.detach().cpu().numpy()
    obj = tsplib_euc2d_length(pos_np, list(nlkh_res.order)) if nlkh_res.feasible else None
    del node_feat, e_idx_nlkh, e_feat_nlkh, inv_e_idx, y_pred_edges_log
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metric_entry(
        feasible=bool(nlkh_res.feasible),
        obj=obj,
        euclidean_length=(None if not nlkh_res.feasible else float(nlkh_res.length)),
        order=(list(nlkh_res.order) if nlkh_res.feasible else None),
        time_sec=duration,
        optimum=optimum,
        extra={"timing": timing},
    )


def run_onepass_instance(
    *,
    data_cpu,
    coords_np: np.ndarray,
    device: torch.device,
    packer: NodeTokenPacker,
    runner: OnePassDPRunner,
    model_bundle,
    catalog,
    optimum: Optional[float],
) -> Dict[str, Any]:
    timing = {key: 0.0 for key in ONEPASS_TIMING_KEYS}
    cpu_device = torch.device("cpu")

    xfer_t0 = time.perf_counter()
    data_dev = to_device(data_cpu, device)
    sync_device(device)
    timing["device_transfer_sec"] = time.perf_counter() - xfer_t0

    pack_t0 = time.perf_counter()
    with torch.no_grad():
        packed = packer.pack_batch([data_dev])
        packed = to_device(packed, device)
    sync_device(device)
    timing["token_packing_sec"] = time.perf_counter() - pack_t0

    run_t0 = time.perf_counter()
    with torch.no_grad():
        dp_result = runner.run_single(
            tokens=packed.tokens,
            leaves=packed.leaves,
            leaf_encoder=model_bundle.leaf_encoder,
            merge_encoder=model_bundle.merge_encoder,
            merge_decoder=model_bundle.merge_decoder,
            catalog=catalog,
        )
    sync_device(device)
    timing["total_sec"] = time.perf_counter() - run_t0

    stats = dict(dp_result.stats)
    timing["leaf_encode_sec"] = float(stats.get("leaf_encode_sec", 0.0))
    timing["leaf_exact_sec"] = float(stats.get("leaf_exact_sec", 0.0))
    timing["internal_encode_sec"] = float(stats.get("internal_encode_sec", 0.0))
    timing["parent_memory_sec"] = float(stats.get("parent_memory_sec", 0.0))
    timing["decode_sec"] = float(stats.get("decode_sec", 0.0))
    timing["merge_dp_sec"] = float(stats.get("merge_dp_sec", 0.0))
    timing["bottom_up_total_sec"] = float(stats.get("bottom_up_total_sec", 0.0))
    timing["traceback_state_sec"] = float(stats.get("traceback_state_sec", 0.0))
    timing["direct_reconstruct_sec"] = float(stats.get("direct_reconstruct_time_s", 0.0))
    timing["other_overhead_sec"] = max(
        0.0,
        timing["total_sec"]
        - (
            timing["bottom_up_total_sec"]
            + timing["traceback_state_sec"]
            + timing["direct_reconstruct_sec"]
        ),
    )

    obj = tsplib_euc2d_length(coords_np, list(dp_result.tour_order)) if dp_result.tour_feasible else None
    payload = metric_entry(
        feasible=bool(dp_result.tour_feasible),
        obj=obj,
        euclidean_length=(None if not dp_result.tour_feasible else float(dp_result.tour_length)),
        order=(list(dp_result.tour_order) if dp_result.tour_feasible else None),
        time_sec=timing["total_sec"],
        optimum=optimum,
        extra={
            "dp_cost": (None if not math.isfinite(float(dp_result.tour_cost)) else float(dp_result.tour_cost)),
            "num_sigma_total": int(stats.get("num_sigma_total", 0.0)),
            "timing": timing,
            "dp_stats": stats,
            "tour_stats": dict(dp_result.tour_stats),
        },
    )

    to_device(data_dev, cpu_device)
    to_device(packed, cpu_device)
    del packed, dp_result, data_dev
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def run_twopass_guided_instance(
    *,
    data_cpu,
    coords_np: np.ndarray,
    device: torch.device,
    packer: NodeTokenPacker,
    model_bundle,
    bu_runner: BottomUpTreeRunner,
    td_runner: TopDownTreeRunner,
    use_iface_in_decode: bool,
    guided_top_k: int,
    lkh_exe: str,
    lkh_runs: int,
    lkh_timeout: Optional[float],
    optimum: Optional[float],
    warm_start_max_n: Optional[int] = None,
    warm_start_refine_max_n: Optional[int] = None,
    warm_start_fallback_max_n: Optional[int] = None,
    progress_log_fn=None,
) -> Dict[str, Any]:
    timing = {key: 0.0 for key in TWOPASS_TIMING_KEYS}
    cpu_device = torch.device("cpu")
    progress = progress_log_fn

    if progress is not None:
        progress("guided stage: transfer/pack")

    sample_t0 = time.perf_counter()
    xfer_t0 = time.perf_counter()
    data_dev = to_device(data_cpu, device)
    sync_device(device)
    timing["device_transfer_sec"] = time.perf_counter() - xfer_t0

    pack_t0 = time.perf_counter()
    with torch.no_grad():
        packed = packer.pack_batch([data_dev])
    sync_device(device)
    timing["token_packing_sec"] = time.perf_counter() - pack_t0

    bu_t0 = time.perf_counter()
    if progress is not None:
        progress("guided stage: bottom-up")
    with torch.no_grad():
        out_bu = bu_runner.run_batch(
            batch=packed,
            leaf_encoder=model_bundle.leaf_encoder,
            merge_encoder=model_bundle.merge_encoder,
        )
    sync_device(device)
    timing["bottom_up_sec"] = time.perf_counter() - bu_t0

    td_t0 = time.perf_counter()
    if progress is not None:
        progress("guided stage: top-down")
    with torch.no_grad():
        out_td = td_runner.run_batch(
            packed=packed,
            z=out_bu.z,
            decoder=model_bundle.decoder,
        )
    sync_device(device)
    timing["top_down_sec"] = time.perf_counter() - td_t0

    agg_t0 = time.perf_counter()
    if progress is not None:
        progress("guided stage: score aggregation")
    with torch.no_grad():
        edge_scores = aggregate_logits_to_edges(
            tokens=packed.tokens,
            cross_logit=out_td.cross_logit,
            iface_logit=out_td.iface_logit if use_iface_in_decode else None,
            reduce="mean",
            num_edges=data_dev.spanner_edge_index.shape[1],
        )
        el = edge_scores.edge_logit.clone()
        em = edge_scores.edge_mask.bool()
        el[~em] = -1e9
    sync_device(device)
    timing["score_aggregation_sec"] = time.perf_counter() - agg_t0

    warm_t0 = time.perf_counter()
    skip_warm_start = (
        warm_start_max_n is not None
        and int(warm_start_max_n) >= 0
        and int(data_dev.pos.shape[0]) > int(warm_start_max_n)
    )
    if skip_warm_start:
        if progress is not None:
            progress(
                f"guided stage: skip warm start "
                f"(N={int(data_dev.pos.shape[0])} > warm_start_max_n={int(warm_start_max_n)})"
            )
        greedy_res = TourDecodeResult(
            order=[],
            length=float("inf"),
            feasible=False,
            num_off_spanner_edges=0,
            num_components_initial=0,
            num_edges_broken=0,
            fallback_used=False,
            num_patching_steps=0,
            duration=0.0,
        )
        timing["warm_start_sec"] = 0.0
    else:
        if progress is not None:
            progress("guided stage: greedy warm start")
        greedy_res = decode_tour(
            pos=data_dev.pos.detach().cpu(),
            spanner_edge_index=data_dev.spanner_edge_index.detach().cpu(),
            edge_logit=el.detach().cpu(),
            backend="greedy",
            prefer_spanner_only=True,
            allow_off_spanner_patch=True,
            greedy_refine_max_n=warm_start_refine_max_n,
            greedy_fallback_max_n=warm_start_fallback_max_n,
        )
        timing["warm_start_sec"] = time.perf_counter() - warm_t0

    cand_t0 = time.perf_counter()
    if progress is not None:
        progress("guided stage: candidate construction")
    candidates = build_guided_candidates(
        num_nodes=int(data_dev.pos.shape[0]),
        edge_index=data_dev.spanner_edge_index.detach().cpu(),
        edge_logit=el.detach().cpu(),
        logit_scale=1e3,
        top_k=int(guided_top_k),
    )
    timing["candidate_construction_sec"] = time.perf_counter() - cand_t0

    if progress is not None:
        progress("guided stage: LKH search")
    guided_res, lkh_break = run_guided_lkh_timed(
        pos=data_dev.pos.detach().cpu(),
        candidates=candidates,
        initial_tour=greedy_res.order if greedy_res.feasible else None,
        lkh_executable=lkh_exe,
        num_runs=int(lkh_runs),
        seed=1234,
        timeout=lkh_timeout,
    )
    timing["lkh_setup_io_sec"] = float(lkh_break.get("lkh_setup_io_sec", 0.0))
    timing["lkh_search_sec"] = float(lkh_break.get("lkh_search_sec", 0.0))
    timing["lkh_parse_sec"] = float(lkh_break.get("lkh_parse_sec", 0.0))
    timing["total_sec"] = time.perf_counter() - sample_t0
    timing["other_overhead_sec"] = max(
        0.0,
        timing["total_sec"]
        - (
            timing["device_transfer_sec"]
            + timing["token_packing_sec"]
            + timing["bottom_up_sec"]
            + timing["top_down_sec"]
            + timing["score_aggregation_sec"]
            + timing["warm_start_sec"]
            + timing["candidate_construction_sec"]
            + timing["lkh_setup_io_sec"]
            + timing["lkh_search_sec"]
            + timing["lkh_parse_sec"]
        ),
    )

    obj = tsplib_euc2d_length(coords_np, list(guided_res.order)) if guided_res.feasible else None
    payload = metric_entry(
        feasible=bool(guided_res.feasible),
        obj=obj,
        euclidean_length=(None if not guided_res.feasible else float(guided_res.length)),
        order=(list(guided_res.order) if guided_res.feasible else None),
        time_sec=timing["total_sec"],
        optimum=optimum,
        extra={
            "timing": timing,
            "warm_start_feasible": bool(greedy_res.feasible),
            "warm_start_length": (None if not greedy_res.feasible else float(greedy_res.length)),
        },
    )

    to_device(data_dev, cpu_device)
    to_device(packed, cpu_device)
    del packed, out_bu, out_td, edge_scores, el, em, data_dev
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def print_summary(records: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 90)
    print("TSPLIB 100-500 Comparison Summary")
    print(f"{'Method':<20} {'Feasible':>10} {'Avg Obj':>12} {'Avg Gap':>10} {'Avg Time':>12}")
    print("-" * 90)
    for method in METHODS:
        feasible = [rec[method] for rec in records if rec.get(method, {}).get("obj") is not None]
        if not feasible:
            print(f"{method:<20} {0:>10} {'N/A':>12} {'N/A':>10} {'N/A':>12}")
            continue
        avg_obj = float(np.mean([x["obj"] for x in feasible]))
        gaps = [x["gap_pct"] for x in feasible if x.get("gap_pct") is not None]
        avg_gap = None if not gaps else float(np.mean(gaps))
        avg_time = float(np.mean([x["time_sec"] for x in feasible]))
        print(
            f"{method:<20} {len(feasible):>10} "
            f"{avg_obj:>12.0f} "
            f"{(f'{avg_gap:.2f}%' if avg_gap is not None else 'N/A'):>10} "
            f"{avg_time:>11.2f}s"
        )
    print("=" * 90)

    if records:
        onepass_avg = {
            key: float(np.mean([rec["onepass"]["timing"].get(key, 0.0) for rec in records]))
            for key in ONEPASS_TIMING_KEYS
        }
        twopass_avg = {
            key: float(np.mean([rec["twopass_guided_lkh"]["timing"].get(key, 0.0) for rec in records]))
            for key in TWOPASS_TIMING_KEYS
        }
        print("Average 1-pass timing (s):")
        print(
            f"  transfer={onepass_avg['device_transfer_sec']:.3f}, "
            f"pack={onepass_avg['token_packing_sec']:.3f}, "
            f"bottom_up={onepass_avg['bottom_up_total_sec']:.3f}, "
            f"traceback={onepass_avg['traceback_state_sec']:.3f}, "
            f"reconstruct={onepass_avg['direct_reconstruct_sec']:.3f}, "
            f"total={onepass_avg['total_sec']:.3f}"
        )
        print("Average 2-pass timing (s):")
        print(
            f"  transfer={twopass_avg['device_transfer_sec']:.3f}, "
            f"pack={twopass_avg['token_packing_sec']:.3f}, "
            f"bottom_up={twopass_avg['bottom_up_sec']:.3f}, "
            f"top_down={twopass_avg['top_down_sec']:.3f}, "
            f"score_agg={twopass_avg['score_aggregation_sec']:.3f}, "
            f"warm_start={twopass_avg['warm_start_sec']:.3f}, "
            f"candidate={twopass_avg['candidate_construction_sec']:.3f}, "
            f"lkh_search={twopass_avg['lkh_search_sec']:.3f}, "
            f"total={twopass_avg['total_sec']:.3f}"
        )


def main(argv: Optional[List[str]] = None) -> None:
    from src.utils.lkh_solver import default_lkh_executable, resolve_lkh_executable

    parser = argparse.ArgumentParser(
        description="Compare 1-pass, 2-pass+LKH, LKH, NeuroLKH, and POMO on TSPLIB instances."
    )
    parser.add_argument("--onepass_ckpt", type=str, required=False, help="path to 1-pass checkpoint")
    parser.add_argument("--twopass_ckpt", type=str, required=False, help="path to 2-pass checkpoint")
    parser.add_argument("--tsplib_dir", type=str, default="benchmarks/tsplib", help="directory with TSPLIB .tsp files")
    parser.add_argument("--instance_preset", type=str, default="n100_500", help="TSPLIB instance preset; " + describe_instance_presets())
    parser.add_argument("--instances", type=str, default=None, help="comma-separated explicit TSPLIB instance names")
    parser.add_argument("--num_instances", type=int, default=10, help="fallback count when using largest-K selection")
    parser.add_argument("--list_instance_presets", action="store_true", help="print supported TSPLIB presets and exit")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--lkh_exe", type=str, default=default_lkh_executable())
    parser.add_argument("--lkh_runs", type=int, default=1)
    parser.add_argument("--lkh_timeout", type=float, default=0.0)
    parser.add_argument("--guided_top_k", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=1, help="workers used by the spanner builder and optional baselines")
    parser.add_argument("--spanner_mode", type=str, default="delaunay", choices=("delaunay", "theta"))
    parser.add_argument("--theta_k", type=int, default=14, help="theta spanner cone count when --spanner_mode theta")
    parser.add_argument("--patching_mode", type=str, default="prune", choices=("prune", "arora"))
    parser.add_argument("--use_iface_in_decode", type=parse_bool_arg, default=True)
    parser.add_argument("--pomo_ckpt", type=str, default=None, help="path to POMO checkpoint")
    parser.add_argument("--neurolkh_ckpt", type=str, default=None, help="path to NeuroLKH checkpoint")
    parser.add_argument("--save_dir", type=str, default="outputs/eval_tsplib_compare")
    parser.add_argument("--run_tag", type=str, default=None)

    parser.add_argument("--dp_max_used", type=int, default=4)
    parser.add_argument("--dp_topk", type=int, default=5)
    parser.add_argument("--dp_max_sigma", type=int, default=0)
    parser.add_argument("--dp_child_catalog_cap", type=int, default=0)
    parser.add_argument("--dp_fallback_exact", type=parse_bool_arg, default=True)
    parser.add_argument("--dp_leaf_workers", type=int, default=16)
    parser.add_argument("--dp_parse_mode", type=str, default="catalog_enum", choices=("catalog_enum", "heuristic"))

    args = parser.parse_args(argv)

    if args.list_instance_presets:
        print(describe_instance_presets())
        return
    if not args.onepass_ckpt or not args.twopass_ckpt:
        parser.error("--onepass_ckpt and --twopass_ckpt are required unless --list_instance_presets is used")

    device = resolve_device(str(args.device))
    lkh_exe = resolve_lkh_executable(str(args.lkh_exe))
    lkh_timeout = None if float(args.lkh_timeout) <= 0 else float(args.lkh_timeout)
    print(f"[env] device={device}")
    print(f"[lkh] executable={lkh_exe}")

    print(f"[ckpt] loading 1-pass from {args.onepass_ckpt}")
    onepass_bundle = load_onepass_eval_models(
        ckpt_path=str(args.onepass_ckpt),
        device=device,
        r=int(args.r),
        default_matching_max_used=int(args.dp_max_used),
    )
    if not onepass_bundle.merge_decoder_loaded:
        print("[warn] 1-pass checkpoint has no merge_decoder; results will be invalid.")

    print(f"[ckpt] loading 2-pass from {args.twopass_ckpt}")
    twopass_bundle = load_twopass_eval_models(
        ckpt_path=str(args.twopass_ckpt),
        device=device,
        r=int(args.r),
    )

    onepass_packer = NodeTokenPacker(
        r=int(args.r),
        state_mode="matching",
        matching_max_used=onepass_bundle.matching_max_used,
    )
    twopass_packer = NodeTokenPacker(
        r=int(args.r),
        state_mode=twopass_bundle.state_mode,
        matching_max_used=twopass_bundle.matching_max_used,
    )
    onepass_catalog = build_boundary_state_catalog(
        num_slots=onepass_bundle.num_iface_slots,
        max_used=onepass_bundle.matching_max_used,
        device=device,
    )
    onepass_runner = OnePassDPRunner(
        r=int(args.r),
        max_used=onepass_bundle.matching_max_used,
        topk=int(args.dp_topk),
        max_sigma_enumerate=int(args.dp_max_sigma),
        max_child_catalog_states=int(args.dp_child_catalog_cap),
        fallback_exact=bool(args.dp_fallback_exact),
        num_leaf_workers=int(args.dp_leaf_workers),
        parse_mode=str(args.dp_parse_mode),
    )
    bu_runner = BottomUpTreeRunner()
    td_runner = TopDownTreeRunner()
    spanner_builder = build_spanner_builder(
        spanner_mode=str(args.spanner_mode),
        theta_k=int(args.theta_k),
    )
    raw_builder = RawPyramidBuilder(max_points_per_leaf=4, max_depth=20)

    pomo_model = None
    pomo_env = None
    if args.pomo_ckpt and os.path.exists(args.pomo_ckpt):
        print(f"[baseline] loading POMO from {args.pomo_ckpt}")
        pomo_model, pomo_env = load_pomo(device, str(args.pomo_ckpt))
    else:
        print("[warn] POMO checkpoint missing; POMO results will be N/A.")

    nlkh_model = None
    if args.neurolkh_ckpt and os.path.exists(args.neurolkh_ckpt):
        print(f"[baseline] loading NeuroLKH from {args.neurolkh_ckpt}")
        nlkh_model = load_neurolkh(device, str(args.neurolkh_ckpt))
    else:
        print("[warn] NeuroLKH checkpoint missing; NeuroLKH results will be N/A.")

    tsplib_path = Path(args.tsplib_dir)
    all_tsp_files = collect_tsplib_files(tsplib_path)
    tsp_files, selection_desc, selection_tag = select_tsplib_files(
        all_tsp_files=all_tsp_files,
        instance_preset=args.instance_preset,
        instance_names=args.instances,
        num_instances=int(args.num_instances),
    )
    print(f"[data] Selected {selection_desc} from {len(all_tsp_files)} total instances.")

    optima = load_tsplib_optima(tsplib_path)

    if args.run_tag:
        run_tag = sanitize_tag(str(args.run_tag))
    else:
        run_tag = build_run_tag(
            onepass_ckpt=str(args.onepass_ckpt),
            twopass_ckpt=str(args.twopass_ckpt),
            preset_tag=selection_tag,
            r=int(args.r),
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / f"{run_tag}.csv"
    jsonl_path = save_dir / f"{run_tag}.jsonl"
    summary_path = save_dir / f"{run_tag}_summary.json"
    fieldnames = build_fieldnames()
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    jsonl_path.write_text("", encoding="utf-8")

    skipped_instances: List[Dict[str, Any]] = []
    metadata = {
        "created_at_utc": utc_now_iso(),
        "onepass_ckpt": str(Path(args.onepass_ckpt).resolve()),
        "twopass_ckpt": str(Path(args.twopass_ckpt).resolve()),
        "tsplib_dir": str(tsplib_path.resolve()),
        "selection_desc": selection_desc,
        "instance_preset": args.instance_preset,
        "instances": args.instances,
        "num_instances_selected": len(tsp_files),
        "num_instances_skipped": 0,
        "skipped_instances": skipped_instances,
        "r": int(args.r),
        "device": str(device),
        "lkh_exe": str(lkh_exe),
        "lkh_runs": int(args.lkh_runs),
        "lkh_timeout": (None if lkh_timeout is None else float(lkh_timeout)),
        "guided_top_k": int(args.guided_top_k),
        "use_iface_in_decode": bool(args.use_iface_in_decode),
        "spanner_mode": str(args.spanner_mode),
        "theta_k": int(args.theta_k),
        "patching_mode": str(args.patching_mode),
    }

    records: List[Dict[str, Any]] = []
    print(f"[save] csv={csv_path}")
    print(f"[save] jsonl={jsonl_path}")
    print(f"[save] summary={summary_path}")

    for inst_idx, (n, tsp_file) in enumerate(tsp_files, start=1):
        name = tsp_file.stem
        prefix = f"[compare {inst_idx}/{len(tsp_files)}] {name} (N={n})"
        inst_t0 = time.perf_counter()
        shared_timing = {key: 0.0 for key in SHARED_TIMING_KEYS}

        parse_t0 = time.perf_counter()
        try:
            coords = parse_tsp_file(str(tsp_file))
        except ValueError as exc:
            skipped_instances.append(
                {
                    "instance_index": inst_idx,
                    "instance": name,
                    "n": int(n),
                    "reason": str(exc),
                }
            )
            metadata["num_instances_skipped"] = len(skipped_instances)
            write_summary(summary_path, metadata, records)
            log_progress(prefix, f"skip unsupported instance: {exc}")
            continue
        shared_timing["parse_sec"] = time.perf_counter() - parse_t0
        points_cpu = torch.from_numpy(coords).float()
        optimum = optima.get(name)

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

        log_progress(prefix, "run 1-pass direct traceback...")
        onepass_res = run_onepass_instance(
            data_cpu=data_cpu,
            coords_np=coords,
            device=device,
            packer=onepass_packer,
            runner=onepass_runner,
            model_bundle=onepass_bundle,
            catalog=onepass_catalog,
            optimum=optimum,
        )

        log_progress(prefix, "run 2-pass + guided LKH...")
        twopass_res = run_twopass_guided_instance(
            data_cpu=data_cpu,
            coords_np=coords,
            device=device,
            packer=twopass_packer,
            model_bundle=twopass_bundle,
            bu_runner=bu_runner,
            td_runner=td_runner,
            use_iface_in_decode=bool(args.use_iface_in_decode),
            guided_top_k=int(args.guided_top_k),
            lkh_exe=lkh_exe,
            lkh_runs=int(args.lkh_runs),
            lkh_timeout=lkh_timeout,
            optimum=optimum,
        )

        log_progress(prefix, "run pure LKH...")
        pure_lkh_res = run_pure_lkh(points_cpu, optimum=optimum, lkh_exe=lkh_exe)

        if nlkh_model is not None:
            log_progress(prefix, "run NeuroLKH...")
            neurolkh_res = run_neurolkh_instance(
                points_cpu=points_cpu,
                device=device,
                nlkh_model=nlkh_model,
                lkh_exe=lkh_exe,
                num_workers=int(args.num_workers),
                optimum=optimum,
            )
        else:
            neurolkh_res = metric_entry(
                feasible=False,
                obj=None,
                euclidean_length=None,
                order=None,
                time_sec=0.0,
                optimum=optimum,
            )

        if pomo_model is not None and pomo_env is not None:
            log_progress(prefix, "run POMO...")
            pomo_res = run_pomo_instance(
                points_cpu=points_cpu,
                device=device,
                pomo_model=pomo_model,
                pomo_env=pomo_env,
                optimum=optimum,
            )
        else:
            pomo_res = metric_entry(
                feasible=False,
                obj=None,
                euclidean_length=None,
                order=None,
                time_sec=0.0,
                optimum=optimum,
            )

        elapsed_sec = time.perf_counter() - inst_t0
        record = {
            "instance_index": inst_idx,
            "instance": name,
            "n": int(n),
            "opt_obj": optimum,
            "elapsed_sec": float(elapsed_sec),
            "completed_at_utc": utc_now_iso(),
            "shared_timing": shared_timing,
            "onepass": onepass_res,
            "twopass_guided_lkh": twopass_res,
            "pure_lkh": pure_lkh_res,
            "neurolkh": neurolkh_res,
            "pomo": pomo_res,
        }
        records.append(record)

        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(flatten_record(record))
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_summary(summary_path, metadata, records)

        log_progress(
            prefix,
            (
                f"done total={elapsed_sec:.2f}s | "
                f"1-pass={format_obj(onepass_res['obj'])}/{format_gap(onepass_res['gap_pct'])}/{format_time(onepass_res['time_sec'])}, "
                f"2-pass+LKH={format_obj(twopass_res['obj'])}/{format_gap(twopass_res['gap_pct'])}/{format_time(twopass_res['time_sec'])}, "
                f"LKH={format_obj(pure_lkh_res['obj'])}/{format_gap(pure_lkh_res['gap_pct'])}/{format_time(pure_lkh_res['time_sec'])}"
            ),
        )

    write_summary(summary_path, metadata, records)
    print_summary(records)
    if skipped_instances:
        print(f"[done] Skipped unsupported instances: {len(skipped_instances)}")
    print(f"[done] Results -> {csv_path}")
    print(f"[done] Details -> {jsonl_path}")
    print(f"[done] Summary -> {summary_path}")


if __name__ == "__main__":
    main()
