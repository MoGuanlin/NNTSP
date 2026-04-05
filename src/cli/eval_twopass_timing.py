# src/cli/eval_twopass_timing.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from src.cli.common import load_dataset, log_progress, move_data_tensors_to_device, parse_bool_arg, resolve_device
from src.cli.guided_lkh_args import add_guided_lkh_args, guided_lkh_config_from_args
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
from src.models.lkh_decode import GuidedLKHConfig, LKHDecodeResult, build_guided_candidates, run_candidate_lkh_timed
from src.models.node_token_packer import NodeTokenPacker
from src.models.top_down_runner import TopDownTreeRunner
from src.utils.lkh_solver import default_lkh_executable, resolve_lkh_executable


def resolve_synthetic_raw_data_path(*, synthetic_n: int, synthetic_data_root: str) -> str:
    return str(Path(synthetic_data_root) / f"N{int(synthetic_n)}" / "test.pt")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_run_tag(*, ckpt_path: str, data_pt: str) -> str:
    ckpt_obj = Path(ckpt_path)
    data_obj = Path(data_pt)
    return f"{utc_timestamp()}_{ckpt_obj.stem}_{data_obj.stem}_twopass_timing"


def load_points_dataset(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    # Raw synthetic test sets are stored as [B, N, 2] tensors.
    if p.suffix == ".pt" and "r_light" not in p.name and "raw_pyramid" not in p.name and "spanner" not in p.name:
        obj = torch.load(str(p), map_location="cpu", weights_only=False)
        if isinstance(obj, torch.Tensor):
            if obj.dim() != 3 or obj.shape[-1] != 2:
                raise ValueError(f"Expected raw point tensor [B,N,2], got {tuple(obj.shape)}")
            return obj

    return load_dataset(str(p))


def get_num_samples(dataset: Any) -> int:
    if isinstance(dataset, torch.Tensor):
        return int(dataset.shape[0])
    return int(len(dataset))


def get_points_from_dataset(dataset: Any, idx: int) -> torch.Tensor:
    if isinstance(dataset, torch.Tensor):
        pts = dataset[idx]
    else:
        sample = dataset[idx]
        if not hasattr(sample, "pos"):
            raise ValueError(f"Dataset sample at idx={idx} has no `pos` attribute.")
        pts = sample.pos
    pts = torch.as_tensor(pts, dtype=torch.float32).detach().cpu()
    if pts.dim() != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected points [N,2], got {tuple(pts.shape)}")
    return pts


def build_spanner_single(points: torch.Tensor, *, builder, num_workers: int) -> tuple[torch.Tensor, torch.Tensor]:
    silent = io.StringIO()
    with contextlib.redirect_stdout(silent):
        edge_index, edge_attr, _ = builder.build_batch(points.unsqueeze(0), num_workers=max(1, int(num_workers)))
    return edge_index.detach().cpu(), edge_attr.detach().cpu()


def run_guided_lkh_timed(
    *,
    pos: torch.Tensor,
    candidates: List[List[tuple[int, int]]],
    initial_tour: Sequence[int] | None,
    lkh_executable: str,
    num_runs: int,
    seed: int,
    timeout: float | None,
    guided_config: GuidedLKHConfig | None = None,
) -> tuple[LKHDecodeResult, Dict[str, float]]:
    return run_candidate_lkh_timed(
        pos=pos,
        candidates=candidates,
        initial_tour=initial_tour,
        lkh_executable=lkh_executable,
        num_runs=num_runs,
        seed=seed,
        timeout=timeout,
        mode="guided",
        candidate_config=guided_config,
    )


def summarize_records(records: List[Dict[str, Any]], *, stage_keys: List[str]) -> Dict[str, Any]:
    if not records:
        return {
            "num_samples": 0,
            "total": {k: 0.0 for k in stage_keys},
            "avg_per_sample": {k: 0.0 for k in stage_keys},
            "median_per_sample": {k: 0.0 for k in stage_keys},
            "share_of_total": {k: 0.0 for k in stage_keys if k != "total_sec"},
        }

    arr = {k: np.array([float(rec.get(k, 0.0)) for rec in records], dtype=np.float64) for k in stage_keys}
    total = {k: float(v.sum()) for k, v in arr.items()}
    avg = {k: float(v.mean()) for k, v in arr.items()}
    median = {k: float(np.median(v)) for k, v in arr.items()}

    total_total = max(total.get("total_sec", 0.0), 1e-12)
    share = {
        k: float(total[k] / total_total)
        for k in stage_keys
        if k != "total_sec"
    }
    return {
        "num_samples": len(records),
        "total": total,
        "avg_per_sample": avg,
        "median_per_sample": median,
        "share_of_total": share,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    stage_rows = [
        ("spanner_construction_sec", "Spanner construction"),
        ("quadtree_building_sec", "Quadtree building"),
        ("patching_sec", "Patching"),
        ("neural_inference_sec", "Neural inference"),
        ("score_agg_candidate_sec", "Score agg + candidates"),
        ("lkh_search_sec", "LKH-3 search"),
        ("other_overhead_sec", "Other overhead"),
        ("total_sec", "Total"),
    ]

    print("\n" + "=" * 78)
    print("2-Pass Practical Timing Breakdown")
    print(f"{'Stage':<28} {'Total(s)':>12} {'Avg/sample(s)':>15} {'Share':>10}")
    print("-" * 78)
    for key, label in stage_rows:
        total_v = float(summary["total"].get(key, 0.0))
        avg_v = float(summary["avg_per_sample"].get(key, 0.0))
        if key == "total_sec":
            share_text = "100.00%"
        else:
            share_text = f"{summary['share_of_total'].get(key, 0.0) * 100.0:>8.2f}%"
        print(f"{label:<28} {total_v:>12.4f} {avg_v:>15.4f} {share_text:>10}")
    print("=" * 78)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Dataset-level timing breakdown for the 2-pass practical pipeline (guided LKH)."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="path to 2-pass checkpoint")
    parser.add_argument("--data_pt", type=str, default=None, help="dataset path; can be raw points or preprocessed samples")
    parser.add_argument("--synthetic_n", type=int, default=None, help="standard synthetic dataset size N, used when --data_pt is omitted")
    parser.add_argument("--synthetic_data_root", type=str, default="data/std", help="root directory for standard synthetic datasets")
    parser.add_argument("--sample_idx", type=int, default=0, help="start sample index")
    parser.add_argument("--sample_idx_end", type=int, default=None, help="end sample index (exclusive); default is the full dataset")
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_iface_in_decode", type=parse_bool_arg, default=True)
    parser.add_argument("--lkh_exe", type=str, default=default_lkh_executable(), help="path to LKH executable")
    parser.add_argument("--lkh_runs", type=int, default=1, help="number of LKH runs")
    parser.add_argument("--lkh_timeout", type=float, default=0.0, help="timeout in seconds for LKH; 0 disables timeout")
    add_guided_lkh_args(parser)
    parser.add_argument("--num_workers", type=int, default=1, help="workers used by spanner builder when timing that stage")
    parser.add_argument("--spanner_mode", type=str, default="delaunay", choices=("delaunay", "theta"))
    parser.add_argument("--theta_k", type=int, default=14, help="theta spanner cone count when --spanner_mode theta")
    parser.add_argument("--patching_mode", type=str, default="prune", choices=("prune", "arora"))
    parser.add_argument("--output_dir", type=str, default="outputs/eval_twopass_timing", help="directory for timing reports")
    parser.add_argument("--run_tag", type=str, default=None, help="optional output file prefix")
    args = parser.parse_args(argv)

    data_pt = args.data_pt
    if not data_pt:
        if args.synthetic_n is None:
            parser.error("--data_pt or --synthetic_n is required")
        data_pt = resolve_synthetic_raw_data_path(
            synthetic_n=int(args.synthetic_n),
            synthetic_data_root=str(args.synthetic_data_root),
        )

    device = resolve_device(str(args.device))
    guided_config = guided_lkh_config_from_args(args)
    print(f"[env] device={device}")
    dataset = load_points_dataset(str(data_pt))
    total_samples = get_num_samples(dataset)
    start_idx = max(0, int(args.sample_idx))
    end_idx = total_samples if args.sample_idx_end is None else min(int(args.sample_idx_end), total_samples)
    if end_idx <= start_idx:
        raise ValueError(f"Empty sample range: [{start_idx}, {end_idx}) out of {total_samples}")
    selected_indices = list(range(start_idx, end_idx))

    print(f"[data] timing dataset={data_pt}")
    print(f"[data] selected samples: {start_idx}:{end_idx} (count={len(selected_indices)})")

    print(f"[ckpt] loading from {args.ckpt}")
    model_bundle = load_twopass_eval_models(
        ckpt_path=args.ckpt,
        device=device,
        r=int(args.r),
    )
    print(f"[model] detected d_model={model_bundle.d_model}")

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
    raw_builder = RawPyramidBuilder()
    lkh_exe = resolve_lkh_executable(str(args.lkh_exe))
    lkh_timeout = None if float(args.lkh_timeout) <= 0 else float(args.lkh_timeout)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_tag = str(args.run_tag or build_run_tag(ckpt_path=str(args.ckpt), data_pt=str(data_pt)))
    summary_path = output_dir / f"{run_tag}_summary.json"
    jsonl_path = output_dir / f"{run_tag}.jsonl"

    stage_keys = [
        "spanner_construction_sec",
        "quadtree_building_sec",
        "patching_sec",
        "neural_inference_sec",
        "score_agg_candidate_sec",
        "lkh_search_sec",
        "other_overhead_sec",
        "total_sec",
    ]
    detail_keys = [
        "device_transfer_sec",
        "token_packing_sec",
        "bottom_up_sec",
        "top_down_sec",
        "score_aggregation_sec",
        "warm_start_sec",
        "candidate_construction_sec",
        "lkh_setup_io_sec",
        "lkh_parse_sec",
    ]

    records: List[Dict[str, Any]] = []
    jsonl_path.write_text("", encoding="utf-8")

    for offset, sample_idx in enumerate(selected_indices, start=1):
        prefix = f"[timing {offset}/{len(selected_indices)}] sample={sample_idx}"
        points = get_points_from_dataset(dataset, sample_idx)
        sample_total_t0 = time.perf_counter()

        prep = preprocess_points_to_hierarchy(
            points,
            r=int(args.r),
            num_workers=int(args.num_workers),
            raw_builder=raw_builder,
            spanner_builder=spanner_builder,
            patching_mode=str(args.patching_mode),
        )
        raw_data = prep["raw_data"]
        data = prep["data_cpu"]
        edge_index_sp = prep["spanner_edge_index"]
        spanner_sec = float(prep["shared_timing"]["spanner_construction_sec"])
        quadtree_sec = float(prep["shared_timing"]["quadtree_building_sec"])
        patching_sec = float(prep["shared_timing"]["patching_sec"])

        xfer_t0 = time.perf_counter()
        data_dev = move_data_tensors_to_device(data, device)
        sync_device(device)
        device_transfer_sec = time.perf_counter() - xfer_t0

        pack_t0 = time.perf_counter()
        with torch.no_grad():
            packed = packer.pack_batch([data_dev])
        sync_device(device)
        token_packing_sec = time.perf_counter() - pack_t0

        bu_t0 = time.perf_counter()
        with torch.no_grad():
            out_bu = bu_runner.run_batch(
                batch=packed,
                leaf_encoder=model_bundle.leaf_encoder,
                merge_encoder=model_bundle.merge_encoder,
            )
        sync_device(device)
        bottom_up_sec = time.perf_counter() - bu_t0

        td_t0 = time.perf_counter()
        with torch.no_grad():
            out_td = td_runner.run_batch(
                packed=packed,
                z=out_bu.z,
                decoder=model_bundle.decoder,
            )
        sync_device(device)
        top_down_sec = time.perf_counter() - td_t0
        neural_inference_sec = bottom_up_sec + top_down_sec

        agg_t0 = time.perf_counter()
        with torch.no_grad():
            edge_scores = aggregate_logits_to_edges(
                tokens=packed.tokens,
                cross_logit=out_td.cross_logit,
                iface_logit=out_td.iface_logit if args.use_iface_in_decode else None,
                reduce="mean",
                num_edges=data_dev.spanner_edge_index.shape[1],
            )
            el = edge_scores.edge_logit.clone()
            em = edge_scores.edge_mask.bool()
            el[~em] = -1e9
        sync_device(device)
        score_aggregation_sec = time.perf_counter() - agg_t0

        warm_t0 = time.perf_counter()
        greedy_res = decode_tour(
            pos=data_dev.pos.detach().cpu(),
            spanner_edge_index=data_dev.spanner_edge_index.detach().cpu(),
            edge_logit=el.detach().cpu(),
            backend="greedy",
            allow_off_spanner_patch=True,
        )
        warm_start_sec = time.perf_counter() - warm_t0

        cand_t0 = time.perf_counter()
        candidates = build_guided_candidates(
            num_nodes=int(data_dev.pos.shape[0]),
            edge_index=data_dev.spanner_edge_index.detach().cpu(),
            edge_logit=el.detach().cpu(),
            logit_scale=float(guided_config.logit_scale),
            top_k=int(guided_config.top_k),
        )
        candidate_construction_sec = time.perf_counter() - cand_t0

        guided_res, lkh_break = run_guided_lkh_timed(
            pos=data_dev.pos.detach().cpu(),
            candidates=candidates,
            initial_tour=greedy_res.order if greedy_res.feasible else None,
            lkh_executable=lkh_exe,
            num_runs=int(args.lkh_runs),
            seed=1234,
            timeout=lkh_timeout,
            guided_config=guided_config,
        )

        score_agg_candidate_sec = (
            score_aggregation_sec
            + warm_start_sec
            + candidate_construction_sec
            + float(lkh_break["lkh_setup_io_sec"])
        )
        lkh_search_sec = float(lkh_break["lkh_search_sec"])
        total_sec = time.perf_counter() - sample_total_t0
        other_overhead_sec = max(
            0.0,
            total_sec
            - (
                spanner_sec
                + quadtree_sec
                + patching_sec
                + neural_inference_sec
                + score_agg_candidate_sec
                + lkh_search_sec
            ),
        )

        record = {
            "sample_idx": int(sample_idx),
            "num_points": int(points.shape[0]),
            "num_spanner_edges": int(edge_index_sp.shape[1]),
            "num_alive_edges": int(select_effective_edge_index(data).shape[1]),
            "num_tree_nodes": int(getattr(raw_data, "num_tree_nodes", -1)),
            "guided_lkh_feasible": bool(guided_res.feasible),
            "guided_lkh_length": (None if not guided_res.feasible else float(guided_res.length)),
            "warm_start_feasible": bool(greedy_res.feasible),
            "warm_start_length": (None if not greedy_res.feasible else float(greedy_res.length)),
            "spanner_construction_sec": float(spanner_sec),
            "quadtree_building_sec": float(quadtree_sec),
            "patching_sec": float(patching_sec),
            "neural_inference_sec": float(neural_inference_sec),
            "score_agg_candidate_sec": float(score_agg_candidate_sec),
            "lkh_search_sec": float(lkh_search_sec),
            "other_overhead_sec": float(other_overhead_sec),
            "total_sec": float(total_sec),
            "device_transfer_sec": float(device_transfer_sec),
            "token_packing_sec": float(token_packing_sec),
            "bottom_up_sec": float(bottom_up_sec),
            "top_down_sec": float(top_down_sec),
            "score_aggregation_sec": float(score_aggregation_sec),
            "warm_start_sec": float(warm_start_sec),
            "candidate_construction_sec": float(candidate_construction_sec),
            "lkh_setup_io_sec": float(lkh_break["lkh_setup_io_sec"]),
            "lkh_parse_sec": float(lkh_break["lkh_parse_sec"]),
        }
        records.append(record)
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        log_progress(
            prefix,
            (
                f"done total={total_sec:.3f}s "
                f"(spanner={spanner_sec:.3f}s, quadtree={quadtree_sec:.3f}s, patch={patching_sec:.3f}s, "
                f"nn={neural_inference_sec:.3f}s, score+candidate={score_agg_candidate_sec:.3f}s, "
                f"lkh={lkh_search_sec:.3f}s)"
            ),
        )

        # Free device memory between samples.
        move_data_tensors_to_device(data_dev, torch.device("cpu"))
        del packed, out_bu, out_td, edge_scores, el, em, data_dev
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = summarize_records(records, stage_keys=stage_keys + detail_keys)
    payload = {
        "meta": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "ckpt": str(Path(args.ckpt).resolve()),
            "data_pt": str(Path(data_pt).resolve()),
            "sample_idx": int(start_idx),
            "sample_idx_end": int(end_idx),
            "num_selected_samples": len(selected_indices),
            "device": str(device),
            "r": int(args.r),
            "use_iface_in_decode": bool(args.use_iface_in_decode),
            "guided_lkh": {
                "top_k": int(guided_config.top_k),
                "logit_scale": float(guided_config.logit_scale),
                "subgradient": bool(guided_config.subgradient),
                "max_candidates": guided_config.max_candidates,
                "max_trials": guided_config.max_trials,
                "use_initial_tour": bool(guided_config.use_initial_tour),
            },
            "spanner_mode": str(args.spanner_mode),
            "theta_k": int(args.theta_k),
            "patching_mode": str(args.patching_mode),
            "lkh_exe": str(lkh_exe),
            "lkh_runs": int(args.lkh_runs),
            "lkh_timeout": None if lkh_timeout is None else float(lkh_timeout),
            "state_mode": model_bundle.state_mode,
            "matching_max_used": int(model_bundle.matching_max_used),
        },
        "summary": summary,
        "records": records,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print_summary(summary)
    print(f"[save] per-sample JSONL -> {jsonl_path}")
    print(f"[save] summary JSON    -> {summary_path}")


if __name__ == "__main__":
    main()
