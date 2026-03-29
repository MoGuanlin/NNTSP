# src/cli/eval_onepass.py
# -*- coding: utf-8 -*-
"""
Evaluation script for the 1-pass DP pipeline.

Replaces the 2-pass neural inference (BU + TD → edge_logit) with:
  OnePassDPRunner → dp_result_to_edge_scores → edge_logit

Then reuses the same greedy / exact / LKH decoding backends.

Usage:
  python -m src.cli.eval_onepass --ckpt <path> --data_pt <path> [options]
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch

from src.cli.eval_and_vis import (
    AVAILABLE_SETTINGS,
    DEFAULT_SETTINGS,
    SETTING_ALIASES,
    SETTING_COLORS,
    SETTING_DISPLAY_NAMES,
    SETTING_GROUPS,
    load_dataset,
    log_progress,
    move_data_tensors_to_device,
    parse_bool_arg,
    placeholder_result,
    resolve_device,
)
from src.cli.eval_settings import describe_settings, resolve_eval_settings


def _sort_depth_items(depth_stats: Any) -> List[tuple[str, Dict[str, float]]]:
    """Sort per-depth stats from deepest layer to root."""
    if not isinstance(depth_stats, dict):
        return []
    items: List[tuple[str, Dict[str, float]]] = []
    for depth_key, bucket in depth_stats.items():
        if not isinstance(bucket, dict):
            continue
        items.append((str(depth_key), bucket))
    return sorted(items, key=lambda kv: int(kv[0]), reverse=True)


def _extract_depth_fallback_rates(dp_stats: Dict[str, Any]) -> Dict[str, float]:
    """Get a stable depth -> fallback_rate dict from runner stats."""
    raw_rates = dp_stats.get("depth_fallback_rates", {})
    if isinstance(raw_rates, dict) and raw_rates:
        return {
            str(depth_key): float(rate)
            for depth_key, rate in sorted(raw_rates.items(), key=lambda kv: int(kv[0]), reverse=True)
        }

    rates: Dict[str, float] = {}
    for depth_key, bucket in _sort_depth_items(dp_stats.get("depth_stats", {})):
        total = float(bucket.get("num_sigma_total", 0.0))
        fallback = float(bucket.get("num_fallback", 0.0))
        rates[depth_key] = (fallback / total) if total > 0.0 else 0.0
    return rates


def _format_depth_fallback_summary(dp_stats: Dict[str, Any]) -> str:
    """Format a compact per-depth fallback summary for console logs."""
    parts = []
    for depth_key, bucket in _sort_depth_items(dp_stats.get("depth_stats", {})):
        total = int(bucket.get("num_sigma_total", 0.0))
        fallback = int(bucket.get("num_fallback", 0.0))
        rate = float(bucket.get("fallback_rate", 0.0))
        parts.append(f"d{depth_key}={fallback}/{total} ({rate * 100.0:.1f}%)")
    return ", ".join(parts) if parts else "none"


def main(argv: List[str] | None = None):
    from src.utils.lkh_solver import default_lkh_executable

    parser = argparse.ArgumentParser(description="1-pass DP evaluation")
    default_lkh = default_lkh_executable()
    parser.add_argument("--ckpt", type=str, required=True, help="path to 1-pass checkpoint")
    parser.add_argument("--data_pt", type=str, required=True, help="path to data .pt")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--sample_idx_end", type=int, default=None)
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_onepass")

    # DP parameters
    parser.add_argument("--dp_max_used", type=int, default=4)
    parser.add_argument("--dp_topk", type=int, default=5)
    parser.add_argument(
        "--dp_max_sigma",
        type=int,
        default=0,
        help="max parent-state candidates per node; 0 disables truncation, >0 is heuristic pruning",
    )
    parser.add_argument(
        "--dp_fallback_exact",
        type=parse_bool_arg,
        default=True,
        help="enable exact fallback: heuristic parse falls back to exact search, and catalog_enum child-cap falls back to uncapped enumeration",
    )
    parser.add_argument("--dp_leaf_workers", type=int, default=16,
                        help="number of parallel workers for leaf_exact_solve (0=sequential)")
    parser.add_argument("--dp_parse_mode", type=str, default="catalog_enum",
                        choices=["catalog_enum", "heuristic"],
                        help="PARSE mode: 'catalog_enum' (default, exact) or 'heuristic' (legacy)")
    parser.add_argument(
        "--dp_child_catalog_cap",
        type=int,
        default=0,
        help="after C1 filtering and ranking in catalog_enum, keep at most this many child states; 0 disables truncation",
    )

    # Teacher
    parser.add_argument("--two_opt_passes", type=int, default=30, help="deprecated; ignored by spanner-LKH teacher generation")
    parser.add_argument("--use_lkh", action="store_true", help="deprecated; teacher generation always uses LKH on the sparse spanner")
    parser.add_argument("--lkh_exe", type=str, default=default_lkh)
    parser.add_argument("--teacher_lkh_runs", type=int, default=1)
    parser.add_argument("--teacher_lkh_timeout", type=float, default=0.0, help="0 disables timeout")

    # Decode settings
    parser.add_argument("--settings", type=str, default=None,
                        help="comma-separated decode settings (greedy, exact, guided_lkh, ...)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--exact_time_limit", type=float, default=30.0)
    parser.add_argument("--exact_length_weight", type=float, default=0.0)
    parser.add_argument("--list_settings", action="store_true")

    args = parser.parse_args(argv)

    if args.list_settings:
        print(describe_settings(available=AVAILABLE_SETTINGS, groups=SETTING_GROUPS))
        return

    selected_settings = resolve_eval_settings(
        requested=args.settings,
        available=AVAILABLE_SETTINGS,
        default=("greedy",),
        aliases=SETTING_ALIASES,
        groups=SETTING_GROUPS,
        enable_exact=False,
    )
    # For 1-pass eval, we add "dp_onepass" as the primary method
    # and allow downstream re-decoding via greedy/exact/LKH
    print(f"[eval] decode settings: {', '.join(selected_settings)}")

    device = resolve_device(str(args.device))
    print(f"[env] device={device}")

    # ─── Imports ──────────────────────────────────────────────────────
    from src.models.bc_state_catalog import build_boundary_state_catalog
    from src.models.decode_backend import DecodingDataset
    from src.models.dp_runner import OnePassDPRunner
    from src.models.labeler import PseudoLabeler
    from src.models.leaf_encoder import LeafEncoder
    from src.models.merge_decoder import MergeDecoder
    from src.models.merge_encoder import MergeEncoder
    from src.models.node_token_packer import NodeTokenPacker
    from src.models.tour_reconstruct import dp_result_to_edge_scores

    # ─── Load checkpoint ─────────────────────────────────────────────
    print(f"[ckpt] loading from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})

    d_model = int(ckpt_args.get("d_model", 128))
    r = int(args.r)
    Ti = 4 * r
    matching_max_used = int(ckpt_args.get("matching_max_used", args.dp_max_used))

    leaf_encoder = LeafEncoder(d_model=d_model).to(device)
    merge_encoder = MergeEncoder(d_model=d_model).to(device)
    merge_decoder = MergeDecoder(
        d_model=d_model,
        n_heads=max(4, d_model // 32),
        num_iface_slots=Ti,
        parent_num_layers=int(ckpt_args.get("parent_num_layers", 3)),
        cross_num_layers=int(ckpt_args.get("cross_num_layers", 2)),
        max_depth=64,
    ).to(device)

    leaf_encoder.load_state_dict(ckpt["leaf_encoder"])
    merge_encoder.load_state_dict(ckpt["merge_encoder"])
    if "merge_decoder" in ckpt:
        merge_decoder.load_state_dict(ckpt["merge_decoder"])
    else:
        print("[warn] No merge_decoder in checkpoint, using random weights")

    leaf_encoder.eval()
    merge_encoder.eval()
    merge_decoder.eval()
    print(f"[ckpt] loaded (d={d_model}, r={r}, Ti={Ti})")

    # ─── Dataset ──────────────────────────────────────────────────────
    dataset = load_dataset(args.data_pt)
    start_idx = args.sample_idx
    end_idx = args.sample_idx_end if args.sample_idx_end is not None else start_idx + 1
    end_idx = min(end_idx, len(dataset))
    num_samples = end_idx - start_idx

    packer = NodeTokenPacker(
        r=r,
        state_mode="matching",
        matching_max_used=matching_max_used,
    )
    labeler = PseudoLabeler(
        two_opt_passes=int(args.two_opt_passes),
        use_lkh=bool(args.use_lkh),
        lkh_exe=str(args.lkh_exe),
        prefer_cpu=True,
        teacher_mode="spanner_lkh",
        teacher_lkh_runs=int(args.teacher_lkh_runs),
        teacher_lkh_timeout=(None if float(args.teacher_lkh_timeout) <= 0 else float(args.teacher_lkh_timeout)),
    )
    print(f"[teacher] using LKH executable: {labeler.lkh_exe}")

    dp_runner = OnePassDPRunner(
        r=r,
        max_used=matching_max_used,
        topk=int(args.dp_topk),
        max_sigma_enumerate=int(args.dp_max_sigma),
        max_child_catalog_states=int(args.dp_child_catalog_cap),
        fallback_exact=bool(args.dp_fallback_exact),
        num_leaf_workers=int(args.dp_leaf_workers),
        parse_mode=str(args.dp_parse_mode),
    )
    catalog = build_boundary_state_catalog(
        num_slots=Ti,
        max_used=matching_max_used,
        device=device,
    )

    # Auto-generate unique output dir: {base}/{ckpt_run}_{step}_{timestamp}/
    import json
    from datetime import datetime
    if args.output_dir == "outputs/eval_onepass":
        ckpt_path = Path(args.ckpt)
        run_name = ckpt_path.parent.name          # e.g. onepass_r4_20260327_091905
        step_name = ckpt_path.stem                 # e.g. ckpt_step_51500
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs/eval_onepass") / f"{run_name}_{step_name}_{ts}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-sample results log (append-mode, survives crashes)
    results_log = output_dir / "results.jsonl"
    def _log_result(record: dict) -> None:
        with open(results_log, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    print(f"[eval] Results will be written to {results_log}")

    stats_dp = {"len": 0.0, "gap": 0.0, "time": 0.0, "cnt": 0}
    stats_direct = {"len": 0.0, "gap": 0.0, "time": 0.0, "cnt": 0}
    stats = {s: {"len": 0.0, "gap": 0.0, "time": 0.0, "cnt": 0} for s in selected_settings}

    # ─── Phase 1: 1-pass DP inference ─────────────────────────────────
    print(f"\n[eval] Phase 1: Running 1-pass DP on {num_samples} samples...")
    model_outputs = []

    for offset, s_idx in enumerate(range(start_idx, end_idx), start=1):
        prefix = f"[eval {offset}/{num_samples}] sample={s_idx}"
        data = dataset[s_idx]
        data = move_data_tensors_to_device(data, device)

        # ── Teacher tour (compute FIRST, before expensive DP) ────────
        t0 = time.time()
        teacher_len = float("nan")
        stored_tour_len = getattr(data, "tour_len", None)
        if stored_tour_len is not None:
            teacher_len = float(torch.as_tensor(stored_tour_len).item())
        if not (teacher_len > 0):
            try:
                log_progress(prefix, "computing teacher tour...")
                # Pack minimally just for labeler (need eid/mask tokens)
                with torch.no_grad():
                    packed_tmp = packer.pack_batch([data])
                ts = SimpleNamespace(
                    cross_eid=packed_tmp.tokens.cross_eid.cpu(),
                    cross_mask=packed_tmp.tokens.cross_mask.cpu(),
                    iface_eid=packed_tmp.tokens.iface_eid.cpu(),
                    iface_mask=packed_tmp.tokens.iface_mask.cpu(),
                )
                teacher_lab = labeler.label_one(
                    data=data, tokens_slice=ts,
                    device=torch.device("cpu"), eid_offset=0,
                )
                teacher_len = float(teacher_lab.stats["tour_len"].item())
                del packed_tmp
            except Exception as exc:
                log_progress(prefix, f"teacher tour failed ({exc}), gap will be N/A")
                teacher_len = float("nan")
        t_teacher = time.time()
        teacher_str = f"{teacher_len:.4f}" if teacher_len == teacher_len else "N/A"
        log_progress(prefix, f"teacher={teacher_str} ({t_teacher - t0:.1f}s)")

        # ── Pack + 1-pass DP ─────────────────────────────────────────
        log_progress(prefix, "packing...")
        with torch.no_grad():
            packed = packer.pack_batch([data])
            packed = move_data_tensors_to_device(packed, device)
            t_pack = time.time()

            # Run 1-pass DP
            log_progress(prefix, "running 1-pass DP...")
            dp_result = dp_runner.run_single(
                tokens=packed.tokens,
                leaves=packed.leaves,
                leaf_encoder=leaf_encoder,
                merge_encoder=merge_encoder,
                merge_decoder=merge_decoder,
                catalog=catalog,
            )
            t_dp = time.time()

            # Convert to edge scores
            edge_scores = dp_result_to_edge_scores(
                result=dp_result,
                tokens=packed.tokens,
                catalog=catalog,
                num_edges=data.spanner_edge_index.shape[1],
            )

            el = edge_scores.edge_logit.clone()
            em = edge_scores.edge_mask.bool()
            el[~em] = -1e9
            t_total = time.time()

        mo = {
            "s_idx": s_idx,
            "pos": data.pos.cpu(),
            "edge_index": data.spanner_edge_index.cpu(),
            "edge_logit": el.cpu(),
            "dp_cost": dp_result.tour_cost,
            "direct_tour_length": dp_result.tour_length,
            "direct_tour_feasible": dp_result.tour_feasible,
            "direct_tour_order": list(dp_result.tour_order),
            "direct_tour_stats": dp_result.tour_stats,
            "dp_stats": dp_result.stats,
            "depth_fallback_rates": _extract_depth_fallback_rates(dp_result.stats),
            "teacher_len": teacher_len,
        }
        model_outputs.append(mo)

        # Write result to disk immediately
        dp_gap = (dp_result.tour_cost / teacher_len - 1.0) if dp_result.tour_cost < float("inf") and teacher_len > 1e-9 else None
        direct_gap = (
            dp_result.tour_length / teacher_len - 1.0
            if dp_result.tour_feasible and dp_result.tour_length < float("inf") and teacher_len > 1e-9
            else None
        )
        depth_fallback_rates = _extract_depth_fallback_rates(dp_result.stats)
        _log_result({
            "sample_idx": s_idx,
            "teacher_len": teacher_len,
            "dp_cost": dp_result.tour_cost,
            "dp_gap": dp_gap,
            "direct_tour_length": dp_result.tour_length if dp_result.tour_feasible else None,
            "direct_tour_gap": direct_gap,
            "direct_tour_feasible": dp_result.tour_feasible,
            "direct_tour_stats": dp_result.tour_stats,
            "dp_stats": dp_result.stats,
            "depth_fallback_rates": depth_fallback_rates,
            "teacher_time": t_teacher - t0,
            "pack_time": t_pack - t_teacher,
            "dp_time": t_dp - t_pack,
        })

        gap_str = f"{dp_gap*100:.2f}%" if dp_gap is not None else "N/A"
        direct_str = (
            f"{dp_result.tour_length:.4f} ({direct_gap*100:.2f}%)"
            if direct_gap is not None
            else "N/A"
        )
        log_progress(
            prefix,
            f"done: dp_cost={dp_result.tour_cost:.4f}, teacher={teacher_str}, "
            f"direct={direct_str}, "
            f"gap={gap_str}, "
            f"fallback={_format_depth_fallback_summary(dp_result.stats)}, "
            f"teacher={t_teacher - t0:.1f}s, pack={t_pack - t_teacher:.1f}s, "
            f"dp={t_dp - t_pack:.1f}s, total={t_total - t0:.1f}s",
        )

        # Free GPU memory: move_data_tensors_to_device mutates data in-place
        _cpu = torch.device("cpu")
        move_data_tensors_to_device(data, _cpu)
        del packed, dp_result, edge_scores

    # ─── Phase 2: Greedy decoding ─────────────────────────────────────
    need_greedy = ("greedy" in selected_settings) or ("guided_lkh" in selected_settings)
    greedy_results = [placeholder_result() for _ in model_outputs]

    if need_greedy:
        print("[eval] Phase 2: Running Greedy decoding...")
        greedy_tasks = [
            (mo["pos"], mo["edge_index"], mo["edge_logit"], True, True, mo["teacher_len"])
            for mo in model_outputs
        ]
        g_dataset = DecodingDataset(greedy_tasks)
        g_loader = torch.utils.data.DataLoader(
            g_dataset, batch_size=1,
            num_workers=min(args.num_workers, len(greedy_tasks)),
            shuffle=False, collate_fn=lambda x: x[0],
        )
        for i, res_pair in enumerate(g_loader):
            greedy_results[i] = res_pair[0]

    # ─── Phase 3: Exact decoding (optional) ───────────────────────────
    exact_results = [placeholder_result() for _ in model_outputs]
    if "exact" in selected_settings:
        print("[eval] Phase 3: Running Exact sparse decoding...")
        exact_tasks = [
            (mo["pos"], mo["edge_index"], mo["edge_logit"], True, False, mo["teacher_len"])
            for mo in model_outputs
        ]
        ex_dataset = DecodingDataset(
            exact_tasks, decode_backend="exact",
            exact_time_limit=float(args.exact_time_limit),
            exact_length_weight=float(args.exact_length_weight),
        )
        ex_loader = torch.utils.data.DataLoader(
            ex_dataset, batch_size=1,
            num_workers=min(args.num_workers, len(exact_tasks)),
            shuffle=False, collate_fn=lambda x: x[0],
        )
        for i, res_pair in enumerate(ex_loader):
            exact_results[i] = res_pair[0]

    # ─── Phase 4: LKH (optional) ─────────────────────────────────────
    lkh_results: Dict[str, list] = {s: [placeholder_result() for _ in model_outputs]
                                     for s in selected_settings if s.endswith("_lkh")}

    for s in ["guided_lkh", "pure_lkh"]:
        if s not in selected_settings:
            continue
        print(f"[eval] Phase 4: Running {SETTING_DISPLAY_NAMES.get(s, s)}...")
        from src.models.lkh_decode import solve_with_lkh_parallel
        for i, mo in enumerate(model_outputs):
            if s == "guided_lkh":
                gr = greedy_results[i]
                initial_tour = mo["direct_tour_order"] if mo["direct_tour_feasible"] else (
                    gr.order if gr.feasible else None
                )
                tasks = [{
                    "pos": mo["pos"],
                    "mode": "guided",
                    "edge_index": mo["edge_index"],
                    "edge_logit": mo["edge_logit"],
                    "teacher_len": mo["teacher_len"],
                    "initial_tour": initial_tour,
                }]
            else:
                tasks = [{"pos": mo["pos"], "mode": "pure", "teacher_len": mo["teacher_len"]}]
            results = solve_with_lkh_parallel(tasks, lkh_executable=args.lkh_exe, num_workers=1)
            lkh_results[s][i] = results[0][0]

    # ─── Collect, log per-sample, & print summary ──────────────────────
    # Write per-sample results to jsonl (survives crashes)
    for i, mo in enumerate(model_outputs):
        teacher_len = mo["teacher_len"]
        dp_cost = mo["dp_cost"]
        direct_len = mo["direct_tour_length"]
        record = {
            "sample_idx": mo["s_idx"],
            "teacher_len": teacher_len,
            "dp_cost": dp_cost,
            "dp_gap": (dp_cost / teacher_len - 1.0) if dp_cost < float("inf") and teacher_len > 1e-9 else None,
            "direct_tour_length": direct_len if mo["direct_tour_feasible"] else None,
            "direct_tour_gap": (
                direct_len / teacher_len - 1.0
                if mo["direct_tour_feasible"] and direct_len < float("inf") and teacher_len > 1e-9
                else None
            ),
            "direct_tour_feasible": mo["direct_tour_feasible"],
            "direct_tour_stats": mo["direct_tour_stats"],
            "dp_stats": mo["dp_stats"],
            "depth_fallback_rates": mo["depth_fallback_rates"],
        }

        if dp_cost < float("inf") and teacher_len > 1e-9:
            stats_dp["len"] += dp_cost
            stats_dp["gap"] += dp_cost / teacher_len - 1.0
            stats_dp["cnt"] += 1

        if mo["direct_tour_feasible"] and direct_len < float("inf") and teacher_len > 1e-9:
            stats_direct["len"] += direct_len
            stats_direct["gap"] += direct_len / teacher_len - 1.0
            stats_direct["cnt"] += 1

        for setting in selected_settings:
            if setting == "greedy":
                res = greedy_results[i]
            elif setting == "exact":
                res = exact_results[i]
            elif setting in lkh_results:
                res = lkh_results[setting][i]
            else:
                continue

            if res.feasible and res.length < float("inf") and teacher_len > 1e-9:
                gap = res.length / teacher_len - 1.0
                stats[setting]["len"] += res.length
                stats[setting]["gap"] += gap
                stats[setting]["time"] += getattr(res, "duration", 0.0)
                stats[setting]["cnt"] += 1
                record[setting] = {"length": res.length, "gap": gap, "feasible": True}
            else:
                record[setting] = {"length": None, "gap": None, "feasible": getattr(res, "feasible", False)}

        _log_result(record)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Method':<20} {'Avg Length':>12} {'Avg Gap':>10} {'Total Time':>12}")
    print("-" * 70)

    if stats_dp["cnt"] > 0:
        n = stats_dp["cnt"]
        print(f"{'DP (cost table)':<20} {stats_dp['len']/n:>12.4f} {stats_dp['gap']/n*100:>9.4f}% {'N/A':>12}")
    else:
        print(f"{'DP (cost table)':<20} {'N/A':>12} {'N/A':>10} {'N/A':>12}")

    if stats_direct["cnt"] > 0:
        n = stats_direct["cnt"]
        print(f"{'1-pass direct':<20} {stats_direct['len']/n:>12.4f} {stats_direct['gap']/n*100:>9.4f}% {'N/A':>12}")
    else:
        print(f"{'1-pass direct':<20} {'N/A':>12} {'N/A':>10} {'N/A':>12}")

    for setting in selected_settings:
        n = stats[setting]["cnt"]
        if n > 0:
            name = SETTING_DISPLAY_NAMES.get(setting, setting)
            print(f"{name:<20} {stats[setting]['len']/n:>12.4f} {stats[setting]['gap']/n*100:>9.4f}% {stats[setting]['time']/n:>11.3f}s")
        else:
            name = SETTING_DISPLAY_NAMES.get(setting, setting)
            print(f"{name:<20} {'N/A':>12} {'N/A':>10} {'N/A':>12}")

    print("=" * 70)
    print(f"[done] Evaluated {num_samples} samples. Results → {results_log}")


if __name__ == "__main__":
    main()
