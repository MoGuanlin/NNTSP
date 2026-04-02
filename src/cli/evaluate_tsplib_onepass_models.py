# src/cli/evaluate_tsplib_onepass_models.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.cli.common import log_progress, resolve_device
from src.cli.evaluate_tsplib import (
    collect_tsplib_files,
    normalize_tsplib_instance_name,
    parse_tsp_file,
    sanitize_tag,
)
from src.cli.evaluate_tsplib_compare import (
    ONEPASS_TIMING_KEYS,
    format_gap,
    format_obj,
    format_time,
    load_tsplib_optima,
    run_onepass_instance,
)
from src.cli.model_factory import load_onepass_eval_models
from src.graph.build_raw_pyramid import RawPyramidBuilder
from src.graph.prune_pyramid import prune_r_light_single
from src.graph.spanner import SpannerBuilder
from src.models.bc_state_catalog import build_boundary_state_catalog
from src.models.dp_runner import OnePassDPRunner
from src.models.node_token_packer import NodeTokenPacker


DEFAULT_INSTANCES = ("eil51", "berlin52", "st70")
HEADER_RE = re.compile(r"^([A-Z_]+)\s*:\s*(.*)$")
STEP_RE = re.compile(r"ckpt_(?:final_)?step_(\d+)\.pt$")


@dataclass
class ModelSpec:
    ckpt_path: Path
    model_name: str
    r: int
    matching_max_used: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_run_tag(*, model_specs: Sequence[ModelSpec], instances: Sequence[str]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_tag = f"models{len(model_specs)}"
    inst_tag = "-".join(str(x) for x in instances)
    return sanitize_tag(f"{ts}_onepass_tsplib_{model_tag}_{inst_tag}")


def parse_instance_name_list(text: Optional[str]) -> List[str]:
    if not text:
        return list(DEFAULT_INSTANCES)
    names = []
    for part in re.split(r"[,\s]+", str(text).strip()):
        name = normalize_tsplib_instance_name(part)
        if name:
            names.append(name)
    return names


def read_tsplib_header_fields(path: Path) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            upper = text.upper()
            if upper in {"NODE_COORD_SECTION", "DISPLAY_DATA_SECTION", "EDGE_WEIGHT_SECTION", "EOF"}:
                break
            match = HEADER_RE.match(upper)
            if match:
                fields[match.group(1).strip()] = match.group(2).strip()
    return fields


def format_step_label(path: Path) -> str:
    match = STEP_RE.search(path.name)
    if match:
        return match.group(1)
    return path.stem


def checkpoint_sort_key(path: Path) -> Tuple[int, int, str]:
    match = STEP_RE.search(path.name)
    if match:
        is_final = 1 if "final" in path.name else 0
        return (is_final, int(match.group(1)), path.name)
    return (0, -1, path.name)


def resolve_checkpoint_path(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")
    if not path.is_dir():
        raise ValueError(f"Unsupported checkpoint path: {path}")

    candidates = sorted(path.glob("*.pt"), key=checkpoint_sort_key)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found in directory: {path}")
    return candidates[-1]


def expand_ckpt_specs(specs: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for spec in specs:
        matches = [Path(p) for p in glob.glob(str(spec))]
        if not matches:
            matches = [Path(spec)]
        for path in matches:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(path)
    return out


def infer_r_from_path(path: Path, default_r: int) -> int:
    for text in (path.name, path.parent.name):
        match = re.search(r"r(\d+)", text)
        if match:
            return int(match.group(1))
    return int(default_r)


def make_model_name(ckpt_path: Path) -> str:
    parent = ckpt_path.parent.name or ckpt_path.stem
    step = format_step_label(ckpt_path)
    return f"{parent}@{step}"


def discover_model_specs(*, ckpt_specs: Sequence[str], default_r: int, default_matching_max_used: int) -> List[ModelSpec]:
    model_specs: List[ModelSpec] = []
    used_names = set()
    for raw_path in expand_ckpt_specs(ckpt_specs):
        ckpt_path = resolve_checkpoint_path(raw_path).resolve()
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        ckpt_args = dict(ckpt.get("args", {}))
        r = int(ckpt_args.get("r", infer_r_from_path(ckpt_path, default_r)))
        matching_max_used = int(ckpt_args.get("matching_max_used", default_matching_max_used))
        model_name = make_model_name(ckpt_path)
        if model_name in used_names:
            suffix = 2
            while f"{model_name}_{suffix}" in used_names:
                suffix += 1
            model_name = f"{model_name}_{suffix}"
        used_names.add(model_name)
        model_specs.append(
            ModelSpec(
                ckpt_path=ckpt_path,
                model_name=model_name,
                r=r,
                matching_max_used=matching_max_used,
            )
        )
    if not model_specs:
        raise ValueError("No valid checkpoint specs resolved.")
    return model_specs


def select_instances(*, tsplib_dir: Path, instance_names: Sequence[str]) -> List[Tuple[int, Path]]:
    all_tsp_files = collect_tsplib_files(tsplib_dir)
    by_name = {path.stem: (n, path) for n, path in all_tsp_files}
    selected: List[Tuple[int, Path]] = []
    missing = []
    for name in instance_names:
        norm = normalize_tsplib_instance_name(name)
        if norm not in by_name:
            missing.append(norm)
            continue
        selected.append(by_name[norm])
    if missing:
        raise ValueError(f"Unknown TSPLIB instances: {', '.join(missing)}")
    return selected


def format_metric(value: Optional[float], *, kind: str) -> str:
    if kind == "obj":
        return format_obj(value)
    if kind == "gap":
        return format_gap(value)
    if kind == "time":
        return format_time(value)
    return "N/A" if value is None else str(value)


def flatten_record(record: Dict[str, Any]) -> Dict[str, str]:
    row = {
        "model": str(record["model"]),
        "ckpt_path": str(record["ckpt_path"]),
        "r": str(record["r"]),
        "matching_max_used": str(record["matching_max_used"]),
        "instance": str(record["instance"]),
        "n": str(record["n"]),
        "edge_weight_type": str(record["edge_weight_type"]),
        "opt_obj": "" if record["opt_obj"] is None else f"{float(record['opt_obj']):.0f}",
        "shared_parse_sec": f"{float(record['shared_timing']['parse_sec']):.6f}",
        "shared_spanner_construction_sec": f"{float(record['shared_timing']['spanner_construction_sec']):.6f}",
        "shared_quadtree_building_sec": f"{float(record['shared_timing']['quadtree_building_sec']):.6f}",
        "shared_preprocess_sec": f"{float(record['shared_preprocess_sec']):.6f}",
        "patching_sec": f"{float(record['patching_sec']):.6f}",
        "patch_plus_model_sec": f"{float(record['patch_plus_model_sec']):.6f}",
        "feasible": str(bool(record["feasible"])),
        "obj": "" if record["obj"] is None else f"{float(record['obj']):.0f}",
        "gap_pct": "" if record["gap_pct"] is None else f"{float(record['gap_pct']):.6f}",
        "time_sec": f"{float(record['time_sec']):.6f}",
        "dp_cost": "" if record["dp_cost"] is None else f"{float(record['dp_cost']):.6f}",
        "num_sigma_total": str(int(record["num_sigma_total"])),
    }
    for key in ONEPASS_TIMING_KEYS:
        row[f"onepass_{key}"] = f"{float(record['timing'].get(key, 0.0)):.6f}"
    return row


def build_fieldnames() -> List[str]:
    fields = [
        "model",
        "ckpt_path",
        "r",
        "matching_max_used",
        "instance",
        "n",
        "edge_weight_type",
        "opt_obj",
        "shared_parse_sec",
        "shared_spanner_construction_sec",
        "shared_quadtree_building_sec",
        "shared_preprocess_sec",
        "patching_sec",
        "patch_plus_model_sec",
        "feasible",
        "obj",
        "gap_pct",
        "time_sec",
        "dp_cost",
        "num_sigma_total",
    ]
    for key in ONEPASS_TIMING_KEYS:
        fields.append(f"onepass_{key}")
    return fields


def build_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_model: Dict[str, Dict[str, Any]] = {}
    per_instance: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        per_instance.setdefault(str(record["instance"]), []).append(record)

    model_names = sorted({str(record["model"]) for record in records})
    for model_name in model_names:
        group = [record for record in records if str(record["model"]) == model_name]
        feasible = [record for record in group if record.get("obj") is not None]
        per_model[model_name] = {
            "num_records": len(group),
            "num_feasible": len(feasible),
            "avg_obj": (None if not feasible else float(np.mean([record["obj"] for record in feasible]))),
            "avg_gap_pct": (
                None
                if not feasible or not any(record.get("gap_pct") is not None for record in feasible)
                else float(np.mean([record["gap_pct"] for record in feasible if record.get("gap_pct") is not None]))
            ),
            "avg_time_sec": (None if not group else float(np.mean([record["time_sec"] for record in group]))),
            "avg_patching_sec": (None if not group else float(np.mean([record["patching_sec"] for record in group]))),
            "avg_patch_plus_model_sec": (
                None if not group else float(np.mean([record["patch_plus_model_sec"] for record in group]))
            ),
        }

    best_by_instance: Dict[str, Dict[str, Any]] = {}
    for instance_name, group in per_instance.items():
        feasible = [record for record in group if record.get("obj") is not None]
        if not feasible:
            continue
        best = min(feasible, key=lambda record: (float(record["obj"]), float(record["time_sec"])))
        best_by_instance[instance_name] = {
            "model": best["model"],
            "obj": float(best["obj"]),
            "gap_pct": (None if best.get("gap_pct") is None else float(best["gap_pct"])),
            "time_sec": float(best["time_sec"]),
        }

    return {
        "num_records": len(records),
        "per_model": per_model,
        "best_by_instance": best_by_instance,
    }


def build_report_lines(*, metadata: Dict[str, Any], records: List[Dict[str, Any]], summary: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    lines.append("")
    lines.append("=" * 108)
    lines.append("1-Pass TSPLIB Model Comparison")
    lines.append("=" * 108)
    lines.append(f"Instances: {', '.join(str(x) for x in metadata['instance_names'])}")
    lines.append(f"Models   : {len(metadata['models'])}")
    lines.append("-" * 108)
    lines.append(f"{'Instance':<12} {'Model':<34} {'Obj':>10} {'Gap':>10} {'Time':>10} {'Patch':>10} {'r':>4}")
    lines.append("-" * 108)
    for record in records:
        lines.append(
            f"{str(record['instance']):<12} "
            f"{str(record['model'])[:34]:<34} "
            f"{format_metric(record.get('obj'), kind='obj'):>10} "
            f"{format_metric(record.get('gap_pct'), kind='gap'):>10} "
            f"{format_metric(record.get('time_sec'), kind='time'):>10} "
            f"{format_metric(record.get('patching_sec'), kind='time'):>10} "
            f"{int(record['r']):>4}"
        )
    lines.append("-" * 108)
    lines.append("Average By Model")
    lines.append(f"{'Model':<34} {'Feasible':>10} {'Avg Obj':>12} {'Avg Gap':>10} {'Avg Time':>12} {'Patch+Model':>14}")
    for model_name, payload in summary["per_model"].items():
        avg_obj = payload.get("avg_obj")
        avg_gap = payload.get("avg_gap_pct")
        avg_time = payload.get("avg_time_sec")
        avg_patch_model = payload.get("avg_patch_plus_model_sec")
        lines.append(
            f"{model_name[:34]:<34} "
            f"{int(payload['num_feasible']):>10} "
            f"{('N/A' if avg_obj is None else f'{avg_obj:.0f}'):>12} "
            f"{('N/A' if avg_gap is None else f'{avg_gap:.2f}%'):>10} "
            f"{('N/A' if avg_time is None else f'{avg_time:.2f}s'):>12} "
            f"{('N/A' if avg_patch_model is None else f'{avg_patch_model:.2f}s'):>14}"
        )
    if summary["best_by_instance"]:
        lines.append("-" * 108)
        lines.append("Best By Instance")
        for instance_name, payload in summary["best_by_instance"].items():
            gap_text = "N/A" if payload["gap_pct"] is None else f"{float(payload['gap_pct']):.2f}%"
            lines.append(
                f"{instance_name}: {payload['model']} | obj={float(payload['obj']):.0f} | gap={gap_text} | time={float(payload['time_sec']):.2f}s"
            )
    lines.append("=" * 108)
    return lines


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare several trained 1-pass checkpoints on selected small TSPLIB instances."
    )
    parser.add_argument("--ckpts", type=str, nargs="+", required=True, help="checkpoint files, directories, or glob patterns")
    parser.add_argument("--instances", type=str, default=",".join(DEFAULT_INSTANCES), help="comma-separated TSPLIB instance names")
    parser.add_argument("--tsplib_dir", type=str, default="benchmarks/tsplib")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=1, help="workers used by the spanner builder")
    parser.add_argument("--save_dir", type=str, default="outputs/eval_tsplib_onepass_models")
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--allow_non_euc2d", action="store_true", help="allow non-EUC_2D TSPLIB instances")
    parser.add_argument("--default_r", type=int, default=4, help="fallback r when the checkpoint args do not store it")
    parser.add_argument("--dp_max_used", type=int, default=4)
    parser.add_argument("--dp_topk", type=int, default=5)
    parser.add_argument("--dp_max_sigma", type=int, default=0)
    parser.add_argument("--dp_child_catalog_cap", type=int, default=0)
    parser.add_argument("--dp_fallback_exact", action="store_true", default=True)
    parser.add_argument("--no_dp_fallback_exact", dest="dp_fallback_exact", action="store_false")
    parser.add_argument("--dp_leaf_workers", type=int, default=16)
    parser.add_argument("--dp_parse_mode", type=str, default="catalog_enum", choices=("catalog_enum", "heuristic"))
    args = parser.parse_args(argv)

    device = resolve_device(str(args.device))
    tsplib_dir = Path(args.tsplib_dir)
    instance_names = parse_instance_name_list(args.instances)
    model_specs = discover_model_specs(
        ckpt_specs=args.ckpts,
        default_r=int(args.default_r),
        default_matching_max_used=int(args.dp_max_used),
    )
    instances = select_instances(tsplib_dir=tsplib_dir, instance_names=instance_names)
    optima = load_tsplib_optima(tsplib_dir)

    run_tag = str(args.run_tag or build_run_tag(model_specs=model_specs, instances=instance_names))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / f"{run_tag}.csv"
    json_path = save_dir / f"{run_tag}.json"
    report_path = save_dir / f"{run_tag}.txt"

    print(f"[env] device={device}")
    print(f"[save] csv={csv_path}")
    print(f"[save] json={json_path}")
    print(f"[save] report={report_path}")
    print("[models]")
    for idx, spec in enumerate(model_specs, start=1):
        print(
            f"  {idx}. {spec.model_name} | r={spec.r} | matching_max_used={spec.matching_max_used} | {spec.ckpt_path}"
        )

    spanner_builder = SpannerBuilder(mode="delaunay")
    raw_builder = RawPyramidBuilder(max_points_per_leaf=4, max_depth=20)

    instance_cache: List[Dict[str, Any]] = []
    for inst_idx, (n, tsp_path) in enumerate(instances, start=1):
        name = tsp_path.stem
        prefix = f"[prep {inst_idx}/{len(instances)}] {name} (N={n})"
        header_fields = read_tsplib_header_fields(tsp_path)
        edge_weight_type = str(header_fields.get("EDGE_WEIGHT_TYPE", "UNKNOWN"))
        if (not args.allow_non_euc2d) and edge_weight_type != "EUC_2D":
            raise ValueError(
                f"Instance {name} uses EDGE_WEIGHT_TYPE={edge_weight_type}. "
                "This script compares 1-pass models in the Euclidean coordinate pipeline, "
                "so please use EUC_2D instances or pass --allow_non_euc2d explicitly."
            )

        parse_t0 = time.perf_counter()
        coords = parse_tsp_file(str(tsp_path))
        parse_sec = time.perf_counter() - parse_t0
        points_cpu = torch.from_numpy(coords).float()

        log_progress(prefix, "build Delaunay spanner...")
        sp_t0 = time.perf_counter()
        edge_index_sp, edge_attr_sp, _ = spanner_builder.build_batch(points_cpu.unsqueeze(0), num_workers=max(1, int(args.num_workers)))
        spanner_sec = time.perf_counter() - sp_t0

        log_progress(prefix, "build raw pyramid...")
        qt_t0 = time.perf_counter()
        raw_data = raw_builder.process_sample(points_cpu, edge_index_sp, edge_attr_sp)
        quadtree_sec = time.perf_counter() - qt_t0

        instance_cache.append(
            {
                "name": name,
                "n": int(n),
                "path": tsp_path.resolve(),
                "edge_weight_type": edge_weight_type,
                "optimum": optima.get(name),
                "coords_np": coords,
                "raw_data": raw_data,
                "shared_timing": {
                    "parse_sec": float(parse_sec),
                    "spanner_construction_sec": float(spanner_sec),
                    "quadtree_building_sec": float(quadtree_sec),
                },
                "patched_by_r": {},
            }
        )
        log_progress(
            prefix,
            f"ready: parse={parse_sec:.3f}s, spanner={spanner_sec:.3f}s, quadtree={quadtree_sec:.3f}s",
        )

    records: List[Dict[str, Any]] = []
    for model_idx, model_spec in enumerate(model_specs, start=1):
        prefix = f"[model {model_idx}/{len(model_specs)}] {model_spec.model_name}"
        print(f"{prefix} load checkpoint...")
        model_bundle = load_onepass_eval_models(
            ckpt_path=str(model_spec.ckpt_path),
            device=device,
            r=int(model_spec.r),
            default_matching_max_used=int(model_spec.matching_max_used),
        )
        if not model_bundle.merge_decoder_loaded:
            print(f"{prefix} missing merge_decoder in checkpoint; skip.")
            continue

        packer = NodeTokenPacker(
            r=int(model_spec.r),
            state_mode="matching",
            matching_max_used=int(model_bundle.matching_max_used),
        )
        catalog = build_boundary_state_catalog(
            num_slots=int(model_bundle.num_iface_slots),
            max_used=int(model_bundle.matching_max_used),
            device=device,
        )
        runner = OnePassDPRunner(
            r=int(model_spec.r),
            max_used=int(model_bundle.matching_max_used),
            topk=int(args.dp_topk),
            max_sigma_enumerate=int(args.dp_max_sigma),
            max_child_catalog_states=int(args.dp_child_catalog_cap),
            fallback_exact=bool(args.dp_fallback_exact),
            num_leaf_workers=int(args.dp_leaf_workers),
            parse_mode=str(args.dp_parse_mode),
        )

        for inst_info in instance_cache:
            inst_prefix = f"{prefix} | {inst_info['name']} (N={inst_info['n']})"
            if int(model_spec.r) not in inst_info["patched_by_r"]:
                log_progress(inst_prefix, f"apply r-light pruning (r={model_spec.r})...")
                patch_t0 = time.perf_counter()
                patched_data = prune_r_light_single(inst_info["raw_data"], r=int(model_spec.r))
                patch_sec = time.perf_counter() - patch_t0
                inst_info["patched_by_r"][int(model_spec.r)] = (patched_data, float(patch_sec))
            patched_data, patch_sec = inst_info["patched_by_r"][int(model_spec.r)]

            log_progress(inst_prefix, "run 1-pass inference...")
            res = run_onepass_instance(
                data_cpu=patched_data,
                coords_np=inst_info["coords_np"],
                device=device,
                packer=packer,
                runner=runner,
                model_bundle=model_bundle,
                catalog=catalog,
                optimum=inst_info["optimum"],
            )

            shared_preprocess_sec = float(
                inst_info["shared_timing"]["parse_sec"]
                + inst_info["shared_timing"]["spanner_construction_sec"]
                + inst_info["shared_timing"]["quadtree_building_sec"]
            )
            record = {
                "model": model_spec.model_name,
                "ckpt_path": str(model_spec.ckpt_path),
                "r": int(model_spec.r),
                "matching_max_used": int(model_bundle.matching_max_used),
                "instance": str(inst_info["name"]),
                "n": int(inst_info["n"]),
                "edge_weight_type": str(inst_info["edge_weight_type"]),
                "opt_obj": inst_info["optimum"],
                "shared_timing": dict(inst_info["shared_timing"]),
                "shared_preprocess_sec": shared_preprocess_sec,
                "patching_sec": float(patch_sec),
                "patch_plus_model_sec": float(patch_sec + res["time_sec"]),
                "feasible": bool(res["feasible"]),
                "obj": res["obj"],
                "gap_pct": res["gap_pct"],
                "time_sec": float(res["time_sec"]),
                "euclidean_length": res["euclidean_length"],
                "order": list(res["order"]),
                "dp_cost": res.get("dp_cost"),
                "num_sigma_total": int(res.get("num_sigma_total", 0)),
                "timing": dict(res.get("timing", {})),
                "dp_stats": dict(res.get("dp_stats", {})),
                "tour_stats": dict(res.get("tour_stats", {})),
            }
            records.append(record)
            log_progress(
                inst_prefix,
                f"done obj={format_obj(record['obj'])} gap={format_gap(record['gap_pct'])} time={format_time(record['time_sec'])}",
            )

        del runner, catalog, packer, model_bundle
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fieldnames = build_fieldnames()
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(flatten_record(record))

    metadata = {
        "created_at_utc": utc_now_iso(),
        "tsplib_dir": str(tsplib_dir.resolve()),
        "instance_names": list(instance_names),
        "instances": [
            {
                "name": inst["name"],
                "n": int(inst["n"]),
                "path": str(inst["path"]),
                "edge_weight_type": str(inst["edge_weight_type"]),
                "optimum": inst["optimum"],
                "shared_timing": dict(inst["shared_timing"]),
            }
            for inst in instance_cache
        ],
        "models": [
            {
                "model_name": spec.model_name,
                "ckpt_path": str(spec.ckpt_path),
                "r": int(spec.r),
                "matching_max_used": int(spec.matching_max_used),
            }
            for spec in model_specs
        ],
        "device": str(device),
        "dp_topk": int(args.dp_topk),
        "dp_max_sigma": int(args.dp_max_sigma),
        "dp_child_catalog_cap": int(args.dp_child_catalog_cap),
        "dp_fallback_exact": bool(args.dp_fallback_exact),
        "dp_leaf_workers": int(args.dp_leaf_workers),
        "dp_parse_mode": str(args.dp_parse_mode),
    }
    summary = build_summary(records)
    payload = {
        "meta": metadata,
        "summary": summary,
        "records": records,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    report_lines = build_report_lines(metadata=metadata, records=records, summary=summary)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    for line in report_lines:
        print(line)


if __name__ == "__main__":
    main()
