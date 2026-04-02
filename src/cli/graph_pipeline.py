# src/cli/graph_pipeline.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import contextlib
import io
import time
from typing import Any, Dict

import torch

from src.graph.build_raw_pyramid import RawPyramidBuilder
from src.graph.prune_pyramid import prune_r_light_single
from src.graph.rao_smith_patch import patch_batch
from src.graph.spanner import SpannerBuilder


def normalize_spanner_mode(value: str) -> str:
    text = str(value).strip().lower()
    aliases = {
        "delaunay": "delaunay",
        "del": "delaunay",
        "theta": "theta",
        "theta_graph": "theta",
        "theta-graph": "theta",
    }
    if text not in aliases:
        raise ValueError(f"Unsupported spanner mode: {value}")
    return aliases[text]


def normalize_patching_mode(value: str) -> str:
    text = str(value).strip().lower()
    aliases = {
        "prune": "prune",
        "r_light": "prune",
        "r-light": "prune",
        "engineering": "prune",
        "arora": "arora",
        "rao": "arora",
        "rao_smith": "arora",
        "rao-smith": "arora",
        "smith": "arora",
    }
    if text not in aliases:
        raise ValueError(f"Unsupported patching mode: {value}")
    return aliases[text]


def build_spanner_builder(*, spanner_mode: str, theta_k: int) -> SpannerBuilder:
    return SpannerBuilder(mode=normalize_spanner_mode(spanner_mode), theta_k=int(theta_k))


def select_effective_edge_index(data_cpu) -> torch.Tensor:
    edge_index = data_cpu.spanner_edge_index.detach().cpu()
    if hasattr(data_cpu, "alive_edge_id") and getattr(data_cpu, "alive_edge_id") is not None:
        alive_eids = getattr(data_cpu, "alive_edge_id").detach().cpu().long().view(-1)
        return edge_index[:, alive_eids]
    if hasattr(data_cpu, "edge_alive_mask") and getattr(data_cpu, "edge_alive_mask") is not None:
        alive_mask = getattr(data_cpu, "edge_alive_mask").detach().cpu().bool().view(-1)
        return edge_index[:, alive_mask]
    return edge_index


def preprocess_points_to_hierarchy(
    points_cpu: torch.Tensor,
    *,
    r: int,
    num_workers: int,
    raw_builder: RawPyramidBuilder,
    spanner_builder: SpannerBuilder,
    patching_mode: str,
) -> Dict[str, Any]:
    patching_mode = normalize_patching_mode(patching_mode)
    points_cpu = torch.as_tensor(points_cpu, dtype=torch.float32).detach().cpu()
    points_batch = points_cpu.unsqueeze(0)

    shared_timing = {
        "spanner_construction_sec": 0.0,
        "quadtree_building_sec": 0.0,
        "patching_sec": 0.0,
    }

    sp_t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        edge_index_sp, edge_attr_sp, batch_idx_sp = spanner_builder.build_batch(
            points_batch,
            num_workers=max(1, int(num_workers)),
        )
    shared_timing["spanner_construction_sec"] = time.perf_counter() - sp_t0
    edge_index_sp = edge_index_sp.detach().cpu()
    edge_attr_sp = edge_attr_sp.detach().cpu()
    batch_idx_sp = batch_idx_sp.detach().cpu()

    if patching_mode == "arora":
        patch_t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            patched_edge_index, patched_edge_attr, _ = patch_batch(
                points_batch,
                edge_index_sp,
                edge_attr_sp,
                batch_idx_sp,
                r=int(r),
                max_points_per_leaf=int(raw_builder.max_points),
                max_depth=int(raw_builder.max_depth),
            )
        shared_timing["patching_sec"] = time.perf_counter() - patch_t0

        qt_t0 = time.perf_counter()
        raw_data = raw_builder.process_sample(
            points_cpu,
            patched_edge_index.detach().cpu(),
            patched_edge_attr.detach().cpu(),
        )
        shared_timing["quadtree_building_sec"] = time.perf_counter() - qt_t0
        data_cpu = raw_data
    else:
        qt_t0 = time.perf_counter()
        raw_data = raw_builder.process_sample(points_cpu, edge_index_sp, edge_attr_sp)
        shared_timing["quadtree_building_sec"] = time.perf_counter() - qt_t0

        patch_t0 = time.perf_counter()
        data_cpu = prune_r_light_single(raw_data, r=int(r))
        shared_timing["patching_sec"] = time.perf_counter() - patch_t0

    return {
        "raw_data": raw_data,
        "data_cpu": data_cpu,
        "shared_timing": shared_timing,
        "spanner_edge_index": edge_index_sp,
        "spanner_edge_attr": edge_attr_sp,
        "spanner_batch_idx": batch_idx_sp,
    }


__all__ = [
    "build_spanner_builder",
    "normalize_patching_mode",
    "normalize_spanner_mode",
    "preprocess_points_to_hierarchy",
    "select_effective_edge_index",
]
