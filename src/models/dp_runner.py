# src/models/dp_runner.py
# -*- coding: utf-8 -*-
"""
1-Pass DP Runner for Neural DP (Rao'98-style TSP).

Implements the theoretical algorithm from Section 3 / Figure 2 of the paper:

  For each node B (bottom-up):
    if leaf:
      C_B = leaf_exact_solve(B)
      h_B = LeafEncoder(B)
    else:
      h_B = MergeEncoder(B, {h_{B_i}})
      parent_memory = MergeDecoder.build_parent_memory(h_B, {h_{B_i}}, ...)
      for sigma in Omega_cand(B):
        tau_tilde = MergeDecoder.decode_sigma(sigma, parent_memory)
        tau = PARSE(tau_tilde)
        if VERIFYTUPLE(sigma, tau):
          C_B[sigma] = sum(C_{B_i}[tau_i])
          backptr[sigma] = tau
        else:
          try top-K or exact fallback

  Traceback from root: sigma* = argmin C_root, follow backpointers.

IMPORTANT: This is a NEW module for the 1-pass DP pipeline. It does NOT modify
or replace any existing 2-pass code (BottomUpTreeRunner, TopDownTreeRunner, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

# Use 'spawn' context to avoid CUDA + fork() deadlocks.
# After CUDA is initialized in the parent, fork() inherits a broken
# CUDA context in child processes, causing silent hangs.
_MP_CTX = mp.get_context("spawn")

import torch
from torch import Tensor

from .bc_state_catalog import BoundaryStateCatalog, build_boundary_state_catalog, state_mask_from_iface_mask
from .dp_core import (
    CorrespondenceMaps,
    batch_check_c1c2,
    build_correspondence_maps,
    leaf_exact_solve,
    parse_activation_batch,
    parse_by_catalog_enum,
    parse_continuous,
    parse_continuous_topk,
    propagate_c1_constraints,
    verify_tuple,
)
from .merge_decoder import MergeDecoder, ParentMemory
from .node_token_packer import PackedBatch, PackedLeafPoints, PackedNodeTokens


# ─── Parallel leaf solver worker ──────────────────────────────────────────────

def _leaf_solve_worker(args):
    """Worker function for parallel leaf_exact_solve (runs in subprocess)."""
    (points_xy, point_mask, iface_eid, iface_mask,
     iface_boundary_dir, iface_feat6, state_used_iface,
     state_mate, state_mask_row, box_xy, is_root) = args
    return leaf_exact_solve(
        points_xy=points_xy,
        point_mask=point_mask,
        iface_eid=iface_eid,
        iface_mask=iface_mask,
        iface_boundary_dir=iface_boundary_dir,
        iface_feat6=iface_feat6,
        state_used_iface=state_used_iface,
        state_mate=state_mate,
        state_mask=state_mask_row,
        box_xy=box_xy,
        is_root=is_root,
    )


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class CostTableEntry:
    """Cost table for a single node.

    costs[s]:     cost of state s, +inf if infeasible
    backptr[s]:   (child_state_indices) tuple for traceback, None if leaf or infeasible
    """
    costs: Tensor               # [S] float
    backptr: Dict[int, Tuple[int, int, int, int]]  # sigma_idx -> (s1, s2, s3, s4) child state indices


@dataclass
class OnePassDPResult:
    """Result of the 1-pass DP.

    tour_cost:    scalar, cost of the best tour found in the original Euclidean
                  scale of the instance (not root-normalized)
    tour_order:   explicit Hamiltonian cycle reconstructed directly from traceback
    tour_length:  Euclidean length of `tour_order` in the original instance scale
    tour_feasible:whether direct reconstruction produced a legal Hamiltonian cycle
    root_sigma:   index of the optimal root state
    leaf_states:  mapping leaf_node_id -> state_index in the catalog
    cost_tables:  mapping node_id -> CostTableEntry (for debugging)
    stats:        diagnostic statistics
    """
    tour_cost: float
    root_sigma: int
    leaf_states: Dict[int, int]
    cost_tables: Dict[int, CostTableEntry]
    stats: Dict[str, Any]
    tour_order: List[int] = field(default_factory=list)
    tour_length: float = float("inf")
    tour_feasible: bool = False
    tour_stats: Dict[str, Any] = field(default_factory=dict)


# ─── Helper: state index lookup ──────────────────────────────────────────────

def _find_state_index(
    child_a: Tensor,        # [Ti] bool
    child_mate: Tensor,     # [Ti] long
    catalog: BoundaryStateCatalog,
) -> int:
    """Find the catalog index matching a discrete (a, mate) state.

    Returns -1 if no exact match found.
    Uses vectorized comparison instead of Python loop.
    """
    device = child_a.device
    cat_used = catalog.used_iface   # [S, Ti]
    cat_mate = catalog.mate         # [S, Ti]
    if cat_used.device != device:
        cat_used = cat_used.to(device)
        cat_mate = cat_mate.to(device)

    match_a = (cat_used == child_a.unsqueeze(0)).all(dim=1)     # [S]
    match_m = (cat_mate == child_mate.unsqueeze(0)).all(dim=1)  # [S]
    match = match_a & match_m
    indices = torch.nonzero(match, as_tuple=False).flatten()
    if indices.numel() > 0:
        return int(indices[0].item())
    return -1


# ─── Main Runner ─────────────────────────────────────────────────────────────

class OnePassDPRunner:
    """1-Pass bottom-up DP runner with sigma-conditioned neural merge.

    Usage:
        runner = OnePassDPRunner(r=4, max_used=4, topk=5)
        result = runner.run_single(
            tokens=..., leaves=...,
            leaf_encoder=..., merge_encoder=..., merge_decoder=...,
        )

    The runner:
      1. Builds a boundary state catalog Omega(B)
      2. Traverses bottom-up: leaf exact solve, then neural merge with DP
      3. Returns cost tables, backpointers, and traceback
    """

    def __init__(
        self,
        *,
        r: int = 4,
        max_used: int = 4,
        topk: int = 5,
        max_sigma_enumerate: Optional[int] = None,
        max_child_catalog_states: Optional[int] = None,
        fallback_exact: bool = True,
        num_leaf_workers: int = 0,
        parse_mode: str = "catalog_enum",
        max_decode_batch_size: Optional[int] = 60000,
    ) -> None:
        self.r = r
        self.max_used = max_used
        self.topk = topk
        if max_sigma_enumerate is None:
            self.max_sigma_enumerate = None
        else:
            cap = int(max_sigma_enumerate)
            self.max_sigma_enumerate = cap if cap > 0 else None
        if max_child_catalog_states is None:
            self.max_child_catalog_states = None
        else:
            cap = int(max_child_catalog_states)
            self.max_child_catalog_states = cap if cap > 0 else None
        self.fallback_exact = fallback_exact
        self.num_leaf_workers = num_leaf_workers
        if max_decode_batch_size is None:
            self.max_decode_batch_size = None
        else:
            cap = int(max_decode_batch_size)
            self.max_decode_batch_size = cap if cap > 0 else None
        if parse_mode not in ("catalog_enum", "heuristic"):
            raise ValueError(f"parse_mode must be 'catalog_enum' or 'heuristic', got '{parse_mode}'")
        self.parse_mode = parse_mode

    def _apply_sigma_cap(self, candidate_indices: Tensor) -> Tensor:
        """Apply an explicit heuristic sigma cap.

        IMPORTANT:
          - `None` / `<= 0` means "do not truncate" and preserves the certified
            per-state DP semantics from the paper.
          - A positive cap is an opt-in heuristic approximation. We keep the
            current catalog order instead of pretending it is safe/certified.
        """
        if self.max_sigma_enumerate is None:
            return candidate_indices
        if candidate_indices.numel() <= self.max_sigma_enumerate:
            return candidate_indices
        return candidate_indices[:self.max_sigma_enumerate]

    @staticmethod
    def _new_depth_stats_bucket() -> Dict[str, float]:
        """Create an empty per-depth statistics bucket."""
        return {
            "num_internal_nodes": 0.0,
            "num_sigma_total": 0.0,
            "num_parse_ok": 0.0,
            "num_topk_ok": 0.0,
            "num_fallback": 0.0,
            "num_infeasible": 0.0,
            "fallback_rate": 0.0,
        }

    @classmethod
    def _ensure_depth_stats_bucket(
        cls,
        stats: Dict[str, Any],
        depth: int,
    ) -> Dict[str, float]:
        """Return the mutable per-depth stats bucket for one merge depth."""
        depth_stats = stats.setdefault("depth_stats", {})
        depth_key = str(int(depth))
        bucket = depth_stats.get(depth_key)
        if bucket is None:
            bucket = cls._new_depth_stats_bucket()
            depth_stats[depth_key] = bucket
        return bucket

    @staticmethod
    def _bump_stat(
        stats: Dict[str, Any],
        key: str,
        *,
        amount: float = 1.0,
        depth_bucket: Optional[Dict[str, float]] = None,
    ) -> None:
        """Accumulate one global stat and, optionally, the current depth bucket."""
        stats[key] = float(stats.get(key, 0.0) + amount)
        if depth_bucket is not None:
            depth_bucket[key] = float(depth_bucket.get(key, 0.0) + amount)

    @staticmethod
    def _finalize_depth_stats_bucket(bucket: Dict[str, float]) -> None:
        """Update derived metrics for a finished depth bucket."""
        total = float(bucket.get("num_sigma_total", 0.0))
        fallback = float(bucket.get("num_fallback", 0.0))
        bucket["fallback_rate"] = (fallback / total) if total > 0.0 else 0.0

    @classmethod
    def _refresh_depth_fallback_rates(cls, stats: Dict[str, Any]) -> None:
        """Materialize a compact depth->fallback_rate summary in stats."""
        depth_stats = stats.get("depth_stats", {})
        if not isinstance(depth_stats, dict):
            stats["depth_fallback_rates"] = {}
            return
        stats["depth_fallback_rates"] = {
            depth_key: float(bucket.get("fallback_rate", 0.0))
            for depth_key, bucket in sorted(
                depth_stats.items(),
                key=lambda kv: int(kv[0]),
                reverse=True,
            )
        }

    @staticmethod
    def _root_scale(tokens: PackedNodeTokens) -> float:
        """Return the original root-box scale for this single graph."""
        root_scale = getattr(tokens, "root_scale_s", None)
        if root_scale is None:
            return 1.0
        if torch.is_tensor(root_scale):
            if root_scale.numel() == 0:
                return 1.0
            return float(root_scale.reshape(-1)[0].item())
        return float(root_scale)

    @torch.no_grad()
    def run_single(
        self,
        *,
        tokens: PackedNodeTokens,
        leaves: PackedLeafPoints,
        leaf_encoder,     # LeafEncoder protocol
        merge_encoder,    # MergeEncoder protocol
        merge_decoder: MergeDecoder,
        catalog: Optional[BoundaryStateCatalog] = None,
    ) -> OnePassDPResult:
        """Run 1-pass DP on a single graph.

        All encoders/decoder should be in eval mode. This runs with no_grad.
        """
        device = tokens.tree_parent_index.device
        total_M = int(tokens.tree_parent_index.numel())
        Ti = int(tokens.iface_mask.shape[1])
        tokens_cpu = self._make_cpu_token_cache(tokens)

        # Build catalog if not provided
        if catalog is None:
            catalog = build_boundary_state_catalog(
                num_slots=Ti, max_used=self.max_used, device=device,
            )
        S = int(catalog.used_iface.shape[0])
        cat_used_cpu = catalog.used_iface.detach().cpu()
        cat_mate_cpu = catalog.mate.detach().cpu()

        # Per-node valid state mask
        node_state_mask = state_mask_from_iface_mask(
            iface_mask=tokens_cpu.iface_mask,
            state_used_iface=cat_used_cpu,
        )  # [M, S]

        # Storage
        cost_tables: Dict[int, CostTableEntry] = {}
        z_storage: Optional[Tensor] = None
        computed = torch.zeros(total_M, dtype=torch.bool, device=device)

        # Build leaf_row_for_node mapping
        leaf_row_for_node = torch.full((total_M,), -1, dtype=torch.long, device=device)
        if leaves.leaf_node_id.numel() > 0:
            leaf_row_for_node[leaves.leaf_node_id] = torch.arange(
                leaves.leaf_node_id.numel(), device=device, dtype=torch.long,
            )

        depth = tokens.tree_node_depth.long()
        max_depth = int(depth.max().item()) if depth.numel() > 0 else 0

        # Stats
        stats: Dict[str, Any] = {
            "num_nodes": float(total_M),
            "num_leaves": 0.0,
            "num_internal": 0.0,
            "num_sigma_total": 0.0,
            "num_parse_ok": 0.0,
            "num_topk_ok": 0.0,
            "num_fallback": 0.0,
            "num_infeasible": 0.0,
            "num_exact_too_large": 0.0,
            "depth_stats": {},
            "depth_fallback_rates": {},
        }

        root_id = int(tokens_cpu.root_id[0].item())

        # ─── Bottom-up traversal ─────────────────────────────────────────
        t_start = time.time()
        print(f"  [dp] total_M={total_M}, max_depth={max_depth}, S={S}, Ti={Ti}")
        for d in range(max_depth, -1, -1):
            nids_at_d = torch.nonzero(depth == d, as_tuple=False).flatten()
            if nids_at_d.numel() == 0:
                continue

            is_leaf_d = tokens.is_leaf[nids_at_d]
            leaf_nids = nids_at_d[is_leaf_d]
            internal_nids = nids_at_d[~is_leaf_d]
            t_depth = time.time()
            print(f"  [dp] depth={d}: {leaf_nids.numel()} leaves, {internal_nids.numel()} internal")

            # ──── Process leaves ─────────────────────────────────────────
            if leaf_nids.numel() > 0:
                # Encode leaves (batched)
                leaf_inputs = self._gather_node_fields(tokens, leaf_nids)
                rows = leaf_row_for_node[leaf_nids]
                leaf_inputs["leaf_points_xy"] = leaves.point_xy[rows]
                leaf_inputs["leaf_points_mask"] = leaves.point_mask[rows]

                z_leaf = leaf_encoder(**leaf_inputs)
                if isinstance(z_leaf, tuple):
                    z_leaf = z_leaf[0]

                if z_storage is None:
                    z_storage = torch.zeros(total_M, z_leaf.shape[1], dtype=z_leaf.dtype, device=device)
                z_storage[leaf_nids] = z_leaf

                t_enc = time.time()
                print(f"    [leaf] encode {leaf_nids.numel()} leaves: {t_enc - t_depth:.2f}s")

                # Prepare args for leaf_exact_solve (all on CPU for efficiency)
                cat_used_cpu = catalog.used_iface.cpu()
                cat_mate_cpu = catalog.mate.cpu()
                leaf_args_list = []
                leaf_nid_list = []
                for nid_t in leaf_nids:
                    nid = int(nid_t.item())
                    row = int(leaf_row_for_node[nid].item())
                    leaf_nid_list.append(nid)
                    leaf_args_list.append((
                        leaves.point_xy[row].cpu(),
                        leaves.point_mask[row].cpu(),
                        tokens_cpu.iface_eid[nid],
                        tokens_cpu.iface_mask[nid],
                        tokens_cpu.iface_boundary_dir[nid],
                        tokens_cpu.iface_feat6[nid],
                        cat_used_cpu,
                        cat_mate_cpu,
                        node_state_mask[nid],
                        tokens_cpu.tree_node_feat_rel[nid],
                        nid == root_id,
                    ))

                num_w = self.num_leaf_workers
                num_leaves_d = len(leaf_nid_list)
                if num_w > 0 and num_leaves_d > 1:
                    # Parallel leaf exact solve
                    print(f"    [leaf] exact_solve {num_leaves_d} leaves with {num_w} workers...")
                    with ProcessPoolExecutor(max_workers=num_w, mp_context=_MP_CTX) as pool:
                        results = list(pool.map(_leaf_solve_worker, leaf_args_list))
                    for nid, costs in zip(leaf_nid_list, results):
                        cost_tables[nid] = CostTableEntry(costs=costs.cpu(), backptr={})
                        stats["num_leaves"] += 1
                    print(f"    [leaf] exact_solve {num_leaves_d}/{num_leaves_d} done ({time.time()-t_enc:.1f}s)")
                else:
                    # Sequential fallback
                    for li, (nid, args) in enumerate(zip(leaf_nid_list, leaf_args_list)):
                        costs = _leaf_solve_worker(args)
                        cost_tables[nid] = CostTableEntry(costs=costs.cpu(), backptr={})
                        stats["num_leaves"] += 1
                        if (li + 1) % 50 == 0 or (li + 1) == num_leaves_d:
                            print(f"    [leaf] exact_solve {li+1}/{num_leaves_d} done ({time.time()-t_enc:.1f}s)")

                computed[leaf_nids] = True

            # ──── Process internal nodes ─────────────────────────────────
            if internal_nids.numel() > 0:
                depth_bucket = self._ensure_depth_stats_bucket(stats, d)
                if z_storage is None:
                    raise RuntimeError("Internal node encountered before any leaf.")

                # Encode internal nodes (batched)
                ch = tokens.tree_children_index[internal_nids].long()
                child_mask_batch = ch >= 0
                ch_clamped = ch.clamp_min(0)
                child_z_batch = z_storage[ch_clamped]
                child_z_batch = child_z_batch * child_mask_batch.unsqueeze(-1).float()

                merge_inputs = self._gather_node_fields(tokens, internal_nids)
                merge_inputs["child_z"] = child_z_batch
                merge_inputs["child_mask"] = child_mask_batch

                z_parent = merge_encoder(**merge_inputs)
                if isinstance(z_parent, tuple):
                    z_parent = z_parent[0]
                z_storage[internal_nids] = z_parent

                t_merge_enc = time.time()
                N_int = internal_nids.numel()
                depth_bucket["num_internal_nodes"] += float(N_int)
                print(f"    [internal] encode {N_int} nodes: {t_merge_enc - t_depth:.2f}s")

                # Build parent memory for ALL internal nodes in ONE batched GPU call
                batch_parent_mem = merge_decoder.build_parent_memory(
                    node_feat_rel=tokens.tree_node_feat_rel[internal_nids],
                    node_depth=tokens.tree_node_depth[internal_nids],
                    z_node=z_storage[internal_nids],
                    iface_feat6=tokens.iface_feat6[internal_nids],
                    iface_mask=tokens.iface_mask[internal_nids],
                    iface_boundary_dir=tokens.iface_boundary_dir[internal_nids],
                    iface_inside_endpoint=tokens.iface_inside_endpoint[internal_nids],
                    iface_inside_quadrant=tokens.iface_inside_quadrant[internal_nids],
                    cross_feat6=tokens.cross_feat6[internal_nids],
                    cross_mask=tokens.cross_mask[internal_nids],
                    cross_child_pair=tokens.cross_child_pair[internal_nids],
                    cross_is_leaf_internal=tokens.cross_is_leaf_internal[internal_nids],
                    child_z=child_z_batch,
                    child_exists_mask=child_mask_batch,
                )
                t_mem = time.time()
                print(f"    [internal] parent_memory({N_int}): {t_mem - t_merge_enc:.2f}s")

                # ── Batched decode: collect all candidates from ALL nodes,
                #    one GPU forward pass ──────────────────────────────────
                cat_used_dev = catalog.used_iface.to(device)
                cat_mate_dev = catalog.mate.to(device)

                per_node_info = []   # [(nid, candidates, maps, child_iface_mask_4, child_iface_bdir, children, child_exists)]
                all_sigma_a = []
                all_sigma_mate = []
                all_iface_mask = []
                all_child_iface_mask = []
                repeat_counts = []

                for idx in range(N_int):
                    nid = int(internal_nids[idx].item())
                    children_n = tokens_cpu.tree_children_index[nid].long()
                    child_exists_n = children_n >= 0
                    ch_clamped_n = children_n.clamp_min(0)

                    maps_n = build_correspondence_maps(
                        parent_iface_eid=tokens_cpu.iface_eid[nid],
                        parent_iface_mask=tokens_cpu.iface_mask[nid].bool(),
                        parent_iface_bdir=tokens_cpu.iface_boundary_dir[nid],
                        parent_cross_eid=tokens_cpu.cross_eid[nid],
                        parent_cross_mask=tokens_cpu.cross_mask[nid].bool(),
                        parent_cross_child_pair=tokens_cpu.cross_child_pair[nid],
                        children_iface_eid=tokens_cpu.iface_eid[ch_clamped_n],
                        children_iface_mask=tokens_cpu.iface_mask[ch_clamped_n].bool(),
                        children_iface_bdir=tokens_cpu.iface_boundary_dir[ch_clamped_n],
                        child_exists=child_exists_n,
                    )
                    child_iface_mask_4 = tokens_cpu.iface_mask[ch_clamped_n].bool() & child_exists_n.unsqueeze(-1)
                    child_iface_bdir_n = tokens_cpu.iface_boundary_dir[ch_clamped_n]
                    parent_iface_mask_n = tokens_cpu.iface_mask[nid].bool()

                    # Get + pre-filter candidates
                    valid_mask = node_state_mask[nid]
                    cands = torch.nonzero(valid_mask, as_tuple=False).flatten()
                    has_child = maps_n.phi_out_child >= 0
                    unmapped = parent_iface_mask_n & ~has_child
                    if unmapped.any() and cands.numel() > 0:
                        c1_pre = ~(cat_used_cpu[cands].bool() & unmapped.unsqueeze(0)).any(dim=1)
                        cands = cands[c1_pre]
                    cands = self._apply_sigma_cap(cands)

                    per_node_info.append((
                        nid, cands, maps_n, child_iface_mask_4,
                        child_iface_bdir_n, children_n, child_exists_n,
                    ))

                    K_i = cands.numel()
                    self._bump_stat(
                        stats,
                        "num_sigma_total",
                        amount=float(K_i),
                        depth_bucket=depth_bucket,
                    )
                    if K_i > 0:
                        cands_dev = cands.to(device=device, dtype=torch.long)
                        all_sigma_a.append(cat_used_dev[cands_dev].float())
                        all_sigma_mate.append(cat_mate_dev[cands_dev])
                        all_iface_mask.append(
                            parent_iface_mask_n.to(device=device).unsqueeze(0).expand(K_i, -1)
                        )
                        all_child_iface_mask.append(
                            child_iface_mask_4.to(device=device).unsqueeze(0).expand(K_i, -1, -1)
                        )
                    repeat_counts.append(K_i)

                # One big GPU decode
                K_total = sum(repeat_counts)
                all_decode_scores = None
                if K_total > 0:
                    big_sigma_a = torch.cat(all_sigma_a)           # [K_total, Ti]
                    big_sigma_mate = torch.cat(all_sigma_mate)     # [K_total, Ti]
                    big_iface_mask = torch.cat(all_iface_mask)     # [K_total, Ti]
                    big_child_mask = torch.cat(all_child_iface_mask)  # [K_total, 4, Ti]

                    # Expand parent memory: repeat each node's memory K_i times
                    mem_tokens_list, mem_mask_list = [], []
                    for idx, rc in enumerate(repeat_counts):
                        if rc > 0:
                            mem_tokens_list.append(batch_parent_mem.tokens[idx:idx+1].expand(rc, -1, -1))
                            mem_mask_list.append(batch_parent_mem.mask[idx:idx+1].expand(rc, -1))
                    big_mem = ParentMemory(
                        tokens=torch.cat(mem_tokens_list),
                        mask=torch.cat(mem_mask_list),
                        iface_slice=batch_parent_mem.iface_slice,
                        cross_slice=batch_parent_mem.cross_slice,
                    )

                    out = merge_decoder.decode_sigma_chunked(
                        sigma_a=big_sigma_a,
                        sigma_mate=big_sigma_mate,
                        sigma_iface_mask=big_iface_mask.bool(),
                        parent_memory=big_mem,
                        child_iface_mask=big_child_mask.bool(),
                        max_batch_size=self.max_decode_batch_size,
                    )
                    all_decode_scores = torch.sigmoid(out.child_scores).cpu()  # [K_total, 4, Ti]

                t_decode = time.time()
                print(f"    [internal] decode({N_int} nodes, {K_total} sigmas): {t_decode - t_mem:.2f}s")

                # ── Per-node: PARSE + VERIFY + fallback (CPU-bound) ──
                offset = 0
                for idx, (nid, cands, maps_n, child_iface_mask_4,
                          child_iface_bdir_n, children_n, child_exists_n) in enumerate(per_node_info):
                    K_i = repeat_counts[idx]
                    node_scores = all_decode_scores[offset:offset+K_i] if K_i > 0 else None
                    offset += K_i

                    self._process_internal_node_post_decode(
                        nid=nid, tokens=tokens_cpu,
                        candidate_indices=cands,
                        decode_scores=node_scores,
                        maps=maps_n,
                        child_iface_mask_4=child_iface_mask_4,
                        child_iface_bdir=child_iface_bdir_n,
                        children=children_n,
                        child_exists=child_exists_n,
                        catalog=catalog,
                        cat_used_dev=cat_used_cpu,
                        cat_mate_dev=cat_mate_cpu,
                        node_state_mask=node_state_mask,
                        cost_tables=cost_tables,
                        stats=stats,
                        depth_bucket=depth_bucket,
                    )
                    stats["num_internal"] += 1
                    ct = cost_tables[nid]
                    n_feasible = int((ct.costs < float("inf")).sum().item())
                    print(
                        f"    [internal] DP {idx+1}/{N_int} "
                        f"nid={nid} feasible={n_feasible}/{int(node_state_mask[nid].sum().item())} "
                        f"({time.time() - t_decode:.1f}s)"
                    )

                computed[internal_nids] = True
                self._finalize_depth_stats_bucket(depth_bucket)
                self._refresh_depth_fallback_rates(stats)
                print(
                    f"  [dp] depth={d} total: {time.time()-t_depth:.2f}s | "
                    f"sigma={int(depth_bucket['num_sigma_total'])} "
                    f"fallback={int(depth_bucket['num_fallback'])}/{int(depth_bucket['num_sigma_total'])} "
                    f"({depth_bucket['fallback_rate'] * 100.0:.1f}%)"
                )

        self._refresh_depth_fallback_rates(stats)
        print(f"  [dp] bottom-up done: {time.time()-t_start:.2f}s, stats={stats}")
        # ─── Find best root state ────────────────────────────────────────
        root_ct = cost_tables.get(root_id)
        if root_ct is None:
            return OnePassDPResult(
                tour_cost=float("inf"), root_sigma=-1,
                leaf_states={}, cost_tables=cost_tables, stats=stats,
            )

        valid_costs = root_ct.costs.clone()
        valid_costs[~node_state_mask[root_id]] = float("inf")
        best_sigma = int(valid_costs.argmin().item())
        best_cost_norm = float(valid_costs[best_sigma].item())

        # If all states are infeasible, return failure explicitly
        if best_cost_norm == float("inf"):
            return OnePassDPResult(
                tour_cost=float("inf"), root_sigma=-1,
                leaf_states={}, cost_tables=cost_tables, stats=stats,
            )

        root_scale = self._root_scale(tokens)
        stats["root_scale_s"] = root_scale
        stats["tour_cost_normalized"] = best_cost_norm
        best_cost = best_cost_norm * root_scale

        # ─── Top-down traceback ──────────────────────────────────────────
        leaf_states = self._traceback(
            root_id=root_id,
            root_sigma=best_sigma,
            tokens=tokens,
            cost_tables=cost_tables,
        )

        result = OnePassDPResult(
            tour_cost=best_cost,
            root_sigma=best_sigma,
            leaf_states=leaf_states,
            cost_tables=cost_tables,
            stats=stats,
        )
        if hasattr(leaves, "point_idx"):
            try:
                from .tour_reconstruct import reconstruct_tour_direct

                t_reconstruct = time.time()
                direct_tour = reconstruct_tour_direct(
                    result=result,
                    tokens=tokens,
                    leaves=leaves,
                    catalog=catalog,
                )
                result.tour_order = direct_tour.order
                result.tour_length = direct_tour.length
                result.tour_feasible = direct_tour.feasible
                result.tour_stats = dict(direct_tour.stats)
                result.tour_stats["duration"] = time.time() - t_reconstruct
                stats["direct_reconstruct_time_s"] = result.tour_stats["duration"]
            except Exception as exc:  # pragma: no cover - defensive only
                result.tour_order = []
                result.tour_length = float("inf")
                result.tour_feasible = False
                result.tour_stats = {"error": f"direct_reconstruction_failed: {exc}"}
        else:
            result.tour_stats = {"error": "missing_point_idx"}

        return result

    # ─── Internal: process one internal node ─────────────────────────────

    def _process_internal_node(
        self,
        *,
        nid: int,
        tokens: PackedNodeTokens,
        z_storage: Tensor,
        merge_decoder: MergeDecoder,
        catalog: BoundaryStateCatalog,
        node_state_mask: Tensor,
        cost_tables: Dict[int, CostTableEntry],
        stats: Dict[str, Any],
        parent_mem: Optional[ParentMemory] = None,
    ) -> None:
        """Run the DP merge for one internal node."""
        device = tokens.tree_parent_index.device
        Ti = int(tokens.iface_mask.shape[1])
        S = int(catalog.used_iface.shape[0])

        # Get children
        children = tokens.tree_children_index[nid].long()  # [4]
        child_exists = children >= 0                         # [4]
        ch_clamped = children.clamp_min(0)

        # Build correspondence maps
        maps = build_correspondence_maps(
            parent_iface_eid=tokens.iface_eid[nid],
            parent_iface_mask=tokens.iface_mask[nid].bool(),
            parent_iface_bdir=tokens.iface_boundary_dir[nid],
            parent_cross_eid=tokens.cross_eid[nid],
            parent_cross_mask=tokens.cross_mask[nid].bool(),
            parent_cross_child_pair=tokens.cross_child_pair[nid],
            children_iface_eid=tokens.iface_eid[ch_clamped],
            children_iface_mask=tokens.iface_mask[ch_clamped].bool(),
            children_iface_bdir=tokens.iface_boundary_dir[ch_clamped],
            child_exists=child_exists,
        )

        # Build parent memory if not pre-built
        if parent_mem is None:
            parent_mem = merge_decoder.build_parent_memory(
                node_feat_rel=tokens.tree_node_feat_rel[nid].unsqueeze(0),
                node_depth=tokens.tree_node_depth[nid].unsqueeze(0),
                z_node=z_storage[nid].unsqueeze(0),
                iface_feat6=tokens.iface_feat6[nid].unsqueeze(0),
                iface_mask=tokens.iface_mask[nid].unsqueeze(0),
                iface_boundary_dir=tokens.iface_boundary_dir[nid].unsqueeze(0),
                iface_inside_endpoint=tokens.iface_inside_endpoint[nid].unsqueeze(0),
                iface_inside_quadrant=tokens.iface_inside_quadrant[nid].unsqueeze(0),
                cross_feat6=tokens.cross_feat6[nid].unsqueeze(0),
                cross_mask=tokens.cross_mask[nid].unsqueeze(0),
                cross_child_pair=tokens.cross_child_pair[nid].unsqueeze(0),
                cross_is_leaf_internal=tokens.cross_is_leaf_internal[nid].unsqueeze(0),
                child_z=z_storage[ch_clamped].unsqueeze(0),
                child_exists_mask=child_exists.unsqueeze(0),
            )

        # Prepare child iface mask for decode
        child_iface_mask_4 = tokens.iface_mask[ch_clamped].bool()  # [4, Ti]
        child_iface_mask_4 = child_iface_mask_4 & child_exists.unsqueeze(-1)
        child_iface_bdir = tokens.iface_boundary_dir[ch_clamped]   # [4, Ti]

        # Get candidate states for this node
        valid_mask = node_state_mask[nid]  # [S]
        candidate_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
        num_candidates = candidate_indices.numel()

        # ── Pre-filter: reject parent states that activate unmapped outer
        #    boundary slots (they can never satisfy C1). ──
        parent_iface_mask = tokens.iface_mask[nid].bool()                    # [Ti]
        has_child = maps.phi_out_child >= 0                                  # [Ti]
        unmapped_active = parent_iface_mask & ~has_child                     # [Ti]
        if unmapped_active.any():
            cat_used_dev = catalog.used_iface.to(device)                     # [S, Ti]
            # Any sigma activating an unmapped slot will fail C1
            c1_ok = ~(cat_used_dev[candidate_indices].bool()
                      & unmapped_active.unsqueeze(0)).any(dim=1)             # [K]
            candidate_indices = candidate_indices[c1_ok]
            num_candidates = candidate_indices.numel()

        candidate_indices = self._apply_sigma_cap(candidate_indices)
        num_candidates = candidate_indices.numel()
        self._bump_stat(stats, "num_sigma_total", amount=float(num_candidates))

        # Initialize cost table
        costs = torch.full((S,), float("inf"), device=device)
        backptr: Dict[int, Tuple[int, int, int, int]] = {}

        # Pre-transfer catalog tensors to device (avoid per-sigma transfers)
        cat_used_dev = catalog.used_iface.to(device)     # [S, Ti]
        cat_mate_dev = catalog.mate.to(device)           # [S, Ti]

        # Batch decode all candidate sigmas
        if candidate_indices.numel() > 0:
            sigma_a_batch = cat_used_dev[candidate_indices].float()          # [K, Ti]
            sigma_mate_batch = cat_mate_dev[candidate_indices]               # [K, Ti]

            out = merge_decoder.decode_sigma_batch(
                sigma_a=sigma_a_batch,
                sigma_mate=sigma_mate_batch,
                sigma_iface_mask=parent_iface_mask,
                parent_memory=parent_mem,
                child_iface_mask=child_iface_mask_4,
            )
            # out.child_scores: [K, 4, Ti]

            all_scores = torch.sigmoid(out.child_scores)  # [K, 4, Ti]

            # ── Phase A: batch PARSE (rounding only) + batch C1+C2 filter ──
            child_a_all = parse_activation_batch(
                scores_batch=all_scores,
                child_iface_mask=child_iface_mask_4,
                child_iface_bdir=child_iface_bdir,
                child_exists=child_exists,
                maps=maps,
                r=self.r,
            )  # [K, 4, Ti] bool

            parent_a_batch = cat_used_dev[candidate_indices].bool()  # [K, Ti]
            c12_ok = batch_check_c1c2(
                parent_a_batch=parent_a_batch,
                child_a_batch=child_a_all,
                child_iface_mask=child_iface_mask_4,
                child_exists=child_exists,
                maps=maps,
            )  # [K] bool

            # ── Phase B: C1+C2 survivors get PARSE + VERIFY + top-K ──
            survivor_indices = torch.nonzero(c12_ok, as_tuple=False).flatten()
            failed_indices = torch.nonzero(~c12_ok, as_tuple=False).flatten()

            for k_idx_t in survivor_indices:
                k_idx = int(k_idx_t.item())
                si = int(candidate_indices[k_idx].item())
                parent_a = cat_used_dev[si]
                parent_mate = cat_mate_dev[si]
                scores = all_scores[k_idx]

                # Full PARSE (with matching) on this single candidate
                child_a, child_mate = parse_continuous(
                    scores=scores,
                    child_iface_mask=child_iface_mask_4,
                    child_iface_bdir=child_iface_bdir,
                    child_exists=child_exists,
                    maps=maps,
                    r=self.r,
                    parent_a=parent_a.bool(),
                    parent_iface_mask=parent_iface_mask,
                )

                # Full VERIFY (C1+C2 passed in batch, but PARSE may change
                # activations during matching; re-check all)
                ok = verify_tuple(
                    parent_a=parent_a.bool(),
                    parent_mate=parent_mate,
                    parent_iface_mask=parent_iface_mask,
                    child_a=child_a,
                    child_mate=child_mate,
                    child_iface_mask=child_iface_mask_4,
                    child_exists=child_exists,
                    maps=maps,
                )

                if ok:
                    cost, child_state_indices = self._lookup_child_costs(
                        child_a=child_a,
                        child_mate=child_mate,
                        children=children,
                        child_exists=child_exists,
                        cost_tables=cost_tables,
                        catalog=catalog,
                    )
                    if cost < float("inf"):
                        costs[si] = cost
                        backptr[si] = child_state_indices
                        self._bump_stat(stats, "num_parse_ok")
                        continue

                # Top-K fallback
                topk_results = parse_continuous_topk(
                    scores=scores,
                    child_iface_mask=child_iface_mask_4,
                    child_iface_bdir=child_iface_bdir,
                    child_exists=child_exists,
                    maps=maps,
                    parent_a=parent_a.bool(),
                    parent_mate=parent_mate,
                    parent_iface_mask=parent_iface_mask,
                    r=self.r,
                    K=self.topk,
                )

                found = False
                for ca_k, cm_k in topk_results:
                    cost, child_state_indices = self._lookup_child_costs(
                        child_a=ca_k,
                        child_mate=cm_k,
                        children=children,
                        child_exists=child_exists,
                        cost_tables=cost_tables,
                        catalog=catalog,
                    )
                    if cost < float("inf"):
                        costs[si] = cost
                        backptr[si] = child_state_indices
                        self._bump_stat(stats, "num_topk_ok")
                        found = True
                        break

                if not found and self.fallback_exact:
                    cost, child_si = self._exact_fallback(
                        si=si, catalog=catalog, tokens=tokens, nid=nid,
                        children=children, child_exists=child_exists,
                        maps=maps, cost_tables=cost_tables,
                    )
                    if cost < float("inf"):
                        costs[si] = cost
                        backptr[si] = child_si
                        self._bump_stat(stats, "num_fallback")
                    else:
                        self._bump_stat(stats, "num_infeasible")
                elif not found:
                    self._bump_stat(stats, "num_infeasible")

            # ── Phase C: C1+C2 failures — skip PARSE/top-K but try exact ──
            # Only worthwhile if all children have at least one finite-cost state,
            # otherwise exact fallback can never succeed.
            children_have_costs = all(
                (not child_exists[q].item()) or
                (int(children[q].item()) in cost_tables and
                 (cost_tables[int(children[q].item())].costs < float("inf")).any().item())
                for q in range(4)
            )
            if self.fallback_exact and children_have_costs and failed_indices.numel() > 0:
                # Constraint-propagated fallback: no candidate limit needed
                # (C1+C2 pruning keeps search space small, typically ~16 combos)
                for idx_i in range(int(failed_indices.numel())):
                    k_idx = int(failed_indices[idx_i].item())
                    si = int(candidate_indices[k_idx].item())
                    if costs[si] < float("inf"):
                        continue
                    cost, child_si = self._exact_fallback(
                        si=si, catalog=catalog, tokens=tokens, nid=nid,
                        children=children, child_exists=child_exists,
                        maps=maps, cost_tables=cost_tables,
                    )
                    if cost < float("inf"):
                        costs[si] = cost
                        backptr[si] = child_si
                        self._bump_stat(stats, "num_fallback")
                    else:
                        self._bump_stat(stats, "num_infeasible")
            else:
                self._bump_stat(stats, "num_infeasible", amount=float(int(failed_indices.numel())))

        cost_tables[nid] = CostTableEntry(costs=costs, backptr=backptr)

    def _process_internal_node_post_decode(
        self,
        *,
        nid: int,
        tokens: PackedNodeTokens,
        candidate_indices: Tensor,
        decode_scores: Optional[Tensor],   # [K, 4, Ti] sigmoid scores, or None
        maps: CorrespondenceMaps,
        child_iface_mask_4: Tensor,
        child_iface_bdir: Tensor,
        children: Tensor,
        child_exists: Tensor,
        catalog: BoundaryStateCatalog,
        cat_used_dev: Tensor,
        cat_mate_dev: Tensor,
        node_state_mask: Tensor,
        cost_tables: Dict[int, CostTableEntry],
        stats: Dict[str, Any],
        depth_bucket: Optional[Dict[str, float]] = None,
    ) -> None:
        """Post-decode DP merge: PARSE + VERIFY + fallback. Scores already computed."""
        device = tokens.tree_parent_index.device
        Ti = int(tokens.iface_mask.shape[1])
        S = int(catalog.used_iface.shape[0])

        costs = torch.full((S,), float("inf"), device=device)
        backptr: Dict[int, Tuple[int, int, int, int]] = {}

        K = candidate_indices.numel()
        if K > 0 and decode_scores is not None:
            parent_iface_mask = tokens.iface_mask[nid].bool()

            # ── Gather child cost tables (shared across all sigmas for this node) ──
            child_cost_list: List[Optional[Tensor]] = [None] * 4
            for q in range(4):
                if not child_exists[q].item():
                    continue
                cid = int(children[q].item())
                ct = cost_tables.get(cid)
                if ct is not None:
                    child_cost_list[q] = ct.costs

            if self.parse_mode == "catalog_enum":
                # ── Catalog-enumeration PARSE: direct enumeration with neural guidance ──
                for k_idx in range(K):
                    si = int(candidate_indices[k_idx].item())
                    if costs[si] < float("inf"):
                        continue

                    cost, child_si, used_fallback = self._parse_catalog_enum_with_optional_fallback(
                        scores=decode_scores[k_idx],
                        parent_a=cat_used_dev[si],
                        parent_mate=cat_mate_dev[si],
                        parent_iface_mask=parent_iface_mask,
                        child_iface_mask=child_iface_mask_4,
                        child_exists=child_exists,
                        maps=maps,
                        cat_used=cat_used_dev,
                        cat_mate=cat_mate_dev,
                        child_costs=child_cost_list,
                    )
                    if cost < float("inf"):
                        costs[si] = cost
                        backptr[si] = child_si
                        stat_key = "num_fallback" if used_fallback else "num_parse_ok"
                        self._bump_stat(stats, stat_key, depth_bucket=depth_bucket)
                    else:
                        self._bump_stat(stats, "num_infeasible", depth_bucket=depth_bucket)

            else:
                # ── Heuristic PARSE (legacy): threshold rounding + top-K + exact fallback ──
                child_a_all = parse_activation_batch(
                    scores_batch=decode_scores,
                    child_iface_mask=child_iface_mask_4,
                    child_iface_bdir=child_iface_bdir,
                    child_exists=child_exists,
                    maps=maps,
                    r=self.r,
                )
                parent_a_batch = cat_used_dev[candidate_indices].bool()
                c12_ok = batch_check_c1c2(
                    parent_a_batch=parent_a_batch,
                    child_a_batch=child_a_all,
                    child_iface_mask=child_iface_mask_4,
                    child_exists=child_exists,
                    maps=maps,
                )

                survivor_indices = torch.nonzero(c12_ok, as_tuple=False).flatten()
                failed_indices = torch.nonzero(~c12_ok, as_tuple=False).flatten()

                for k_idx_t in survivor_indices:
                    k_idx = int(k_idx_t.item())
                    si = int(candidate_indices[k_idx].item())
                    if costs[si] < float("inf"):
                        continue
                    parent_a = cat_used_dev[si]
                    parent_mate = cat_mate_dev[si]
                    scores = decode_scores[k_idx]

                    child_a, child_mate = parse_continuous(
                        scores=scores, child_iface_mask=child_iface_mask_4,
                        child_iface_bdir=child_iface_bdir, child_exists=child_exists,
                        maps=maps, r=self.r,
                        parent_a=parent_a.bool(),
                        parent_iface_mask=parent_iface_mask,
                    )
                    ok = verify_tuple(
                        parent_a=parent_a.bool(), parent_mate=parent_mate,
                        parent_iface_mask=parent_iface_mask,
                        child_a=child_a, child_mate=child_mate,
                        child_iface_mask=child_iface_mask_4,
                        child_exists=child_exists, maps=maps,
                    )
                    if ok:
                        cost, csi = self._lookup_child_costs(
                            child_a=child_a, child_mate=child_mate,
                            children=children, child_exists=child_exists,
                            cost_tables=cost_tables, catalog=catalog,
                        )
                        if cost < float("inf"):
                            costs[si] = cost
                            backptr[si] = csi
                            self._bump_stat(stats, "num_parse_ok", depth_bucket=depth_bucket)
                            continue

                    topk_results = parse_continuous_topk(
                        scores=scores, child_iface_mask=child_iface_mask_4,
                        child_iface_bdir=child_iface_bdir, child_exists=child_exists,
                        maps=maps, parent_a=parent_a.bool(), parent_mate=parent_mate,
                        parent_iface_mask=parent_iface_mask, r=self.r, K=self.topk,
                    )
                    found = False
                    for ca_k, cm_k in topk_results:
                        cost, csi = self._lookup_child_costs(
                            child_a=ca_k, child_mate=cm_k,
                            children=children, child_exists=child_exists,
                            cost_tables=cost_tables, catalog=catalog,
                        )
                        if cost < float("inf"):
                            costs[si] = cost
                            backptr[si] = csi
                            self._bump_stat(stats, "num_topk_ok", depth_bucket=depth_bucket)
                            found = True
                            break
                    if not found and self.fallback_exact:
                        cost, child_si = self._exact_fallback(
                            si=si, catalog=catalog, tokens=tokens, nid=nid,
                            children=children, child_exists=child_exists,
                            maps=maps, cost_tables=cost_tables,
                        )
                        if cost < float("inf"):
                            costs[si] = cost
                            backptr[si] = child_si
                            self._bump_stat(stats, "num_fallback", depth_bucket=depth_bucket)
                        else:
                            self._bump_stat(stats, "num_infeasible", depth_bucket=depth_bucket)
                    elif not found:
                        self._bump_stat(stats, "num_infeasible", depth_bucket=depth_bucket)

                # C1+C2 failures: constraint-propagated exact fallback
                children_have_costs = all(
                    (not child_exists[q].item()) or
                    (int(children[q].item()) in cost_tables and
                     (cost_tables[int(children[q].item())].costs < float("inf")).any().item())
                    for q in range(4)
                )
                n_failed = int(failed_indices.numel())
                if self.fallback_exact and children_have_costs and n_failed > 0:
                    for ii in range(n_failed):
                        k_idx = int(failed_indices[ii].item())
                        si = int(candidate_indices[k_idx].item())
                        if costs[si] < float("inf"):
                            continue
                        cost, child_si = self._exact_fallback(
                            si=si, catalog=catalog, tokens=tokens, nid=nid,
                            children=children, child_exists=child_exists,
                            maps=maps, cost_tables=cost_tables,
                        )
                        if cost < float("inf"):
                            costs[si] = cost
                            backptr[si] = child_si
                            self._bump_stat(stats, "num_fallback", depth_bucket=depth_bucket)
                        else:
                            self._bump_stat(stats, "num_infeasible", depth_bucket=depth_bucket)
                else:
                    self._bump_stat(
                        stats,
                        "num_infeasible",
                        amount=float(n_failed),
                        depth_bucket=depth_bucket,
                    )

        cost_tables[nid] = CostTableEntry(costs=costs, backptr=backptr)

    def _parse_catalog_enum_with_optional_fallback(
        self,
        *,
        scores: Tensor,
        parent_a: Tensor,
        parent_mate: Tensor,
        parent_iface_mask: Tensor,
        child_iface_mask: Tensor,
        child_exists: Tensor,
        maps: CorrespondenceMaps,
        cat_used: Tensor,
        cat_mate: Tensor,
        child_costs: List[Optional[Tensor]],
    ) -> Tuple[float, Tuple[int, int, int, int], bool]:
        """Run capped catalog_enum first, then exact enumeration if requested."""
        cost, child_si = parse_by_catalog_enum(
            scores=scores,
            parent_a=parent_a,
            parent_mate=parent_mate,
            parent_iface_mask=parent_iface_mask,
            child_iface_mask=child_iface_mask,
            child_exists=child_exists,
            maps=maps,
            cat_used=cat_used,
            cat_mate=cat_mate,
            child_costs=child_costs,
            max_child_states=self.max_child_catalog_states,
        )
        if cost < float("inf"):
            return cost, child_si, False

        should_retry_exact = (
            self.fallback_exact
            and self.max_child_catalog_states is not None
        )
        if not should_retry_exact:
            return cost, child_si, False

        exact_cost, exact_child_si = parse_by_catalog_enum(
            scores=scores,
            parent_a=parent_a,
            parent_mate=parent_mate,
            parent_iface_mask=parent_iface_mask,
            child_iface_mask=child_iface_mask,
            child_exists=child_exists,
            maps=maps,
            cat_used=cat_used,
            cat_mate=cat_mate,
            child_costs=child_costs,
            max_child_states=None,
        )
        return exact_cost, exact_child_si, exact_cost < float("inf")

    # ─── Cost lookup ─────────────────────────────────────────────────────

    def _lookup_child_costs(
        self,
        *,
        child_a: Tensor,          # [4, Ti]
        child_mate: Tensor,       # [4, Ti]
        children: Tensor,         # [4] long, -1 = missing
        child_exists: Tensor,     # [4] bool
        cost_tables: Dict[int, CostTableEntry],
        catalog: BoundaryStateCatalog,
    ) -> Tuple[float, Tuple[int, int, int, int]]:
        """Look up total cost from child cost tables. Returns (cost, child_state_indices)."""
        total_cost = 0.0
        child_state_indices = [-1, -1, -1, -1]

        for q in range(4):
            if not child_exists[q].item():
                child_state_indices[q] = -1
                continue

            cid = int(children[q].item())
            ct = cost_tables.get(cid)
            if ct is None:
                return float("inf"), (-1, -1, -1, -1)

            # Find matching state in catalog
            si = _find_state_index(child_a[q], child_mate[q], catalog)
            if si < 0:
                return float("inf"), (-1, -1, -1, -1)

            c = float(ct.costs[si].item())
            if c == float("inf"):
                return float("inf"), (-1, -1, -1, -1)

            total_cost += c
            child_state_indices[q] = si

        return total_cost, tuple(child_state_indices)

    # ─── Exact fallback (constraint-propagated) ─────────────────────────

    def _exact_fallback(
        self,
        *,
        si: int,
        catalog: BoundaryStateCatalog,
        tokens: PackedNodeTokens,
        nid: int,
        children: Tensor,
        child_exists: Tensor,
        maps: CorrespondenceMaps,
        cost_tables: Dict[int, CostTableEntry],
    ) -> Tuple[float, Tuple[int, int, int, int]]:
        """Constraint-propagated exact fallback for one parent state.

        Instead of blind Cartesian-product enumeration (old: S^4, often > 10000),
        we use C1 + C2 to prune child candidates:

          1. C1 propagation: parent σ determines each child's outer-boundary
             activation via φ^out → filter catalog to matching states.
          2. C2 pruning during backtracking: when choosing child q's state,
             verify child↔child glue activation agrees with already-chosen peers.
          3. C3 + C4: full verify_tuple on surviving combinations.

        Typical search space: Catalan(max_used/2)^4 ≈ 2^4 = 16 for max_used=4.
        No hard combination limit — always runs to completion.
        """
        device = tokens.tree_parent_index.device
        Ti = int(tokens.iface_mask.shape[1])

        cat_used_dev = catalog.used_iface if catalog.used_iface.device == device \
            else catalog.used_iface.to(device)
        cat_mate_dev = catalog.mate if catalog.mate.device == device \
            else catalog.mate.to(device)

        parent_a = cat_used_dev[si]
        parent_mate = cat_mate_dev[si]
        parent_iface_mask = tokens.iface_mask[nid].bool()
        ch_clamped = children.clamp_min(0)
        child_iface_mask_4 = tokens.iface_mask[ch_clamped].bool()
        child_iface_mask_4 = child_iface_mask_4 & child_exists.unsqueeze(-1)

        # ── Step 1: Propagate C1 constraints ──
        c1_required, c1_constrained = propagate_c1_constraints(
            parent_a=parent_a.bool(),
            parent_iface_mask=parent_iface_mask,
            maps=maps,
            child_exists=child_exists,
            child_iface_mask=child_iface_mask_4,
        )

        # ── Step 2: Filter child states by C1 + finite cost ──
        child_valid_states: List[List[int]] = []
        for q in range(4):
            if not child_exists[q].item():
                child_valid_states.append([-1])
                continue
            cid = int(children[q].item())
            ct = cost_tables.get(cid)
            if ct is None:
                child_valid_states.append([])
                continue

            # Vectorized C1 filter: all constrained slots must match
            constrained_q = c1_constrained[q]           # [Ti] bool
            finite_mask = ct.costs < float("inf")       # [S]

            if constrained_q.any():
                required_q = c1_required[q]              # [Ti] bool
                # Compare only constrained columns
                state_vals = cat_used_dev[:, constrained_q].bool()   # [S, K]
                required_vals = required_q[constrained_q].unsqueeze(0)  # [1, K]
                c1_match = (state_vals == required_vals).all(dim=1)     # [S]
                valid_mask = c1_match & finite_mask
            else:
                valid_mask = finite_mask

            valid_indices = valid_mask.nonzero(as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                # No state matches C1 constraints → σ infeasible
                return float("inf"), (-1, -1, -1, -1)
            child_valid_states.append(valid_indices.tolist())

        # ── Step 3: Backtracking enumeration with C2 pruning + C3/C4 verify ──

        # Pre-build glue lookup per child for fast C2 checks.
        # sh_links[q] = list of (slot_in_q, peer_q, peer_slot)
        #   only for peers with index < q (already chosen in enumeration order)
        sh_links: List[List[Tuple[int, int, int]]] = [[] for _ in range(4)]
        for q in range(4):
            if not child_exists[q].item():
                continue
            for s in range(Ti):
                if not child_iface_mask_4[q, s].item():
                    continue
                peer_q = int(maps.phi_glue_peer_child[q, s].item())
                peer_s = int(maps.phi_glue_peer_slot[q, s].item())
                if peer_q < 0 or peer_s < 0:
                    continue
                if peer_q < q and child_exists[peer_q].item():
                    sh_links[q].append((s, peer_q, peer_s))

        best_cost = float("inf")
        best_indices: Tuple[int, int, int, int] = (-1, -1, -1, -1)

        def _enum(q: int, current_cost: float, current_indices: List[int]) -> None:
            nonlocal best_cost, best_indices

            if current_cost >= best_cost:
                return  # cost pruning

            if q == 4:
                # All children chosen — full C3+C4 verify
                child_a = torch.zeros(4, Ti, dtype=torch.bool, device=device)
                child_mate_t = torch.full((4, Ti), -1, dtype=torch.long, device=device)
                for qq in range(4):
                    if child_exists[qq].item() and current_indices[qq] >= 0:
                        child_a[qq] = cat_used_dev[current_indices[qq]]
                        child_mate_t[qq] = cat_mate_dev[current_indices[qq]]

                ok = verify_tuple(
                    parent_a=parent_a.bool(),
                    parent_mate=parent_mate,
                    parent_iface_mask=parent_iface_mask,
                    child_a=child_a,
                    child_mate=child_mate_t,
                    child_iface_mask=child_iface_mask_4,
                    child_exists=child_exists,
                    maps=maps,
                )
                if ok and current_cost < best_cost:
                    best_cost = current_cost
                    best_indices = tuple(current_indices)
                return

            if not child_exists[q].item():
                current_indices.append(-1)
                _enum(q + 1, current_cost, current_indices)
                current_indices.pop()
                return

            cid = int(children[q].item())
            ct = cost_tables.get(cid)
            if ct is None:
                return

            for s in child_valid_states[q]:
                # C2 pruning: check glue agreement with chosen peers
                c2_ok = True
                for my_slot, peer_q, peer_slot in sh_links[q]:
                    peer_si = current_indices[peer_q]
                    if peer_si < 0:
                        continue
                    if bool(cat_used_dev[s, my_slot].item()) != \
                       bool(cat_used_dev[peer_si, peer_slot].item()):
                        c2_ok = False
                        break
                if not c2_ok:
                    continue

                c = float(ct.costs[s].item())
                current_indices.append(s)
                _enum(q + 1, current_cost + c, current_indices)
                current_indices.pop()

        _enum(0, 0.0, [])
        return best_cost, best_indices

    # ─── Traceback ───────────────────────────────────────────────────────

    def _traceback(
        self,
        *,
        root_id: int,
        root_sigma: int,
        tokens: PackedNodeTokens,
        cost_tables: Dict[int, CostTableEntry],
    ) -> Dict[int, int]:
        """Top-down traceback to recover leaf states."""
        leaf_states: Dict[int, int] = {}

        if root_sigma < 0:
            return leaf_states

        queue: List[Tuple[int, int]] = [(root_id, root_sigma)]

        while queue:
            nid, sigma_idx = queue.pop()

            if tokens.is_leaf[nid].item():
                leaf_states[nid] = sigma_idx
                continue

            ct = cost_tables.get(nid)
            if ct is None:
                continue

            child_indices = ct.backptr.get(sigma_idx)
            if child_indices is None:
                continue

            children = tokens.tree_children_index[nid].long()
            for q in range(4):
                cid = int(children[q].item())
                if cid < 0:
                    continue
                c_si = child_indices[q]
                if c_si < 0:
                    continue
                queue.append((cid, c_si))

        return leaf_states

    # ─── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _gather_node_fields(tokens: PackedNodeTokens, nids: Tensor) -> Dict[str, Tensor]:
        return {
            "node_feat_rel": tokens.tree_node_feat_rel[nids],
            "node_depth": tokens.tree_node_depth[nids],
            "iface_feat6": tokens.iface_feat6[nids],
            "iface_mask": tokens.iface_mask[nids],
            "iface_boundary_dir": tokens.iface_boundary_dir[nids],
            "iface_inside_endpoint": tokens.iface_inside_endpoint[nids],
            "iface_inside_quadrant": tokens.iface_inside_quadrant[nids],
            "cross_feat6": tokens.cross_feat6[nids],
            "cross_mask": tokens.cross_mask[nids],
            "cross_child_pair": tokens.cross_child_pair[nids],
            "cross_is_leaf_internal": tokens.cross_is_leaf_internal[nids],
        }

    @staticmethod
    def _make_cpu_token_cache(tokens: PackedNodeTokens) -> SimpleNamespace:
        """Cache the DP-combinatorial token fields on CPU.

        The neural forward stays on the requested device, but all post-decode
        DP routines are Python-heavy and call `.item()` frequently. Keeping
        those tensors on CPU avoids thousands of tiny CUDA sync points.
        """
        field_names = [
            "tree_parent_index",
            "tree_children_index",
            "tree_node_depth",
            "tree_node_feat_rel",
            "is_leaf",
            "root_id",
            "iface_eid",
            "iface_mask",
            "iface_boundary_dir",
            "iface_feat6",
            "cross_eid",
            "cross_mask",
            "cross_child_pair",
        ]
        payload: Dict[str, Any] = {}
        for name in field_names:
            value = getattr(tokens, name)
            payload[name] = value.detach().cpu() if torch.is_tensor(value) else value
        return SimpleNamespace(**payload)


__all__ = ["OnePassDPRunner", "OnePassDPResult", "CostTableEntry"]
