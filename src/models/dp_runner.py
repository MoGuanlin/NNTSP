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
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Use 'spawn' context to avoid CUDA + fork() deadlocks.
# After CUDA is initialized in the parent, fork() inherits a broken
# CUDA context in child processes, causing silent hangs.
_MP_CTX = mp.get_context("spawn")

import torch
from torch import Tensor

from .bc_state_catalog import BoundaryStateCatalog, build_boundary_state_catalog, state_mask_from_iface_mask
from .boundary_state_structured import (
    StructuredBoundaryState,
    build_state_index,
    enumerate_structured_states_for_iface_mask,
    stack_state_tensors,
)
from .dp_core import (
    CorrespondenceMaps,
    batch_check_c1c2,
    build_correspondence_maps,
    leaf_exact_solve,
    parse_activation_batch,
    parse_by_catalog_enum,
    parse_continuous,
    parse_continuous_topk,
    verify_tuple,
)
from .dp_fallback import exact_fallback, lookup_child_costs
from .dp_leaf_solver_structured import leaf_exact_solve_structured
from .dp_parse_factorized import (
    parse_by_factorized_widening,
    prepare_factorized_child_rankings,
    rank_parent_states_by_child_lower_bound,
    solve_factorized_widening_from_ranked_child_states,
)
from .dp_stats import (
    bump_stat,
    ensure_depth_stats_bucket,
    finalize_depth_stats_bucket,
    refresh_depth_fallback_rates,
)
from .dp_traceback import make_cpu_token_cache, traceback_leaf_states
from .merge_decoder import MergeDecoder, ParentMemory
from .node_token_packer import PackedBatch, PackedLeafPoints, PackedNodeTokens
from .shared_tree import build_leaf_row_for_node, extract_z, gather_node_fields


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


def _leaf_solve_structured_worker(args):
    """Worker function for parallel structured leaf exact solve."""
    (points_xy, point_mask, iface_mask, iface_feat6, box_xy, is_root) = args
    costs, state_list, state_used, state_mate, _state_index = leaf_exact_solve_structured(
        points_xy=points_xy,
        point_mask=point_mask,
        iface_mask=iface_mask,
        iface_feat6=iface_feat6,
        box_xy=box_xy,
        is_root=is_root,
    )
    return costs, state_list, state_used, state_mate


def _factorized_node_parse_worker(args):
    """CPU worker for factorized ranking prep + exact widening on one chunk."""
    (
        candidate_indices,
        decode_scores,
        decode_mate_scores,
        parent_iface_mask,
        child_iface_mask,
        child_exists,
        maps,
        child_tables,
        state_used,
        state_mate,
        widening_schedule,
        ranking_mode,
        lambda_mate,
        fallback_exact,
    ) = args
    solved_costs: Dict[int, float] = {}
    backptr: Dict[int, Tuple[int, int, int, int]] = {}
    outcome_counts: Dict[str, float] = {}
    prep_sec = 0.0
    solve_sec = 0.0
    num_rank_failed = 0

    for local_idx, si in enumerate(candidate_indices):
        t_prep = time.time()
        ranked_child_states, status = prepare_factorized_child_rankings(
            scores=decode_scores[local_idx],
            mate_scores=(
                decode_mate_scores[local_idx]
                if decode_mate_scores is not None
                else None
            ),
            parent_a=state_used[int(si)],
            parent_iface_mask=parent_iface_mask,
            child_iface_mask=child_iface_mask,
            child_exists=child_exists,
            maps=maps,
            child_tables=child_tables,
            ranking_mode=ranking_mode,
            lambda_mate=lambda_mate,
        )
        prep_sec += float(time.time() - t_prep)

        if status != "ok" or ranked_child_states is None:
            num_rank_failed += 1
            outcome_counts["num_infeasible"] = float(outcome_counts.get("num_infeasible", 0.0) + 1.0)
            continue

        t_solve = time.time()
        cost, child_si, outcome_key = solve_factorized_widening_from_ranked_child_states(
            ranked_child_states=ranked_child_states,
            parent_a=state_used[int(si)],
            parent_mate=state_mate[int(si)],
            parent_iface_mask=parent_iface_mask,
            child_iface_mask=child_iface_mask,
            child_exists=child_exists,
            maps=maps,
            child_tables=child_tables,
            widening_schedule=widening_schedule,
            fallback_exact=fallback_exact,
        )
        solve_sec += float(time.time() - t_solve)
        if cost < float("inf"):
            solved_costs[int(si)] = float(cost)
            backptr[int(si)] = child_si
            outcome_counts[outcome_key] = float(outcome_counts.get(outcome_key, 0.0) + 1.0)
        else:
            outcome_counts["num_infeasible"] = float(outcome_counts.get("num_infeasible", 0.0) + 1.0)

    return solved_costs, backptr, outcome_counts, {
        "num_candidates": int(len(candidate_indices)),
        "num_rank_failed": int(num_rank_failed),
        "prep_sec": float(prep_sec),
        "solve_sec": float(solve_sec),
    }


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class CostTableEntry:
    """Cost table for a single node.

    costs[s]:     cost of state s, +inf if infeasible
    backptr[s]:   (child_state_indices) tuple for traceback, None if leaf or infeasible
    """
    costs: Tensor               # [S] float
    backptr: Dict[int, Tuple[int, int, int, int]]  # sigma_idx -> (s1, s2, s3, s4) child state indices
    state_list: Optional[List[StructuredBoundaryState]] = None
    state_used_iface: Optional[Tensor] = None  # [S, Ti] bool, node-local structured states
    state_mate: Optional[Tensor] = None        # [S, Ti] long, node-local structured states
    state_index: Optional[Dict[StructuredBoundaryState, int]] = None


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


# ─── Main Runner ─────────────────────────────────────────────────────────────

class OnePassDPRunner:
    """1-Pass bottom-up DP runner with sigma-conditioned neural merge.

    Usage:
        runner = OnePassDPRunner(r=4, max_used=4)
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
        num_parse_workers: int = 0,
        parse_mode: str = "catalog_enum",
        catalog_mate_lambda: float = 1.0,
        child_catalog_widening: Optional[Union[Iterable[int], str]] = None,
        max_decode_batch_size: Optional[int] = 60000,
    ) -> None:
        self.r = r
        self.max_used = max_used
        self.topk = int(topk)
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
        self.num_parse_workers = max(0, int(num_parse_workers))
        if max_decode_batch_size is None:
            self.max_decode_batch_size = None
        else:
            cap = int(max_decode_batch_size)
            self.max_decode_batch_size = cap if cap > 0 else None
        if parse_mode not in (
            "catalog_enum",
            "catalog_enum_iface_mate",
            "catalog_widening",
            "catalog_widening_iface_mate",
            "factorized_widening",
            "factorized_widening_iface_mate",
            "heuristic",
        ):
            raise ValueError(
                "parse_mode must be 'catalog_enum', 'catalog_enum_iface_mate', "
                "'catalog_widening', 'catalog_widening_iface_mate', "
                "'factorized_widening', 'factorized_widening_iface_mate' or 'heuristic', "
                f"got '{parse_mode}'"
            )
        self.parse_mode = str(parse_mode)
        self.catalog_mate_lambda = float(catalog_mate_lambda)
        self.child_catalog_widening = self._normalize_widening_schedule(child_catalog_widening)

    @staticmethod
    def _normalize_widening_schedule(
        schedule: Optional[Union[Iterable[int], str]],
    ) -> Tuple[int, ...]:
        """Parse a widening schedule from iterable or comma-separated CLI text."""
        if schedule is None:
            return ()
        if isinstance(schedule, str):
            text = schedule.strip()
            if text == "" or text.lower() in {"none", "off", "disabled"}:
                return ()
            raw_values = [chunk.strip() for chunk in text.split(",")]
        else:
            raw_values = list(schedule)

        normalized: List[int] = []
        seen = set()
        for raw in raw_values:
            if raw is None:
                continue
            value = int(raw)
            if value <= 0 or value in seen:
                continue
            normalized.append(value)
            seen.add(value)
        normalized.sort()
        return tuple(normalized)

    def _effective_catalog_widening_schedule(self) -> Tuple[int, ...]:
        """Return the widening caps used by catalog_widening modes."""
        if self.child_catalog_widening:
            return self.child_catalog_widening
        if self.max_child_catalog_states is not None:
            base = max(1, int(self.max_child_catalog_states))
        else:
            base = 8
        schedule = []
        cap = base
        for _ in range(4):
            schedule.append(cap)
            cap *= 2
        return self._normalize_widening_schedule(schedule)

    def _is_factorized_mode(self) -> bool:
        return self.parse_mode in ("factorized_widening", "factorized_widening_iface_mate")

    def _decode_iface_uses_logits(self) -> bool:
        """Return whether the current parse mode expects raw iface logits."""
        if self._is_factorized_mode():
            return True
        return self.parse_mode in ("catalog_enum_iface_mate", "catalog_widening_iface_mate")

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
        tokens_cpu = make_cpu_token_cache(tokens)
        structured_mode = self._is_factorized_mode()

        # Build catalog if not provided
        if (not structured_mode) and catalog is None:
            catalog = build_boundary_state_catalog(
                num_slots=Ti, max_used=self.max_used, device=device,
            )
        if structured_mode:
            S = -1
            cat_used_cpu = None
            cat_mate_cpu = None
            node_state_mask = None
        else:
            assert catalog is not None
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
        leaf_row_for_node = build_leaf_row_for_node(total_M, leaves.leaf_node_id)

        depth = tokens.tree_node_depth.long()
        max_depth = int(depth.max().item()) if depth.numel() > 0 else 0

        # Stats
        stats: Dict[str, Any] = {
            "num_nodes": float(total_M),
            "num_leaves": 0.0,
            "num_internal": 0.0,
            "num_sigma_total": 0.0,
            "num_sigma_pruned_lb": 0.0,
            "num_parse_ok": 0.0,
            "num_widening_ok": 0.0,
            "num_topk_ok": 0.0,
            "num_fallback": 0.0,
            "num_infeasible": 0.0,
            "num_exact_too_large": 0.0,
            "depth_stats": {},
            "depth_fallback_rates": {},
            "leaf_encode_sec": 0.0,
            "leaf_exact_sec": 0.0,
            "internal_encode_sec": 0.0,
            "parent_memory_sec": 0.0,
            "decode_sec": 0.0,
            "merge_dp_sec": 0.0,
            "bottom_up_total_sec": 0.0,
            "traceback_state_sec": 0.0,
        }

        root_id = int(tokens_cpu.root_id[0].item())

        # ─── Bottom-up traversal ─────────────────────────────────────────
        t_start = time.time()
        print(f"  [dp] total_M={total_M}, max_depth={max_depth}, S={S}, Ti={Ti}, structured={structured_mode}")
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
                leaf_stage_t0 = time.time()
                # Encode leaves (batched)
                leaf_inputs = gather_node_fields(tokens, leaf_nids)
                rows = leaf_row_for_node[leaf_nids]
                leaf_inputs["leaf_points_xy"] = leaves.point_xy[rows]
                leaf_inputs["leaf_points_mask"] = leaves.point_mask[rows]

                z_leaf = extract_z(leaf_encoder(**leaf_inputs))[0]

                if z_storage is None:
                    z_storage = torch.zeros(total_M, z_leaf.shape[1], dtype=z_leaf.dtype, device=device)
                z_storage[leaf_nids] = z_leaf

                t_enc = time.time()
                stats["leaf_encode_sec"] += float(t_enc - leaf_stage_t0)
                print(f"    [leaf] encode {leaf_nids.numel()} leaves: {t_enc - t_depth:.2f}s")

                leaf_nid_list = []
                num_leaves_d = int(leaf_nids.numel())
                leaf_exact_t0 = time.time()
                if structured_mode:
                    leaf_args_list = []
                    for nid_t in leaf_nids:
                        nid = int(nid_t.item())
                        row = int(leaf_row_for_node[nid].item())
                        leaf_nid_list.append(nid)
                        leaf_args_list.append((
                            leaves.point_xy[row].cpu(),
                            leaves.point_mask[row].cpu(),
                            tokens_cpu.iface_mask[nid],
                            tokens_cpu.iface_feat6[nid],
                            tokens_cpu.tree_node_feat_rel[nid],
                            nid == root_id,
                        ))
                    num_w = self.num_leaf_workers
                    if num_w > 0 and num_leaves_d > 1:
                        print(f"    [leaf] structured_exact_solve {num_leaves_d} leaves with {num_w} workers...")
                        with ProcessPoolExecutor(max_workers=num_w, mp_context=_MP_CTX) as pool:
                            results = list(pool.map(_leaf_solve_structured_worker, leaf_args_list))
                        for nid, (costs, state_list, state_used, state_mate) in zip(leaf_nid_list, results):
                            cost_tables[nid] = CostTableEntry(
                                costs=costs.cpu(),
                                backptr={},
                                state_list=state_list,
                                state_used_iface=state_used.cpu(),
                                state_mate=state_mate.cpu(),
                                state_index=build_state_index(state_list),
                            )
                            stats["num_leaves"] += 1
                        print(f"    [leaf] structured_exact_solve {num_leaves_d}/{num_leaves_d} done ({time.time()-t_enc:.1f}s)")
                    else:
                        for li, (nid, args) in enumerate(zip(leaf_nid_list, leaf_args_list)):
                            costs, state_list, state_used, state_mate = _leaf_solve_structured_worker(args)
                            cost_tables[nid] = CostTableEntry(
                                costs=costs.cpu(),
                                backptr={},
                                state_list=state_list,
                                state_used_iface=state_used.cpu(),
                                state_mate=state_mate.cpu(),
                                state_index=build_state_index(state_list),
                            )
                            stats["num_leaves"] += 1
                            if (li + 1) % 50 == 0 or (li + 1) == num_leaves_d:
                                print(f"    [leaf] structured_exact_solve {li+1}/{num_leaves_d} done ({time.time()-t_enc:.1f}s)")
                else:
                    # Prepare args for leaf_exact_solve (all on CPU for efficiency)
                    assert catalog is not None and cat_used_cpu is not None and cat_mate_cpu is not None and node_state_mask is not None
                    leaf_args_list = []
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
                stats["leaf_exact_sec"] += float(time.time() - leaf_exact_t0)

                computed[leaf_nids] = True

            # ──── Process internal nodes ─────────────────────────────────
            if internal_nids.numel() > 0:
                depth_bucket = ensure_depth_stats_bucket(stats, d)
                if z_storage is None:
                    raise RuntimeError("Internal node encountered before any leaf.")

                # Encode internal nodes (batched)
                ch = tokens.tree_children_index[internal_nids].long()
                child_mask_batch = ch >= 0
                ch_clamped = ch.clamp_min(0)
                child_z_batch = z_storage[ch_clamped]
                child_z_batch = child_z_batch * child_mask_batch.unsqueeze(-1).float()

                merge_inputs = gather_node_fields(tokens, internal_nids)
                merge_inputs["child_z"] = child_z_batch
                merge_inputs["child_mask"] = child_mask_batch

                z_parent = extract_z(merge_encoder(**merge_inputs))[0]
                z_storage[internal_nids] = z_parent

                t_merge_enc = time.time()
                stats["internal_encode_sec"] += float(t_merge_enc - t_depth)
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
                stats["parent_memory_sec"] += float(t_mem - t_merge_enc)
                print(f"    [internal] parent_memory({N_int}): {t_mem - t_merge_enc:.2f}s")

                # ── Batched decode: collect all candidates from ALL nodes,
                #    one GPU forward pass ──────────────────────────────────
                cat_used_dev = catalog.used_iface.to(device) if (catalog is not None and not structured_mode) else None
                cat_mate_dev = catalog.mate.to(device) if (catalog is not None and not structured_mode) else None

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

                    if structured_mode:
                        state_list_n = enumerate_structured_states_for_iface_mask(
                            iface_mask=tokens_cpu.iface_mask[nid].bool()
                        )
                        state_used_n_cpu, state_mate_n_cpu = stack_state_tensors(
                            states=state_list_n,
                            num_slots=Ti,
                            device=torch.device("cpu"),
                        )
                        cands = torch.arange(len(state_list_n), dtype=torch.long)
                        has_child = maps_n.phi_out_child >= 0
                        unmapped = parent_iface_mask_n & ~has_child
                        if unmapped.any() and cands.numel() > 0:
                            c1_pre = ~(state_used_n_cpu[cands].bool() & unmapped.unsqueeze(0)).any(dim=1)
                            cands = cands[c1_pre]
                        child_tables_n: List[Optional[CostTableEntry]] = [None] * 4
                        for q in range(4):
                            if not child_exists_n[q].item():
                                continue
                            cid = int(children_n[q].item())
                            child_tables_n[q] = cost_tables.get(cid)
                        if cands.numel() > 0:
                            ordered_local_indices, _ = rank_parent_states_by_child_lower_bound(
                                parent_state_used=state_used_n_cpu[cands],
                                parent_iface_mask=parent_iface_mask_n,
                                child_iface_mask=child_iface_mask_4,
                                child_exists=child_exists_n,
                                maps=maps_n,
                                child_tables=child_tables_n,
                            )
                            pruned_lb = int(cands.numel()) - int(len(ordered_local_indices))
                            if pruned_lb > 0:
                                bump_stat(
                                    stats,
                                    "num_sigma_pruned_lb",
                                    amount=float(pruned_lb),
                                    depth_bucket=depth_bucket,
                                )
                            if ordered_local_indices:
                                reorder = torch.tensor(ordered_local_indices, dtype=torch.long)
                                cands = cands[reorder]
                            else:
                                cands = cands[:0]
                        cands = self._apply_sigma_cap(cands)
                        per_node_info.append((
                            nid, cands, maps_n, child_iface_mask_4, child_iface_bdir_n,
                            children_n, child_exists_n, state_list_n, state_used_n_cpu, state_mate_n_cpu,
                        ))
                    else:
                        assert node_state_mask is not None and cat_used_cpu is not None
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
                            nid, cands, maps_n, child_iface_mask_4, child_iface_bdir_n,
                            children_n, child_exists_n, None, None, None,
                        ))

                    K_i = cands.numel()
                    bump_stat(
                        stats,
                        "num_sigma_total",
                        amount=float(K_i),
                        depth_bucket=depth_bucket,
                    )
                    if K_i > 0:
                        cands_dev = cands.to(device=device, dtype=torch.long)
                        if structured_mode:
                            assert per_node_info[-1][8] is not None and per_node_info[-1][9] is not None
                            state_used_n_dev = per_node_info[-1][8].to(device=device)
                            state_mate_n_dev = per_node_info[-1][9].to(device=device)
                            all_sigma_a.append(state_used_n_dev[cands_dev].float())
                            all_sigma_mate.append(state_mate_n_dev[cands_dev])
                        else:
                            assert cat_used_dev is not None and cat_mate_dev is not None
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
                all_decode_logits = None
                all_decode_probs = None
                all_decode_mate_scores = None
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
                    all_decode_logits = out.child_scores  # [K_total, 4, Ti]
                    all_decode_probs = torch.sigmoid(out.child_scores)  # [K_total, 4, Ti]
                    if out.child_mate_scores is not None:
                        all_decode_mate_scores = out.child_mate_scores  # [K_total, 4, Ti, Ti]

                t_decode = time.time()
                stats["decode_sec"] += float(t_decode - t_mem)
                print(f"    [internal] decode({N_int} nodes, {K_total} sigmas): {t_decode - t_mem:.2f}s")

                # ── Per-node: PARSE + VERIFY + fallback (CPU-bound) ──
                merge_dp_t0 = time.time()
                offset = 0
                if structured_mode and self.num_parse_workers > 0 and K_total > 0:
                    ranking_mode = "iface_mate" if self.parse_mode.endswith("iface_mate") else "iface"
                    if ranking_mode == "iface_mate" and all_decode_mate_scores is None:
                        raise ValueError(
                            f"parse_mode='{self.parse_mode}' requires decoder outputs with mate scores"
                        )
                    schedule = self._effective_catalog_widening_schedule()
                    target_jobs = max(self.num_parse_workers, self.num_parse_workers * 4)
                    chunk_size = max(1, (K_total + target_jobs - 1) // target_jobs)
                    total_chunks = 0
                    worker_specs = []
                    node_accumulators: Dict[int, Dict[str, Any]] = {}

                    def _finalize_parallel_structured_node(nid: int) -> None:
                        acc = node_accumulators[nid]
                        self._store_structured_cost_table(
                            nid=nid,
                            state_list=acc["state_list"],
                            state_used=acc["state_used"],
                            state_mate=acc["state_mate"],
                            costs=acc["costs"],
                            backptr=acc["backptr"],
                            cost_tables=cost_tables,
                        )
                        stats["num_internal"] += 1
                        ct = cost_tables[nid]
                        feasible_mask = ct.costs < float("inf")
                        n_feasible = int(feasible_mask.sum().item())
                        denom = int(ct.costs.numel())
                        print(
                            f"    [internal] DP {acc['node_idx']+1}/{N_int} "
                            f"nid={nid} feasible={n_feasible}/{denom} "
                            f"prep={acc['prep_sec']:.2f}s solve={acc['solve_sec']:.2f}s "
                            f"rank_failed={acc['num_rank_failed']} "
                            f"({time.time() - t_decode:.1f}s)"
                        )

                    for idx, (nid, cands, maps_n, child_iface_mask_4,
                              child_iface_bdir_n, children_n, child_exists_n,
                              state_list_n, state_used_n_cpu, state_mate_n_cpu) in enumerate(per_node_info):
                        K_i = repeat_counts[idx]
                        node_scores_cpu = None
                        if K_i > 0:
                            source = all_decode_logits if self._decode_iface_uses_logits() else all_decode_probs
                            node_scores_cpu = source[offset:offset+K_i].detach().cpu()
                        node_mate_scores_cpu = (
                            all_decode_mate_scores[offset:offset+K_i].detach().cpu()
                            if (K_i > 0 and all_decode_mate_scores is not None)
                            else None
                        )
                        offset += K_i
                        assert state_list_n is not None and state_used_n_cpu is not None and state_mate_n_cpu is not None

                        node_accumulators[nid] = {
                            "node_idx": idx,
                            "state_list": state_list_n,
                            "state_used": state_used_n_cpu.cpu(),
                            "state_mate": state_mate_n_cpu.cpu(),
                            "costs": torch.full(
                                (int(len(state_list_n)),),
                                float("inf"),
                                dtype=torch.float32,
                                device=torch.device("cpu"),
                            ),
                            "backptr": {},
                            "num_pending_chunks": 0,
                            "num_rank_failed": 0,
                            "prep_sec": 0.0,
                            "solve_sec": 0.0,
                        }

                        child_tables_n: List[Optional[CostTableEntry]] = [None] * 4
                        for q in range(4):
                            if not child_exists_n[q].item():
                                continue
                            cid = int(children_n[q].item())
                            child_tables_n[q] = cost_tables.get(cid)

                        if K_i > 0:
                            for start in range(0, K_i, chunk_size):
                                end = min(K_i, start + chunk_size)
                                chunk_candidates = [int(v) for v in cands[start:end].tolist()]
                                worker_specs.append((
                                    (
                                        chunk_candidates,
                                        node_scores_cpu[start:end],
                                        (
                                            node_mate_scores_cpu[start:end]
                                            if node_mate_scores_cpu is not None
                                            else None
                                        ),
                                        tokens_cpu.iface_mask[nid].bool().cpu(),
                                        child_iface_mask_4.cpu(),
                                        child_exists_n.cpu(),
                                        maps_n,
                                        child_tables_n,
                                        state_used_n_cpu.cpu(),
                                        state_mate_n_cpu.cpu(),
                                        schedule,
                                        ranking_mode,
                                        self.catalog_mate_lambda,
                                        self.fallback_exact,
                                    ),
                                    {
                                        "nid": nid,
                                        "node_idx": idx,
                                        "chunk_idx": total_chunks,
                                    },
                                ))
                                node_accumulators[nid]["num_pending_chunks"] += 1
                                total_chunks += 1
                        else:
                            _finalize_parallel_structured_node(nid)

                    if worker_specs:
                        print(
                            f"    [internal] factorized prep+solve: "
                            f"{N_int} nodes, {K_total} sigmas, workers={self.num_parse_workers}, "
                            f"chunks={total_chunks}, chunk_size~{chunk_size}"
                        )
                        with ProcessPoolExecutor(max_workers=self.num_parse_workers, mp_context=_MP_CTX) as pool:
                            futures = {
                                pool.submit(_factorized_node_parse_worker, args): meta
                                for args, meta in worker_specs
                            }
                            completed_chunks = 0
                            for future in as_completed(futures):
                                meta = futures[future]
                                nid = int(meta["nid"])
                                costs_n, backptr_n, outcome_counts, worker_stats = future.result()
                                acc = node_accumulators[nid]
                                for si, cost in costs_n.items():
                                    acc["costs"][int(si)] = float(cost)
                                acc["backptr"].update(backptr_n)
                                acc["num_rank_failed"] += int(worker_stats["num_rank_failed"])
                                acc["prep_sec"] += float(worker_stats["prep_sec"])
                                acc["solve_sec"] += float(worker_stats["solve_sec"])
                                acc["num_pending_chunks"] -= 1

                                completed_chunks += 1
                                print(
                                    f"      [factorized] chunk {completed_chunks}/{total_chunks} "
                                    f"nid={nid} sigmas={int(worker_stats['num_candidates'])} "
                                    f"prep={float(worker_stats['prep_sec']):.2f}s "
                                    f"solve={float(worker_stats['solve_sec']):.2f}s "
                                    f"rank_failed={int(worker_stats['num_rank_failed'])} "
                                    f"({time.time() - t_decode:.1f}s)"
                                )

                                if acc["num_pending_chunks"] == 0:
                                    _finalize_parallel_structured_node(nid)
                                for key, amount in outcome_counts.items():
                                    bump_stat(stats, key, amount=float(amount), depth_bucket=depth_bucket)
                else:
                    for idx, (nid, cands, maps_n, child_iface_mask_4,
                              child_iface_bdir_n, children_n, child_exists_n,
                              state_list_n, state_used_n_cpu, state_mate_n_cpu) in enumerate(per_node_info):
                        K_i = repeat_counts[idx]
                        node_scores = None
                        if K_i > 0:
                            source = all_decode_logits if self._decode_iface_uses_logits() else all_decode_probs
                            node_scores = source[offset:offset+K_i]
                        node_mate_scores = all_decode_mate_scores[offset:offset+K_i] if (K_i > 0 and all_decode_mate_scores is not None) else None
                        offset += K_i

                        if structured_mode:
                            assert state_list_n is not None and state_used_n_cpu is not None and state_mate_n_cpu is not None
                            factorized_summary = self._process_internal_node_post_decode_factorized(
                                nid=nid,
                                tokens=tokens_cpu,
                                candidate_indices=cands,
                                decode_scores=node_scores,
                                decode_mate_scores=node_mate_scores,
                                maps=maps_n,
                                child_iface_mask_4=child_iface_mask_4,
                                child_iface_bdir=child_iface_bdir_n,
                                children=children_n,
                                child_exists=child_exists_n,
                                state_list=state_list_n,
                                state_used=state_used_n_cpu,
                                state_mate=state_mate_n_cpu,
                                cost_tables=cost_tables,
                                stats=stats,
                                depth_bucket=depth_bucket,
                            )
                        else:
                            assert catalog is not None and cat_used_cpu is not None and cat_mate_cpu is not None and node_state_mask is not None
                            self._process_internal_node_post_decode(
                                nid=nid, tokens=tokens_cpu,
                                candidate_indices=cands,
                                decode_scores=node_scores,
                                decode_mate_scores=node_mate_scores,
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
                        feasible_mask = ct.costs < float("inf")
                        n_feasible = int(feasible_mask.sum().item())
                        denom = int(ct.costs.numel())
                        if structured_mode:
                            print(
                                f"    [internal] DP {idx+1}/{N_int} "
                                f"nid={nid} feasible={n_feasible}/{denom} "
                                f"prep={factorized_summary['prep_sec']:.2f}s "
                                f"solve={factorized_summary['solve_sec']:.2f}s "
                                f"rank_failed={int(factorized_summary['num_rank_failed'])} "
                                f"({time.time() - t_decode:.1f}s)"
                            )
                        else:
                            print(
                                f"    [internal] DP {idx+1}/{N_int} "
                                f"nid={nid} feasible={n_feasible}/{denom} "
                                f"({time.time() - t_decode:.1f}s)"
                            )

                computed[internal_nids] = True
                stats["merge_dp_sec"] += float(time.time() - merge_dp_t0)
                finalize_depth_stats_bucket(depth_bucket)
                refresh_depth_fallback_rates(stats)
                print(
                    f"  [dp] depth={d} total: {time.time()-t_depth:.2f}s | "
                    f"sigma={int(depth_bucket['num_sigma_total'])} "
                    f"fallback={int(depth_bucket['num_fallback'])}/{int(depth_bucket['num_sigma_total'])} "
                    f"({depth_bucket['fallback_rate'] * 100.0:.1f}%)"
                )

        refresh_depth_fallback_rates(stats)
        stats["bottom_up_total_sec"] = float(time.time() - t_start)
        print(f"  [dp] bottom-up done: {stats['bottom_up_total_sec']:.2f}s, stats={stats}")
        # ─── Find best root state ────────────────────────────────────────
        root_ct = cost_tables.get(root_id)
        if root_ct is None:
            return OnePassDPResult(
                tour_cost=float("inf"), root_sigma=-1,
                leaf_states={}, cost_tables=cost_tables, stats=stats,
            )

        valid_costs = root_ct.costs.clone()
        if not structured_mode:
            assert node_state_mask is not None
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
        t_traceback = time.time()
        leaf_states = traceback_leaf_states(
            root_id=root_id,
            root_sigma=best_sigma,
            tokens=tokens,
            cost_tables=cost_tables,
        )
        stats["traceback_state_sec"] = float(time.time() - t_traceback)

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

    def _process_internal_node_post_decode(
        self,
        *,
        nid: int,
        tokens: PackedNodeTokens,
        candidate_indices: Tensor,
        decode_scores: Optional[Tensor],   # [K, 4, Ti] sigmoid scores, or None
        decode_mate_scores: Optional[Tensor],  # [K, 4, Ti, Ti] mate logits, or None
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

            if self.parse_mode in (
                "catalog_enum",
                "catalog_enum_iface_mate",
                "catalog_widening",
                "catalog_widening_iface_mate",
            ):
                # Catalog-enumeration PARSE: direct enumeration with neural guidance.
                ranking_mode = "iface_mate" if self.parse_mode.endswith("iface_mate") else "iface"
                if ranking_mode == "iface_mate" and decode_mate_scores is None:
                    raise ValueError(
                        f"parse_mode='{self.parse_mode}' requires decoder outputs with mate scores"
                    )
                for k_idx in range(K):
                    si = int(candidate_indices[k_idx].item())
                    if costs[si] < float("inf"):
                        continue

                    if self.parse_mode.startswith("catalog_widening"):
                        cost, child_si, outcome_key = self._parse_catalog_enum_with_widening(
                            scores=decode_scores[k_idx],
                            mate_scores=(decode_mate_scores[k_idx] if decode_mate_scores is not None else None),
                            parent_a=cat_used_dev[si],
                            parent_mate=cat_mate_dev[si],
                            parent_iface_mask=parent_iface_mask,
                            child_iface_mask=child_iface_mask_4,
                            child_exists=child_exists,
                            maps=maps,
                            cat_used=cat_used_dev,
                            cat_mate=cat_mate_dev,
                            child_costs=child_cost_list,
                            ranking_mode=ranking_mode,
                        )
                    else:
                        cost, child_si, used_fallback = self._parse_catalog_enum_with_optional_fallback(
                            scores=decode_scores[k_idx],
                            mate_scores=(decode_mate_scores[k_idx] if decode_mate_scores is not None else None),
                            parent_a=cat_used_dev[si],
                            parent_mate=cat_mate_dev[si],
                            parent_iface_mask=parent_iface_mask,
                            child_iface_mask=child_iface_mask_4,
                            child_exists=child_exists,
                            maps=maps,
                            cat_used=cat_used_dev,
                            cat_mate=cat_mate_dev,
                            child_costs=child_cost_list,
                            ranking_mode=ranking_mode,
                        )
                        outcome_key = "num_fallback" if used_fallback else "num_parse_ok"
                    if cost < float("inf"):
                        costs[si] = cost
                        backptr[si] = child_si
                        bump_stat(stats, outcome_key, depth_bucket=depth_bucket)
                    else:
                        bump_stat(stats, "num_infeasible", depth_bucket=depth_bucket)
            else:
                # Legacy heuristic PARSE: threshold rounding + top-K + exact fallback.
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
                        scores=scores,
                        child_iface_mask=child_iface_mask_4,
                        child_iface_bdir=child_iface_bdir,
                        child_exists=child_exists,
                        maps=maps,
                        r=self.r,
                        parent_a=parent_a.bool(),
                        parent_iface_mask=parent_iface_mask,
                    )
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
                        cost, csi = lookup_child_costs(
                            child_a=child_a,
                            child_mate=child_mate,
                            children=children,
                            child_exists=child_exists,
                            cost_tables=cost_tables,
                            catalog=catalog,
                        )
                        if cost < float("inf"):
                            costs[si] = cost
                            backptr[si] = csi
                            bump_stat(stats, "num_parse_ok", depth_bucket=depth_bucket)
                            continue

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
                        cost, csi = lookup_child_costs(
                            child_a=ca_k,
                            child_mate=cm_k,
                            children=children,
                            child_exists=child_exists,
                            cost_tables=cost_tables,
                            catalog=catalog,
                        )
                        if cost < float("inf"):
                            costs[si] = cost
                            backptr[si] = csi
                            bump_stat(stats, "num_topk_ok", depth_bucket=depth_bucket)
                            found = True
                            break
                    if not found and self.fallback_exact:
                        cost, child_si = exact_fallback(
                            si=si, catalog=catalog, tokens=tokens, nid=nid,
                            children=children, child_exists=child_exists,
                            maps=maps, cost_tables=cost_tables,
                        )
                        if cost < float("inf"):
                            costs[si] = cost
                            backptr[si] = child_si
                            bump_stat(stats, "num_fallback", depth_bucket=depth_bucket)
                        else:
                            bump_stat(stats, "num_infeasible", depth_bucket=depth_bucket)
                    elif not found:
                        bump_stat(stats, "num_infeasible", depth_bucket=depth_bucket)

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
                        cost, child_si = exact_fallback(
                            si=si, catalog=catalog, tokens=tokens, nid=nid,
                            children=children, child_exists=child_exists,
                            maps=maps, cost_tables=cost_tables,
                        )
                        if cost < float("inf"):
                            costs[si] = cost
                            backptr[si] = child_si
                            bump_stat(stats, "num_fallback", depth_bucket=depth_bucket)
                        else:
                            bump_stat(stats, "num_infeasible", depth_bucket=depth_bucket)
                else:
                    bump_stat(
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
        mate_scores: Optional[Tensor] = None,
        parent_a: Tensor,
        parent_mate: Tensor,
        parent_iface_mask: Tensor,
        child_iface_mask: Tensor,
        child_exists: Tensor,
        maps: CorrespondenceMaps,
        cat_used: Tensor,
        cat_mate: Tensor,
        child_costs: List[Optional[Tensor]],
        ranking_mode: str = "iface",
    ) -> Tuple[float, Tuple[int, int, int, int], bool]:
        """Run capped catalog_enum first, then exact enumeration if requested."""
        cost, child_si = parse_by_catalog_enum(
            scores=scores,
            mate_scores=mate_scores,
            ranking_mode=ranking_mode,
            lambda_mate=self.catalog_mate_lambda,
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
            mate_scores=mate_scores,
            ranking_mode=ranking_mode,
            lambda_mate=self.catalog_mate_lambda,
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

    def _prepare_factorized_node_rankings(
        self,
        *,
        candidate_indices: Tensor,
        decode_scores: Optional[Tensor],
        decode_mate_scores: Optional[Tensor],
        parent_iface_mask: Tensor,
        child_iface_mask_4: Tensor,
        child_exists: Tensor,
        maps: CorrespondenceMaps,
        child_tables: List[Optional[CostTableEntry]],
        state_used: Tensor,
    ) -> List[Tuple[int, Optional[List[List[int]]]]]:
        """Prepare ranked child candidate lists per parent sigma, using GPU when available."""
        if decode_scores is None or int(candidate_indices.numel()) == 0:
            return []
        device = decode_scores.device
        parent_state_used_dev = state_used.to(device=device, non_blocking=True)
        parent_iface_mask_dev = parent_iface_mask.to(device=device, non_blocking=True)
        child_iface_mask_dev = child_iface_mask_4.to(device=device, non_blocking=True)
        child_exists_dev = child_exists.to(device=device, non_blocking=True)
        child_state_used_dev: List[Optional[Tensor]] = [None] * 4
        child_state_mate_dev: List[Optional[Tensor]] = [None] * 4
        child_cost_dev: List[Optional[Tensor]] = [None] * 4
        for q in range(4):
            ct = child_tables[q]
            if ct is None or ct.state_used_iface is None or ct.state_mate is None or ct.costs is None:
                continue
            child_state_used_dev[q] = ct.state_used_iface.to(device=device, non_blocking=True)
            child_state_mate_dev[q] = ct.state_mate.to(device=device, non_blocking=True)
            child_cost_dev[q] = ct.costs.to(device=device, non_blocking=True)

        ranking_mode = "iface_mate" if self.parse_mode.endswith("iface_mate") else "iface"
        prepared: List[Tuple[int, Optional[List[List[int]]]]] = []
        for k_idx in range(int(candidate_indices.numel())):
            si = int(candidate_indices[k_idx].item())
            ranked_child_states, status = prepare_factorized_child_rankings(
                scores=decode_scores[k_idx],
                mate_scores=(decode_mate_scores[k_idx] if decode_mate_scores is not None else None),
                parent_a=parent_state_used_dev[si],
                parent_iface_mask=parent_iface_mask_dev,
                child_iface_mask=child_iface_mask_dev,
                child_exists=child_exists_dev,
                maps=maps,
                child_tables=child_tables,
                ranking_mode=ranking_mode,
                lambda_mate=self.catalog_mate_lambda,
                child_state_used_dev=child_state_used_dev,
                child_state_mate_dev=child_state_mate_dev,
                child_cost_dev=child_cost_dev,
            )
            prepared.append((si, ranked_child_states if status == "ok" else None))
        return prepared

    def _store_structured_cost_table(
        self,
        *,
        nid: int,
        state_list: List[StructuredBoundaryState],
        state_used: Tensor,
        state_mate: Tensor,
        costs: Tensor,
        backptr: Dict[int, Tuple[int, int, int, int]],
        cost_tables: Dict[int, CostTableEntry],
    ) -> None:
        """Store one structured-state DP table on CPU."""
        cost_tables[nid] = CostTableEntry(
            costs=costs,
            backptr=backptr,
            state_list=state_list,
            state_used_iface=state_used.cpu(),
            state_mate=state_mate.cpu(),
            state_index=build_state_index(state_list),
        )

    def _process_internal_node_post_decode_factorized(
        self,
        *,
        nid: int,
        tokens: PackedNodeTokens,
        candidate_indices: Tensor,
        decode_scores: Optional[Tensor],
        decode_mate_scores: Optional[Tensor],
        maps: CorrespondenceMaps,
        child_iface_mask_4: Tensor,
        child_iface_bdir: Tensor,
        children: Tensor,
        child_exists: Tensor,
        state_list: List[StructuredBoundaryState],
        state_used: Tensor,
        state_mate: Tensor,
        cost_tables: Dict[int, CostTableEntry],
        stats: Dict[str, Any],
        depth_bucket: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Structured-state DP merge without a global capped catalog."""
        del child_iface_bdir  # reserved for future factorized child generation heuristics
        parent_iface_mask = tokens.iface_mask[nid].bool()
        num_states = int(len(state_list))
        costs = torch.full((num_states,), float("inf"), dtype=torch.float32, device=torch.device("cpu"))
        backptr: Dict[int, Tuple[int, int, int, int]] = {}

        child_tables: List[Optional[CostTableEntry]] = [None] * 4
        for q in range(4):
            if not child_exists[q].item():
                continue
            cid = int(children[q].item())
            child_tables[q] = cost_tables.get(cid)

        ranking_mode = "iface_mate" if self.parse_mode.endswith("iface_mate") else "iface"
        if ranking_mode == "iface_mate" and candidate_indices.numel() > 0 and decode_mate_scores is None:
            raise ValueError(
                f"parse_mode='{self.parse_mode}' requires decoder outputs with mate scores"
            )

        t_prep = time.time()
        prepared = self._prepare_factorized_node_rankings(
            candidate_indices=candidate_indices,
            decode_scores=decode_scores,
            decode_mate_scores=decode_mate_scores,
            parent_iface_mask=parent_iface_mask,
            child_iface_mask_4=child_iface_mask_4,
            child_exists=child_exists,
            maps=maps,
            child_tables=child_tables,
            state_used=state_used,
        )
        prep_sec = float(time.time() - t_prep)
        num_rank_failed = sum(1 for _, ranked_child_states in prepared if ranked_child_states is None)
        if prepared:
            print(
                f"      [factorized] nid={nid} prep "
                f"sigmas={len(prepared)} ok={len(prepared) - num_rank_failed} "
                f"failed={num_rank_failed} ({prep_sec:.2f}s)"
            )

        schedule = self._effective_catalog_widening_schedule()
        t_solve = time.time()
        for si, ranked_child_states in prepared:
            if costs[si] < float("inf"):
                continue
            if ranked_child_states is None:
                bump_stat(stats, "num_infeasible", depth_bucket=depth_bucket)
                continue
            cost, child_si, outcome_key = solve_factorized_widening_from_ranked_child_states(
                ranked_child_states=ranked_child_states,
                parent_a=state_used[si],
                parent_mate=state_mate[si],
                parent_iface_mask=parent_iface_mask,
                child_iface_mask=child_iface_mask_4,
                child_exists=child_exists,
                maps=maps,
                child_tables=child_tables,
                widening_schedule=schedule,
                fallback_exact=self.fallback_exact,
            )
            if cost < float("inf"):
                costs[si] = float(cost)
                backptr[si] = child_si
                bump_stat(stats, outcome_key, depth_bucket=depth_bucket)
            else:
                bump_stat(stats, "num_infeasible", depth_bucket=depth_bucket)
        solve_sec = float(time.time() - t_solve)
        if prepared:
            print(
                f"      [factorized] nid={nid} solve "
                f"ranked={len(prepared) - num_rank_failed}/{len(prepared)} "
                f"({solve_sec:.2f}s)"
            )

        self._store_structured_cost_table(
            nid=nid,
            state_list=state_list,
            state_used=state_used,
            state_mate=state_mate,
            costs=costs,
            backptr=backptr,
            cost_tables=cost_tables,
        )
        return {
            "prep_sec": prep_sec,
            "solve_sec": solve_sec,
            "num_rank_failed": float(num_rank_failed),
        }

    def _parse_catalog_enum_with_widening(
        self,
        *,
        scores: Tensor,
        mate_scores: Optional[Tensor] = None,
        parent_a: Tensor,
        parent_mate: Tensor,
        parent_iface_mask: Tensor,
        child_iface_mask: Tensor,
        child_exists: Tensor,
        maps: CorrespondenceMaps,
        cat_used: Tensor,
        cat_mate: Tensor,
        child_costs: List[Optional[Tensor]],
        ranking_mode: str = "iface",
    ) -> Tuple[float, Tuple[int, int, int, int], str]:
        """Run catalog-enum with widening rounds before optional exact fallback."""
        schedule = list(self._effective_catalog_widening_schedule())
        if not schedule and self.max_child_catalog_states is not None:
            schedule = [int(self.max_child_catalog_states)]

        if not schedule:
            cost, child_si = parse_by_catalog_enum(
                scores=scores,
                mate_scores=mate_scores,
                ranking_mode=ranking_mode,
                lambda_mate=self.catalog_mate_lambda,
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
            if cost < float("inf"):
                return cost, child_si, "num_parse_ok"
            return cost, child_si, "num_infeasible"

        for attempt_idx, cap in enumerate(schedule):
            cost, child_si = parse_by_catalog_enum(
                scores=scores,
                mate_scores=mate_scores,
                ranking_mode=ranking_mode,
                lambda_mate=self.catalog_mate_lambda,
                parent_a=parent_a,
                parent_mate=parent_mate,
                parent_iface_mask=parent_iface_mask,
                child_iface_mask=child_iface_mask,
                child_exists=child_exists,
                maps=maps,
                cat_used=cat_used,
                cat_mate=cat_mate,
                child_costs=child_costs,
                max_child_states=cap,
            )
            if cost < float("inf"):
                outcome = "num_parse_ok" if attempt_idx == 0 else "num_widening_ok"
                return cost, child_si, outcome

        if not self.fallback_exact:
            return float("inf"), (-1, -1, -1, -1), "num_infeasible"

        exact_cost, exact_child_si = parse_by_catalog_enum(
            scores=scores,
            mate_scores=mate_scores,
            ranking_mode=ranking_mode,
            lambda_mate=self.catalog_mate_lambda,
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
        if exact_cost < float("inf"):
            return exact_cost, exact_child_si, "num_fallback"
        return float("inf"), (-1, -1, -1, -1), "num_infeasible"

__all__ = ["OnePassDPRunner", "OnePassDPResult", "CostTableEntry"]
