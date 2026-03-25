# train.py
# -*- coding: utf-8 -*-

"""
================================================================================
  [IMPORTANT] [DO NOT DELETE] PERFORMANCE CRITICAL NOTES
================================================================================
HIGH PERFORMANCE DATA LOADING STRATEGY (LESSONS LEARNED):

1. DATASET ARCHITECTURE:
   The `FastTSPDataset` holds the entire dataset (~20GB+) in memory as a consolidated
   dictionary of CPU tensors (`self.c`).

2. MULTIPROCESSING HAZARD (PICKLING):
   - PROBLEM: If the dataset object is pickled/serialized when spawning workers (Window/Spawn
     or explicit `pickle` calls), Python copies the entire 20GB structure for EACH worker.
     This causes initialization to take >30s PER BATCH and massive memory bloat.
   - SOLUTION: We define `FastTSPDataset` to NOT implement `__getstate__` or use explicit
     pickling logic. We rely on Linux `fork()` Copy-On-Write (COW) memory sharing.

3. WORKER CONFIGURATION:
   - `num_workers > 0`: SAFE ONLY IF running on Linux (Fork). Workers inherit the
     memory pointer to the huge dict without copying.
   - `num_workers = 0`: Safe fallback, but slow (single usage of packing/labeling).
   - `use_pickle=False`: MANDATORY for `FastTSPDataset`. Never enable pickling for
     fetching items from this dataset.

4. IPC OPTIMIZATION:
   - Inputs to workers (Dataset): Shared via Fork (Zero cost).
   - Outputs from workers (Batch): Serialized to `bytes` using `torch.save(batch, buf)`
     inside `TrainingWorker` to avoid "Too many open files" / shared memory limits
     when passing many small tensors.

DO NOT RE-ENABLE "use_pickle" IN DATASET OR REMOVE WORKER IPC OPTIMIZATIONS.
================================================================================
"""

from __future__ import annotations

import argparse
import random
import time
import warnings
import datetime
import pickle
import os
import io
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm



def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_bool_arg(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_device(device_arg: str) -> torch.device:
    if str(device_arg).startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def move_data_tensors_to_device(data: Any, device: torch.device) -> Any:
    # Handle Dict
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(device)
            elif v is not None:
                move_data_tensors_to_device(v, device)
        return data

    # Handle List
    if isinstance(data, list):
        for i, v in enumerate(data):
            if isinstance(v, torch.Tensor):
                data[i] = v.to(device)
            elif v is not None:
                move_data_tensors_to_device(v, device)
        return data

    # Avoid touching PyG deprecated virtual attrs (e.g., num_faces)
    skip = {"num_faces"}
    for k in dir(data):
        if k.startswith("_") or k in skip:
            continue
        try:
            v = getattr(data, k, None)
        except Exception:
            continue
        if isinstance(v, torch.Tensor):
            try:
                setattr(data, k, v.to(device))
            except Exception:
                pass
        elif v is not None and (hasattr(v, "__dict__") or isinstance(v, (dict, list))):
            # Recurse into nested objects, dicts or lists (e.g., PackedBatch.tokens or TokenLabels.stats)
            move_data_tensors_to_device(v, device)
    return data


def load_list_pt(path: str) -> FastTSPDataset:
    from src.dataprep.dataset import smart_load_dataset
    return smart_load_dataset(path)



@dataclass
class LiteData:
    pos: torch.Tensor
    spanner_edge_index: torch.Tensor
    target_edges: Optional[torch.Tensor] = None

@dataclass
class PrecomputedBatch:
    datas: List[LiteData]
    packed: Any
    labels: Any

class ValidationWorker:
    def __init__(self, packer, labeler, device, num_workers=0):
        self.packer = packer
        self.labeler = labeler
        self.device = device
        self.num_workers = int(num_workers)

    def __call__(self, datas):
        # In workers (especially if num_workers > 0), we should keep tensors on CPU
        # for safer IPC, then move to device in the main thread.
        # However, the current setup expects tiles on 'device'.
        # We will do the computation on CPU and return.
        
        # Ensure datas are on CPU for worker processing
        datas = [pickle.loads(d) if isinstance(d, bytes) else d for d in datas]
        
        # Pack and Label (on CPU)
        packed = self.packer.pack_batch(datas)
        labels = self.labeler.label_batch(datas=datas, packed=packed, device=torch.device("cpu"))
        
        # Convert to LiteData to save memory (discard heavy PyG Data objects)
        lite_datas = [
            LiteData(
                pos=d.pos.detach().cpu().clone(),
                spanner_edge_index=d.spanner_edge_index.detach().cpu().clone()
            ) 
            for d in datas
        ]
        
        # Explicitly delete heavy datas to free memory immediately
        del datas
        
        batch = PrecomputedBatch(datas=lite_datas, packed=packed, labels=labels)
        
        # If we are in a worker process, standard PyTorch IPC for many small tensors 
        # often leads to shared memory exhaustion (unable to mmap). 
        # Returning bytes bypasses this by copying data into a single byte stream.
        # We use it for num_workers > 1 to be safe on shared memory/FD limits,
        # but skip it for 0 or 1 to avoid unnecessary serialization overhead.
        if self.num_workers > 1:
            buf = io.BytesIO()
            torch.save(batch, buf)
            return buf.getvalue()
        else:
            return batch

class TrainingWorker:
    def __init__(self, packer, labeler, device, num_workers=0):
        self.packer = packer
        self.labeler = labeler
        self.device = device
        self.num_workers = int(num_workers)

    def __call__(self, datas):
        t0 = time.time()
        # In workers, ensure datas are deserialized
        datas = [pickle.loads(d) if isinstance(d, bytes) else d for d in datas]
        t1 = time.time()
        
        # Pack and Label (on CPU)
        packed = self.packer.pack_batch(datas)
        t2 = time.time()
        
        labels = self.labeler.label_batch(datas=datas, packed=packed, device=torch.device("cpu"))
        t3 = time.time()
        
        # For training, we don't need the raw 'datas' anymore, only the packed inputs and labels.
        # Returning empty datas saves massive IPC cost.
        batch = PrecomputedBatch(datas=[], packed=packed, labels=labels)
        
        dur = t3 - t0
        if dur > 1.0:
            print(f"[debug] Worker slow: Load={t1-t0:.4f}s Pack={t2-t1:.4f}s Label={t3-t2:.4f}s Total={dur:.4f}s")
        
        # Serialize to bytes to avoid shared memory issues with many small tensors
        if self.num_workers > 0:
            buf = io.BytesIO()
            torch.save(batch, buf)
            return buf.getvalue()
        else:
            return batch

@torch.no_grad()
def precompute_validation_batches(
    val_loader: DataLoader,
    device: torch.device,
) -> List[PrecomputedBatch]:
    print("[val] Start pre-computing validation batches (Parallel Pack + Label)...")
    t0 = time.time()
    batches = []
    pbar = tqdm(total=len(val_loader), desc="[val] Pre-computing batches")
    for batch_raw in val_loader:
        # If batch is bytes (from worker), deserialize it
        if isinstance(batch_raw, bytes):
            try:
                batch = torch.load(io.BytesIO(batch_raw), map_location="cpu", weights_only=False)
            except TypeError:
                batch = torch.load(io.BytesIO(batch_raw), map_location="cpu")
        else:
            batch = batch_raw

        # Keep pre-computed batch on CPU to save VRAM/RAM
        # We will move to device on-demand during validation
        
        batches.append(batch)
        pbar.update(1)

    pbar.close()
    print(f"[val] Pre-computed {len(batches)} batches in {time.time()-t0:.2f}s.")
    return batches

def _label_worker_fn(args_tuple):
    idx, data_simplified, labeler = args_tuple
    # data_simplified is a dict of numpy arrays, which are safe for IPC
    te_np, tlen = labeler.extract_target_edges(data_simplified)
    return idx, te_np, tlen

def precompute_labels_for_dataset(
    data_list: List[Any],
    labeler: Any,
    num_workers: int = 0,
    desc: str = "train",
) -> List[Any]:
    if not data_list:
        return []
    
    # If it's a dict, it's a consolidated/fast format, skip labeling
    if isinstance(data_list, dict):
        return data_list

    # Check if first item already has labels
    if hasattr(data_list[0], "target_edges") and data_list[0].target_edges is not None:
        print(f"[{desc}] Labels already present, skipping pre-computation.")
        return data_list

    algo_name = "LKH" if getattr(labeler, "use_lkh", False) else "2-opt"
    print(f"[{desc}] Pre-computing TSP labels ({algo_name} + projection) for {len(data_list)} samples...")
    t0 = time.time()
    
    from tqdm import tqdm
    if num_workers <= 0:
        pbar = tqdm(total=len(data_list), desc=f"[{desc}] Processing")
        for i, data in enumerate(data_list):
            te_np, tlen = labeler.extract_target_edges(data)
            data.target_edges = torch.from_numpy(te_np)
            data.tour_len = torch.tensor(tlen)
            pbar.update(1)
        pbar.close()
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        # Use a list of tuples for map. Simplify data to numpy to avoid FD exhaustion.
        tasks = [
            (i, labeler.simplify_data_for_ipc(d), labeler) 
            for i, d in enumerate(data_list)
        ]
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # We use as_completed to update progress bar in real-time
            futures = [executor.submit(_label_worker_fn, task) for task in tasks]
            
            pbar = tqdm(total=len(data_list), desc=f"[{desc}] Parallel Processing")
            for future in as_completed(futures):
                idx, te_np, tlen = future.result()
                data_list[idx].target_edges = torch.from_numpy(te_np)
                data_list[idx].tour_len = torch.tensor(tlen)
                pbar.update(1)
            pbar.close()

    print(f"[{desc}] Done pre-computing labels in {time.time()-t0:.2f}s.")
    return data_list


def dataset_has_precomputed_labels(dataset: Any) -> bool:
    """Return True if the dataset already carries target_edges and tour_len."""
    if hasattr(dataset, "c") and isinstance(getattr(dataset, "c"), dict):
        consolidated = dataset.c
        keys = set(consolidated.get("keys", []))
        return (
            "target_edges" in keys
            and "tour_len" in keys
            and "target_edges" in consolidated
            and "target_edges_ptr" in consolidated
            and "tour_len" in consolidated
        )

    if isinstance(dataset, list) and dataset:
        return (
            hasattr(dataset[0], "target_edges")
            and dataset[0].target_edges is not None
            and hasattr(dataset[0], "tour_len")
        )

    return False


def ensure_dataset_labels(
    dataset: Any,
    *,
    source_path: str,
    labeler: Any,
    num_workers: int = 0,
    desc: str = "train",
) -> Any:
    """
    Ensure the dataset carries target_edges/tour_len.

    Older .fast.pt files may have been created before labels were attached,
    which would otherwise make training degenerate to all-zero supervision.
    """
    if dataset_has_precomputed_labels(dataset):
        print(f"[{desc}] Dataset labels already available.")
        return dataset

    if hasattr(dataset, "c") and isinstance(getattr(dataset, "c"), dict):
        from src.dataprep.dataset import FastTSPDataset

        total = len(dataset)
        print(f"[{desc}] Fast dataset is missing labels. Pre-computing and caching labels for {total} samples...")
        t0 = time.time()

        target_edges_list: List[Optional[torch.Tensor]] = [None] * total
        tour_len_list: List[Optional[torch.Tensor]] = [None] * total

        if num_workers <= 0:
            pbar = tqdm(total=total, desc=f"[{desc}] Labeling fast dataset")
            for i in range(total):
                te_np, tlen = labeler.extract_target_edges(dataset[i])
                target_edges_list[i] = torch.from_numpy(te_np).long()
                tour_len_list[i] = torch.tensor(float(tlen), dtype=torch.float32)
                pbar.update(1)
            pbar.close()
        else:
            from concurrent.futures import ProcessPoolExecutor

            def task_iter():
                for i in range(total):
                    yield (i, labeler.simplify_data_for_ipc(dataset[i]), labeler)

            pbar = tqdm(total=total, desc=f"[{desc}] Parallel labeling fast dataset")
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for idx, te_np, tlen in executor.map(_label_worker_fn, task_iter(), chunksize=16):
                    target_edges_list[idx] = torch.from_numpy(te_np).long()
                    tour_len_list[idx] = torch.tensor(float(tlen), dtype=torch.float32)
                    pbar.update(1)
            pbar.close()

        consolidated = dict(dataset.c)
        target_ptr = [0]
        target_chunks: List[torch.Tensor] = []
        for te in target_edges_list:
            cur = te if te is not None else torch.empty((0,), dtype=torch.long)
            target_chunks.append(cur)
            target_ptr.append(target_ptr[-1] + int(cur.numel()))

        consolidated["target_edges"] = (
            torch.cat(target_chunks, dim=0) if target_chunks else torch.empty((0,), dtype=torch.long)
        )
        consolidated["target_edges_ptr"] = torch.tensor(target_ptr, dtype=torch.long)
        consolidated["tour_len"] = torch.stack(
            [tl if tl is not None else torch.tensor(0.0, dtype=torch.float32) for tl in tour_len_list]
        )

        keys = list(consolidated.get("keys", []))
        for key in ["target_edges", "tour_len"]:
            if key not in keys:
                keys.append(key)
        consolidated["keys"] = keys

        save_path = Path(source_path)
        if ".fast.pt" not in save_path.name:
            save_path = save_path.parent / (save_path.stem + ".fast.pt")
        torch.save(consolidated, save_path)
        print(f"[{desc}] Saved labeled fast dataset to {save_path} in {time.time()-t0:.2f}s.")
        return FastTSPDataset(consolidated)

    print(f"[{desc}] List dataset is missing labels. Pre-computing now...")
    return precompute_labels_for_dataset(dataset, labeler, num_workers=num_workers, desc=desc)


def compute_bc_loss(
    *,
    packed: Any,
    labels: Any,
    out_td: Any,
    device: torch.device,
    state_mode: str,
    masked_bce_with_logits,
    masked_ce_with_logits,
) -> torch.Tensor:
    total_M = int(packed.tokens.tree_node_depth.numel())
    root_ids = packed.tokens.root_id.long().to(device)
    is_root = torch.zeros((total_M,), device=device, dtype=torch.bool)
    is_root[root_ids] = True

    if str(state_mode) == "matching":
        if getattr(out_td, "bc_state_logit", None) is None:
            raise RuntimeError("matching mode requires out_td.bc_state_logit.")
        bc_mask = labels.m_state & (~is_root)
        return masked_ce_with_logits(out_td.bc_state_logit, labels.target_state_idx, bc_mask)

    bc_mask = labels.m_iface & (~is_root.unsqueeze(1))
    return masked_bce_with_logits(out_td.bc_iface_logit, labels.y_iface, bc_mask, pos_weight=None)


@torch.no_grad()
def run_validation(
    *,
    val_batches: List[PrecomputedBatch],
    device: torch.device,
    leaf_encoder,
    merge_encoder,
    decoder,
    bu_runner,
    td_runner,
    # loss weights
    w_token: float,
    w_iface_aux: float,
    w_bc: float,
    use_iface_in_decode: bool = True,
    decode_backend: str = "greedy",
    exact_time_limit: float = 30.0,
    exact_length_weight: float = 0.0,
    num_workers: int = 0,
) -> Dict[str, float]:
    """
    Validation returns:
      - token-level BCE losses (cross + optional iface aux)
      - BC loss (supervise out_td.bc_iface_logit vs teacher y_iface on non-root nodes)
      - teacher projection stats (direct/projected/unreachable)
      - decode metrics from edge_logit:
          feasible_rate, gap_mean (vs teacher), off_spanner_mean, components0_mean
    """
    from src.models.edge_aggregation import aggregate_logits_to_edges
    from src.models.decode_backend import DecodingDataset
    from src.models.losses import dp_token_losses, masked_bce_with_logits, masked_ce_with_logits

    leaf_encoder.eval()
    merge_encoder.eval()
    decoder.eval()

    # losses
    loss_total_list: List[float] = []
    loss_token_list: List[float] = []
    loss_cross_list: List[float] = []
    loss_iface_list: List[float] = []
    loss_bc_list: List[float] = []

    # teacher projection stats (per graph)
    direct_list: List[int] = []
    proj_list: List[int] = []
    unr_list: List[int] = []

    # decode metrics
    dec_feasible: List[float] = []
    dec_gap: List[float] = []
    dec_off: List[float] = []
    dec_comp0: List[float] = []

    # Collected tasks for parallel decoding [pos_cpu, sp_cpu, el_cpu, prefer, patch, tlen]
    decode_tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, bool, float]] = []
    # Timers
    t_infer_accum = 0.0
    t_label_accum = 0.0 # Should be 0 now
    t_loss_accum = 0.0
    t_agg_accum = 0.0
    t_cpu_accum = 0.0
    t_decode_accum = 0.0

    t0_loop = time.time()

    for batch in val_batches:
        t_start_batch = time.time()
        
        datas = batch.datas
        packed = batch.packed
        labels = batch.labels
        
        # Move to device on-demand (Validation batches kept on CPU)
        packed = move_data_tensors_to_device(batch.packed, device)
        labels = move_data_tensors_to_device(batch.labels, device)
        
        out_bu = bu_runner.run_batch(batch=packed, leaf_encoder=leaf_encoder, merge_encoder=merge_encoder)
        z = out_bu.z
        out_td = td_runner.run_batch(packed=packed, z=z, decoder=decoder)
        
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_infer_accum += (time.time() - t_start_batch)
        t_mid1 = time.time()

        # Labels are pre-computed
        
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_label_accum += (time.time() - t_mid1)
        t_mid2 = time.time()

        # token loss (cross + optional iface)
        token_out = dp_token_losses(
            cross_logit=out_td.cross_logit,
            y_cross=labels.y_cross,
            m_cross=labels.m_cross,
            iface_logit=out_td.iface_logit,
            y_iface=labels.y_iface,
            m_iface=labels.m_iface,
            w_iface=float(w_iface_aux),
            pos_weight_cross=None,
        )
        L_token = token_out.loss

        # BC loss: supervise propagated parent->child state on non-root nodes.
        L_bc = compute_bc_loss(
            packed=packed,
            labels=labels,
            out_td=out_td,
            device=device,
            state_mode=str(getattr(decoder, "state_mode", "iface")),
            masked_bce_with_logits=masked_bce_with_logits,
            masked_ce_with_logits=masked_ce_with_logits,
        )

        loss = float(w_token) * L_token + float(w_bc) * L_bc
        
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_loss_accum += (time.time() - t_mid2)
        t_mid3 = time.time()

        loss_total_list.append(float(loss.item()))
        loss_token_list.append(float(L_token.item()))
        loss_cross_list.append(float(token_out.parts["loss_cross"].item()))
        loss_iface_list.append(float(token_out.parts.get("loss_iface", torch.tensor(0.0, device=device)).item()))
        loss_bc_list.append(float(L_bc.item()))

        # stats per graph (direct/proj/unr + teacher len for gap)
        # Optimization: use batch stats directly instead of re-running label_one
        direct_list.append(int(labels.stats["num_direct_sum"].item()))
        proj_list.append(int(labels.stats["num_projected_sum"].item()))
        unr_list.append(int(labels.stats["num_unreachable_sum"].item()))
        
        teacher_len_tensor = labels.stats["tour_len"] # [B]
        
        # ---------- decode metrics collection ----------
        if use_iface_in_decode:
            edge_scores = aggregate_logits_to_edges(
                tokens=packed.tokens, 
                cross_logit=out_td.cross_logit,
                iface_logit=out_td.iface_logit,
                reduce="mean",
                num_edges=int(packed.edge_ptr[-1].item()),
            )
        else:
            edge_scores = aggregate_logits_to_edges(
                tokens=packed.tokens, 
                cross_logit=out_td.cross_logit,
                num_edges=int(packed.edge_ptr[-1].item()),
            )
        edge_logit_g = edge_scores.edge_logit
        edge_mask_g = edge_scores.edge_mask.bool()
        
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_agg_accum += (time.time() - t_mid3)
        t_mid4 = time.time()

        B = int(packed.node_ptr.numel() - 1)
        for b in range(B):
            e0 = int(packed.edge_ptr[b].item())
            e1 = int(packed.edge_ptr[b + 1].item())
            if e1 <= e0:
                continue

            # local edge logits for this graph
            el = edge_logit_g[e0:e1].clone()
            em = edge_mask_g[e0:e1]

            # uncovered edges: push down so decode prefers covered edges
            el[~em] = -1e9

            # Pack tasks for parallel decoding (Move to CPU now)
            pos_cpu = datas[b].pos.detach().cpu()
            sp_edge_cpu = datas[b].spanner_edge_index.detach().cpu()
            el_cpu = el.detach().cpu()
            tlen = float(teacher_len_tensor[b].item()) if b < teacher_len_tensor.numel() else 0.0
            
            decode_tasks.append((pos_cpu, sp_edge_cpu, el_cpu, True, True, tlen))
            
        t_cpu_accum += (time.time() - t_mid4)

    # ---- Phase 2: Parallel Decoding on CPU ----
    t_start_decode = time.time()
    if decode_tasks:
        ds = DecodingDataset(
            decode_tasks,
            decode_backend=str(decode_backend),
            exact_time_limit=float(exact_time_limit),
            exact_length_weight=float(exact_length_weight),
        )
        # Use num_workers for parallel decoding
        # Since decoding is CPU bound, multiprocessing is effective.
        decode_loader = DataLoader(
            ds, 
            batch_size=1, 
            shuffle=False, 
            num_workers=int(num_workers), 
            collate_fn=lambda x: x[0]
        )
        
        for res, tlen in decode_loader:
            is_valid = bool(res.feasible) and math.isfinite(float(res.length))
            dec_feasible.append(1.0 if is_valid else 0.0)
            if is_valid:
                dec_off.append(float(res.num_off_spanner_edges))
                dec_comp0.append(float(res.num_components_initial))
                gap = (res.length / tlen - 1.0) if tlen > 1e-9 else 0.0
                dec_gap.append(float(gap))
            
    t_decode_accum = time.time() - t_start_decode

    print(f"[Profiling] Validation Breakdown (Total={time.time()-t0_loop:.2f}s):")
    print(f"  Infer & Pack: {t_infer_accum:.2f}s")
    print(f"  Labeling    : {t_label_accum:.2f}s")
    print(f"  Loss Calc   : {t_loss_accum:.2f}s")
    print(f"  Edge Agg    : {t_agg_accum:.2f}s")
    print(f"  CPU Move    : {t_cpu_accum:.2f}s")
    print(f"  Par Decode  : {t_decode_accum:.2f}s")

    total_graphs = len(dec_feasible)

    def avg(xs: List[float]) -> float:
        return sum(xs) / max(len(xs), 1)
        
    def avg_sum(xs: List[int]) -> float:
         # xs contains sums per batch, so sum(xs) is total count. Divide by total_graphs.
         return sum(xs) / max(total_graphs, 1)

    return {
        "val_loss": avg(loss_total_list),
        "val_loss_token": avg(loss_token_list),
        "val_loss_cross": avg(loss_cross_list),
        "val_loss_iface": avg(loss_iface_list),
        "val_loss_bc": avg(loss_bc_list),
        "val_direct": avg_sum(direct_list),
        "val_projected": avg_sum(proj_list),
        "val_unreachable": avg_sum(unr_list),
        "val_decode_feasible_rate": avg(dec_feasible) if dec_feasible else 0.0,
        "val_decode_gap_mean": avg(dec_gap) if dec_gap else 0.0,
        "val_decode_off_spanner_mean": avg(dec_off) if dec_off else 0.0,
        "val_decode_components0_mean": avg(dec_comp0) if dec_comp0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_pt", type=str, required=True)
    parser.add_argument("--val_pt", type=str, default="")
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--val_interval", type=int, default=200)

    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--num_checkpoints", type=int, default=10, help="max number of recent checkpoints to keep")
    # On Linux, num_workers=4 is safe and faster with consolidated FastTSPDataset.
    # On Windows, we default to 0 to avoid SHM/IPC friction unless the user overrides.
    import platform
    is_linux = platform.system().lower() == "linux"
    default_workers = 4 if is_linux else 0
    parser.add_argument("--num_workers", type=int, default=default_workers, help="number of workers for DataLoader")
    parser.add_argument("--no_shuffle", action="store_true", help="disable shuffling in training")

    # Teacher / labeler
    parser.add_argument("--two_opt_passes", type=int, default=30)
    parser.add_argument("--use_lkh", action="store_true", help="use LKH-3 for TSP ground truth instead of 2-opt")
    parser.add_argument("--lkh_exe", type=str, default="LKH", help="path to LKH-3 executable")

    # Top-down decoder
    parser.add_argument("--td_mode", type=str, default="two_stage", choices=["two_stage", "one_stage"])
    parser.add_argument("--state_mode", type=str, default="iface", choices=["iface", "matching"], help="boundary-condition state representation")
    parser.add_argument("--matching_max_used", type=int, default=4, help="max number of active interfaces enumerated in matching state catalog")

    # Loss weights
    parser.add_argument("--w_token", type=float, default=1.0, help="weight for token-level losses (cross + iface aux)")
    parser.add_argument("--w_iface_aux", type=float, default=0.05, help="aux weight for iface token BCE inside token loss")
    parser.add_argument("--w_bc", type=float, default=1.0, help="weight for BC loss on out_td.bc_iface_logit")
    parser.add_argument("--use_iface_in_decode", type=parse_bool_arg, default=True, help="fusion of iface logit (mean) into edge score during decode")
    parser.add_argument("--decode_backend", type=str, default="greedy", choices=["greedy", "exact"], help="post-processing backend used for validation decoding")
    parser.add_argument("--exact_time_limit", type=float, default=30.0, help="time limit in seconds for exact sparse decoding")
    parser.add_argument("--exact_length_weight", type=float, default=0.0, help="optional Euclidean tie-break weight for exact sparse decoding")

    # Inference / Eval
    parser.add_argument("--ckpt", type=str, default="", help="path to checkpoint to load")
    parser.add_argument("--eval_only", action="store_true", help="run validation only and exit")
    parser.add_argument("--prepare_labels_only", action="store_true", help="materialize and cache target_edges/tour_len, then exit")

    args = parser.parse_args()

    device = resolve_device(str(args.device))
    print(f"[env] device={device}")
    set_seed(int(args.seed))

    resume_ckpt = None
    if args.ckpt and str(args.ckpt).lower() != "none":
        print(f"[ckpt] inspecting architecture config from {args.ckpt}")
        resume_ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        ckpt_args = dict(resume_ckpt.get("args", {}))
        if "td_mode" in ckpt_args and str(args.td_mode) != str(ckpt_args["td_mode"]):
            print(f"[ckpt] overriding td_mode: {args.td_mode} -> {ckpt_args['td_mode']}")
            args.td_mode = ckpt_args["td_mode"]
        if "state_mode" in ckpt_args and str(args.state_mode) != str(ckpt_args["state_mode"]):
            print(f"[ckpt] overriding state_mode: {args.state_mode} -> {ckpt_args['state_mode']}")
            args.state_mode = ckpt_args["state_mode"]
        if "matching_max_used" in ckpt_args and int(args.matching_max_used) != int(ckpt_args["matching_max_used"]):
            print(f"[ckpt] overriding matching_max_used: {args.matching_max_used} -> {ckpt_args['matching_max_used']}")
            args.matching_max_used = int(ckpt_args["matching_max_used"])

    from src.models.bc_state_catalog import infer_boundary_state_count
    from src.models.node_token_packer import NodeTokenPacker
    from src.models.leaf_encoder import LeafEncoder
    from src.models.merge_encoder import MergeEncoder
    from src.models.bottom_up_runner import BottomUpTreeRunner
    from src.models.top_down_decoder import TopDownDecoder
    from src.models.top_down_runner import TopDownTreeRunner
    from src.models.labeler import PseudoLabeler
    from src.models.losses import dp_token_losses, masked_bce_with_logits, masked_ce_with_logits
    from src.dataprep.dataset import TSPDataset, FastTSPDataset, consolidate_data_list, smart_load_dataset

    packer = NodeTokenPacker(
        r=int(args.r),
        state_mode=str(args.state_mode),
        matching_max_used=int(args.matching_max_used),
    )

    leaf_encoder = LeafEncoder(d_model=128).to(device)
    merge_encoder = MergeEncoder(d_model=128).to(device)

    # NOTE: new TopDownDecoder no longer takes "r=..."; mode switch here.
    num_states = None
    if str(args.state_mode) == "matching":
        num_states = infer_boundary_state_count(num_slots=4 * int(args.r), max_used=int(args.matching_max_used))
    decoder = TopDownDecoder(
        d_model=128,
        mode=str(args.td_mode),
        state_mode=str(args.state_mode),
        num_states=num_states,
    ).to(device)

    bu_runner = BottomUpTreeRunner()
    td_runner = TopDownTreeRunner()

    labeler = PseudoLabeler(
        two_opt_passes=int(args.two_opt_passes),
        use_lkh=bool(args.use_lkh),
        lkh_exe=str(args.lkh_exe),
        prefer_cpu=True
    )

    # 1. Load Datasets
    print(f"[data] Loading training dataset...")
    train_dataset = smart_load_dataset(args.train_pt)
    
    val_dataset = None
    if args.val_pt:
        print(f"[data] Loading validation dataset...")
        val_dataset = smart_load_dataset(args.val_pt)

    train_dataset = ensure_dataset_labels(
        train_dataset,
        source_path=args.train_pt,
        labeler=labeler,
        num_workers=int(args.num_workers),
        desc="train",
    )
    if val_dataset is not None:
        val_dataset = ensure_dataset_labels(
            val_dataset,
            source_path=args.val_pt,
            labeler=labeler,
            num_workers=int(args.num_workers),
            desc="val",
        )

    if args.prepare_labels_only:
        print("[labels] Dataset label preparation finished. Exiting without training.")
        return

    params = list(leaf_encoder.parameters()) + list(merge_encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.wd))

    # Safe Checkpointing: Create unique run ID to avoid overwriting
    # Format: run_r{r}_{timestamp}
    run_id = f"run_r{args.r}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ckpt_dir = Path(args.ckpt_dir) / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ckpt] Checkpoints will be saved to: {ckpt_dir}")

    if resume_ckpt is not None:
        print(f"[ckpt] loading from {args.ckpt}")
        ckpt = resume_ckpt
        leaf_encoder.load_state_dict(ckpt["leaf_encoder"])
        merge_encoder.load_state_dict(ckpt["merge_encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        if "opt" in ckpt and not args.eval_only:
            opt.load_state_dict(ckpt["opt"])
        print("[ckpt] loaded.")

    if args.eval_only:
        print("[eval] starting evaluation only...")
        eval_loader = DataLoader(
            val_dataset if val_dataset else train_dataset, 
            batch_size=int(args.batch_size), 
            shuffle=False, 
            num_workers=int(args.num_workers),
            collate_fn=ValidationWorker(packer, labeler, torch.device("cpu"), num_workers=int(args.num_workers))
        )
        
        # Pre-compute
        val_batches = precompute_validation_batches(eval_loader, device=device)
        
        st = run_validation(
            val_batches=val_batches,
            device=device,
            leaf_encoder=leaf_encoder,
            merge_encoder=merge_encoder,
            decoder=decoder,
            bu_runner=bu_runner,
            td_runner=td_runner,
            w_token=float(args.w_token),
            w_iface_aux=float(args.w_iface_aux),
            w_bc=float(args.w_bc),
            use_iface_in_decode=bool(args.use_iface_in_decode),
            decode_backend=str(args.decode_backend),
            exact_time_limit=float(args.exact_time_limit),
            exact_length_weight=float(args.exact_length_weight),
            num_workers=int(args.num_workers),
        )
        msg = " ".join([f"{k}={v:.6f}" for k, v in st.items()])
        print(f"[val] {msg}")
        return

    global_step = 0
    t0 = time.time()
    
    # Checkpoint management
    # Checkpoint management
    best_val_loss = float("inf")
    recent_ckpts: List[Path] = []
    MAX_RECENT_CKPTS = int(args.num_checkpoints)

    val_batches = []
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            collate_fn=ValidationWorker(packer, labeler, torch.device("cpu"), num_workers=int(args.num_workers))
        )
        val_batches = precompute_validation_batches(val_loader, device)
        
        # Cleanup validation workers explicitly to free memory/handles
        del val_loader
        import gc
        gc.collect()

    # For training with FastTSPDataset (already in memory), workers add massive IPC overhead
    # copying the 20GB+ dict. Main process loading is faster and safer.
    # Use TrainingWorker to offload packing/labeling logic (logic reuse)
    # Reverting to args.num_workers as requested. 
    # With 'use_pickle' removed from dataset, accessing shared memory via Fork (Linux) should be efficient.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=int(args.batch_size), 
        shuffle=not args.no_shuffle, 
        num_workers=int(args.num_workers), 
        collate_fn=TrainingWorker(packer, labeler, torch.device("cpu"), num_workers=int(args.num_workers))
    )

    for epoch in range(int(args.epochs)):
        leaf_encoder.train()
        merge_encoder.train()
        decoder.train()

        pbar = tqdm(train_loader, desc=f"[train] Ep {epoch}/{args.epochs}")
        for i, batch_raw in enumerate(pbar):
            # 1. Deserialize if needed
            if isinstance(batch_raw, bytes):
                try:
                    batch = torch.load(io.BytesIO(batch_raw), map_location="cpu", weights_only=False)
                except TypeError:
                    batch = torch.load(io.BytesIO(batch_raw), map_location="cpu")
            else:
                batch = batch_raw

            # 2. Move to device (packed and labels)
            packed = move_data_tensors_to_device(batch.packed, device)
            labels = move_data_tensors_to_device(batch.labels, device)

            # 3. Forward pass
            out_bu = bu_runner.run_batch(batch=packed, leaf_encoder=leaf_encoder, merge_encoder=merge_encoder)
            z = out_bu.z
            out_td = td_runner.run_batch(packed=packed, z=z, decoder=decoder)

            # ... loss calculation ...
            # labels = labeler.label_batch(datas=datas, packed=packed, device=device) # Done in worker

            # token loss (cross + optional iface aux)
            token_out = dp_token_losses(
                cross_logit=out_td.cross_logit,
                y_cross=labels.y_cross,
                m_cross=labels.m_cross,
                iface_logit=out_td.iface_logit,
                y_iface=labels.y_iface,
                m_iface=labels.m_iface,
                w_iface=float(args.w_iface_aux),
                pos_weight_cross=None,
            )
            L_token = token_out.loss

            # BC loss: supervise propagated parent->child state on non-root nodes.
            L_bc = compute_bc_loss(
                packed=packed,
                labels=labels,
                out_td=out_td,
                device=device,
                state_mode=str(args.state_mode),
                masked_bce_with_logits=masked_bce_with_logits,
                masked_ce_with_logits=masked_ce_with_logits,
            )

            loss = float(args.w_token) * L_token + float(args.w_bc) * L_bc

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(args.grad_clip))

            opt.step()
            global_step += 1

            # Skip redundant b0 stats in hot training loop for better throughput
            direct0 = proj0 = unr0 = 0

            if global_step % int(args.log_interval) == 0:
                dt = time.time() - t0
                parts = ", ".join([f"{k}={float(v.item()):.6f}" for k, v in token_out.parts.items()])
                print(
                    f"\n[train] epoch={epoch} step={global_step} "
                    f"loss={float(loss.item()):.6f} "
                    f"L_token={float(L_token.item()):.6f} L_bc={float(L_bc.item()):.6f} "
                    f"({parts}) "
                    f"b0(direct={direct0},proj={proj0},unr={unr0}) "
                    f"time={dt:.1f}s"
                )

            if global_step % int(args.save_interval) == 0:
                ckpt_path = ckpt_dir / f"ckpt_step_{global_step}.pt"
                torch.save(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "args": vars(args),
                        "leaf_encoder": leaf_encoder.state_dict(),
                        "merge_encoder": merge_encoder.state_dict(),
                        "decoder": decoder.state_dict(),
                        "opt": opt.state_dict(),
                    },
                    ckpt_path,
                )
                print(f"[ckpt] saved to {ckpt_path}")
                
                # Checkpoint rotation (keep max 10)
                recent_ckpts.append(ckpt_path)
                if len(recent_ckpts) > MAX_RECENT_CKPTS:
                    oldest = recent_ckpts.pop(0)
                    if oldest.exists():
                        try:
                            os.remove(oldest)
                            print(f"[ckpt] removed old checkpoint {oldest}")
                        except Exception as e:
                            print(f"[ckpt] failed to remove {oldest}: {e}")

            if val_batches and (global_step % int(args.val_interval) == 0):
                st = run_validation(
                    val_batches=val_batches,
                    device=device,
                    leaf_encoder=leaf_encoder,
                    merge_encoder=merge_encoder,
                    decoder=decoder,
                    bu_runner=bu_runner,
                    td_runner=td_runner,
                    w_token=float(args.w_token),
                    w_iface_aux=float(args.w_iface_aux),
                    w_bc=float(args.w_bc),
                    use_iface_in_decode=bool(args.use_iface_in_decode),
                    decode_backend=str(args.decode_backend),
                    exact_time_limit=float(args.exact_time_limit),
                    exact_length_weight=float(args.exact_length_weight),
                    num_workers=int(args.num_workers),
                )
                msg = " ".join([f"{k}={v:.6f}" for k, v in st.items()])
                print(f"[val] {msg}")

                # Save best checkpoint
                val_loss = st["val_loss"]
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = ckpt_dir / "ckpt_best.pt"
                    torch.save(
                        {
                            "step": global_step,
                            "epoch": epoch,
                            "args": vars(args),
                            "leaf_encoder": leaf_encoder.state_dict(),
                            "merge_encoder": merge_encoder.state_dict(),
                            "decoder": decoder.state_dict(),
                            "opt": opt.state_dict(),
                            "val_metrics": st,
                        },
                        best_path,
                    )
                    print(f"[ckpt] new best val_loss={val_loss:.6f} saved to {best_path}")
    ckpt_path = ckpt_dir / f"ckpt_final_step_{global_step}.pt"
    torch.save(
        {
            "step": global_step,
            "epoch": int(args.epochs),
            "args": vars(args),
            "leaf_encoder": leaf_encoder.state_dict(),
            "merge_encoder": merge_encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "opt": opt.state_dict(),
        },
        ckpt_path,
    )
    print(f"[ckpt] saved to {ckpt_path}")
    print("[done] training finished.")


if __name__ == "__main__":
    main()
