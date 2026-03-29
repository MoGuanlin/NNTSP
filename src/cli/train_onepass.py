# src/cli/train_onepass.py
# -*- coding: utf-8 -*-
"""
Training script for the 1-pass DP pipeline (σ-conditioned MergeDecoder).

This is a SEPARATE entry point from the existing 2-pass train.py.
It reuses the same data loading, packing, and labeling infrastructure,
but swaps the top-down decoder for the σ-conditioned MergeDecoder.

Usage:
  python -m src.cli.train_onepass --train_pt <path> [options]

The MergeDecoder is trained to predict child boundary states given a parent σ.
Teacher σ comes from the PseudoLabeler's target_state_idx (matching mode).
"""

from __future__ import annotations

import argparse
import datetime
import gc
import io
import logging
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# ─── Shared utilities from train.py ──────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if str(device_arg).startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def move_data_tensors_to_device(data: Any, device: torch.device) -> Any:
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(device)
            elif v is not None:
                move_data_tensors_to_device(v, device)
        return data
    if isinstance(data, list):
        for i, v in enumerate(data):
            if isinstance(v, torch.Tensor):
                data[i] = v.to(device)
            elif v is not None:
                move_data_tensors_to_device(v, device)
        return data
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
            move_data_tensors_to_device(v, device)
    return data


# ─── Worker for DataLoader ───────────────────────────────────────────────────

from dataclasses import dataclass


@dataclass
class PrecomputedBatch:
    packed: Any
    labels: Any


def _ensure_labels_compat(labels: Any) -> Any:
    """Backfill fields expected by the current training code for old caches."""
    if not hasattr(labels, "m_state_exact"):
        if hasattr(labels, "m_state") and isinstance(labels.m_state, torch.Tensor):
            labels.m_state_exact = labels.m_state.bool().clone()
        elif hasattr(labels, "target_state_idx") and isinstance(labels.target_state_idx, torch.Tensor):
            labels.m_state_exact = (labels.target_state_idx >= 0).bool().clone()
        elif hasattr(labels, "y_iface") and isinstance(labels.y_iface, torch.Tensor):
            labels.m_state_exact = torch.zeros(
                (int(labels.y_iface.shape[0]),),
                dtype=torch.bool,
                device=labels.y_iface.device,
            )
        else:
            labels.m_state_exact = torch.zeros((0,), dtype=torch.bool)
    return labels


def _load_precomputed_batch(batch_raw: Any) -> PrecomputedBatch:
    """Deserialize a possibly-bytes batch produced by OnePassWorker."""
    if isinstance(batch_raw, bytes):
        try:
            batch = torch.load(io.BytesIO(batch_raw), map_location="cpu", weights_only=False)
        except TypeError:
            batch = torch.load(io.BytesIO(batch_raw), map_location="cpu")
    else:
        batch = batch_raw
    if hasattr(batch, "labels"):
        batch.labels = _ensure_labels_compat(batch.labels)
    return batch


def _hist_add_from_values(hist: Dict[int, int], values: torch.Tensor) -> None:
    """Accumulate an integer histogram from a 1D tensor of counts."""
    if values.numel() == 0:
        return
    values = values.detach().to(dtype=torch.long, device="cpu").view(-1)
    max_v = int(values.max().item())
    binc = torch.bincount(values, minlength=max_v + 1)
    for idx, cnt in enumerate(binc.tolist()):
        if cnt:
            hist[idx] = hist.get(idx, 0) + int(cnt)


def _coverage_from_hist(hist: Dict[int, int]) -> Dict[int, float]:
    """Convert a histogram into cumulative coverage P(num_used <= k)."""
    total = int(sum(hist.values()))
    if total <= 0:
        return {}
    running = 0
    coverage: Dict[int, float] = {}
    for k in range(max(hist.keys()) + 1):
        running += int(hist.get(k, 0))
        coverage[k] = float(running / total)
    return coverage


def _min_k_for_coverage(coverage: Dict[int, float], target: float) -> int:
    """Smallest k whose cumulative coverage reaches target."""
    if not coverage:
        return 0
    tgt = float(target)
    for k in sorted(coverage):
        if coverage[k] >= tgt:
            return int(k)
    return int(max(coverage))


def _coverage_at_k(coverage: Dict[int, float], k: int) -> float:
    """Coverage lookup that stays at 100% once k exceeds the observed max."""
    if not coverage:
        return 0.0
    kk = int(k)
    max_k = int(max(coverage))
    if kk >= max_k:
        return float(coverage[max_k])
    return float(coverage.get(kk, 0.0))


def _format_histogram(hist: Dict[int, int]) -> str:
    """Compact 'k:count (pct%)' histogram formatter."""
    total = int(sum(hist.values()))
    if total <= 0:
        return "none"
    parts = []
    for k in sorted(hist):
        cnt = int(hist[k])
        parts.append(f"{k}:{cnt} ({100.0 * cnt / total:.1f}%)")
    return ", ".join(parts)


def _collect_teacher_max_used_stats(
    batch_iter,
    *,
    configured_max_used: int,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Scan batches and summarize raw teacher interface usage counts."""
    raw_hist_all: Dict[int, int] = {}
    raw_hist_internal: Dict[int, int] = {}
    num_nodes_all = 0
    num_nodes_internal = 0
    num_exact_all = 0
    num_exact_internal = 0
    overflow_all = 0
    overflow_internal = 0

    iterator = tqdm(batch_iter, desc="[teacher] scanning max_used", disable=not show_progress)
    for batch_raw in iterator:
        batch = _load_precomputed_batch(batch_raw)
        packed = batch.packed
        labels = batch.labels

        raw_used = ((labels.y_iface > 0.5) & labels.m_iface.bool()).sum(dim=1)
        raw_used = raw_used.detach().to(dtype=torch.long, device="cpu")

        is_internal = (~packed.tokens.is_leaf.bool()).detach().to(device="cpu")
        exact_mask = labels.m_state_exact.bool().detach().to(device="cpu")

        num_nodes_all += int(raw_used.numel())
        num_nodes_internal += int(is_internal.sum().item())
        num_exact_all += int(exact_mask.sum().item())
        num_exact_internal += int((exact_mask & is_internal).sum().item())
        overflow_all += int((raw_used > int(configured_max_used)).sum().item())
        overflow_internal += int((raw_used[is_internal] > int(configured_max_used)).sum().item())

        _hist_add_from_values(raw_hist_all, raw_used)
        _hist_add_from_values(raw_hist_internal, raw_used[is_internal])

    coverage_all = _coverage_from_hist(raw_hist_all)
    coverage_internal = _coverage_from_hist(raw_hist_internal)
    return {
        "configured_max_used": int(configured_max_used),
        "num_nodes_all": int(num_nodes_all),
        "num_nodes_internal": int(num_nodes_internal),
        "num_exact_all": int(num_exact_all),
        "num_exact_internal": int(num_exact_internal),
        "overflow_all": int(overflow_all),
        "overflow_internal": int(overflow_internal),
        "raw_hist_all": raw_hist_all,
        "raw_hist_internal": raw_hist_internal,
        "coverage_all_by_k": coverage_all,
        "coverage_internal_by_k": coverage_internal,
        "current_coverage_all": _coverage_at_k(coverage_all, int(configured_max_used)),
        "current_coverage_internal": _coverage_at_k(coverage_internal, int(configured_max_used)),
        "exact_rate_all": float(num_exact_all / num_nodes_all) if num_nodes_all > 0 else 0.0,
        "exact_rate_internal": float(num_exact_internal / num_nodes_internal) if num_nodes_internal > 0 else 0.0,
        "suggested_max_used_90": _min_k_for_coverage(coverage_internal, 0.90),
        "suggested_max_used_95": _min_k_for_coverage(coverage_internal, 0.95),
        "suggested_max_used_99": _min_k_for_coverage(coverage_internal, 0.99),
        "max_raw_used_all": int(max(raw_hist_all)) if raw_hist_all else 0,
        "max_raw_used_internal": int(max(raw_hist_internal)) if raw_hist_internal else 0,
    }


def _log_teacher_max_used_stats(summary: Dict[str, Any], log) -> None:
    """Emit the teacher max_used summary at training start."""
    cap = int(summary["configured_max_used"])
    n_all = int(summary["num_nodes_all"])
    n_int = int(summary["num_nodes_internal"])
    ov_all = int(summary["overflow_all"])
    ov_int = int(summary["overflow_internal"])
    log(
        f"[teacher] raw active-iface histogram over all nodes (N={n_all}): "
        f"{_format_histogram(summary['raw_hist_all'])}"
    )
    log(
        f"[teacher] raw active-iface histogram over internal nodes (N={n_int}): "
        f"{_format_histogram(summary['raw_hist_internal'])}"
    )
    log(
        f"[teacher] current matching_max_used={cap}: "
        f"raw coverage all={summary['current_coverage_all'] * 100.0:.2f}% "
        f"(overflow {ov_all}/{n_all}), "
        f"internal={summary['current_coverage_internal'] * 100.0:.2f}% "
        f"(overflow {ov_int}/{n_int})"
    )
    log(
        f"[teacher] exact teacher-state availability: "
        f"all={summary['exact_rate_all'] * 100.0:.2f}% "
        f"({summary['num_exact_all']}/{n_all}), "
        f"internal={summary['exact_rate_internal'] * 100.0:.2f}% "
        f"({summary['num_exact_internal']}/{n_int})"
    )
    log(
        f"[teacher] suggested max_used from raw internal-node coverage: "
        f"90%->{summary['suggested_max_used_90']}, "
        f"95%->{summary['suggested_max_used_95']}, "
        f"99%->{summary['suggested_max_used_99']}, "
        f"max={summary['max_raw_used_internal']}"
    )


class OnePassWorker:
    """Collate function: pack + label a batch for 1-pass training."""

    def __init__(self, packer, labeler, num_workers=0):
        self.packer = packer
        self.labeler = labeler
        self.num_workers = int(num_workers)

    def __call__(self, datas):
        datas = [pickle.loads(d) if isinstance(d, bytes) else d for d in datas]
        packed = self.packer.pack_batch(datas)
        labels = self.labeler.label_batch(datas=datas, packed=packed, device=torch.device("cpu"))
        batch = PrecomputedBatch(packed=packed, labels=labels)
        if self.num_workers > 0:
            buf = io.BytesIO()
            torch.save(batch, buf)
            return buf.getvalue()
        return batch


def _catalog_from_packed(packed) -> "BoundaryStateCatalog":
    """Convert PackedStateCatalog to BoundaryStateCatalog for the DP runner."""
    from src.models.bc_state_catalog import BoundaryStateCatalog
    sc = packed.state_catalog
    return BoundaryStateCatalog(
        used_iface=sc.used_iface,
        mate=sc.mate,
        num_used=sc.num_used,
        empty_index=sc.empty_index,
        max_used=sc.max_used,
    )


def _precompute_cache_path(
    train_pt: str, batch_size: int, r: int, max_used: int,
    two_opt_passes: int = 30, use_lkh: bool = False,
    teacher_signature: str = "",
) -> Path:
    """Derive a deterministic cache path from key parameters."""
    import hashlib
    src = Path(train_pt).resolve()
    fast_path = src if ".fast.pt" in src.name else (src.parent / f"{src.stem}.fast.pt")
    label_source = fast_path if fast_path.exists() else src
    stat = label_source.stat() if label_source.exists() else None
    key = (
        f"{src}|bs={batch_size}|r={r}|mu={max_used}"
        f"|2opt={two_opt_passes}|lkh={use_lkh}"
        f"|teacher={teacher_signature}"
        f"|src_mtime_ns={(stat.st_mtime_ns if stat else 0)}"
        f"|src_size={(stat.st_size if stat else 0)}"
    )
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return Path(train_pt).parent / f".precomputed_bs{batch_size}_r{r}_{h}.pt"


def _precompute_batches(
    dataset,
    batch_size: int,
    packer,
    labeler,
    shuffle: bool,
    num_workers: int,
    cache_path: Optional[Path] = None,
) -> List[PrecomputedBatch]:
    """Pre-compute all packed+labeled batches once at startup.

    If cache_path is provided and exists, load from disk (instant).
    Otherwise compute and save to cache_path for next time.
    """
    # Try loading from disk cache
    if cache_path is not None and cache_path.exists():
        print(f"[precompute] Loading cached batches from {cache_path}")
        try:
            batches = torch.load(cache_path, map_location="cpu", weights_only=False)
            print(f"[precompute] Loaded {len(batches)} batches from cache")
            return batches
        except Exception as e:
            print(f"[precompute] Cache load failed ({e}), recomputing...")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # deterministic order for pre-compute
        num_workers=num_workers,
        collate_fn=OnePassWorker(packer, labeler, num_workers=num_workers),
    )

    batches = []
    for batch_raw in tqdm(loader, desc="[precompute] packing + labeling"):
        if isinstance(batch_raw, bytes):
            try:
                batch = torch.load(io.BytesIO(batch_raw), map_location="cpu", weights_only=False)
            except TypeError:
                batch = torch.load(io.BytesIO(batch_raw), map_location="cpu")
        else:
            batch = batch_raw
        batches.append(batch)

    # Save to disk for next run
    if cache_path is not None:
        print(f"[precompute] Saving {len(batches)} batches to {cache_path}")
        torch.save(batches, cache_path)

    return batches


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    from src.utils.lkh_solver import default_lkh_executable

    parser = argparse.ArgumentParser(description="1-pass DP training with σ-conditioned MergeDecoder")
    default_lkh = default_lkh_executable()

    # Data
    parser.add_argument("--train_pt", type=str, required=True)
    parser.add_argument("--val_pt", type=str, default="")

    # Architecture
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--matching_max_used", type=int, default=4)
    parser.add_argument("--parent_num_layers", type=int, default=3, help="self-attention layers in parent memory")
    parser.add_argument("--cross_num_layers", type=int, default=2, help="cross-attention layers for sigma query")

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--pos_weight", type=float, default=0.0, help="pos_weight for BCE (0 = disabled)")
    parser.add_argument(
        "--child_target_mode",
        type=str,
        default="state_exact",
        choices=["raw_iface", "state_exact", "state_projected"],
        help=(
            "supervision target for decoded children: "
            "'raw_iface' uses raw teacher edge usage, "
            "'state_exact' uses exact child catalog states only, "
            "'state_projected' uses projected child catalog states"
        ),
    )
    parser.add_argument("--amp", action="store_true", help="enable automatic mixed precision (bf16/fp16)")

    # Logging & checkpointing
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--num_checkpoints", type=int, default=10)

    # Infra
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    import platform
    default_workers = 4 if platform.system().lower() == "linux" else 0
    parser.add_argument("--num_workers", type=int, default=default_workers)
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--no_precompute", action="store_true",
                        help="disable batch pre-computation (use streaming DataLoader instead)")

    # Teacher
    parser.add_argument("--two_opt_passes", type=int, default=30, help="deprecated; ignored by spanner-LKH teacher generation")
    parser.add_argument("--use_lkh", action="store_true", help="deprecated; teacher generation always uses LKH on the sparse spanner")
    parser.add_argument("--lkh_exe", type=str, default=default_lkh)
    parser.add_argument("--teacher_lkh_runs", type=int, default=1)
    parser.add_argument("--teacher_lkh_timeout", type=float, default=0.0, help="0 disables timeout")

    # Resume
    parser.add_argument("--ckpt", type=str, default="")

    args = parser.parse_args()

    device = resolve_device(str(args.device))
    print(f"[env] device={device}")
    set_seed(int(args.seed))

    # ─── Imports ──────────────────────────────────────────────────────
    from src.models.node_token_packer import NodeTokenPacker
    from src.models.leaf_encoder import LeafEncoder
    from src.models.merge_encoder import MergeEncoder
    from src.models.merge_decoder import MergeDecoder
    from src.models.onepass_trainer import (
        OnePassTrainRunner,
        build_child_iface_targets_from_states,
        onepass_loss,
    )
    from src.models.labeler import PseudoLabeler
    from src.dataprep.dataset import smart_load_dataset

    d = int(args.d_model)
    r = int(args.r)
    Ti = 4 * r  # interface slots = 4 * r

    # ─── Packer (matching mode required) ──────────────────────────────
    packer = NodeTokenPacker(
        r=r,
        state_mode="matching",
        matching_max_used=int(args.matching_max_used),
    )

    # ─── Models ───────────────────────────────────────────────────────
    leaf_encoder = LeafEncoder(d_model=d).to(device)
    merge_encoder = MergeEncoder(d_model=d).to(device)
    merge_decoder = MergeDecoder(
        d_model=d,
        n_heads=max(4, d // 32),
        num_iface_slots=Ti,
        parent_num_layers=int(args.parent_num_layers),
        cross_num_layers=int(args.cross_num_layers),
        max_depth=64,
    ).to(device)

    runner = OnePassTrainRunner()

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

    # ─── Datasets ─────────────────────────────────────────────────────
    print("[data] Loading training dataset...")
    train_dataset = smart_load_dataset(args.train_pt)

    # Import ensure_dataset_labels from train.py
    from src.cli.train import ensure_dataset_labels
    train_dataset = ensure_dataset_labels(
        train_dataset,
        source_path=args.train_pt,
        labeler=labeler,
        num_workers=int(args.num_workers),
        desc="train",
    )

    # ─── Optimizer ────────────────────────────────────────────────────
    params = (
        list(leaf_encoder.parameters())
        + list(merge_encoder.parameters())
        + list(merge_decoder.parameters())
    )
    opt = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.wd))

    # ─── AMP scaler ──────────────────────────────────────────────────
    use_amp = bool(args.amp) and device.type == "cuda"
    # Prefer bf16 if available (A100 supports it), else fp16
    if use_amp:
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            print("[amp] Using bfloat16 mixed precision")
        else:
            amp_dtype = torch.float16
            print("[amp] Using float16 mixed precision")
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    else:
        amp_dtype = torch.float32
        scaler = None

    # ─── Checkpoint ───────────────────────────────────────────────────
    run_id = f"onepass_r{r}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ckpt_dir = Path(args.ckpt_dir) / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ─── Logging: tee to file + stdout ───────────────────────────────
    log_path = ckpt_dir / "train.log"
    logger = logging.getLogger("onepass_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    def log(msg: str) -> None:
        logger.info(msg)

    log(f"[ckpt] Checkpoints → {ckpt_dir}")
    log(f"[args] {vars(args)}")

    if args.ckpt:
        log(f"[ckpt] Loading from {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        leaf_encoder.load_state_dict(ckpt["leaf_encoder"])
        merge_encoder.load_state_dict(ckpt["merge_encoder"])
        if "merge_decoder" in ckpt:
            merge_decoder.load_state_dict(ckpt["merge_decoder"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        log("[ckpt] Loaded.")

    # ─── Pre-compute or streaming DataLoader ─────────────────────────
    use_precompute = not args.no_precompute
    cached_batches: Optional[List[PrecomputedBatch]] = None

    if use_precompute:
        cache_path = _precompute_cache_path(
            args.train_pt, int(args.batch_size), r, int(args.matching_max_used),
            two_opt_passes=int(args.two_opt_passes), use_lkh=bool(args.use_lkh),
            teacher_signature=labeler.label_signature(),
        )
        log(f"[data] Precompute cache: {cache_path}")
        t_pre = time.time()
        cached_batches = _precompute_batches(
            dataset=train_dataset,
            batch_size=int(args.batch_size),
            packer=packer,
            labeler=labeler,
            shuffle=not args.no_shuffle,
            num_workers=int(args.num_workers),
            cache_path=cache_path,
        )
        dt_pre = time.time() - t_pre
        log(f"[data] Pre-computed {len(cached_batches)} batches in {dt_pre:.1f}s")
        # Free dataset memory since we no longer need it
        del train_dataset
        gc.collect()
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(args.batch_size),
            shuffle=not args.no_shuffle,
            num_workers=int(args.num_workers),
            collate_fn=OnePassWorker(packer, labeler, num_workers=int(args.num_workers)),
        )

    log("[teacher] Scanning training-set teacher boundary usage...")
    t_teacher = time.time()
    teacher_usage_summary = _collect_teacher_max_used_stats(
        cached_batches if use_precompute else train_loader,
        configured_max_used=int(args.matching_max_used),
        show_progress=True,
    )
    log(f"[teacher] max_used stats computed in {time.time() - t_teacher:.1f}s")
    _log_teacher_max_used_stats(teacher_usage_summary, log)

    # ─── Training loop ────────────────────────────────────────────────
    global_step = 0
    t0 = time.time()
    recent_ckpts: List[Path] = []
    MAX_RECENT = int(args.num_checkpoints)
    pw = float(args.pos_weight) if float(args.pos_weight) > 0 else None

    log(f"[train] Starting 1-pass training: {args.epochs} epochs, bs={args.batch_size}")
    log(f"[train] Models: LeafEnc + MergeEnc + MergeDecoder (d={d}, Ti={Ti})")
    log(f"[train] Params: {sum(p.numel() for p in params):,}")
    log(f"[train] Child target mode: {args.child_target_mode}")
    if use_amp:
        log(f"[train] AMP enabled ({amp_dtype})")
    if use_precompute:
        log(f"[train] Pre-computed mode: {len(cached_batches)} batches cached in RAM")

    for epoch in range(int(args.epochs)):
        leaf_encoder.train()
        merge_encoder.train()
        merge_decoder.train()

        # Build iteration source
        if use_precompute:
            if not args.no_shuffle:
                random.shuffle(cached_batches)
            batch_iter = cached_batches
        else:
            batch_iter = train_loader

        pbar = tqdm(batch_iter, desc=f"[train] Ep {epoch}/{args.epochs}")
        for batch_raw in pbar:
            batch = _load_precomputed_batch(batch_raw)

            packed = move_data_tensors_to_device(batch.packed, device)
            labels = move_data_tensors_to_device(batch.labels, device)

            # Check that state catalog exists
            if packed.state_catalog is None:
                raise RuntimeError(
                    "1-pass training requires state_mode='matching'. "
                    "PackedBatch has no state_catalog."
                )

            catalog = _catalog_from_packed(packed)

            # Forward: 1-pass bottom-up with σ-conditioned decode
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                result = runner.run_batch(
                    batch=packed,
                    leaf_encoder=leaf_encoder,
                    merge_encoder=merge_encoder,
                    merge_decoder=merge_decoder,
                    catalog=catalog,
                    target_state_idx=labels.target_state_idx,
                    m_state=labels.m_state_exact,
                )

                if args.child_target_mode == "raw_iface":
                    y_child_target = labels.y_child_iface
                    m_child_target = labels.m_child_iface
                else:
                    child_state_mask = (
                        labels.m_state_exact
                        if args.child_target_mode == "state_exact"
                        else labels.m_state
                    )
                    y_child_target, m_child_target = build_child_iface_targets_from_states(
                        tree_children_index=packed.tokens.tree_children_index,
                        m_child_iface=labels.m_child_iface,
                        target_state_idx=labels.target_state_idx,
                        child_state_mask=child_state_mask,
                        state_used_iface=catalog.used_iface,
                    )

                # Loss
                loss = onepass_loss(
                    child_scores=result.child_scores,
                    decode_mask=result.decode_mask,
                    y_child_iface=y_child_target,
                    m_child_iface=m_child_target,
                    pos_weight=pw,
                )

            # Backward
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                if float(args.grad_clip) > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, max_norm=float(args.grad_clip))
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if float(args.grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=float(args.grad_clip))
                opt.step()

            global_step += 1

            # Log
            if global_step % int(args.log_interval) == 0:
                n_decoded = int(result.decode_mask.sum().item())
                total_nodes = int(result.decode_mask.numel())
                dt = time.time() - t0
                log(
                    f"[train] ep={epoch} step={global_step} "
                    f"loss={loss.item():.6f} "
                    f"decoded={n_decoded}/{total_nodes} "
                    f"time={dt:.1f}s"
                )

            # Save
            if global_step % int(args.save_interval) == 0:
                ckpt_path = ckpt_dir / f"ckpt_step_{global_step}.pt"
                torch.save(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "args": vars(args),
                        "leaf_encoder": leaf_encoder.state_dict(),
                        "merge_encoder": merge_encoder.state_dict(),
                        "merge_decoder": merge_decoder.state_dict(),
                        "opt": opt.state_dict(),
                    },
                    ckpt_path,
                )
                log(f"[ckpt] saved → {ckpt_path}")
                recent_ckpts.append(ckpt_path)
                if len(recent_ckpts) > MAX_RECENT:
                    oldest = recent_ckpts.pop(0)
                    if oldest.exists():
                        try:
                            os.remove(oldest)
                        except Exception:
                            pass

            # Free GPU memory: move_data_tensors_to_device mutates the
            # cached batch in-place, so every processed batch accumulates
            # on GPU.  Move back to CPU to keep only one batch on GPU at
            # a time.
            del result, loss
            if use_precompute:
                _cpu = torch.device("cpu")
                move_data_tensors_to_device(batch.packed, _cpu)
                move_data_tensors_to_device(batch.labels, _cpu)

    # Final save
    ckpt_path = ckpt_dir / f"ckpt_final_step_{global_step}.pt"
    torch.save(
        {
            "step": global_step,
            "epoch": int(args.epochs),
            "args": vars(args),
            "leaf_encoder": leaf_encoder.state_dict(),
            "merge_encoder": merge_encoder.state_dict(),
            "merge_decoder": merge_decoder.state_dict(),
            "opt": opt.state_dict(),
        },
        ckpt_path,
    )
    log(f"[ckpt] Final saved → {ckpt_path}")
    log("[done] 1-pass training finished.")


if __name__ == "__main__":
    main()
