# src/cli/analyze_teacher_max_used.py
# -*- coding: utf-8 -*-
"""
Scan teacher-tour boundary usage for a dataset/sample range.

This is useful for answering:
  - under alive-edge-only teacher tours, what max_used do we actually see?
  - how much of the data is covered by matching_max_used = 4 / 6 / ...?
  - how often do exact matching states exist?
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, Subset

from src.cli.train_onepass import (
    PrecomputedBatch,
    OnePassWorker,
    _collect_teacher_max_used_stats,
    _log_teacher_max_used_stats,
)
from src.cli.eval_and_vis import load_dataset
from src.models.labeler import PseudoLabeler
from src.models.node_token_packer import NodeTokenPacker
from src.utils.lkh_solver import default_lkh_executable


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze teacher max_used on a dataset")
    parser.add_argument("--data_pt", type=str, required=True, help="dataset .pt path")
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--matching_max_used", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--sample_idx_end", type=int, default=None)
    parser.add_argument("--teacher_lkh_runs", type=int, default=1)
    parser.add_argument("--teacher_lkh_timeout", type=float, default=0.0, help="0 disables timeout")
    parser.add_argument("--lkh_exe", type=str, default=default_lkh_executable())
    parser.add_argument(
        "--skip_teacher_failures",
        action="store_true",
        help="skip samples whose alive-only teacher cannot be validated instead of aborting",
    )
    args = parser.parse_args(argv)

    dataset = load_dataset(args.data_pt)
    start_idx = max(0, int(args.sample_idx))
    end_idx = len(dataset) if args.sample_idx_end is None else min(len(dataset), int(args.sample_idx_end))
    if end_idx <= start_idx:
        raise ValueError(f"empty sample range: [{start_idx}, {end_idx})")

    subset = Subset(dataset, list(range(start_idx, end_idx)))
    print(f"[data] {args.data_pt} samples=[{start_idx}, {end_idx}) count={len(subset)}")

    packer = NodeTokenPacker(
        r=int(args.r),
        state_mode="matching",
        matching_max_used=int(args.matching_max_used),
    )
    labeler = PseudoLabeler(
        two_opt_passes=30,
        use_lkh=False,
        lkh_exe=str(args.lkh_exe),
        prefer_cpu=True,
        teacher_mode="spanner_lkh",
        teacher_lkh_runs=int(args.teacher_lkh_runs),
        teacher_lkh_timeout=(None if float(args.teacher_lkh_timeout) <= 0 else float(args.teacher_lkh_timeout)),
    )
    print(f"[teacher] signature={labeler.label_signature()}")
    print(f"[teacher] lkh={labeler.lkh_exe}")

    if args.skip_teacher_failures:
        batches: List[PrecomputedBatch] = []
        failed = 0
        worker = OnePassWorker(packer, labeler, num_workers=0)
        for rel_idx in range(len(subset)):
            ds_idx = start_idx + rel_idx
            try:
                batch = worker([subset[rel_idx]])
                batches.append(batch)
            except Exception as exc:
                failed += 1
                print(f"[warn] sample={ds_idx} teacher failed, skipped: {exc}")
        print(f"[teacher] valid_samples={len(batches)} failed_samples={failed}")
        summary = _collect_teacher_max_used_stats(
            batches,
            configured_max_used=int(args.matching_max_used),
            show_progress=True,
        )
        summary["num_failed_samples"] = int(failed)
    else:
        loader = DataLoader(
            subset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            collate_fn=OnePassWorker(packer, labeler, num_workers=int(args.num_workers)),
        )
        summary = _collect_teacher_max_used_stats(
            loader,
            configured_max_used=int(args.matching_max_used),
            show_progress=True,
        )

    def _log(msg: str) -> None:
        print(msg)

    _log_teacher_max_used_stats(summary, _log)

    out_path = Path("outputs") / "teacher_max_used_summary.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "args": vars(args),
            "summary": summary,
        },
        out_path,
    )
    print(f"[done] summary saved to {out_path}")


if __name__ == "__main__":
    main()
