# src/cli/eval_task_factory.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch
from torch import Tensor


MASKED_EDGE_LOGIT = -1.0e9


def to_cpu_tensor(x: Tensor) -> Tensor:
    return x.detach().cpu()


def mask_edge_logits_with_coverage(
    edge_logit: Tensor,
    edge_mask: Tensor,
    *,
    masked_value: float = MASKED_EDGE_LOGIT,
) -> Tensor:
    masked = edge_logit.clone()
    masked[~edge_mask.bool()] = float(masked_value)
    return masked


def prepare_decode_inputs(
    *,
    pos: Tensor,
    spanner_edge_index: Tensor,
    edge_logit: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    return to_cpu_tensor(pos), to_cpu_tensor(spanner_edge_index), to_cpu_tensor(edge_logit)


def build_decode_task(
    *,
    pos: Tensor,
    spanner_edge_index: Tensor,
    edge_logit: Tensor,
    teacher_len: float,
    allow_off_spanner_patch: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, bool, float]:
    pos_cpu, edge_index_cpu, edge_logit_cpu = prepare_decode_inputs(
        pos=pos,
        spanner_edge_index=spanner_edge_index,
        edge_logit=edge_logit,
    )
    return (
        pos_cpu,
        edge_index_cpu,
        edge_logit_cpu,
        bool(allow_off_spanner_patch),
        float(teacher_len),
    )


def build_lkh_task(
    *,
    pos: Tensor,
    mode: str,
    teacher_len: float | None,
    edge_index: Tensor | None = None,
    edge_logit: Tensor | None = None,
    initial_tour: Sequence[int] | None = None,
    uniform_alpha: int | None = None,
) -> Dict[str, Any]:
    task: Dict[str, Any] = {
        "pos": to_cpu_tensor(pos),
        "mode": str(mode),
        "teacher_len": float(teacher_len) if teacher_len is not None else 0.0,
    }
    if edge_index is not None:
        task["edge_index"] = to_cpu_tensor(edge_index)
    if edge_logit is not None:
        task["edge_logit"] = to_cpu_tensor(edge_logit)
    if initial_tour is not None:
        task["initial_tour"] = [int(x) for x in initial_tour]
    if uniform_alpha is not None:
        task["uniform_alpha"] = int(uniform_alpha)
    return task


__all__ = [
    "MASKED_EDGE_LOGIT",
    "build_decode_task",
    "build_lkh_task",
    "mask_edge_logits_with_coverage",
    "prepare_decode_inputs",
    "to_cpu_tensor",
]
