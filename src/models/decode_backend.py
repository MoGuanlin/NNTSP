# src/models/decode_backend.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor

from src.models.edge_decode import TourDecodeResult, decode_tour_from_edge_logits
from src.models.exact_decode import decode_tour_exact_from_edge_logits


def decode_tour(
    *,
    pos: Tensor,
    spanner_edge_index: Tensor,
    edge_logit: Tensor,
    backend: str = "greedy",
    prefer_spanner_only: bool = True,
    allow_off_spanner_patch: bool = True,
    exact_time_limit: float = 30.0,
    exact_length_weight: float = 0.0,
) -> TourDecodeResult:
    backend = str(backend).lower()
    if backend == "greedy":
        return decode_tour_from_edge_logits(
            pos=pos,
            spanner_edge_index=spanner_edge_index,
            edge_logit=edge_logit,
            prefer_spanner_only=prefer_spanner_only,
            allow_off_spanner_patch=allow_off_spanner_patch,
        )
    if backend == "exact":
        return decode_tour_exact_from_edge_logits(
            pos=pos,
            spanner_edge_index=spanner_edge_index,
            edge_logit=edge_logit,
            time_limit=float(exact_time_limit) if exact_time_limit is not None else None,
            length_weight=float(exact_length_weight),
        )
    raise ValueError(f"Unknown decode backend: {backend}")


class DecodingDataset(torch.utils.data.Dataset):
    """Dataset wrapper for parallel CPU decoding via DataLoader."""

    def __init__(
        self,
        tasks: List[Tuple[Tensor, Tensor, Tensor, bool, bool, float]],
        *,
        decode_backend: str = "greedy",
        exact_time_limit: float = 30.0,
        exact_length_weight: float = 0.0,
    ) -> None:
        self.tasks = tasks
        self.decode_backend = str(decode_backend)
        self.exact_time_limit = float(exact_time_limit)
        self.exact_length_weight = float(exact_length_weight)

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> Tuple[TourDecodeResult, float]:
        pos, spanner_edge_index, edge_logit, prefer_spanner, allow_patch, tlen = self.tasks[idx]
        res = decode_tour(
            pos=pos,
            spanner_edge_index=spanner_edge_index,
            edge_logit=edge_logit,
            backend=self.decode_backend,
            prefer_spanner_only=prefer_spanner,
            allow_off_spanner_patch=allow_patch,
            exact_time_limit=self.exact_time_limit,
            exact_length_weight=self.exact_length_weight,
        )
        return res, tlen


__all__ = ["decode_tour", "DecodingDataset", "TourDecodeResult"]
