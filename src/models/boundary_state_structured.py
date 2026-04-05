# -*- coding: utf-8 -*-
"""Structured boundary-state helpers for uncapped 1-pass DP.

This module replaces the legacy "global catalog index" view with node-local
structured states represented directly as `(used, mate)` on the fixed `Ti`
interface slots of one node.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor

from .bc_state_catalog import _enumerate_noncrossing_pairings


@dataclass(frozen=True)
class StructuredBoundaryState:
    """Hashable node-local boundary state."""

    used_mask: int
    mate_tuple: Tuple[int, ...]


def state_from_tensors(*, used: Tensor, mate: Tensor) -> StructuredBoundaryState:
    """Build a hashable structured state from full-slot tensors."""
    if used.dim() != 1 or mate.dim() != 1 or used.shape != mate.shape:
        raise ValueError("used and mate must be 1D tensors with identical shape.")
    num_slots = int(used.numel())
    used_mask = 0
    mate_values: List[int] = []
    used_b = used.bool()
    mate_l = mate.long()
    for slot in range(num_slots):
        if bool(used_b[slot].item()):
            used_mask |= (1 << slot)
            mate_values.append(int(mate_l[slot].item()))
        else:
            mate_values.append(-1)
    return StructuredBoundaryState(used_mask=used_mask, mate_tuple=tuple(mate_values))


def state_to_tensors(
    *,
    state: StructuredBoundaryState,
    num_slots: int,
    device: torch.device | None = None,
) -> Tuple[Tensor, Tensor]:
    """Convert a structured state back to full-slot tensors."""
    dev = torch.device("cpu") if device is None else device
    used = torch.zeros((int(num_slots),), dtype=torch.bool, device=dev)
    mate = torch.full((int(num_slots),), -1, dtype=torch.long, device=dev)
    if len(state.mate_tuple) != int(num_slots):
        raise ValueError("state.mate_tuple length must equal num_slots.")
    for slot in range(int(num_slots)):
        if (int(state.used_mask) >> slot) & 1:
            used[slot] = True
            mate[slot] = int(state.mate_tuple[slot])
    return used, mate


def stack_state_tensors(
    *,
    states: Sequence[StructuredBoundaryState],
    num_slots: int,
    device: torch.device | None = None,
) -> Tuple[Tensor, Tensor]:
    """Stack multiple structured states into `[S, Ti]` tensors."""
    dev = torch.device("cpu") if device is None else device
    if len(states) == 0:
        return (
            torch.zeros((0, int(num_slots)), dtype=torch.bool, device=dev),
            torch.full((0, int(num_slots)), -1, dtype=torch.long, device=dev),
        )

    used_rows = torch.zeros((len(states), int(num_slots)), dtype=torch.bool, device=dev)
    mate_rows = torch.full((len(states), int(num_slots)), -1, dtype=torch.long, device=dev)
    for idx, state in enumerate(states):
        used, mate = state_to_tensors(state=state, num_slots=num_slots, device=dev)
        used_rows[idx] = used
        mate_rows[idx] = mate
    return used_rows, mate_rows


def build_state_index(
    states: Iterable[StructuredBoundaryState],
) -> Dict[StructuredBoundaryState, int]:
    """Create a structured-state -> local-index lookup."""
    return {state: idx for idx, state in enumerate(states)}


@lru_cache(maxsize=256)
def _enumerate_states_for_active_slots(
    active_slots: Tuple[int, ...],
    num_slots: int,
) -> Tuple[StructuredBoundaryState, ...]:
    states: List[StructuredBoundaryState] = [
        StructuredBoundaryState(
            used_mask=0,
            mate_tuple=tuple(-1 for _ in range(int(num_slots))),
        )
    ]
    num_active = len(active_slots)
    for used_count in range(2, num_active + 1, 2):
        for subset in combinations(active_slots, used_count):
            for pairing in _enumerate_noncrossing_pairings(subset):
                used_mask = 0
                mate = [-1] * int(num_slots)
                for a, b in pairing:
                    used_mask |= (1 << int(a))
                    used_mask |= (1 << int(b))
                    mate[int(a)] = int(b)
                    mate[int(b)] = int(a)
                states.append(
                    StructuredBoundaryState(
                        used_mask=used_mask,
                        mate_tuple=tuple(mate),
                    )
                )
    return tuple(states)


def enumerate_structured_states_for_iface_mask(
    *,
    iface_mask: Tensor,
) -> List[StructuredBoundaryState]:
    """Enumerate all non-crossing structured states allowed by a node iface mask."""
    if iface_mask.dim() != 1:
        raise ValueError("iface_mask must be 1D [Ti].")
    active_slots = tuple(int(i) for i in torch.nonzero(iface_mask.bool(), as_tuple=False).flatten().tolist())
    return list(_enumerate_states_for_active_slots(active_slots, int(iface_mask.numel())))


__all__ = [
    "StructuredBoundaryState",
    "build_state_index",
    "enumerate_structured_states_for_iface_mask",
    "stack_state_tensors",
    "state_from_tensors",
    "state_to_tensors",
]
