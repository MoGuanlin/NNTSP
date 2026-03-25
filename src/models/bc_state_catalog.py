# src/models/bc_state_catalog.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Iterable, Sequence

import torch
from torch import Tensor


@dataclass
class BoundaryStateCatalog:
    """
    Boundary-condition state catalog over a fixed number of interface slots.

    A state is represented by:
      - used_iface[s, i] in {0,1}
      - mate[s, i] = partner index within the same state, or -1 if unused

    Notes:
      - This prototype enumerates non-crossing perfect matchings on subsets of
        the interface order, capped by `max_used`.
      - The empty state is always state 0.
    """

    used_iface: Tensor          # [S, Ti] bool
    mate: Tensor                # [S, Ti] long
    num_used: Tensor            # [S] long
    empty_index: int
    max_used: int


def _enumerate_noncrossing_pairings(indices: Sequence[int]) -> list[tuple[tuple[int, int], ...]]:
    if len(indices) == 0:
        return [tuple()]
    if len(indices) % 2 != 0:
        return []

    first = int(indices[0])
    out: list[tuple[tuple[int, int], ...]] = []
    for j in range(1, len(indices), 2):
        partner = int(indices[j])
        left = indices[1:j]
        right = indices[j + 1 :]
        for left_pairs in _enumerate_noncrossing_pairings(left):
            for right_pairs in _enumerate_noncrossing_pairings(right):
                out.append(((first, partner),) + left_pairs + right_pairs)
    return out


@lru_cache(maxsize=32)
def _build_catalog_cpu(num_slots: int, max_used: int) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    if num_slots <= 0:
        raise ValueError("num_slots must be positive.")
    if max_used <= 0:
        raise ValueError("max_used must be positive.")

    capped = min(int(max_used), int(num_slots))
    capped = capped if capped % 2 == 0 else capped - 1

    used_states: list[tuple[int, ...]] = [tuple(0 for _ in range(num_slots))]
    mate_states: list[tuple[int, ...]] = [tuple(-1 for _ in range(num_slots))]

    for used_count in range(2, capped + 1, 2):
        for subset in combinations(range(num_slots), used_count):
            for pairing in _enumerate_noncrossing_pairings(subset):
                used = [0] * num_slots
                mate = [-1] * num_slots
                for a, b in pairing:
                    used[a] = 1
                    used[b] = 1
                    mate[a] = b
                    mate[b] = a
                used_states.append(tuple(used))
                mate_states.append(tuple(mate))

    return tuple(used_states), tuple(mate_states)


def build_boundary_state_catalog(
    *,
    num_slots: int,
    max_used: int = 4,
    device: torch.device | None = None,
) -> BoundaryStateCatalog:
    used_states, mate_states = _build_catalog_cpu(int(num_slots), int(max_used))
    dev = torch.device("cpu") if device is None else device

    used_iface = torch.tensor(used_states, dtype=torch.bool, device=dev)
    mate = torch.tensor(mate_states, dtype=torch.long, device=dev)
    num_used = used_iface.sum(dim=1).to(dtype=torch.long)

    return BoundaryStateCatalog(
        used_iface=used_iface,
        mate=mate,
        num_used=num_used,
        empty_index=0,
        max_used=int(max_used),
    )


def infer_boundary_state_count(*, num_slots: int, max_used: int = 4) -> int:
    used_states, _ = _build_catalog_cpu(int(num_slots), int(max_used))
    return len(used_states)


def state_mask_from_iface_mask(
    *,
    iface_mask: Tensor,          # [M, Ti] bool
    state_used_iface: Tensor,    # [S, Ti] bool
) -> Tensor:
    if iface_mask.dim() != 2:
        raise ValueError(f"iface_mask must be [M,Ti], got {tuple(iface_mask.shape)}")
    if state_used_iface.dim() != 2:
        raise ValueError(f"state_used_iface must be [S,Ti], got {tuple(state_used_iface.shape)}")
    if iface_mask.shape[1] != state_used_iface.shape[1]:
        raise ValueError("iface_mask and state_used_iface must share Ti.")

    valid = iface_mask.bool().unsqueeze(1)           # [M,1,Ti]
    used = state_used_iface.bool().unsqueeze(0)      # [1,S,Ti]
    return ~(used & (~valid)).any(dim=-1)            # [M,S]


def project_iface_usage_to_state_index(
    *,
    iface_target: Tensor,        # [Ti] float/bool
    iface_mask: Tensor,          # [Ti] bool
    state_mask: Tensor,          # [S] bool
    state_used_iface: Tensor,    # [S,Ti] bool
) -> int:
    if iface_target.dim() != 1 or iface_mask.dim() != 1:
        raise ValueError("iface_target and iface_mask must be 1D [Ti].")
    if state_mask.dim() != 1:
        raise ValueError("state_mask must be 1D [S].")
    if state_used_iface.dim() != 2:
        raise ValueError("state_used_iface must be [S,Ti].")

    allowed = state_mask.bool()
    if not allowed.any().item():
        return -1

    used_target = ((iface_target > 0.5) & iface_mask.bool()).to(dtype=torch.bool)
    used_table = state_used_iface.bool()

    mismatch = torch.logical_xor(used_table, used_target.unsqueeze(0)).sum(dim=1)
    # States using invalid interfaces are never allowed.
    mismatch = torch.where(allowed, mismatch, torch.full_like(mismatch, int(1e9)))

    best = int(torch.argmin(mismatch).item())
    return best


def project_matching_to_state_index(
    *,
    iface_used: Tensor,          # [Ti] bool
    iface_mate: Tensor,          # [Ti] long, -1 if unused
    iface_mask: Tensor,          # [Ti] bool
    state_mask: Tensor,          # [S] bool
    state_used_iface: Tensor,    # [S,Ti] bool
    state_mate: Tensor,          # [S,Ti] long
) -> int:
    if iface_used.dim() != 1 or iface_mate.dim() != 1 or iface_mask.dim() != 1:
        raise ValueError("iface_used/iface_mate/iface_mask must be 1D [Ti].")
    if state_mask.dim() != 1:
        raise ValueError("state_mask must be 1D [S].")
    if state_used_iface.dim() != 2 or state_mate.dim() != 2:
        raise ValueError("state_used_iface/state_mate must be [S,Ti].")
    if state_used_iface.shape != state_mate.shape:
        raise ValueError("state_used_iface and state_mate must share shape [S,Ti].")

    allowed = state_mask.bool()
    if not allowed.any().item():
        return -1

    used = (iface_used.bool() & iface_mask.bool())
    mate = torch.where(used, iface_mate.long(), torch.full_like(iface_mate.long(), -1))
    exact = allowed & (state_used_iface.bool() == used.unsqueeze(0)).all(dim=1) & (state_mate.long() == mate.unsqueeze(0)).all(dim=1)
    if exact.any().item():
        return int(torch.nonzero(exact, as_tuple=False).view(-1)[0].item())

    return project_iface_usage_to_state_index(
        iface_target=used.to(dtype=torch.float32),
        iface_mask=iface_mask,
        state_mask=state_mask,
        state_used_iface=state_used_iface,
    )


def state_logits_to_expected_iface_usage(
    *,
    state_logit: Tensor,         # [B,S] or [K,S]
    state_mask: Tensor,          # same leading shape, bool
    state_used_iface: Tensor,    # [S,Ti] bool/float
) -> Tensor:
    if state_logit.dim() != 2:
        raise ValueError("state_logit must be 2D [B,S].")
    if state_mask.shape != state_logit.shape:
        raise ValueError("state_mask must match state_logit shape.")
    if state_used_iface.dim() != 2 or state_used_iface.shape[0] != state_logit.shape[1]:
        raise ValueError("state_used_iface must be [S,Ti] and match state_logit.")

    neg_inf = torch.tensor(-1.0e9, device=state_logit.device, dtype=state_logit.dtype)
    masked = torch.where(state_mask.bool(), state_logit, neg_inf)
    has_valid = state_mask.any(dim=1, keepdim=True)
    masked = torch.where(has_valid, masked, torch.zeros_like(masked))
    probs = torch.softmax(masked, dim=-1)
    probs = probs * has_valid.to(dtype=probs.dtype)
    used = state_used_iface.to(device=state_logit.device, dtype=state_logit.dtype)
    return probs @ used


__all__ = [
    "BoundaryStateCatalog",
    "build_boundary_state_catalog",
    "infer_boundary_state_count",
    "project_matching_to_state_index",
    "project_iface_usage_to_state_index",
    "state_logits_to_expected_iface_usage",
    "state_mask_from_iface_mask",
]
