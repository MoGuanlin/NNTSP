# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.boundary_state_structured import (
    enumerate_structured_states_for_iface_mask,
    stack_state_tensors,
    state_from_tensors,
    state_to_tensors,
)


def test_enumerate_structured_states_for_four_active_slots() -> None:
    iface_mask = torch.tensor([True, True, True, True], dtype=torch.bool)
    states = enumerate_structured_states_for_iface_mask(iface_mask=iface_mask)
    assert len(states) == 9


def test_structured_state_roundtrip() -> None:
    used = torch.tensor([True, False, True, False], dtype=torch.bool)
    mate = torch.tensor([2, -1, 0, -1], dtype=torch.long)
    state = state_from_tensors(used=used, mate=mate)
    used2, mate2 = state_to_tensors(state=state, num_slots=4)
    assert torch.equal(used, used2)
    assert torch.equal(mate, mate2)


def test_stack_state_tensors_shape() -> None:
    iface_mask = torch.tensor([True, True, True, True], dtype=torch.bool)
    states = enumerate_structured_states_for_iface_mask(iface_mask=iface_mask)
    used, mate = stack_state_tensors(states=states, num_slots=4)
    assert used.shape == (9, 4)
    assert mate.shape == (9, 4)
