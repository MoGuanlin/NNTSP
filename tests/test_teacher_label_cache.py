# tests/test_teacher_label_cache.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli.train import dataset_has_precomputed_labels


class _FakeLabeler:
    def __init__(self, signature: str):
        self._signature = signature

    def label_signature(self) -> str:
        return self._signature

    def data_has_compatible_teacher(self, data) -> bool:
        return getattr(data, "teacher_label_signature", None) == self._signature


def test_dataset_has_precomputed_labels_requires_matching_signature_for_fast_dataset():
    dataset = SimpleNamespace(
        c={
            "keys": ["target_edges", "tour_len", "teacher_order"],
            "target_edges": torch.tensor([0, 1, 2], dtype=torch.long),
            "target_edges_ptr": torch.tensor([0, 3], dtype=torch.long),
            "tour_len": torch.tensor([3.0], dtype=torch.float32),
            "teacher_order": torch.tensor([0, 1, 2], dtype=torch.long),
            "teacher_order_ptr": torch.tensor([0, 3], dtype=torch.long),
            "teacher_label_signature": "new_sig",
        }
    )

    assert dataset_has_precomputed_labels(dataset, labeler=_FakeLabeler("new_sig"))
    assert not dataset_has_precomputed_labels(dataset, labeler=_FakeLabeler("old_sig"))


def test_dataset_has_precomputed_labels_requires_teacher_order_for_list_dataset():
    item = SimpleNamespace(
        target_edges=torch.tensor([0, 1], dtype=torch.long),
        tour_len=torch.tensor(2.0),
        teacher_label_signature="sig",
    )
    assert not dataset_has_precomputed_labels([item], labeler=_FakeLabeler("sig"))

    item.teacher_order = torch.tensor([0, 1], dtype=torch.long)
    assert dataset_has_precomputed_labels([item], labeler=_FakeLabeler("sig"))
