from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.cli import model_factory
from src.experiments.evaluate_tsplib_onepass_models import discover_model_specs


class _DummyModule:
    def __init__(self) -> None:
        self.loaded = None
        self.eval_called = False

    def load_state_dict(self, state_dict):
        self.loaded = state_dict

    def eval(self):
        self.eval_called = True
        return self


def _onepass_bundle() -> model_factory.OnePassTrainingModels:
    return model_factory.OnePassTrainingModels(
        d_model=64,
        matching_max_used=4,
        num_iface_slots=16,
        decoder_variant="iface",
        leaf_encoder=_DummyModule(),
        merge_encoder=_DummyModule(),
        merge_decoder=_DummyModule(),
    )


def _twopass_bundle() -> model_factory.TwoPassTrainingModels:
    return model_factory.TwoPassTrainingModels(
        d_model=64,
        state_mode="matching",
        matching_max_used=4,
        num_states=17,
        leaf_encoder=_DummyModule(),
        merge_encoder=_DummyModule(),
        decoder=_DummyModule(),
    )


def test_load_onepass_eval_models_requires_merge_decoder(monkeypatch) -> None:
    ckpt = {
        "args": {
            "d_model": 64,
            "matching_max_used": 4,
            "parent_num_layers": 3,
            "cross_num_layers": 2,
        },
        "leaf_encoder": {"w": torch.tensor([1.0])},
        "merge_encoder": {"w": torch.tensor([2.0])},
    }

    monkeypatch.setattr(model_factory.torch, "load", lambda *args, **kwargs: ckpt)
    monkeypatch.setattr(model_factory, "build_onepass_training_models", lambda **kwargs: _onepass_bundle())

    with pytest.raises(KeyError, match="merge_decoder"):
        model_factory.load_onepass_eval_models(
            ckpt_path="dummy.pt",
            device=torch.device("cpu"),
            r=4,
        )


def test_load_onepass_eval_models_defaults_catalog_checkpoint_iface_order_to_clockwise(monkeypatch) -> None:
    ckpt = {
        "args": {
            "d_model": 64,
            "matching_max_used": 4,
            "parent_num_layers": 3,
            "cross_num_layers": 2,
        },
        "leaf_encoder": {"w": torch.tensor([1.0])},
        "merge_encoder": {"w": torch.tensor([2.0])},
        "merge_decoder": {"w": torch.tensor([3.0])},
    }

    monkeypatch.setattr(model_factory.torch, "load", lambda *args, **kwargs: ckpt)
    monkeypatch.setattr(model_factory, "build_onepass_training_models", lambda **kwargs: _onepass_bundle())

    bundle = model_factory.load_onepass_eval_models(
        ckpt_path="dummy.pt",
        device=torch.device("cpu"),
        r=4,
    )

    assert bundle.iface_order == "clockwise"


def test_load_onepass_eval_models_defaults_direct_structured_checkpoint_iface_order_to_legacy(monkeypatch) -> None:
    ckpt = {
        "args": {
            "d_model": 64,
            "matching_max_used": 4,
            "parent_num_layers": 3,
            "cross_num_layers": 2,
            "supervision_mode": "direct_structured",
        },
        "leaf_encoder": {"w": torch.tensor([1.0])},
        "merge_encoder": {"w": torch.tensor([2.0])},
        "merge_decoder": {"w": torch.tensor([3.0])},
    }

    monkeypatch.setattr(model_factory.torch, "load", lambda *args, **kwargs: ckpt)
    monkeypatch.setattr(model_factory, "build_onepass_training_models", lambda **kwargs: _onepass_bundle())

    bundle = model_factory.load_onepass_eval_models(
        ckpt_path="dummy.pt",
        device=torch.device("cpu"),
        r=4,
    )

    assert bundle.iface_order == "legacy"


def test_load_twopass_eval_models_rejects_one_stage_checkpoint(monkeypatch) -> None:
    ckpt = {
        "args": {
            "td_mode": "one_stage",
            "state_mode": "matching",
            "matching_max_used": 4,
        },
        "leaf_encoder": {"emb_type.weight": torch.zeros((4, 64), dtype=torch.float32)},
    }

    monkeypatch.setattr(model_factory.torch, "load", lambda *args, **kwargs: ckpt)
    monkeypatch.setattr(model_factory, "build_twopass_training_models", lambda **kwargs: _twopass_bundle())

    with pytest.raises(ValueError, match="one_stage"):
        model_factory.load_twopass_eval_models(
            ckpt_path="dummy.pt",
            device=torch.device("cpu"),
            r=4,
        )


def test_inspect_twopass_resume_checkpoint_rejects_one_stage_checkpoint(monkeypatch) -> None:
    ckpt = {
        "args": {
            "td_mode": "one_stage",
            "state_mode": "iface",
            "matching_max_used": 4,
        }
    }

    monkeypatch.setattr(model_factory.torch, "load", lambda *args, **kwargs: ckpt)

    with pytest.raises(ValueError, match="one_stage"):
        model_factory.inspect_twopass_resume_checkpoint(
            ckpt_path="resume.pt",
            args=SimpleNamespace(state_mode="iface", matching_max_used=4),
            emit=lambda _: None,
        )


def test_discover_model_specs_requires_checkpoint_metadata(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "ckpt_step_12.pt"
    torch.save({"args": {"r": 4}}, ckpt_path)

    with pytest.raises(KeyError, match="matching_max_used"):
        discover_model_specs(ckpt_specs=[str(ckpt_path)])
