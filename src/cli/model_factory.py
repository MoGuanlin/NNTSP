# src/cli/model_factory.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch


@dataclass
class TwoPassEvalModels:
    ckpt: Dict[str, Any]
    ckpt_args: Dict[str, Any]
    d_model: int
    state_mode: str
    matching_max_used: int
    num_states: Optional[int]
    leaf_encoder: Any
    merge_encoder: Any
    decoder: Any


@dataclass
class OnePassEvalModels:
    ckpt: Dict[str, Any]
    ckpt_args: Dict[str, Any]
    d_model: int
    matching_max_used: int
    num_iface_slots: int
    leaf_encoder: Any
    merge_encoder: Any
    merge_decoder: Any
    merge_decoder_loaded: bool


@dataclass
class TwoPassTrainingModels:
    d_model: int
    state_mode: str
    matching_max_used: int
    num_states: Optional[int]
    leaf_encoder: Any
    merge_encoder: Any
    decoder: Any


@dataclass
class OnePassTrainingModels:
    d_model: int
    matching_max_used: int
    num_iface_slots: int
    leaf_encoder: Any
    merge_encoder: Any
    merge_decoder: Any


def _infer_twopass_num_states(*, r: int, state_mode: str, matching_max_used: int) -> Optional[int]:
    if str(state_mode) != "matching":
        return None
    from src.models.bc_state_catalog import infer_boundary_state_count

    return infer_boundary_state_count(num_slots=4 * int(r), max_used=int(matching_max_used))


def _detect_twopass_d_model(ckpt: Dict[str, Any]) -> int:
    if "leaf_encoder" in ckpt and "emb_type.weight" in ckpt["leaf_encoder"]:
        return int(ckpt["leaf_encoder"]["emb_type.weight"].shape[1])
    return 128


def inspect_twopass_resume_checkpoint(
    *,
    ckpt_path: str,
    args: Any,
    emit: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = dict(ckpt.get("args", {}))
    log = emit if emit is not None else print

    if "td_mode" in ckpt_args and str(args.td_mode) != str(ckpt_args["td_mode"]):
        log(f"[ckpt] overriding td_mode: {args.td_mode} -> {ckpt_args['td_mode']}")
        args.td_mode = ckpt_args["td_mode"]
    if "state_mode" in ckpt_args and str(args.state_mode) != str(ckpt_args["state_mode"]):
        log(f"[ckpt] overriding state_mode: {args.state_mode} -> {ckpt_args['state_mode']}")
        args.state_mode = ckpt_args["state_mode"]
    if "matching_max_used" in ckpt_args and int(args.matching_max_used) != int(ckpt_args["matching_max_used"]):
        log(f"[ckpt] overriding matching_max_used: {args.matching_max_used} -> {ckpt_args['matching_max_used']}")
        args.matching_max_used = int(ckpt_args["matching_max_used"])
    return ckpt


def build_twopass_training_models(
    *,
    device: torch.device,
    r: int,
    td_mode: str,
    state_mode: str,
    matching_max_used: int,
    d_model: int = 128,
) -> TwoPassTrainingModels:
    from src.models.leaf_encoder import LeafEncoder
    from src.models.merge_encoder import MergeEncoder
    from src.models.top_down_decoder import TopDownDecoder

    num_states = _infer_twopass_num_states(
        r=int(r),
        state_mode=str(state_mode),
        matching_max_used=int(matching_max_used),
    )
    leaf_encoder = LeafEncoder(d_model=int(d_model)).to(device)
    merge_encoder = MergeEncoder(d_model=int(d_model)).to(device)
    decoder = TopDownDecoder(
        d_model=int(d_model),
        mode=str(td_mode),
        state_mode=str(state_mode),
        num_states=num_states,
    ).to(device)
    return TwoPassTrainingModels(
        d_model=int(d_model),
        state_mode=str(state_mode),
        matching_max_used=int(matching_max_used),
        num_states=num_states,
        leaf_encoder=leaf_encoder,
        merge_encoder=merge_encoder,
        decoder=decoder,
    )


def restore_twopass_checkpoint_state(
    *,
    ckpt: Dict[str, Any],
    leaf_encoder: Any,
    merge_encoder: Any,
    decoder: Any,
    opt: Any | None = None,
    load_optimizer: bool = False,
) -> None:
    leaf_encoder.load_state_dict(ckpt["leaf_encoder"])
    merge_encoder.load_state_dict(ckpt["merge_encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    if load_optimizer and opt is not None and "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])


def build_onepass_training_models(
    *,
    device: torch.device,
    d_model: int,
    r: int,
    matching_max_used: int,
    parent_num_layers: int,
    cross_num_layers: int,
) -> OnePassTrainingModels:
    from src.models.leaf_encoder import LeafEncoder
    from src.models.merge_decoder import MergeDecoder
    from src.models.merge_encoder import MergeEncoder

    num_iface_slots = 4 * int(r)
    leaf_encoder = LeafEncoder(d_model=int(d_model)).to(device)
    merge_encoder = MergeEncoder(d_model=int(d_model)).to(device)
    merge_decoder = MergeDecoder(
        d_model=int(d_model),
        n_heads=max(4, int(d_model) // 32),
        num_iface_slots=num_iface_slots,
        parent_num_layers=int(parent_num_layers),
        cross_num_layers=int(cross_num_layers),
        max_depth=64,
    ).to(device)
    return OnePassTrainingModels(
        d_model=int(d_model),
        matching_max_used=int(matching_max_used),
        num_iface_slots=num_iface_slots,
        leaf_encoder=leaf_encoder,
        merge_encoder=merge_encoder,
        merge_decoder=merge_decoder,
    )


def restore_onepass_checkpoint_state(
    *,
    ckpt: Dict[str, Any],
    leaf_encoder: Any,
    merge_encoder: Any,
    merge_decoder: Any,
    opt: Any | None = None,
    load_optimizer: bool = False,
) -> bool:
    leaf_encoder.load_state_dict(ckpt["leaf_encoder"])
    merge_encoder.load_state_dict(ckpt["merge_encoder"])
    merge_decoder_loaded = "merge_decoder" in ckpt
    if merge_decoder_loaded:
        merge_decoder.load_state_dict(ckpt["merge_decoder"])
    if load_optimizer and opt is not None and "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])
    return merge_decoder_loaded


def load_twopass_eval_models(*, ckpt_path: str, device: torch.device, r: int) -> TwoPassEvalModels:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    d_model = _detect_twopass_d_model(ckpt)
    ckpt_args = ckpt.get("args", {})
    state_mode = str(ckpt_args.get("state_mode", "iface"))
    matching_max_used = int(ckpt_args.get("matching_max_used", 4))
    models = build_twopass_training_models(
        device=device,
        r=int(r),
        td_mode=str(ckpt_args.get("td_mode", "two_stage")),
        state_mode=state_mode,
        matching_max_used=matching_max_used,
        d_model=d_model,
    )
    restore_twopass_checkpoint_state(
        ckpt=ckpt,
        leaf_encoder=models.leaf_encoder,
        merge_encoder=models.merge_encoder,
        decoder=models.decoder,
    )
    models.leaf_encoder.eval()
    models.merge_encoder.eval()
    models.decoder.eval()
    return TwoPassEvalModels(
        ckpt=ckpt,
        ckpt_args=ckpt_args,
        d_model=d_model,
        state_mode=state_mode,
        matching_max_used=matching_max_used,
        num_states=models.num_states,
        leaf_encoder=models.leaf_encoder,
        merge_encoder=models.merge_encoder,
        decoder=models.decoder,
    )


def load_onepass_eval_models(
    *,
    ckpt_path: str,
    device: torch.device,
    r: int,
    default_matching_max_used: int,
) -> OnePassEvalModels:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    d_model = int(ckpt_args.get("d_model", 128))
    matching_max_used = int(ckpt_args.get("matching_max_used", default_matching_max_used))
    models = build_onepass_training_models(
        device=device,
        d_model=d_model,
        r=int(r),
        matching_max_used=matching_max_used,
        parent_num_layers=int(ckpt_args.get("parent_num_layers", 3)),
        cross_num_layers=int(ckpt_args.get("cross_num_layers", 2)),
    )
    merge_decoder_loaded = restore_onepass_checkpoint_state(
        ckpt=ckpt,
        leaf_encoder=models.leaf_encoder,
        merge_encoder=models.merge_encoder,
        merge_decoder=models.merge_decoder,
    )
    models.leaf_encoder.eval()
    models.merge_encoder.eval()
    models.merge_decoder.eval()
    return OnePassEvalModels(
        ckpt=ckpt,
        ckpt_args=ckpt_args,
        d_model=d_model,
        matching_max_used=matching_max_used,
        num_iface_slots=models.num_iface_slots,
        leaf_encoder=models.leaf_encoder,
        merge_encoder=models.merge_encoder,
        merge_decoder=models.merge_decoder,
        merge_decoder_loaded=merge_decoder_loaded,
    )


__all__ = [
    "OnePassEvalModels",
    "OnePassTrainingModels",
    "TwoPassEvalModels",
    "TwoPassTrainingModels",
    "build_onepass_training_models",
    "build_twopass_training_models",
    "inspect_twopass_resume_checkpoint",
    "load_onepass_eval_models",
    "load_twopass_eval_models",
    "restore_onepass_checkpoint_state",
    "restore_twopass_checkpoint_state",
]
