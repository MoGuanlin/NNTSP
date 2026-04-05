# tests/test_decode_tour.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import inspect
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

import torch


def _add_repo_root_to_syspath() -> None:
    """
    When running `python tests/xxx.py`, sys.path[0] is `tests`,
    so repo root is NOT importable. Add it explicitly.
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[1]  # .../NNTSP
    sys.path.insert(0, str(repo_root))


def _instantiate_best_effort(cls, preferred_kwargs: Dict[str, Any]):
    sig = inspect.signature(cls.__init__)
    usable = {k: v for k, v in preferred_kwargs.items() if k in sig.parameters}
    return cls(**usable)


def _move_data_tensors_to_device(data: Any, device: torch.device) -> Any:
    skip = {"num_faces"}
    for k in dir(data):
        if k.startswith("_") or k in skip:
            continue
        v = getattr(data, k, None)
        if isinstance(v, torch.Tensor):
            try:
                setattr(data, k, v.to(device))
            except Exception:
                pass
    return data


def main() -> None:
    _add_repo_root_to_syspath()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_pt", type=str, required=True)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ckpt", type=str, default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[env] device={device}")

    warnings.filterwarnings("ignore", category=FutureWarning)
    obj = torch.load(args.data_pt, map_location="cpu")
    if not isinstance(obj, list) or len(obj) == 0:
        raise RuntimeError("--data_pt must be a list[Data].")
    data = obj[int(args.idx)]
    data = _move_data_tensors_to_device(data, device)

    from src.models.node_token_packer import NodeTokenPacker
    from src.models.leaf_encoder import LeafEncoder
    from src.models.merge_encoder import MergeEncoder
    from src.models.bottom_up_runner import BottomUpTreeRunner
    from src.models.top_down_decoder import TopDownDecoder
    from src.models.top_down_runner import TopDownTreeRunner
    from src.models.edge_aggregation import aggregate_logits_to_edges
    from src.models.edge_decode import decode_tour_from_edge_logits
    from src.models.labeler import PseudoLabeler

    packer = _instantiate_best_effort(NodeTokenPacker, {"r": int(args.r)})

    leaf_encoder = _instantiate_best_effort(LeafEncoder, {"d_model": 128}).to(device)
    merge_encoder = _instantiate_best_effort(MergeEncoder, {"d_model": 128}).to(device)
    decoder = _instantiate_best_effort(TopDownDecoder, {"r": int(args.r), "d_model": 128}).to(device)

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        leaf_encoder.load_state_dict(ckpt["leaf_encoder"], strict=True)
        merge_encoder.load_state_dict(ckpt["merge_encoder"], strict=True)
        decoder.load_state_dict(ckpt["decoder"], strict=True)
        print(f"[ckpt] loaded: {args.ckpt}")

    bu = BottomUpTreeRunner()
    td = TopDownTreeRunner()

    batch = packer.pack_batch([data])  # B=1, so eid_offset=0
    out_bu = bu.run_batch(batch=batch, leaf_encoder=leaf_encoder, merge_encoder=merge_encoder)
    out_td = td.run_batch(batch=batch, z=out_bu.z, decoder=decoder)

    edge_scores = aggregate_logits_to_edges(tokens=batch.tokens, cross_logit=out_td.cross_logit)
    edge_logit_global = edge_scores.edge_logit
    edge_mask_global = edge_scores.edge_mask.bool()

    E = int(data.spanner_edge_index.shape[1])
    if int(edge_logit_global.numel()) != E:
        raise RuntimeError(f"Expected E=={E} for B=1, got {int(edge_logit_global.numel())}")

    # For uncovered edges, push score down so decoder prefers covered edges.
    edge_logit_local = edge_logit_global.clone()
    edge_logit_local[~edge_mask_global] = -1e9

    res = decode_tour_from_edge_logits(
        pos=data.pos,
        spanner_edge_index=data.spanner_edge_index,
        edge_logit=edge_logit_local,
        allow_off_spanner_patch=True,
    )

    print(
        f"[decode] feasible={res.feasible} len={res.length:.2f} "
        f"off_spanner={res.num_off_spanner_edges} "
        f"components0={res.num_components_initial}"
    )

    # Teacher length for reference
    labeler = PseudoLabeler(prefer_cpu=True)
    t = batch.tokens

    class _Slice:
        pass

    ts = _Slice()
    ts.cross_eid = t.cross_eid
    ts.cross_mask = t.cross_mask
    ts.iface_eid = t.iface_eid
    ts.iface_mask = t.iface_mask

    lab = labeler.label_one(data=data, tokens_slice=ts, device=device, eid_offset=0)
    teacher_len = float(lab.stats["tour_len"].item())
    gap = (res.length / teacher_len - 1.0) if teacher_len > 1e-9 else 0.0
    print(f"[teacher] len={teacher_len:.2f}  pred/teacher-1={gap:.4f}")


if __name__ == "__main__":
    main()
