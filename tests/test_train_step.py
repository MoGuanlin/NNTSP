# tests/test_train_step.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import inspect
import sys
from dataclasses import is_dataclass, replace
from pathlib import Path
import warnings
from typing import Any, Dict, List

import torch


def _add_repo_root_to_syspath() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    sys.path.insert(0, str(repo_root))


def _torch_load(path: str, map_location: str = "cpu") -> Any:
    warnings.filterwarnings("ignore", category=FutureWarning)
    return torch.load(path, map_location=map_location)


def _instantiate_best_effort(cls, preferred_kwargs: Dict[str, Any]):
    sig = inspect.signature(cls.__init__)
    usable = {k: v for k, v in preferred_kwargs.items() if k in sig.parameters}
    return cls(**usable)


def _move_data_tensors_to_device(data: Any, device: torch.device) -> Any:
    """
    Best-effort move all Tensor fields on a PyG Data object to `device`.
    Avoid touching deprecated virtual attributes such as `num_faces`.
    """
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


def _dataclass_to_device(obj: Any, device: torch.device) -> Any:
    if not is_dataclass(obj):
        return obj
    updates = {}
    for k in obj.__dataclass_fields__.keys():
        v = getattr(obj, k)
        if isinstance(v, torch.Tensor):
            updates[k] = v.to(device)
    return replace(obj, **updates) if updates else obj


def _packed_batch_to_device(batch: Any, device: torch.device) -> Any:
    if not is_dataclass(batch):
        return batch
    updates = {}
    if hasattr(batch, "tokens"):
        updates["tokens"] = _dataclass_to_device(batch.tokens, device)
    if hasattr(batch, "leaves"):
        updates["leaves"] = _dataclass_to_device(batch.leaves, device)
    for name in ["node_ptr", "leaf_ptr", "edge_ptr", "graph_id_for_node"]:
        if hasattr(batch, name) and isinstance(getattr(batch, name), torch.Tensor):
            updates[name] = getattr(batch, name).to(device)
    return replace(batch, **updates) if updates else batch


def main() -> None:
    _add_repo_root_to_syspath()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_pt", type=str, required=True)
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    print(f"[env] device={device}")

    from src.models.node_token_packer import NodeTokenPacker
    from src.models.leaf_encoder import LeafEncoder
    from src.models.merge_encoder import MergeEncoder
    from src.models.bottom_up_runner import BottomUpTreeRunner

    from src.models.top_down_decoder import TopDownDecoder
    from src.models.top_down_runner import TopDownTreeRunner

    from src.models.labeler import PseudoLabeler
    from src.models.losses import dp_token_losses

    # batch edge aggregation
    from src.models.edge_aggregation import aggregate_logits_to_edges

    obj = _torch_load(args.data_pt, map_location="cpu")
    if not isinstance(obj, list) or len(obj) == 0:
        raise RuntimeError("--data_pt must be a list[Data] for this test.")

    bs = min(int(args.batch_size), len(obj))
    datas: List[Any] = []
    for i in range(bs):
        d = obj[i]
        d = _move_data_tensors_to_device(d, device)
        datas.append(d)

    packer = _instantiate_best_effort(NodeTokenPacker, {"r": int(args.r)})

    leaf_encoder = _instantiate_best_effort(LeafEncoder, {"d_model": 128}).to(device)
    merge_encoder = _instantiate_best_effort(MergeEncoder, {"d_model": 128}).to(device)
    bu = _instantiate_best_effort(BottomUpTreeRunner, {})

    decoder = _instantiate_best_effort(TopDownDecoder, {"r": int(args.r), "d_model": 128}).to(device)
    td = _instantiate_best_effort(TopDownTreeRunner, {})

    labeler = PseudoLabeler(two_opt_passes=30, use_lkh=False, prefer_cpu=True)

    batch = packer.pack_batch(datas)
    batch = _packed_batch_to_device(batch, device)

    # quick sanity: eid ranges in each graph should be disjoint now
    for b in range(bs):
        lo = int(batch.node_ptr[b].item())
        hi = int(batch.node_ptr[b + 1].item())
        e0 = int(batch.edge_ptr[b].item())
        e1 = int(batch.edge_ptr[b + 1].item())
        ce = batch.tokens.cross_eid[lo:hi]
        cm = batch.tokens.cross_mask[lo:hi].bool() & (ce >= 0)
        if cm.any().item():
            mn = int(ce[cm].min().item())
            mx = int(ce[cm].max().item())
            if not (e0 <= mn <= mx < e1):
                raise RuntimeError(f"[eid] graph{b} cross_eid range [{mn},{mx}] not within [{e0},{e1}).")
    print("[ok] batch eid-offset sanity PASSED.")

    # bottom-up
    out_bu = bu.run_batch(batch=batch, leaf_encoder=leaf_encoder, merge_encoder=merge_encoder)
    if not out_bu.computed.all().item():
        raise RuntimeError("bottom-up did not compute all nodes in batch")
    z = out_bu.z

    # top-down
    out_td = td.run_batch(batch=batch, z=z, decoder=decoder)

    # labels (token-level), graph-by-graph using eid_offset=edge_ptr[b]
    total_M, Tc = out_td.cross_logit.shape
    Ti = out_td.iface_logit.shape[1]
    y_cross = torch.zeros((total_M, Tc), device=device, dtype=torch.float32)
    y_iface = torch.zeros((total_M, Ti), device=device, dtype=torch.float32)

    for b in range(bs):
        lo = int(batch.node_ptr[b].item())
        hi = int(batch.node_ptr[b + 1].item())
        eid_off = int(batch.edge_ptr[b].item())

        t = batch.tokens

        class _Slice:
            pass

        ts = _Slice()
        for name in ["cross_eid", "cross_mask", "iface_eid", "iface_mask"]:
            setattr(ts, name, getattr(t, name)[lo:hi])

        lab = labeler.label_one(data=datas[b], tokens_slice=ts, device=device, eid_offset=eid_off)
        y_cross[lo:hi] = lab.y_cross
        y_iface[lo:hi] = lab.y_iface

        if b == 0:
            st = lab.stats
            print(
                f"[label] b0 tour_len={float(st['tour_len'].item()):.2f}, "
                f"alive_edges={int(st['alive_edges'].item())}, "
                f"selected_alive_edges={int(st['selected_alive_edges'].item())}, "
                f"direct={int(st['num_direct'].item())}, projected={int(st['num_projected'].item())}, "
                f"unreachable={int(st['num_unreachable'].item())}"
            )

    # loss
    loss_out = dp_token_losses(
        cross_logit=out_td.cross_logit,
        y_cross=y_cross,
        m_cross=batch.tokens.cross_mask.bool(),
        iface_logit=out_td.iface_logit,
        y_iface=y_iface,
        m_iface=batch.tokens.iface_mask.bool(),
        w_iface=0.05,
        pos_weight_cross=None,
    )

    # backward one step
    opt = torch.optim.AdamW(
        list(leaf_encoder.parameters()) + list(merge_encoder.parameters()) + list(decoder.parameters()),
        lr=1e-4,
    )
    opt.zero_grad(set_to_none=True)
    loss_out.loss.backward()

    # grad sanity
    bad = 0
    tot = 0
    for p in list(leaf_encoder.parameters()) + list(merge_encoder.parameters()) + list(decoder.parameters()):
        if p.grad is None:
            continue
        tot += 1
        if not torch.isfinite(p.grad).all().item():
            bad += 1
    if bad > 0:
        raise RuntimeError(f"Found NaN/Inf gradients in {bad}/{tot} parameter tensors.")

    opt.step()

    parts_str = ", ".join([f"{k}={float(v.item()):.6f}" for k, v in loss_out.parts.items()])
    print(f"[ok] train step loss={float(loss_out.loss.item()):.6f} parts={parts_str}")

    # batch edge aggregation (now valid because eids are global-unique)
    edge_scores = aggregate_logits_to_edges(
        tokens=batch.tokens, 
        cross_logit=out_td.cross_logit,
        iface_logit=out_td.iface_logit,
        reduce="mean"
    )
    edge_logit = edge_scores.edge_logit
    edge_mask = edge_scores.edge_mask.bool()
    print(f"[edge] global edge_logit.shape={tuple(edge_logit.shape)}, edges_covered={int(edge_mask.sum().item())}")

    # per-graph slicing via edge_ptr
    for b in range(bs):
        e0 = int(batch.edge_ptr[b].item())
        e1 = int(batch.edge_ptr[b + 1].item())
        covered_b = int(edge_mask[e0:e1].sum().item()) if e1 > e0 else 0
        print(f"[edge] graph{b} edges_total={e1-e0}, edges_covered={covered_b}")

    print("[done] one-step closed-loop train test PASSED.")


if __name__ == "__main__":
    main()
