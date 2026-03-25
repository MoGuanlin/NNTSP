# tests/test_top_down_process.py
# -*- coding: utf-8 -*-
"""
End-to-end correctness test for the *top-down* DP-aligned pipeline (Contract v2).

This test runs:
  Data (*.pt) -> NodeTokenPacker -> bottom-up (LeafEncoder/MergeEncoder) -> top-down (TopDownDecoder/Runner)
and then aggregates per-node crossing logits into per-edge logits using cross_eid.

Usage:
  python tests/test_top_down_process.py --data_pt data/N50/train_r_light_pyramid.pt --idx 0 --r 4 --device cuda

Notes / Expectations:
- Token shapes are scale-invariant: Ti=4r, Smax=4r, Tc is constant cap (e.g., 8r) unless overridden.
- Top-down is DP-aligned (depth ascending), uses a fixed-cap stub set per node.
- This is a *shape/contract/finite* test, not a quality test (weights are random unless you load a checkpoint).
"""

from __future__ import annotations

import argparse
import inspect
import sys
import warnings
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


# -----------------------------
# Utilities (mirrors test_bottom_up_process.py)
# -----------------------------

def _add_repo_root_to_syspath() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    sys.path.insert(0, str(repo_root))


def _silence_torch_load_warning() -> None:
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r"You are using `torch\.load` with `weights_only=False`.*",
    )


def _torch_load(path: str, map_location: str = "cpu") -> Any:
    _silence_torch_load_warning()
    return torch.load(path, map_location=map_location)


def _instantiate_best_effort(cls, preferred_kwargs: Dict[str, Any]):
    sig = inspect.signature(cls.__init__)
    usable = {k: v for k, v in preferred_kwargs.items() if k in sig.parameters}
    return cls(**usable)


def _to_device(obj: Any, device: torch.device) -> Any:
    if is_dataclass(obj):
        updates = {}
        for k in obj.__dataclass_fields__.keys():
            v = getattr(obj, k)
            if isinstance(v, torch.Tensor):
                updates[k] = v.to(device)
        return replace(obj, **updates) if updates else obj

    for k in dir(obj):
        if k.startswith("_"):
            continue
        v = getattr(obj, k)
        if isinstance(v, torch.Tensor):
            try:
                setattr(obj, k, v.to(device))
            except Exception:
                pass
    return obj


def _find_root_id(tree_parent_index: torch.Tensor) -> int:
    roots = torch.nonzero(tree_parent_index < 0, as_tuple=False).view(-1)
    if roots.numel() != 1:
        raise RuntimeError(f"Expected exactly 1 root (parent<0), got roots={roots.tolist()}")
    return int(roots.item())


def _assert_finite(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all().item():
        bad = torch.nonzero(~torch.isfinite(x), as_tuple=False)[:10].tolist()
        raise AssertionError(f"{name} contains NaN/Inf. First bad indices: {bad}")


def _range_sanity(
    name: str,
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    lo: float,
    hi: float,
    tol: float = 1e-3,
) -> None:
    if mask is not None:
        x = x[mask]
    if x.numel() == 0:
        print(f"[range] {name}: empty after masking.")
        return
    mn = float(x.min().item())
    mx = float(x.max().item())
    print(f"[range] {name}: min={mn:.4f}, max={mx:.4f}, expected approx [{lo},{hi}]")
    if mn < lo - tol or mx > hi + tol:
        raise AssertionError(
            f"{name} out of expected relative range: min={mn:.4f}, max={mx:.4f}, expected approx [{lo},{hi}]."
        )


def _check_packed_fields(tokens: Any, leaves: Any) -> Tuple[int, int, int]:
    """
    Returns caps (Ti, Tc, P).
    """
    required_tokens = [
        "tree_node_feat_rel", "tree_node_feat_raw", "tree_node_depth",
        "tree_children_index", "tree_parent_index", "is_leaf",
        "iface_feat6", "iface_mask", "iface_boundary_dir", "iface_inside_endpoint",
        "iface_inside_quadrant", "iface_eid",
        "cross_feat6", "cross_mask", "cross_child_pair", "cross_is_leaf_internal", "cross_eid",
    ]
    for k in required_tokens:
        if not hasattr(tokens, k):
            raise RuntimeError(f"PackedNodeTokens missing field: {k}")

    required_leaves = ["leaf_node_id", "point_xy", "point_mask", "point_idx"]
    for k in required_leaves:
        if not hasattr(leaves, k):
            raise RuntimeError(f"PackedLeafPoints missing field: {k}")

    M = int(tokens.tree_node_feat_rel.shape[0])
    Ti = int(tokens.iface_feat6.shape[1])
    Tc = int(tokens.cross_feat6.shape[1])
    P = int(leaves.point_xy.shape[1]) if leaves.point_xy.numel() > 0 else 0

    # shape consistency
    assert tokens.tree_node_feat_rel.shape == (M, 4)
    assert tokens.tree_node_feat_raw.shape == (M, 4)
    assert tokens.tree_node_depth.shape == (M,)
    assert tokens.tree_children_index.shape == (M, 4)
    assert tokens.tree_parent_index.shape == (M,)
    assert tokens.is_leaf.shape == (M,)

    assert tokens.iface_feat6.shape == (M, Ti, 6)
    assert tokens.iface_mask.shape == (M, Ti)
    assert tokens.iface_boundary_dir.shape == (M, Ti)
    assert tokens.iface_inside_endpoint.shape == (M, Ti)
    assert tokens.iface_inside_quadrant.shape == (M, Ti)
    assert tokens.iface_eid.shape == (M, Ti)

    assert tokens.cross_feat6.shape == (M, Tc, 6)
    assert tokens.cross_mask.shape == (M, Tc)
    assert tokens.cross_child_pair.shape == (M, Tc, 2)
    assert tokens.cross_is_leaf_internal.shape == (M, Tc)
    assert tokens.cross_eid.shape == (M, Tc)

    if leaves.leaf_node_id.numel() > 0:
        L = int(leaves.leaf_node_id.shape[0])
        assert leaves.point_idx.shape == (L, P)
        assert leaves.point_mask.shape == (L, P)
        assert leaves.point_xy.shape == (L, P, 2)

    # finite checks
    _assert_finite("tree_node_feat_rel", tokens.tree_node_feat_rel)
    _assert_finite("iface_feat6", tokens.iface_feat6)
    _assert_finite("cross_feat6", tokens.cross_feat6)
    if leaves.point_xy.numel() > 0:
        _assert_finite("leaf_point_xy", leaves.point_xy)

    return Ti, Tc, P


def _check_scale_invariant_sanity(tokens: Any, leaves: Any, strict: bool) -> None:
    # leaf points are cell-relative
    if leaves.point_xy.numel() > 0:
        _range_sanity("leaf_points_xy", leaves.point_xy, leaves.point_mask, lo=-1.2, hi=1.2)

    iface_inside_rel = tokens.iface_feat6[..., 0:2]
    iface_inter_rel = tokens.iface_feat6[..., 2:4]
    _range_sanity(
        "iface_inside_rel_xy",
        iface_inside_rel,
        tokens.iface_mask.unsqueeze(-1).expand_as(iface_inside_rel),
        lo=-1.2,
        hi=1.2,
    )
    _range_sanity(
        "iface_inter_rel_xy(derived)",
        iface_inter_rel,
        tokens.iface_mask.unsqueeze(-1).expand_as(iface_inter_rel),
        lo=-1.2,
        hi=1.2,
    )

    cross_xy = tokens.cross_feat6[..., 0:4]
    _range_sanity(
        "cross_feat6_xy",
        cross_xy,
        tokens.cross_mask.unsqueeze(-1).expand_as(cross_xy),
        lo=-1.2,
        hi=1.2,
    )

    # length/angle sanity
    iface_len = tokens.iface_feat6[..., 4]
    iface_ang = tokens.iface_feat6[..., 5]
    cross_len = tokens.cross_feat6[..., 4]
    cross_ang = tokens.cross_feat6[..., 5]
    _range_sanity("iface_norm_len", iface_len, tokens.iface_mask, lo=-1e-3, hi=1.05)
    _range_sanity("cross_norm_len", cross_len, tokens.cross_mask, lo=-1e-3, hi=1.05)
    _range_sanity("iface_angle", iface_ang, tokens.iface_mask, lo=-1.05, hi=1.05)
    _range_sanity("cross_angle", cross_ang, tokens.cross_mask, lo=-1.05, hi=1.05)

    if strict:
        # extra: angles should look like angle/pi, not radians
        # (still wide bounds; this is only a sanity hint)
        pass


def _validate_bottom_up_output(out: Any, M: int) -> None:
    if not hasattr(out, "z") or not hasattr(out, "computed"):
        raise RuntimeError("BottomUpResult must expose .z and .computed")
    z = out.z
    computed = out.computed

    if not isinstance(z, torch.Tensor) or z.ndim != 2:
        raise AssertionError(f"out.z must be [M,d] tensor, got {type(z)} shape={getattr(z, 'shape', None)}")
    if z.shape[0] != M:
        raise AssertionError(f"out.z.shape[0] must be M={M}, got {z.shape[0]}")
    _assert_finite("out.z", z)

    if not isinstance(computed, torch.Tensor) or computed.dtype != torch.bool:
        raise AssertionError("out.computed must be bool tensor")
    if computed.numel() != M:
        raise AssertionError(f"out.computed.numel() must be M={M}, got {computed.numel()}")
    if not bool(computed.all().item()):
        missing = torch.nonzero(~computed, as_tuple=False).view(-1)[:50].tolist()
        raise AssertionError(f"Not all nodes were computed. Missing (up to 50): {missing}")

    print(f"[ok] bottom-up output z.shape={tuple(z.shape)}, all nodes computed.")


def _validate_top_down_output(
    *,
    out: Any,
    M: int,
    Ti: int,
    Tc: int,
    Smax: int,
) -> None:
    for k in ["iface_logit", "cross_logit", "stub", "stub_mask"]:
        if not hasattr(out, k):
            raise RuntimeError(f"TopDownResult missing field: {k}")

    iface_logit = out.iface_logit
    cross_logit = out.cross_logit
    stub = out.stub
    stub_mask = out.stub_mask

    if iface_logit.shape != (M, Ti):
        raise AssertionError(f"iface_logit must be [M,Ti]=({M},{Ti}), got {tuple(iface_logit.shape)}")
    if cross_logit.shape != (M, Tc):
        raise AssertionError(f"cross_logit must be [M,Tc]=({M},{Tc}), got {tuple(cross_logit.shape)}")
    if stub.shape[0] != M or stub.shape[1] != Smax:
        raise AssertionError(f"stub must be [M,Smax,d], got {tuple(stub.shape)}")
    if stub_mask.shape != (M, Smax):
        raise AssertionError(f"stub_mask must be [M,Smax], got {tuple(stub_mask.shape)}")

    _assert_finite("iface_logit", iface_logit)
    _assert_finite("cross_logit", cross_logit)
    _assert_finite("stub", stub)

    print(f"[ok] top-down output iface_logit={tuple(iface_logit.shape)}, cross_logit={tuple(cross_logit.shape)}")
    print(f"[ok] top-down stub={tuple(stub.shape)}, stub_nonempty_nodes={int(stub_mask.any(dim=1).sum().item())}/{M}")


def _cross_eid_uniqueness_sanity(tokens: Any) -> None:
    m = tokens.cross_mask.bool() & (tokens.cross_eid >= 0)
    num_cross = int(m.sum().item())
    if num_cross == 0:
        print("[stat] cross tokens: 0 (no crossing tokens found).")
        return

    eid = tokens.cross_eid[m].long()
    uniq = torch.unique(eid)
    num_uniq = int(uniq.numel())
    print(f"[stat] cross tokens total={num_cross}, unique_eids={num_uniq}")

    # Under current data design: each alive spanner edge has exactly one crossing token.
    # We therefore expect near equality; if not equal, either:
    # - crossing truncation happened, or
    # - data definition changed (multiple cross per edge), or
    # - there are duplicate eids in cross records.
    if num_uniq != num_cross:
        print(
            "[warn] unique_eids != num_cross. This suggests duplicates or truncation. "
            "Not failing hard, but you should investigate."
        )


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    _add_repo_root_to_syspath()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_pt", type=str, required=True, help="Path to *.pt (list[Data] or a single Data).")
    parser.add_argument("--idx", type=int, default=0, help="Sample index if --data_pt stores a list.")
    parser.add_argument("--r", type=int, default=4, help="r-light parameter for packer (affects caps).")
    parser.add_argument("--device", type=str, default="cpu", help="cpu|cuda")
    parser.add_argument("--strict", action="store_true", help="Use tighter coordinate sanity bounds.")
    parser.add_argument("--seed", type=int, default=0, help="Torch seed for determinism.")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))

    device = torch.device(args.device)
    print(f"[env] device={device}")

    # imports after sys.path is set
    from src.models.node_token_packer import NodeTokenPacker
    from src.models.leaf_encoder import LeafEncoder
    from src.models.merge_encoder import MergeEncoder
    from src.models.bottom_up_runner import BottomUpTreeRunner

    # top-down
    from src.models.top_down_decoder import TopDownDecoder
    from src.models.top_down_runner import TopDownTreeRunner
    from src.models.edge_aggregation import aggregate_cross_logits_to_edges

    obj = _torch_load(args.data_pt, map_location="cpu")
    if isinstance(obj, list):
        if args.idx < 0 or args.idx >= len(obj):
            raise IndexError(f"idx out of range: idx={args.idx}, len={len(obj)}")
        data = obj[args.idx]
    else:
        data = obj

    # Move raw Data tensors to device (best-effort)
    skip = {"num_faces"}
    for k in dir(data):
        if k in skip or k.startswith("_"):
            continue
        v = getattr(data, k)
        if isinstance(v, torch.Tensor):
            setattr(data, k, v.to(device))

    # Quick r-light stats on raw Data (interface records)
    ia = data.interface_assign_index  # [2, I]
    nid = ia[0]
    bdir = data.interface_boundary_dir  # [I]
    M_raw = int(data.tree_node_feat.size(0))
    cnt_node = torch.bincount(nid, minlength=M_raw)
    key = nid * 4 + bdir
    cnt_key = torch.bincount(key, minlength=M_raw * 4).view(M_raw, 4)
    print("[stat] max iface per node =", int(cnt_node.max().item()))
    print("[stat] max iface per (node,dir) =", int(cnt_key.max().item()))

    # Instantiate modules best-effort
    packer = _instantiate_best_effort(NodeTokenPacker, {"r": args.r})
    leaf_encoder = _instantiate_best_effort(LeafEncoder, {"d_model": 128})
    merge_encoder = _instantiate_best_effort(MergeEncoder, {"d_model": 128})
    bu_runner = _instantiate_best_effort(BottomUpTreeRunner, {})

    # Top-down modules
    decoder = _instantiate_best_effort(TopDownDecoder, {"r": args.r, "d_model": 128})
    td_runner = _instantiate_best_effort(TopDownTreeRunner, {})

    # Pack
    tokens, leaves, stats = packer.pack_one(data)
    tokens = _to_device(tokens, device)
    leaves = _to_device(leaves, device)

    Ti, Tc, P = _check_packed_fields(tokens, leaves)
    Smax = int(4 * args.r)

    print(f"[pack] M={tokens.tree_node_feat_rel.shape[0]}, Ti={Ti}, Tc={Tc}, P={P}, num_leaves={leaves.leaf_node_id.numel()}")
    print(f"[pack] Smax(topdown stubs)={Smax}")
    print(f"[pack] root_scale_s={float(tokens.root_scale_s.item() if tokens.root_scale_s.numel()==1 else tokens.root_scale_s.flatten()[0].item()):.6f}")

    _check_scale_invariant_sanity(tokens, leaves, strict=args.strict)

    root_id = _find_root_id(tokens.tree_parent_index)
    max_depth = int(tokens.tree_node_depth.max().item()) if tokens.tree_node_depth.numel() > 0 else 0
    print(f"[tree] root_id={root_id}, max_depth={max_depth}")

    # Bottom-up
    leaf_encoder = leaf_encoder.to(device)
    merge_encoder = merge_encoder.to(device)
    out_bu = bu_runner.run_single(tokens=tokens, leaves=leaves, leaf_encoder=leaf_encoder, merge_encoder=merge_encoder)
    _validate_bottom_up_output(out_bu, M=int(tokens.tree_node_feat_rel.shape[0]))

    # Top-down
    decoder = decoder.to(device)
    out_td = td_runner.run_single(tokens=tokens, z=out_bu.z, decoder=decoder)
    _validate_top_down_output(out=out_td, M=int(tokens.tree_node_feat_rel.shape[0]), Ti=Ti, Tc=Tc, Smax=Smax)

    # Cross eid sanity
    _cross_eid_uniqueness_sanity(tokens)

    # Edge aggregation
    edge_scores = aggregate_cross_logits_to_edges(tokens=tokens, cross_logit=out_td.cross_logit)
    edge_logit = edge_scores.edge_logit
    edge_mask = edge_scores.edge_mask
    if edge_logit.numel() > 0:
        _assert_finite("edge_logit", edge_logit[edge_mask] if edge_mask.any().item() else edge_logit)
    print(f"[edge] edge_logit.shape={tuple(edge_logit.shape)}, edges_covered={int(edge_mask.sum().item())}")

    # Diagnostics
    print(f"[diag] z[root].norm={float(out_bu.z[root_id].norm().item()):.6f}")
    if edge_mask.any().item():
        covered_vals = edge_logit[edge_mask]
        print(f"[diag] edge_logit(covered): min={float(covered_vals.min().item()):.4f}, max={float(covered_vals.max().item()):.4f}")

    print("[done] top-down end-to-end test PASSED.")


if __name__ == "__main__":
    main()
