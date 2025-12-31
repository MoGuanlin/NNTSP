# src/test/test_bottom_up_process.py
# -*- coding: utf-8 -*-
"""
End-to-end correctness test for the bottom-up DP-aligned pipeline:

Data (PyG Data)
  -> NodeTokenPacker.pack_one()
      -> PackedNodeTokens + PackedLeafPoints (scale-invariant features)
  -> BottomUpTreeRunner.run_single()
      -> leaf_encoder on leaves (cell-relative leaf points)
      -> merge_encoder on internal nodes (child_z + tokens)
  -> BottomUpResult (z for every node)

What this test checks:
1) Interface compliance: required fields exist and tensor shapes are consistent.
2) Bottom-up correctness: every node latent is computed; child-before-parent ordering holds.
3) Scale-invariant feature sanity:
   - leaf_points_xy is cell-relative (roughly within [-1, 1] on valid points)
   - iface_inter_rel_xy also in a stable relative range
   - no NaN/Inf in packed tensors and outputs

Usage examples:
  python src/test/test_bottom_up_process.py --data_pt data/N50/train_r_light_pyramid.pt --idx 0
  python src/test/test_bottom_up_process.py --data_pt sample_N50_r_light_list.pt --idx 0
"""

from __future__ import annotations

import argparse
import inspect
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


# -----------------------------
# Utilities
# -----------------------------

def _add_repo_root_to_syspath() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]  # .../src/test -> .../src -> repo root
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
    """
    Instantiate a class by passing only kwargs that its __init__ accepts.
    This makes the test robust to minor signature changes.
    """
    sig = inspect.signature(cls.__init__)
    usable = {}
    for k, v in preferred_kwargs.items():
        if k in sig.parameters:
            usable[k] = v
    return cls(**usable)


# def _to_device_inplace(obj: Any, device: torch.device) -> Any:
#     # Best-effort: move every tensor attribute on a dataclass-like container.
#     for k in dir(obj):
#         if k.startswith("_"):
#             continue
#         v = getattr(obj, k)
#         if isinstance(v, torch.Tensor):
#             setattr(obj, k, v.to(device))
#     return obj
from dataclasses import replace, is_dataclass

def _to_device(obj: Any, device: torch.device) -> Any:
    """
    Move all tensor fields of a (possibly frozen) dataclass to device, returning a new object.
    Works for dataclasses with frozen=True.
    """
    if is_dataclass(obj):
        updates = {}
        for k in obj.__dataclass_fields__.keys():
            v = getattr(obj, k)
            if isinstance(v, torch.Tensor):
                updates[k] = v.to(device)
        if len(updates) == 0:
            return obj
        return replace(obj, **updates)

    # fallback: if not a dataclass, try inplace like before
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


def _range_sanity(name: str, x: torch.Tensor, mask: Optional[torch.Tensor], lo: float, hi: float, tol: float = 1e-3):
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
            f"{name} out of expected relative range: min={mn:.4f}, max={mx:.4f}, "
            f"expected approx [{lo},{hi}]. This suggests a coordinate-frame mismatch."
        )


def _check_packed_fields(tokens: Any, leaves: Any) -> Tuple[int, int, int]:
    """
    Returns caps (Ti, Tc, P).
    """
    required_tokens = [
        "tree_node_feat_rel", "tree_node_feat_raw", "tree_node_depth",
        "tree_children_index", "tree_parent_index", "is_leaf",
        "iface_feat6", "iface_mask", "iface_boundary_dir", "iface_inside_endpoint",
        "iface_inter_rel_xy", "iface_inside_quadrant",
        "cross_feat6", "cross_mask", "cross_child_pair", "cross_is_leaf_internal",
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
    assert tokens.tree_children_index.shape[0] == M and tokens.tree_children_index.shape[1] == 4
    assert tokens.tree_parent_index.shape == (M,)
    assert tokens.is_leaf.shape == (M,)

    assert tokens.iface_feat6.shape == (M, Ti, 6)
    assert tokens.iface_mask.shape == (M, Ti)
    assert tokens.iface_boundary_dir.shape == (M, Ti)
    assert tokens.iface_inside_endpoint.shape == (M, Ti)
    assert tokens.iface_inter_rel_xy.shape == (M, Ti, 2)
    assert tokens.iface_inside_quadrant.shape == (M, Ti)

    assert tokens.cross_feat6.shape == (M, Tc, 6)
    assert tokens.cross_mask.shape == (M, Tc)
    assert tokens.cross_child_pair.shape == (M, Tc, 2)
    assert tokens.cross_is_leaf_internal.shape == (M, Tc)

    if leaves.leaf_node_id.numel() > 0:
        L = int(leaves.leaf_node_id.shape[0])
        assert leaves.point_idx.shape == (L, P)
        assert leaves.point_mask.shape == (L, P)
        assert leaves.point_xy.shape == (L, P, 2)

    # finite checks on key tensors
    _assert_finite("tree_node_feat_rel", tokens.tree_node_feat_rel)
    _assert_finite("iface_feat6", tokens.iface_feat6)
    _assert_finite("iface_inter_rel_xy", tokens.iface_inter_rel_xy)
    _assert_finite("cross_feat6", tokens.cross_feat6)
    if leaves.point_xy.numel() > 0:
        _assert_finite("leaf_point_xy", leaves.point_xy)

    return Ti, Tc, P


def _check_scale_invariant_sanity(tokens: Any, leaves: Any, strict: bool) -> None:
    """
    Heuristic sanity checks for relative coordinate frames.
    If strict=True, enforce tighter bounds.
    """
    # leaf points are cell-relative => typically within [-1,1] (allow a little slack)
    if leaves.point_xy.numel() > 0:
        lo, hi = (-1.2, 1.2) if strict else (-2.0, 2.0)
        _range_sanity("leaf_points_xy", leaves.point_xy, leaves.point_mask, lo=lo, hi=hi)

    # interface intersection rel should also be relative to node box
    lo2, hi2 = (-1.2, 1.2) if strict else (-3.0, 3.0)
    _range_sanity("iface_inter_rel_xy", tokens.iface_inter_rel_xy, tokens.iface_mask.unsqueeze(-1).expand_as(tokens.iface_inter_rel_xy), lo=lo2, hi=hi2)

    # endpoints in feat6 are node-relative too (x_u,y_u,x_v,y_v). Allow slack.
    # We mask using iface_mask / cross_mask.
    iface_xy = tokens.iface_feat6[..., 0:4]
    cross_xy = tokens.cross_feat6[..., 0:4]
    _range_sanity("iface_feat6_xy", iface_xy, tokens.iface_mask.unsqueeze(-1).expand_as(iface_xy), lo=-3.0, hi=3.0)
    _range_sanity("cross_feat6_xy", cross_xy, tokens.cross_mask.unsqueeze(-1).expand_as(cross_xy), lo=-3.0, hi=3.0)


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


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    _add_repo_root_to_syspath()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_pt", type=str, required=True, help="Path to *.pt (list[Data] or a single Data).")
    parser.add_argument("--idx", type=int, default=0, help="Sample index if --data_pt stores a list.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--r", type=int, default=0, help="r-light parameter for validation; 0 means 'unset'.")
    parser.add_argument("--ti_cap", type=int, default=32, help="Cap Ti = max interface tokens per node.")
    parser.add_argument("--tc_cap", type=int, default=32, help="Cap Tc = max crossing tokens per node.")
    parser.add_argument("--p_cap", type=int, default=64, help="Cap P = max points per leaf.")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--strict_rel_range", type=int, default=1, help="1 to enforce tighter relative bounds.")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[env] device={device}")

    # Imports from your codebase
    from src.models.node_token_packer import NodeTokenPacker
    from src.models.leaf_encoder import LeafEncoderModule
    from src.models.merge_encoder import MergeEncoderModule
    from src.models.bottom_up_runner import BottomUpTreeRunner

    # Load data
    obj = _torch_load(args.data_pt, map_location="cpu")
    if isinstance(obj, (list, tuple)):
        if not (0 <= args.idx < len(obj)):
            raise IndexError(f"idx out of range: idx={args.idx}, len={len(obj)}")
        data = obj[args.idx]
    else:
        data = obj

    # Build packer (new signature)
    packer = NodeTokenPacker(
        r=(args.r if args.r > 0 else None),
        max_iface_per_node=args.ti_cap,
        max_cross_per_node=args.tc_cap,
        max_points_per_leaf=args.p_cap,
    )

    tokens, leaves, stats = packer.pack_one(data)
    # tokens = _to_device_inplace(tokens, device)
    # leaves = _to_device_inplace(leaves, device)
    tokens = _to_device(tokens, device)
    leaves = _to_device(leaves, device)


    # Basic checks
    M = int(tokens.tree_node_feat_rel.shape[0])
    Ti, Tc, P = _check_packed_fields(tokens, leaves)
    print(f"[pack] M={M}, Ti={Ti}, Tc={Tc}, P={P}, num_leaves={int(leaves.leaf_node_id.numel())}")
    if isinstance(stats, dict) and "root_scale_s" in stats:
        print(f"[pack] root_scale_s={float(stats['root_scale_s'].item()):.6f}")

    # Scale-invariant sanity checks (heuristic)
    _check_scale_invariant_sanity(tokens, leaves, strict=bool(args.strict_rel_range))

    # Instantiate encoders
    leaf_encoder = LeafEncoderModule(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=0.0,
    ).to(device).eval()

    merge_encoder = MergeEncoderModule(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=0.0,
    ).to(device).eval()

    # Runner
    runner = BottomUpTreeRunner(validate_completeness=True)

    # Determine root id
    root_id = _find_root_id(tokens.tree_parent_index)
    print(f"[tree] root_id={root_id}, max_depth={int(tokens.tree_node_depth.max().item())}")

    # Run bottom-up
    with torch.no_grad():
        out = runner.run_single(tokens=tokens, leaves=leaves, leaf_encoder=leaf_encoder, merge_encoder=merge_encoder)

    _validate_bottom_up_output(out, M)

    # Diagnostics
    z = out.z
    print(f"[diag] z[root].norm={float(z[root_id].norm().item()):.6f}")
    # Depth monotonicity spot-check (child depth > parent depth for valid child)
    ch = tokens.tree_children_index
    dep = tokens.tree_node_depth
    child_mask = ch >= 0
    if child_mask.any().item():
        parent_ids, slot_ids = torch.nonzero(child_mask, as_tuple=True)
        child_ids = ch[parent_ids, slot_ids]
        ok = (dep[child_ids] > dep[parent_ids]).all().item()
        if not ok:
            bad = torch.nonzero(dep[child_ids] <= dep[parent_ids], as_tuple=False)[:10]
            raise AssertionError(f"Depth monotonicity violated for some edges. Example pairs idx: {bad.tolist()}")
    print("[ok] depth monotonicity (child depth > parent depth) holds on valid child links.")
    print("[done] bottom-up end-to-end test PASSED.")


if __name__ == "__main__":
    main()
