#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import warnings


def safe_torch_load(path: str) -> Any:
    """
    Load a torch object robustly across torch versions, and suppress the
    FutureWarning about weights_only default flip.

    Note: This dataset is user-generated and trusted in your environment.
    """
    # Suppress the specific FutureWarning noise (you asked to hide warnings previously).
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Newer torch supports weights_only; older may not.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tensor_info(x: Any) -> Optional[Dict[str, Any]]:
    if not torch.is_tensor(x):
        return None
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
    }


def data_field_summary(obj: Any) -> Dict[str, Any]:
    """
    Summarize attributes of a PyG Data-like object. Works even if PyG isn't installed,
    as long as the object has __dict__ or attribute access.
    """
    out: Dict[str, Any] = {}
    keys = list(getattr(obj, "__dict__", {}).keys())
    keys.sort()
    out["fields"] = keys

    tinfo: Dict[str, Any] = {}
    for k in keys:
        v = getattr(obj, k)
        ti = tensor_info(v)
        if ti is not None:
            tinfo[k] = ti
        else:
            # keep small scalars/strings; skip big containers
            if isinstance(v, (int, float, str, bool)) or v is None:
                tinfo[k] = v
            elif isinstance(v, (list, tuple)) and len(v) <= 5:
                tinfo[k] = f"{type(v).__name__}(len={len(v)})"
            else:
                tinfo[k] = str(type(v))
    out["field_info"] = tinfo
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a single sample from *_r_light_pyramid.pt (List[Data])"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/N50/train_r_light_pyramid.pt",
        help="Path to the r-light pyramid dataset (List[Data])",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index to extract (0-based)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sample_N50_r_light.pt",
        help="Output path for the extracted sample (single Data object)",
    )
    parser.add_argument(
        "--output_list",
        type=str,
        default="sample_N50_r_light_list.pt",
        help="Output path for the extracted sample wrapped as a list ([Data])",
    )
    parser.add_argument(
        "--dump_info",
        type=str,
        default="sample_N50_r_light_info.json",
        help="Output path for a JSON summary (fields, shapes, dtypes)",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(f"Input not found: {inp.resolve()}")

    obj = safe_torch_load(str(inp))
    if not isinstance(obj, list) or len(obj) == 0:
        raise RuntimeError(
            f"Expected a non-empty List[Data] in {inp}, got type={type(obj)}"
        )

    if args.index < 0 or args.index >= len(obj):
        raise IndexError(f"index={args.index} out of range: [0, {len(obj)-1}]")

    sample = obj[args.index]

    # Save single Data
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sample, str(out_path))

    # Save as list([Data]) to preserve dataset format
    out_list_path = Path(args.output_list)
    out_list_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save([sample], str(out_list_path))

    # Dump summary json
    info = {
        "input": str(inp),
        "num_samples_in_input": len(obj),
        "index": args.index,
        "saved_single": str(out_path),
        "saved_list": str(out_list_path),
        "summary": data_field_summary(sample),
    }
    dump_path = Path(args.dump_info)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    dump_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[OK] Extracted one sample.")
    print(f"  input      : {inp}")
    print(f"  index      : {args.index} / {len(obj)-1}")
    print(f"  output     : {out_path}")
    print(f"  output_list: {out_list_path}")
    print(f"  dump_info  : {dump_path}")


if __name__ == "__main__":
    main()
