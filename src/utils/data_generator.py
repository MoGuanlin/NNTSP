# src/utils/data_generator.py
import argparse
import os
from pathlib import Path
from typing import Tuple

import torch


def _ensure_unique_points(points_xy: torch.Tensor, grid_size: int, generator: torch.Generator) -> torch.Tensor:
    """Ensure per-sample uniqueness of 2D integer grid points.

    Args:
        points_xy: [N, 2] integer points in [0, grid_size].
        grid_size: max coordinate value (inclusive).
        generator: torch random generator.

    Returns:
        A new tensor [N, 2] with unique rows (as integer coordinates).
    """
    device = points_xy.device
    N = points_xy.shape[0]
    pts = points_xy.clone()

    key_base = grid_size + 1
    keys = pts[:, 0] * key_base + pts[:, 1]
    _, counts = torch.unique(keys, return_counts=True)
    if int(counts.max().item()) == 1:
        return pts

    used = set()
    for i in range(N):
        x = int(pts[i, 0].item())
        y = int(pts[i, 1].item())
        k = x * key_base + y
        while k in used:
            xy = torch.randint(0, key_base, (2,), generator=generator, device=device, dtype=torch.long)
            x, y = int(xy[0].item()), int(xy[1].item())
            k = x * key_base + y
        used.add(k)
        pts[i, 0] = x
        pts[i, 1] = y
    return pts


def generate_tsp_data(
    num_samples: int,
    num_nodes: int,
    grid_size: int = 10000,
    seed: int = 1234,
    ensure_unique: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate Rao'98-style 2D TSP instances on an integer grid.

    Notes:
        - Return **float32** coordinates (with integer values). float32 can exactly represent
          all integers up to 16,777,216, so this preserves integer-grid semantics while
          avoiding dtype churn downstream.
        - Optional uniqueness constraint prevents degenerate edges (zero length) and can
          improve Delaunay robustness.

    Args:
        num_samples: number of instances (B).
        num_nodes: number of points per instance (N).
        grid_size: coordinates are sampled uniformly from [0, grid_size] (inclusive).
        seed: RNG seed.
        ensure_unique: if True, enforce per-instance unique points.
        dtype: output dtype (default float32).

    Returns:
        points: [B, N, 2] tensor.
    """
    if num_samples <= 0 or num_nodes <= 0:
        raise ValueError(f"num_samples and num_nodes must be positive, got {num_samples=}, {num_nodes=}")
    if grid_size < 1:
        raise ValueError(f"grid_size must be >= 1, got {grid_size}")

    g = torch.Generator()
    g.manual_seed(int(seed))
    key_base = grid_size + 1

    pts_int = torch.randint(0, key_base, (num_samples, num_nodes, 2), generator=g, dtype=torch.long)

    if ensure_unique:
        for b in range(num_samples):
            pts_int[b] = _ensure_unique_points(pts_int[b], grid_size=grid_size, generator=g)

    return pts_int.to(dtype)


def save_dataset(data: torch.Tensor, out_dir: str, filename: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    torch.save(data, out_path)
    print(f"Saved: {out_path}  shape={tuple(data.shape)}  dtype={data.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Rao'98-style integer-grid TSP datasets")
    parser.add_argument("--data_dir", type=str, default="data", help="Output root directory")
    parser.add_argument("--sizes", type=str, default="20,50", help="Comma-separated N values, e.g., 20,50")
    parser.add_argument("--num_samples", type=int, default=2000, help="Train samples per N")
    parser.add_argument("--grid_size", type=int, default=10000, help="Grid size (coords in [0, grid_size])")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--no_unique", action="store_true", help="Disable unique-points constraint")
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    if not sizes:
        raise ValueError("--sizes is empty")

    ensure_unique = not args.no_unique

    for N in sizes:
        print(f"Generating data for N={N}...")
        out_dir = os.path.join(args.data_dir, f"N{N}")

        train = generate_tsp_data(args.num_samples, N, args.grid_size, args.seed, ensure_unique=ensure_unique)
        save_dataset(train, out_dir, "train.pt")

        val = generate_tsp_data(max(1, args.num_samples // 10), N, args.grid_size, args.seed + 1, ensure_unique=ensure_unique)
        save_dataset(val, out_dir, "val.pt")

        test = generate_tsp_data(max(1, args.num_samples // 10), N, args.grid_size, args.seed + 2, ensure_unique=ensure_unique)
        save_dataset(test, out_dir, "test.pt")

    print("\n[Done] All datasets generated.")


if __name__ == "__main__":
    main()
