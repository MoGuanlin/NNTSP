# -*- coding: utf-8 -*-
"""Shared statistics helpers for the 1-pass DP runner."""

from __future__ import annotations

from typing import Any, Dict, Optional


def new_depth_stats_bucket() -> Dict[str, float]:
    """Create an empty per-depth statistics bucket."""
    return {
        "num_internal_nodes": 0.0,
        "num_sigma_total": 0.0,
        "num_parse_ok": 0.0,
        "num_topk_ok": 0.0,
        "num_fallback": 0.0,
        "num_infeasible": 0.0,
        "fallback_rate": 0.0,
    }


def ensure_depth_stats_bucket(
    stats: Dict[str, Any],
    depth: int,
) -> Dict[str, float]:
    """Return the mutable per-depth stats bucket for one merge depth."""
    depth_stats = stats.setdefault("depth_stats", {})
    depth_key = str(int(depth))
    bucket = depth_stats.get(depth_key)
    if bucket is None:
        bucket = new_depth_stats_bucket()
        depth_stats[depth_key] = bucket
    return bucket


def bump_stat(
    stats: Dict[str, Any],
    key: str,
    *,
    amount: float = 1.0,
    depth_bucket: Optional[Dict[str, float]] = None,
) -> None:
    """Accumulate one global stat and, optionally, the current depth bucket."""
    stats[key] = float(stats.get(key, 0.0) + amount)
    if depth_bucket is not None:
        depth_bucket[key] = float(depth_bucket.get(key, 0.0) + amount)


def finalize_depth_stats_bucket(bucket: Dict[str, float]) -> None:
    """Update derived metrics for a finished depth bucket."""
    total = float(bucket.get("num_sigma_total", 0.0))
    fallback = float(bucket.get("num_fallback", 0.0))
    bucket["fallback_rate"] = (fallback / total) if total > 0.0 else 0.0


def refresh_depth_fallback_rates(stats: Dict[str, Any]) -> None:
    """Materialize a compact depth->fallback_rate summary in stats."""
    depth_stats = stats.get("depth_stats", {})
    if not isinstance(depth_stats, dict):
        stats["depth_fallback_rates"] = {}
        return
    stats["depth_fallback_rates"] = {
        depth_key: float(bucket.get("fallback_rate", 0.0))
        for depth_key, bucket in sorted(
            depth_stats.items(),
            key=lambda kv: int(kv[0]),
            reverse=True,
        )
    }


__all__ = [
    "bump_stat",
    "ensure_depth_stats_bucket",
    "finalize_depth_stats_bucket",
    "new_depth_stats_bucket",
    "refresh_depth_fallback_rates",
]
