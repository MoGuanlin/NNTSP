# src/cli/eval_settings.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from typing import Mapping, Sequence


def _tokenize_settings(raw: str | None) -> list[str]:
    if raw is None:
        return []
    text = str(raw).strip().lower()
    if not text:
        return []
    return [tok for tok in re.split(r"[\s,]+", text) if tok]


def describe_settings(
    *,
    available: Sequence[str],
    groups: Mapping[str, Sequence[str]] | None = None,
) -> str:
    group_keys = sorted((groups or {}).keys())
    parts = [f"available={','.join(available)}"]
    if group_keys:
        parts.append(f"groups={','.join(group_keys)}")
    return " ; ".join(parts)


def resolve_eval_settings(
    *,
    requested: str | None,
    available: Sequence[str],
    default: Sequence[str],
    aliases: Mapping[str, str] | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
    enable_exact: bool = False,
    disable_guided_lkh: bool = False,
) -> list[str]:
    aliases = dict(aliases or {})
    groups = dict(groups or {})
    available_list = list(dict.fromkeys(str(x) for x in available))
    available_set = set(available_list)

    def expand(token: str, stack: tuple[str, ...] = ()) -> list[str]:
        canonical = aliases.get(token, token)
        if canonical in groups:
            if canonical in stack:
                cycle = " -> ".join(stack + (canonical,))
                raise ValueError(f"Circular setting group detected: {cycle}")
            out: list[str] = []
            for member in groups[canonical]:
                out.extend(expand(str(member), stack + (canonical,)))
            return out
        if canonical not in available_set:
            desc = describe_settings(available=available_list, groups=groups)
            raise ValueError(f"Unknown eval setting '{token}'. {desc}")
        return [canonical]

    tokens = _tokenize_settings(requested)
    if not tokens:
        tokens = list(default)

    resolved: list[str] = []
    for token in tokens:
        for item in expand(token):
            if item not in resolved:
                resolved.append(item)

    if enable_exact and "exact" in available_set and "exact" not in resolved:
        if "greedy" in resolved:
            insert_at = resolved.index("greedy") + 1
            resolved.insert(insert_at, "exact")
        else:
            resolved.append("exact")

    if disable_guided_lkh:
        resolved = [item for item in resolved if item != "guided_lkh"]

    return resolved


__all__ = ["describe_settings", "resolve_eval_settings"]
