# src/models/dp_core.py
# -*- coding: utf-8 -*-
"""
Compatibility facade for 1-pass DP utilities.

The implementation now lives in smaller modules grouped by responsibility:

  - dp_correspondence.py
  - dp_verify.py
  - dp_parse_heuristic.py
  - dp_parse_catalog.py
  - dp_leaf_solver.py

External imports should continue to use `src.models.dp_core` until the rest of
the codebase is migrated.
"""

from __future__ import annotations

from .dp_correspondence import CorrespondenceMaps, build_correspondence_maps, propagate_c1_constraints
from .dp_leaf_solver import LeafPathWitness, LeafStateWitness, leaf_exact_solve, leaf_solve_state
from .dp_parse_catalog import _rank_child_catalog_states_for_parse, parse_by_catalog_enum
from .dp_parse_heuristic import (
    _noncrossing_min_cost_matching,
    parse_activation_batch,
    parse_continuous,
    parse_continuous_topk,
)
from .dp_verify import batch_check_c1c2, verify_tuple


__all__ = [
    "CorrespondenceMaps",
    "LeafPathWitness",
    "LeafStateWitness",
    "_noncrossing_min_cost_matching",
    "_rank_child_catalog_states_for_parse",
    "batch_check_c1c2",
    "build_correspondence_maps",
    "leaf_exact_solve",
    "leaf_solve_state",
    "parse_activation_batch",
    "parse_by_catalog_enum",
    "parse_continuous",
    "parse_continuous_topk",
    "propagate_c1_constraints",
    "verify_tuple",
]
