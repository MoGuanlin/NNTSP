# tests/test_node_token_packer.py
# -*- coding: utf-8 -*-
"""
Unit tests for interface ordering in src/models/node_token_packer.py.

Usage:
  python tests/test_node_token_packer.py
  python -m pytest tests/test_node_token_packer.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.boundary_state_structured import (
    enumerate_structured_states_for_iface_mask,
    state_to_tensors,
)
from src.models.node_token_packer import NodeTokenPacker, _stable_sort_interfaces


class TestStableSortInterfaces:

    def test_clockwise_order_for_matching_mode(self):
        """Clockwise sorting should follow the true perimeter order."""
        iface_nid = torch.zeros(8, dtype=torch.long)
        iface_eid = torch.tensor([50, 10, 70, 30, 20, 60, 40, 80], dtype=torch.long)
        iface_dir = torch.tensor([
            0,  # LEFT  upper
            2,  # BOTTOM left
            3,  # TOP right
            1,  # RIGHT upper
            3,  # TOP left
            0,  # LEFT  lower
            1,  # RIGHT lower
            2,  # BOTTOM right
        ], dtype=torch.long)
        iface_inside_ep = torch.zeros(8, dtype=torch.long)
        iface_inside_quad = torch.zeros(8, dtype=torch.long)
        iface_feat6 = torch.zeros(8, 6, dtype=torch.float32)

        # inter_rel_x lives in feat6[:,2], inter_rel_y in feat6[:,3]
        iface_feat6[0, 3] = 0.8    # LEFT upper
        iface_feat6[1, 2] = -0.7   # BOTTOM left
        iface_feat6[2, 2] = 0.6    # TOP right
        iface_feat6[3, 3] = 0.7    # RIGHT upper
        iface_feat6[4, 2] = -0.8   # TOP left
        iface_feat6[5, 3] = -0.6   # LEFT lower
        iface_feat6[6, 3] = -0.5   # RIGHT lower
        iface_feat6[7, 2] = 0.5    # BOTTOM right

        _, sorted_eid, _, sorted_dir, _, _ = _stable_sort_interfaces(
            iface_nid=iface_nid,
            iface_eid=iface_eid,
            iface_feat6=iface_feat6,
            iface_dir=iface_dir,
            iface_inside_ep=iface_inside_ep,
            iface_inside_quad=iface_inside_quad,
            clockwise=True,
        )

        # Clockwise perimeter order:
        #   TOP left -> TOP right -> RIGHT upper -> RIGHT lower
        #   -> BOTTOM right -> BOTTOM left -> LEFT lower -> LEFT upper
        assert sorted_dir.tolist() == [3, 3, 1, 1, 2, 2, 0, 0]
        assert sorted_eid.tolist() == [20, 70, 30, 40, 80, 10, 60, 50]

    def test_iface_mode_can_force_clockwise_order(self):
        """Structured one-pass DP must be able to use clockwise slot order."""
        legacy_packer = NodeTokenPacker(r=4, state_mode="iface", iface_order="legacy")
        fixed_packer = NodeTokenPacker(r=4, state_mode="iface", iface_order="clockwise")
        matching_packer = NodeTokenPacker(r=4, state_mode="matching", iface_order="legacy")

        assert legacy_packer._use_clockwise_iface_order() is False
        assert fixed_packer._use_clockwise_iface_order() is True
        assert matching_packer._use_clockwise_iface_order() is True

    def test_legacy_iface_order_misses_clockwise_legal_pairing(self):
        """Legacy iface ordering should not be used for structured non-crossing states."""
        iface_nid = torch.zeros(4, dtype=torch.long)
        iface_eid = torch.tensor([100, 200, 300, 400], dtype=torch.long)  # L, R, B, T
        iface_dir = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        iface_inside_ep = torch.zeros(4, dtype=torch.long)
        iface_inside_quad = torch.zeros(4, dtype=torch.long)
        iface_feat6 = torch.zeros(4, 6, dtype=torch.float32)
        iface_feat6[0, 3] = 0.8   # LEFT upper
        iface_feat6[1, 3] = 0.7   # RIGHT upper
        iface_feat6[2, 2] = 0.2   # BOTTOM right
        iface_feat6[3, 2] = -0.2  # TOP left

        _, legacy_eid, _, _, _, _ = _stable_sort_interfaces(
            iface_nid=iface_nid,
            iface_eid=iface_eid,
            iface_feat6=iface_feat6,
            iface_dir=iface_dir,
            iface_inside_ep=iface_inside_ep,
            iface_inside_quad=iface_inside_quad,
            clockwise=False,
        )
        _, clockwise_eid, _, _, _, _ = _stable_sort_interfaces(
            iface_nid=iface_nid,
            iface_eid=iface_eid,
            iface_feat6=iface_feat6,
            iface_dir=iface_dir,
            iface_inside_ep=iface_inside_ep,
            iface_inside_quad=iface_inside_quad,
            clockwise=True,
        )

        states = enumerate_structured_states_for_iface_mask(
            iface_mask=torch.ones(4, dtype=torch.bool)
        )

        def pairing_sets(slot_eids: torch.Tensor) -> set[frozenset[frozenset[int]]]:
            pairings = set()
            for state in states:
                used, mate = state_to_tensors(state=state, num_slots=4)
                edges = []
                for slot in range(4):
                    mate_slot = int(mate[slot].item())
                    if not bool(used[slot].item()) or mate_slot < slot:
                        continue
                    edges.append(
                        frozenset(
                            {
                                int(slot_eids[slot].item()),
                                int(slot_eids[mate_slot].item()),
                            }
                        )
                    )
                pairings.add(frozenset(edges))
            return pairings

        target_pairing = frozenset(
            {
                frozenset({400, 200}),  # TOP-RIGHT
                frozenset({300, 100}),  # BOTTOM-LEFT
            }
        )

        assert target_pairing in pairing_sets(clockwise_eid)
        assert target_pairing not in pairing_sets(legacy_eid)


def run_all_tests():
    test_classes = [TestStableSortInterfaces]
    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        obj = cls()
        methods = [m for m in dir(obj) if m.startswith("test_")]
        for method_name in methods:
            total += 1
            full_name = f"{cls.__name__}.{method_name}"
            try:
                getattr(obj, method_name)()
                passed += 1
                print(f"  PASS  {full_name}")
            except AssertionError as e:
                failed += 1
                errors.append((full_name, e))
                print(f"  FAIL  {full_name}: {e}")
            except Exception as e:
                failed += 1
                errors.append((full_name, e))
                print(f"  ERROR {full_name}: {type(e).__name__}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Total: {total}  Passed: {passed}  Failed: {failed}")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  {name}: {err}")
    return failed == 0


if __name__ == "__main__":
    print("Running node_token_packer.py unit tests...\n")
    success = run_all_tests()
    sys.exit(0 if success else 1)
