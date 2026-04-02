# tests/test_dp_core.py
# -*- coding: utf-8 -*-
"""
Unit tests for src/models/dp_core.py — correspondence maps, VERIFYTUPLE, PARSE,
and leaf exact solver.

These tests construct small synthetic quadtree scenarios and verify correctness.

Usage:
  python tests/test_dp_core.py
  python -m pytest tests/test_dp_core.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.dp_core import (
    CorrespondenceMaps,
    build_correspondence_maps,
    verify_tuple,
    parse_continuous,
    parse_continuous_topk,
    leaf_exact_solve,
    _noncrossing_min_cost_matching,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_simple_scenario(Ti: int = 8):
    """Build a simple 1-level internal node with 4 children.

    Layout (parent box [0,1]x[0,1]):
      TL=[0,0.5]x[0.5,1]   TR=[0.5,1]x[0.5,1]
      BL=[0,0.5]x[0,0.5]   BR=[0.5,1]x[0,0.5]

    We place 4 spanner edges crossing boundaries:
      e0: crosses parent LEFT boundary,  owned by TL (left side)
      e1: crosses parent RIGHT boundary, owned by TR (right side)
      e2: crosses parent TOP boundary,   owned by TL (top side)
      e3: internal crossing TL↔TR (vertical internal boundary)
      e4: internal crossing BL↔BR (vertical internal boundary)
      e5: internal crossing TL↔BL (horizontal internal boundary)

    Each edge gets a unique eid and appears as:
      - crossing at parent
      - interface at the relevant child(ren)
    """
    # Parent iface tokens: e0 (LEFT, TL), e1 (RIGHT, TR), e2 (TOP, TL)
    # Padded to Ti slots
    parent_iface_eid = torch.full((Ti,), -1, dtype=torch.long)
    parent_iface_mask = torch.zeros(Ti, dtype=torch.bool)
    parent_iface_bdir = torch.full((Ti,), -1, dtype=torch.long)

    parent_iface_eid[0] = 0   # e0 on LEFT
    parent_iface_mask[0] = True
    parent_iface_bdir[0] = 0  # LEFT

    parent_iface_eid[1] = 1   # e1 on RIGHT
    parent_iface_mask[1] = True
    parent_iface_bdir[1] = 1  # RIGHT

    parent_iface_eid[2] = 2   # e2 on TOP
    parent_iface_mask[2] = True
    parent_iface_bdir[2] = 3  # TOP

    # Parent crossing tokens: e3 (TL↔TR), e4 (BL↔BR), e5 (TL↔BL)
    Tc = 8
    parent_cross_eid = torch.full((Tc,), -1, dtype=torch.long)
    parent_cross_mask = torch.zeros(Tc, dtype=torch.bool)
    parent_cross_child_pair = torch.full((Tc, 2), -1, dtype=torch.long)

    parent_cross_eid[0] = 3
    parent_cross_mask[0] = True
    parent_cross_child_pair[0] = torch.tensor([0, 1])  # TL↔TR

    parent_cross_eid[1] = 4
    parent_cross_mask[1] = True
    parent_cross_child_pair[1] = torch.tensor([2, 3])  # BL↔BR

    parent_cross_eid[2] = 5
    parent_cross_mask[2] = True
    parent_cross_child_pair[2] = torch.tensor([0, 2])  # TL↔BL

    # Children iface tokens [4, Ti]
    children_iface_eid = torch.full((4, Ti), -1, dtype=torch.long)
    children_iface_mask = torch.zeros(4, Ti, dtype=torch.bool)
    children_iface_bdir = torch.full((4, Ti), -1, dtype=torch.long)

    # TL (child 0): has e0 (LEFT), e2 (TOP), e3 (RIGHT, shared with TR), e5 (BOTTOM, shared with BL)
    children_iface_eid[0, 0] = 0;  children_iface_mask[0, 0] = True;  children_iface_bdir[0, 0] = 0  # e0 LEFT
    children_iface_eid[0, 1] = 2;  children_iface_mask[0, 1] = True;  children_iface_bdir[0, 1] = 3  # e2 TOP
    children_iface_eid[0, 2] = 3;  children_iface_mask[0, 2] = True;  children_iface_bdir[0, 2] = 1  # e3 RIGHT (facing TR)
    children_iface_eid[0, 3] = 5;  children_iface_mask[0, 3] = True;  children_iface_bdir[0, 3] = 2  # e5 BOTTOM (facing BL)

    # TR (child 1): has e1 (RIGHT), e3 (LEFT, shared with TL)
    children_iface_eid[1, 0] = 1;  children_iface_mask[1, 0] = True;  children_iface_bdir[1, 0] = 1  # e1 RIGHT
    children_iface_eid[1, 1] = 3;  children_iface_mask[1, 1] = True;  children_iface_bdir[1, 1] = 0  # e3 LEFT (facing TL)

    # BL (child 2): has e4 (RIGHT, shared with BR), e5 (TOP, shared with TL)
    children_iface_eid[2, 0] = 4;  children_iface_mask[2, 0] = True;  children_iface_bdir[2, 0] = 1  # e4 RIGHT (facing BR)
    children_iface_eid[2, 1] = 5;  children_iface_mask[2, 1] = True;  children_iface_bdir[2, 1] = 3  # e5 TOP (facing TL)

    # BR (child 3): has e4 (LEFT, shared with BL)
    children_iface_eid[3, 0] = 4;  children_iface_mask[3, 0] = True;  children_iface_bdir[3, 0] = 0  # e4 LEFT (facing BL)

    child_exists = torch.tensor([True, True, True, True])

    return {
        "Ti": Ti, "Tc": Tc,
        "parent_iface_eid": parent_iface_eid,
        "parent_iface_mask": parent_iface_mask,
        "parent_iface_bdir": parent_iface_bdir,
        "parent_cross_eid": parent_cross_eid,
        "parent_cross_mask": parent_cross_mask,
        "parent_cross_child_pair": parent_cross_child_pair,
        "children_iface_eid": children_iface_eid,
        "children_iface_mask": children_iface_mask,
        "children_iface_bdir": children_iface_bdir,
        "child_exists": child_exists,
    }


def _make_diagonal_glue_scenario(Ti: int = 8):
    """Build a parent crossing between diagonal children TL and BR."""
    parent_iface_eid = torch.full((Ti,), -1, dtype=torch.long)
    parent_iface_mask = torch.zeros(Ti, dtype=torch.bool)
    parent_iface_bdir = torch.full((Ti,), -1, dtype=torch.long)

    parent_iface_eid[0] = 10
    parent_iface_mask[0] = True
    parent_iface_bdir[0] = 0  # LEFT, owned by TL

    parent_iface_eid[1] = 11
    parent_iface_mask[1] = True
    parent_iface_bdir[1] = 1  # RIGHT, owned by BR

    Tc = 4
    parent_cross_eid = torch.full((Tc,), -1, dtype=torch.long)
    parent_cross_mask = torch.zeros(Tc, dtype=torch.bool)
    parent_cross_child_pair = torch.full((Tc, 2), -1, dtype=torch.long)
    parent_cross_eid[0] = 12
    parent_cross_mask[0] = True
    parent_cross_child_pair[0] = torch.tensor([0, 3])  # TL ↔ BR diagonal

    children_iface_eid = torch.full((4, Ti), -1, dtype=torch.long)
    children_iface_mask = torch.zeros(4, Ti, dtype=torch.bool)
    children_iface_bdir = torch.full((4, Ti), -1, dtype=torch.long)

    # TL: one parent-boundary iface and one diagonal glue iface.
    children_iface_eid[0, 0] = 10; children_iface_mask[0, 0] = True; children_iface_bdir[0, 0] = 0
    children_iface_eid[0, 1] = 12; children_iface_mask[0, 1] = True; children_iface_bdir[0, 1] = 1

    # BR: one parent-boundary iface and one diagonal glue iface.
    children_iface_eid[3, 0] = 11; children_iface_mask[3, 0] = True; children_iface_bdir[3, 0] = 1
    children_iface_eid[3, 1] = 12; children_iface_mask[3, 1] = True; children_iface_bdir[3, 1] = 3

    child_exists = torch.tensor([True, True, True, True])

    return {
        "Ti": Ti,
        "Tc": Tc,
        "parent_iface_eid": parent_iface_eid,
        "parent_iface_mask": parent_iface_mask,
        "parent_iface_bdir": parent_iface_bdir,
        "parent_cross_eid": parent_cross_eid,
        "parent_cross_mask": parent_cross_mask,
        "parent_cross_child_pair": parent_cross_child_pair,
        "children_iface_eid": children_iface_eid,
        "children_iface_mask": children_iface_mask,
        "children_iface_bdir": children_iface_bdir,
        "child_exists": child_exists,
    }


# ─── Test: Correspondence Maps ────────────────────────────────────────────────

class TestCorrespondenceMaps:

    def test_phi_out_basic(self):
        """Parent outer-boundary ifaces should map to the correct child slots."""
        s = _make_simple_scenario()
        maps = build_correspondence_maps(
            parent_iface_eid=s["parent_iface_eid"],
            parent_iface_mask=s["parent_iface_mask"],
            parent_iface_bdir=s["parent_iface_bdir"],
            parent_cross_eid=s["parent_cross_eid"],
            parent_cross_mask=s["parent_cross_mask"],
            parent_cross_child_pair=s["parent_cross_child_pair"],
            children_iface_eid=s["children_iface_eid"],
            children_iface_mask=s["children_iface_mask"],
            children_iface_bdir=s["children_iface_bdir"],
            child_exists=s["child_exists"],
        )

        # e0 (parent slot 0, LEFT) → TL (child 0), TL's slot 0 (eid=0, LEFT)
        assert maps.phi_out_child[0].item() == 0
        assert maps.phi_out_slot[0].item() == 0

        # e1 (parent slot 1, RIGHT) → TR (child 1), TR's slot 0 (eid=1, RIGHT)
        assert maps.phi_out_child[1].item() == 1
        assert maps.phi_out_slot[1].item() == 0

        # e2 (parent slot 2, TOP) → TL (child 0), TL's slot 1 (eid=2, TOP)
        assert maps.phi_out_child[2].item() == 0
        assert maps.phi_out_slot[2].item() == 1

        # Padding slots should be -1
        assert maps.phi_out_child[3].item() == -1

    def test_phi_sh_basic(self):
        """Shared-boundary ifaces should map to peer child slots via crossings."""
        s = _make_simple_scenario()
        maps = build_correspondence_maps(
            parent_iface_eid=s["parent_iface_eid"],
            parent_iface_mask=s["parent_iface_mask"],
            parent_iface_bdir=s["parent_iface_bdir"],
            parent_cross_eid=s["parent_cross_eid"],
            parent_cross_mask=s["parent_cross_mask"],
            parent_cross_child_pair=s["parent_cross_child_pair"],
            children_iface_eid=s["children_iface_eid"],
            children_iface_mask=s["children_iface_mask"],
            children_iface_bdir=s["children_iface_bdir"],
            child_exists=s["child_exists"],
        )

        # e3 (TL↔TR): TL slot 2 (eid=3, RIGHT) ↔ TR slot 1 (eid=3, LEFT)
        assert maps.phi_sh_peer_child[0, 2].item() == 1  # TL slot 2 → peer is TR
        assert maps.phi_sh_peer_slot[0, 2].item() == 1   # TL slot 2 → TR's slot 1
        assert maps.phi_sh_peer_child[1, 1].item() == 0  # TR slot 1 → peer is TL
        assert maps.phi_sh_peer_slot[1, 1].item() == 2   # TR slot 1 → TL's slot 2

        # e5 (TL↔BL): TL slot 3 (eid=5, BOTTOM) ↔ BL slot 1 (eid=5, TOP)
        assert maps.phi_sh_peer_child[0, 3].item() == 2  # TL slot 3 → peer is BL
        assert maps.phi_sh_peer_slot[0, 3].item() == 1   # TL slot 3 → BL's slot 1
        assert maps.phi_sh_peer_child[2, 1].item() == 0  # BL slot 1 → peer is TL
        assert maps.phi_sh_peer_slot[2, 1].item() == 3   # BL slot 1 → TL's slot 3

        # e4 (BL↔BR): BL slot 0 (eid=4, RIGHT) ↔ BR slot 0 (eid=4, LEFT)
        assert maps.phi_sh_peer_child[2, 0].item() == 3
        assert maps.phi_sh_peer_slot[2, 0].item() == 0
        assert maps.phi_sh_peer_child[3, 0].item() == 2
        assert maps.phi_sh_peer_slot[3, 0].item() == 0

    def test_nonexistent_child(self):
        """Missing child should not break map building."""
        s = _make_simple_scenario()
        s["child_exists"][3] = False  # Remove BR
        maps = build_correspondence_maps(
            parent_iface_eid=s["parent_iface_eid"],
            parent_iface_mask=s["parent_iface_mask"],
            parent_iface_bdir=s["parent_iface_bdir"],
            parent_cross_eid=s["parent_cross_eid"],
            parent_cross_mask=s["parent_cross_mask"],
            parent_cross_child_pair=s["parent_cross_child_pair"],
            children_iface_eid=s["children_iface_eid"],
            children_iface_mask=s["children_iface_mask"],
            children_iface_bdir=s["children_iface_bdir"],
            child_exists=s["child_exists"],
        )
        # e4 (BL↔BR) — BR doesn't exist, so BL slot 0 should have no peer
        assert maps.phi_sh_peer_child[2, 0].item() == -1

    def test_diagonal_crossing_uses_glue_not_shared(self):
        """Diagonal child pairs should populate generic glue maps only."""
        s = _make_diagonal_glue_scenario()
        maps = build_correspondence_maps(
            parent_iface_eid=s["parent_iface_eid"],
            parent_iface_mask=s["parent_iface_mask"],
            parent_iface_bdir=s["parent_iface_bdir"],
            parent_cross_eid=s["parent_cross_eid"],
            parent_cross_mask=s["parent_cross_mask"],
            parent_cross_child_pair=s["parent_cross_child_pair"],
            children_iface_eid=s["children_iface_eid"],
            children_iface_mask=s["children_iface_mask"],
            children_iface_bdir=s["children_iface_bdir"],
            child_exists=s["child_exists"],
        )

        assert maps.phi_glue_peer_child[0, 1].item() == 3
        assert maps.phi_glue_peer_slot[0, 1].item() == 1
        assert maps.phi_glue_peer_child[3, 1].item() == 0
        assert maps.phi_glue_peer_slot[3, 1].item() == 1

        assert maps.phi_sh_peer_child[0, 1].item() == -1
        assert maps.phi_sh_peer_slot[0, 1].item() == -1
        assert maps.phi_sh_peer_child[3, 1].item() == -1
        assert maps.phi_sh_peer_slot[3, 1].item() == -1


# ─── Test: VERIFYTUPLE ────────────────────────────────────────────────────────

class TestVerifyTuple:

    def _build_maps(self):
        s = _make_simple_scenario()
        return s, build_correspondence_maps(
            parent_iface_eid=s["parent_iface_eid"],
            parent_iface_mask=s["parent_iface_mask"],
            parent_iface_bdir=s["parent_iface_bdir"],
            parent_cross_eid=s["parent_cross_eid"],
            parent_cross_mask=s["parent_cross_mask"],
            parent_cross_child_pair=s["parent_cross_child_pair"],
            children_iface_eid=s["children_iface_eid"],
            children_iface_mask=s["children_iface_mask"],
            children_iface_bdir=s["children_iface_bdir"],
            child_exists=s["child_exists"],
        )

    def test_empty_state_feasible(self):
        """Empty parent state + empty children → feasible."""
        s, maps = self._build_maps()
        Ti = s["Ti"]

        parent_a = torch.zeros(Ti, dtype=torch.bool)
        parent_mate = torch.full((Ti,), -1, dtype=torch.long)
        child_a = torch.zeros(4, Ti, dtype=torch.bool)
        child_mate = torch.full((4, Ti), -1, dtype=torch.long)

        ok = verify_tuple(
            parent_a=parent_a,
            parent_mate=parent_mate,
            parent_iface_mask=s["parent_iface_mask"],
            child_a=child_a,
            child_mate=child_mate,
            child_iface_mask=s["children_iface_mask"],
            child_exists=s["child_exists"],
            maps=maps,
        )
        assert ok is True

    def test_diagonal_glue_feasible(self):
        """VERIFYTUPLE should compose through diagonal glue, not only shared sides."""
        s = _make_diagonal_glue_scenario()
        maps = build_correspondence_maps(
            parent_iface_eid=s["parent_iface_eid"],
            parent_iface_mask=s["parent_iface_mask"],
            parent_iface_bdir=s["parent_iface_bdir"],
            parent_cross_eid=s["parent_cross_eid"],
            parent_cross_mask=s["parent_cross_mask"],
            parent_cross_child_pair=s["parent_cross_child_pair"],
            children_iface_eid=s["children_iface_eid"],
            children_iface_mask=s["children_iface_mask"],
            children_iface_bdir=s["children_iface_bdir"],
            child_exists=s["child_exists"],
        )

        Ti = s["Ti"]
        parent_a = torch.zeros(Ti, dtype=torch.bool)
        parent_a[0] = True
        parent_a[1] = True
        parent_mate = torch.full((Ti,), -1, dtype=torch.long)
        parent_mate[0] = 1
        parent_mate[1] = 0

        child_a = torch.zeros(4, Ti, dtype=torch.bool)
        child_mate = torch.full((4, Ti), -1, dtype=torch.long)

        child_a[0, 0] = True
        child_a[0, 1] = True
        child_mate[0, 0] = 1
        child_mate[0, 1] = 0

        child_a[3, 0] = True
        child_a[3, 1] = True
        child_mate[3, 0] = 1
        child_mate[3, 1] = 0

        ok = verify_tuple(
            parent_a=parent_a,
            parent_mate=parent_mate,
            parent_iface_mask=s["parent_iface_mask"],
            child_a=child_a,
            child_mate=child_mate,
            child_iface_mask=s["children_iface_mask"],
            child_exists=s["child_exists"],
            maps=maps,
        )
        assert ok is True

    def test_c1_violation(self):
        """Parent activates slot 0 but child TL does not → C1 fail."""
        s, maps = self._build_maps()
        Ti = s["Ti"]

        parent_a = torch.zeros(Ti, dtype=torch.bool)
        parent_mate = torch.full((Ti,), -1, dtype=torch.long)
        # Activate parent slots 0 and 1, pair them
        parent_a[0] = True; parent_a[1] = True
        parent_mate[0] = 1; parent_mate[1] = 0

        child_a = torch.zeros(4, Ti, dtype=torch.bool)
        child_mate = torch.full((4, Ti), -1, dtype=torch.long)
        # Child TL does NOT activate slot 0 → C1 violation

        ok = verify_tuple(
            parent_a=parent_a,
            parent_mate=parent_mate,
            parent_iface_mask=s["parent_iface_mask"],
            child_a=child_a,
            child_mate=child_mate,
            child_iface_mask=s["children_iface_mask"],
            child_exists=s["child_exists"],
            maps=maps,
        )
        assert ok is False

    def test_c2_violation(self):
        """Shared-boundary slots disagree → C2 fail."""
        s, maps = self._build_maps()
        Ti = s["Ti"]

        parent_a = torch.zeros(Ti, dtype=torch.bool)
        parent_mate = torch.full((Ti,), -1, dtype=torch.long)

        child_a = torch.zeros(4, Ti, dtype=torch.bool)
        child_mate = torch.full((4, Ti), -1, dtype=torch.long)
        # TL activates slot 2 (e3 → shared with TR) but TR does NOT activate slot 1
        child_a[0, 2] = True
        # TR slot 1 stays False → C2 violation

        ok = verify_tuple(
            parent_a=parent_a,
            parent_mate=parent_mate,
            parent_iface_mask=s["parent_iface_mask"],
            child_a=child_a,
            child_mate=child_mate,
            child_iface_mask=s["children_iface_mask"],
            child_exists=s["child_exists"],
            maps=maps,
        )
        assert ok is False

    def test_simple_feasible_path(self):
        """A simple feasible configuration: path enters LEFT, crosses TL→TR, exits RIGHT.

        Parent: slot 0 (LEFT) active, slot 1 (RIGHT) active, mate: 0↔1
        TL: slot 0 (e0, LEFT) active, slot 2 (e3, RIGHT→TR) active, mate: 0↔2
        TR: slot 0 (e1, RIGHT) active, slot 1 (e3, LEFT→TL) active, mate: 0↔1
        BL, BR: all inactive
        """
        s, maps = self._build_maps()
        Ti = s["Ti"]

        parent_a = torch.zeros(Ti, dtype=torch.bool)
        parent_mate = torch.full((Ti,), -1, dtype=torch.long)
        parent_a[0] = True; parent_a[1] = True
        parent_mate[0] = 1; parent_mate[1] = 0

        child_a = torch.zeros(4, Ti, dtype=torch.bool)
        child_mate = torch.full((4, Ti), -1, dtype=torch.long)

        # TL: e0 (slot 0) ↔ e3 (slot 2)
        child_a[0, 0] = True; child_a[0, 2] = True
        child_mate[0, 0] = 2; child_mate[0, 2] = 0

        # TR: e1 (slot 0) ↔ e3 (slot 1)
        child_a[1, 0] = True; child_a[1, 1] = True
        child_mate[1, 0] = 1; child_mate[1, 1] = 0

        ok = verify_tuple(
            parent_a=parent_a,
            parent_mate=parent_mate,
            parent_iface_mask=s["parent_iface_mask"],
            child_a=child_a,
            child_mate=child_mate,
            child_iface_mask=s["children_iface_mask"],
            child_exists=s["child_exists"],
            maps=maps,
        )
        assert ok is True

    def test_wrong_pairing_c3_violation(self):
        """Parent expects mate(0)=1 but the trace produces mate(0)=2 → C3 fail.

        Parent: slot 0 (LEFT) ↔ slot 2 (TOP) active, mate: 0↔2
        But we set children so that tracing from slot 0 ends up at slot 1 instead.
        """
        s, maps = self._build_maps()
        Ti = s["Ti"]

        # Parent: 0↔2
        parent_a = torch.zeros(Ti, dtype=torch.bool)
        parent_mate = torch.full((Ti,), -1, dtype=torch.long)
        parent_a[0] = True; parent_a[1] = True
        parent_mate[0] = 1; parent_mate[1] = 0

        child_a = torch.zeros(4, Ti, dtype=torch.bool)
        child_mate = torch.full((4, Ti), -1, dtype=torch.long)

        # TL: e0 (slot 0) ↔ e2 (slot 1, TOP)
        # This makes tracing from parent slot 0 → TL slot 0 → mate → TL slot 1 → outer boundary parent slot 2
        # But parent expects mate(0)=1, not 2 → C3 violation
        child_a[0, 0] = True; child_a[0, 1] = True
        child_mate[0, 0] = 1; child_mate[0, 1] = 0

        ok = verify_tuple(
            parent_a=parent_a,
            parent_mate=parent_mate,
            parent_iface_mask=s["parent_iface_mask"],
            child_a=child_a,
            child_mate=child_mate,
            child_iface_mask=s["children_iface_mask"],
            child_exists=s["child_exists"],
            maps=maps,
        )
        assert ok is False

    def test_orphan_active_component_rejected(self):
        """Active child slots not covered by outer tracing/glue must be rejected."""
        s = _make_simple_scenario()
        # Delete the two crossings incident to BL so BL can carry an orphan component.
        s["parent_cross_mask"][1] = False  # disable e4: BL↔BR
        s["parent_cross_mask"][2] = False  # disable e5: TL↔BL
        maps = build_correspondence_maps(
            parent_iface_eid=s["parent_iface_eid"],
            parent_iface_mask=s["parent_iface_mask"],
            parent_iface_bdir=s["parent_iface_bdir"],
            parent_cross_eid=s["parent_cross_eid"],
            parent_cross_mask=s["parent_cross_mask"],
            parent_cross_child_pair=s["parent_cross_child_pair"],
            children_iface_eid=s["children_iface_eid"],
            children_iface_mask=s["children_iface_mask"],
            children_iface_bdir=s["children_iface_bdir"],
            child_exists=s["child_exists"],
        )

        Ti = s["Ti"]

        # Parent uses a valid TL-only path from LEFT to TOP.
        parent_a = torch.zeros(Ti, dtype=torch.bool)
        parent_mate = torch.full((Ti,), -1, dtype=torch.long)
        parent_a[0] = True
        parent_a[2] = True
        parent_mate[0] = 2
        parent_mate[2] = 0

        child_a = torch.zeros(4, Ti, dtype=torch.bool)
        child_mate = torch.full((4, Ti), -1, dtype=torch.long)

        # Valid TL path realizing parent 0↔2.
        child_a[0, 0] = True
        child_a[0, 1] = True
        child_mate[0, 0] = 1
        child_mate[0, 1] = 0

        # Extra BL-local component on slots that no longer map to glue nor parent.
        child_a[2, 0] = True
        child_a[2, 1] = True
        child_mate[2, 0] = 1
        child_mate[2, 1] = 0

        ok = verify_tuple(
            parent_a=parent_a,
            parent_mate=parent_mate,
            parent_iface_mask=s["parent_iface_mask"],
            child_a=child_a,
            child_mate=child_mate,
            child_iface_mask=s["children_iface_mask"],
            child_exists=s["child_exists"],
            maps=maps,
        )
        assert ok is False

    def test_root_open_path_rejected(self):
        """Root with no outer boundary must reject connected-but-open child paths."""
        s, maps = self._build_maps()
        Ti = s["Ti"]

        # Root state: no outer boundary slots at all.
        parent_iface_mask = torch.zeros_like(s["parent_iface_mask"])
        parent_a = torch.zeros(Ti, dtype=torch.bool)
        parent_mate = torch.full((Ti,), -1, dtype=torch.long)

        child_a = torch.zeros(4, Ti, dtype=torch.bool)
        child_mate = torch.full((4, Ti), -1, dtype=torch.long)

        # TL carries an open path between two root-boundary-facing slots.
        # This is connected locally, but since root has no parent boundary and
        # these slots have no glue peers, the tuple cannot represent a closed tour.
        child_a[0, 0] = True
        child_a[0, 1] = True
        child_mate[0, 0] = 1
        child_mate[0, 1] = 0

        ok = verify_tuple(
            parent_a=parent_a,
            parent_mate=parent_mate,
            parent_iface_mask=parent_iface_mask,
            child_a=child_a,
            child_mate=child_mate,
            child_iface_mask=s["children_iface_mask"],
            child_exists=s["child_exists"],
            maps=maps,
        )
        assert ok is False


# ─── Test: Noncrossing matching DP ────────────────────────────────────────────

class TestNoncrossingMatching:

    def test_empty(self):
        cost = torch.zeros(4, 4)
        pairs = _noncrossing_min_cost_matching([], cost)
        assert pairs == []

    def test_single_pair(self):
        cost = torch.zeros(4, 4)
        cost[0, 1] = 1.0; cost[1, 0] = 1.0
        cost[0, 3] = 2.0; cost[3, 0] = 2.0
        pairs = _noncrossing_min_cost_matching([0, 1], cost)
        assert pairs is not None
        assert len(pairs) == 1
        assert pairs[0] == (0, 1)

    def test_two_pairs_noncrossing(self):
        """Slots in order [0,1,2,3]. Noncrossing options: (0,1)(2,3) or (0,3)(1,2)."""
        cost = torch.zeros(4, 4)
        # Make (0,1)(2,3) cheaper
        cost[0, 1] = 1.0; cost[1, 0] = 1.0
        cost[2, 3] = 1.0; cost[3, 2] = 1.0
        cost[0, 3] = 10.0; cost[3, 0] = 10.0
        cost[1, 2] = 10.0; cost[2, 1] = 10.0
        pairs = _noncrossing_min_cost_matching([0, 1, 2, 3], cost)
        assert pairs is not None
        assert len(pairs) == 2
        pair_set = {tuple(sorted(p)) for p in pairs}
        assert pair_set == {(0, 1), (2, 3)}

    def test_odd_count_returns_none(self):
        cost = torch.zeros(4, 4)
        result = _noncrossing_min_cost_matching([0, 1, 2], cost)
        assert result is None


# ─── Test: PARSE ──────────────────────────────────────────────────────────────

class TestParse:

    def test_all_low_scores(self):
        """All scores below threshold → empty activations."""
        s = _make_simple_scenario()
        Ti = s["Ti"]
        scores = torch.full((4, Ti), 0.1)
        child_a, child_mate = parse_continuous(
            scores=scores,
            child_iface_mask=s["children_iface_mask"],
            child_iface_bdir=s["children_iface_bdir"],
            child_exists=s["child_exists"],
            maps=build_correspondence_maps(**{k: s[k] for k in [
                "parent_iface_eid", "parent_iface_mask", "parent_iface_bdir",
                "parent_cross_eid", "parent_cross_mask", "parent_cross_child_pair",
                "children_iface_eid", "children_iface_mask", "children_iface_bdir",
                "child_exists",
            ]}),
        )
        assert child_a.sum().item() == 0

    def test_high_scores_produce_activations(self):
        """High scores on two shared-boundary slots should activate both."""
        s = _make_simple_scenario()
        Ti = s["Ti"]
        maps = build_correspondence_maps(**{k: s[k] for k in [
            "parent_iface_eid", "parent_iface_mask", "parent_iface_bdir",
            "parent_cross_eid", "parent_cross_mask", "parent_cross_child_pair",
            "children_iface_eid", "children_iface_mask", "children_iface_bdir",
            "child_exists",
        ]})

        scores = torch.full((4, Ti), 0.1)
        # High scores for TL slot 0 (e0) and TL slot 2 (e3)
        scores[0, 0] = 0.9
        scores[0, 2] = 0.9
        # High score for TR slot 1 (e3, shared with TL slot 2) — should be enforced
        scores[1, 1] = 0.9
        # High score for TR slot 0 (e1)
        scores[1, 0] = 0.9

        child_a, child_mate = parse_continuous(
            scores=scores,
            child_iface_mask=s["children_iface_mask"],
            child_iface_bdir=s["children_iface_bdir"],
            child_exists=s["child_exists"],
            maps=maps,
        )

        # TL should have 2 active slots: 0 and 2
        assert child_a[0, 0].item() is True
        assert child_a[0, 2].item() is True
        # TR should have 2 active slots: 0 and 1
        assert child_a[1, 0].item() is True
        assert child_a[1, 1].item() is True
        # Parity should be even
        for qi in range(4):
            assert child_a[qi].sum().item() % 2 == 0

    def test_parity_enforcement(self):
        """Odd number of high-score slots → parity fixed to even."""
        s = _make_simple_scenario()
        Ti = s["Ti"]
        maps = build_correspondence_maps(**{k: s[k] for k in [
            "parent_iface_eid", "parent_iface_mask", "parent_iface_bdir",
            "parent_cross_eid", "parent_cross_mask", "parent_cross_child_pair",
            "children_iface_eid", "children_iface_mask", "children_iface_bdir",
            "child_exists",
        ]})

        scores = torch.full((4, Ti), 0.1)
        # Only 1 high score for TL (odd) → should be fixed to 0 or 2
        scores[0, 0] = 0.9

        child_a, _ = parse_continuous(
            scores=scores,
            child_iface_mask=s["children_iface_mask"],
            child_iface_bdir=s["children_iface_bdir"],
            child_exists=s["child_exists"],
            maps=maps,
        )

        for qi in range(4):
            active_count = child_a[qi].sum().item()
            assert active_count % 2 == 0, f"Child {qi} has odd active count {active_count}"

    def test_c1_forced_slot_not_dropped_by_parity(self):
        """Parity repair should not deactivate a C1-required active slot."""
        s = _make_simple_scenario()
        Ti = s["Ti"]
        maps = build_correspondence_maps(**{k: s[k] for k in [
            "parent_iface_eid", "parent_iface_mask", "parent_iface_bdir",
            "parent_cross_eid", "parent_cross_mask", "parent_cross_child_pair",
            "children_iface_eid", "children_iface_mask", "children_iface_bdir",
            "child_exists",
        ]})

        scores = torch.full((4, Ti), 0.1)
        parent_a = torch.zeros(Ti, dtype=torch.bool)
        parent_a[0] = True  # Forces TL slot 0 active via C1

        child_a, _ = parse_continuous(
            scores=scores,
            child_iface_mask=s["children_iface_mask"],
            child_iface_bdir=s["children_iface_bdir"],
            child_exists=s["child_exists"],
            maps=maps,
            parent_a=parent_a,
            parent_iface_mask=s["parent_iface_mask"],
        )

        assert child_a[0, 0].item() is True, "C1-required slot should remain active"


# ─── Test: Leaf exact solver ──────────────────────────────────────────────────

class TestLeafExactSolve:

    def test_empty_leaf_empty_state(self):
        """Leaf with 0 points, empty state → cost 0."""
        Ti = 4
        S = 2  # state 0: empty, state 1: some active
        points_xy = torch.zeros(0, 2)
        point_mask = torch.zeros(0, dtype=torch.bool)
        iface_eid = torch.full((Ti,), -1, dtype=torch.long)
        iface_mask = torch.zeros(Ti, dtype=torch.bool)
        iface_bdir = torch.full((Ti,), -1, dtype=torch.long)
        iface_feat6 = torch.zeros(Ti, 6)
        state_used = torch.zeros(S, Ti, dtype=torch.bool)
        state_mate = torch.full((S, Ti), -1, dtype=torch.long)
        state_mask = torch.tensor([True, False])
        box_xy = torch.tensor([0.0, 0.0, 1.0, 1.0])

        costs = leaf_exact_solve(
            points_xy=points_xy,
            point_mask=point_mask,
            iface_eid=iface_eid,
            iface_mask=iface_mask,
            iface_boundary_dir=iface_bdir,
            iface_feat6=iface_feat6,
            state_used_iface=state_used,
            state_mate=state_mate,
            state_mask=state_mask,
            box_xy=box_xy,
        )
        assert costs[0].item() == 0.0
        assert costs[1].item() == float("inf")  # masked out

    def test_two_points_closed_tour(self):
        """Root leaf: empty state encodes the closed tour through all points."""
        Ti = 4
        S = 1
        points_xy = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        point_mask = torch.tensor([True, True])
        iface_eid = torch.full((Ti,), -1, dtype=torch.long)
        iface_mask = torch.zeros(Ti, dtype=torch.bool)
        iface_bdir = torch.full((Ti,), -1, dtype=torch.long)
        iface_feat6 = torch.zeros(Ti, 6)
        state_used = torch.zeros(S, Ti, dtype=torch.bool)
        state_mate = torch.full((S, Ti), -1, dtype=torch.long)
        state_mask = torch.tensor([True])
        box_xy = torch.tensor([0.0, 0.0, 2.0, 2.0])

        costs = leaf_exact_solve(
            points_xy=points_xy,
            point_mask=point_mask,
            iface_eid=iface_eid,
            iface_mask=iface_mask,
            iface_boundary_dir=iface_bdir,
            iface_feat6=iface_feat6,
            state_used_iface=state_used,
            state_mate=state_mate,
            state_mask=state_mask,
            box_xy=box_xy,
            is_root=True,
        )
        assert abs(costs[0].item() - 2.0) < 1e-5

    def test_nonroot_empty_state_with_points_is_infeasible(self):
        """Non-root empty state with interior points must be rejected by C4."""
        Ti = 4
        S = 1
        points_xy = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        point_mask = torch.tensor([True, True])
        iface_eid = torch.full((Ti,), -1, dtype=torch.long)
        iface_mask = torch.zeros(Ti, dtype=torch.bool)
        iface_bdir = torch.full((Ti,), -1, dtype=torch.long)
        iface_feat6 = torch.zeros(Ti, 6)
        state_used = torch.zeros(S, Ti, dtype=torch.bool)
        state_mate = torch.full((S, Ti), -1, dtype=torch.long)
        state_mask = torch.tensor([True])
        box_xy = torch.tensor([0.0, 0.0, 2.0, 2.0])

        costs = leaf_exact_solve(
            points_xy=points_xy,
            point_mask=point_mask,
            iface_eid=iface_eid,
            iface_mask=iface_mask,
            iface_boundary_dir=iface_bdir,
            iface_feat6=iface_feat6,
            state_used_iface=state_used,
            state_mate=state_mate,
            state_mask=state_mask,
            box_xy=box_xy,
            is_root=False,
        )
        assert costs[0].item() == float("inf")

    def test_one_path_segment(self):
        """1 path segment through a leaf with 1 point.

        Box from abs (0,0) to (1,1): center=(0.5,0.5), half-size=(0.5,0.5).
        Root-normalized (root = this box): box_xy = [0, 0, 1, 1].
        Two boundary sites at abs (0, 0.5) and abs (1, 0.5).
        One interior point at abs (0.5, 0.5).

        All coordinates passed to leaf_exact_solve are node-relative:
          point  (0.5,0.5) → rel = ((0.5-0.5)/0.5, (0.5-0.5)/0.5) = (0, 0)
          site_a (0, 0.5)  → inter_rel = (-1, 0)
          site_b (1, 0.5)  → inter_rel = ( 1, 0)

        Optimal path: site_a → point → site_b, length = 0.5 + 0.5 = 1.0.
        """
        Ti = 4
        S = 1
        # points_xy is node-relative: abs (0.5,0.5) → rel (0, 0)
        points_xy = torch.tensor([[0.0, 0.0]])
        point_mask = torch.tensor([True])

        iface_eid = torch.full((Ti,), -1, dtype=torch.long)
        iface_mask = torch.zeros(Ti, dtype=torch.bool)
        iface_bdir = torch.full((Ti,), -1, dtype=torch.long)
        iface_feat6 = torch.zeros(Ti, 6)

        # Two active interfaces at slot 0 and 1
        iface_eid[0] = 0; iface_mask[0] = True; iface_bdir[0] = 0  # LEFT
        iface_eid[1] = 1; iface_mask[1] = True; iface_bdir[1] = 1  # RIGHT

        # inter_rel coordinates (node-relative):
        # Site at abs (0, 0.5): inter_rel = ((0-0.5)/0.5, (0.5-0.5)/0.5) = (-1, 0)
        iface_feat6[0, 2] = -1.0; iface_feat6[0, 3] = 0.0
        # Site at abs (1, 0.5): inter_rel = ((1-0.5)/0.5, (0.5-0.5)/0.5) = (1, 0)
        iface_feat6[1, 2] = 1.0; iface_feat6[1, 3] = 0.0

        state_used = torch.zeros(S, Ti, dtype=torch.bool)
        state_used[0, 0] = True; state_used[0, 1] = True
        state_mate = torch.full((S, Ti), -1, dtype=torch.long)
        state_mate[0, 0] = 1; state_mate[0, 1] = 0
        state_mask = torch.tensor([True])
        box_xy = torch.tensor([0.0, 0.0, 1.0, 1.0])

        costs = leaf_exact_solve(
            points_xy=points_xy,
            point_mask=point_mask,
            iface_eid=iface_eid,
            iface_mask=iface_mask,
            iface_boundary_dir=iface_bdir,
            iface_feat6=iface_feat6,
            state_used_iface=state_used,
            state_mate=state_mate,
            state_mask=state_mask,
            box_xy=box_xy,
        )
        assert abs(costs[0].item() - 1.0) < 1e-5, f"Expected 1.0, got {costs[0].item()}"


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_all_tests():
    """Simple test runner (no pytest dependency)."""
    test_classes = [
        TestCorrespondenceMaps,
        TestVerifyTuple,
        TestNoncrossingMatching,
        TestParse,
        TestLeafExactSolve,
    ]
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

    print(f"\n{'='*60}")
    print(f"Total: {total}  Passed: {passed}  Failed: {failed}")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  {name}: {err}")
    return failed == 0


if __name__ == "__main__":
    print("Running dp_core.py unit tests...\n")
    success = run_all_tests()
    sys.exit(0 if success else 1)
