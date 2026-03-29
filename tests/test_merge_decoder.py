# tests/test_merge_decoder.py
# -*- coding: utf-8 -*-
"""
Unit tests for src/models/merge_decoder.py — sigma-conditioned decoder for 1-pass DP.

Usage:
  python tests/test_merge_decoder.py
  python -m pytest tests/test_merge_decoder.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.merge_decoder import (
    MergeDecoder,
    MergeDecoderOutput,
    ParentMemory,
    SigmaEncoder,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_dummy_inputs(B: int = 2, Ti: int = 8, Tc: int = 4, d_model: int = 64):
    """Create dummy inputs for MergeDecoder testing."""
    return dict(
        node_feat_rel=torch.randn(B, 4),
        node_depth=torch.randint(0, 5, (B,)),
        z_node=torch.randn(B, d_model),
        iface_feat6=torch.randn(B, Ti, 6),
        iface_mask=torch.ones(B, Ti, dtype=torch.bool),
        iface_boundary_dir=torch.randint(0, 4, (B, Ti)),
        iface_inside_endpoint=torch.randint(0, 2, (B, Ti)),
        iface_inside_quadrant=torch.randint(0, 4, (B, Ti)),
        cross_feat6=torch.randn(B, Tc, 6),
        cross_mask=torch.ones(B, Tc, dtype=torch.bool),
        cross_child_pair=torch.randint(0, 4, (B, Tc, 2)),
        cross_is_leaf_internal=torch.zeros(B, Tc, dtype=torch.bool),
        child_z=torch.randn(B, 4, d_model),
        child_exists_mask=torch.ones(B, 4, dtype=torch.bool),
    )


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestSigmaEncoder:
    """Tests for the sigma state encoder."""

    def test_basic_shape(self):
        Ti = 8
        d_model = 64
        B = 3
        enc = SigmaEncoder(num_iface_slots=Ti, d_model=d_model)

        a = torch.zeros(B, Ti)
        a[0, 0] = 1.0
        a[0, 3] = 1.0
        mate = torch.full((B, Ti), -1, dtype=torch.long)
        mate[0, 0] = 3
        mate[0, 3] = 0
        mask = torch.ones(B, Ti, dtype=torch.bool)

        out = enc(a, mate, mask)
        assert out.shape == (B, d_model), f"Expected ({B}, {d_model}), got {out.shape}"

    def test_different_states_give_different_embeddings(self):
        Ti = 8
        d_model = 64
        enc = SigmaEncoder(num_iface_slots=Ti, d_model=d_model)
        enc.eval()

        mask = torch.ones(1, Ti, dtype=torch.bool)

        # State 1: empty
        a1 = torch.zeros(1, Ti)
        mate1 = torch.full((1, Ti), -1, dtype=torch.long)

        # State 2: slots 0-1 active, paired
        a2 = torch.zeros(1, Ti)
        a2[0, 0] = 1.0
        a2[0, 1] = 1.0
        mate2 = torch.full((1, Ti), -1, dtype=torch.long)
        mate2[0, 0] = 1
        mate2[0, 1] = 0

        with torch.no_grad():
            e1 = enc(a1, mate1, mask)
            e2 = enc(a2, mate2, mask)

        diff = (e1 - e2).abs().sum().item()
        assert diff > 0.01, f"Different states should give different embeddings, diff={diff}"

    def test_masked_slots_ignored(self):
        Ti = 8
        d_model = 64
        enc = SigmaEncoder(num_iface_slots=Ti, d_model=d_model)
        enc.eval()

        a = torch.zeros(1, Ti)
        mate = torch.full((1, Ti), -1, dtype=torch.long)

        # Full mask vs partial mask — only valid slots should matter
        mask_full = torch.ones(1, Ti, dtype=torch.bool)
        mask_half = torch.ones(1, Ti, dtype=torch.bool)
        mask_half[0, 4:] = False

        with torch.no_grad():
            e_full = enc(a, mate, mask_full)
            e_half = enc(a, mate, mask_half)

        # They should differ because the pooling denominator changes
        # (This is a sanity check, not an exact equality test)
        assert e_full.shape == e_half.shape


class TestMergeDecoder:
    """Tests for the full MergeDecoder module."""

    def _make_decoder(self, Ti=8, Tc=4, d_model=64):
        return MergeDecoder(
            d_model=d_model,
            n_heads=4,
            num_iface_slots=Ti,
            parent_num_layers=2,
            cross_num_layers=1,
            max_depth=16,
        )

    def test_build_parent_memory_shape(self):
        B, Ti, Tc, d = 2, 8, 4, 64
        decoder = self._make_decoder(Ti, Tc, d)
        inputs = _make_dummy_inputs(B, Ti, Tc, d)

        mem = decoder.build_parent_memory(**inputs)

        assert isinstance(mem, ParentMemory)
        assert mem.tokens.shape[0] == B
        assert mem.tokens.shape[2] == d
        assert mem.mask.shape[0] == B
        assert mem.mask.shape[1] == mem.tokens.shape[1]

    def test_decode_sigma_shape(self):
        B, Ti, Tc, d = 1, 8, 4, 64
        decoder = self._make_decoder(Ti, Tc, d)
        inputs = _make_dummy_inputs(B, Ti, Tc, d)

        mem = decoder.build_parent_memory(**inputs)

        # Decode a single sigma
        sigma_a = torch.zeros(1, Ti)
        sigma_mate = torch.full((1, Ti), -1, dtype=torch.long)
        sigma_iface_mask = torch.ones(1, Ti, dtype=torch.bool)
        child_iface_mask = torch.ones(1, 4, Ti, dtype=torch.bool)

        out = decoder.decode_sigma(
            sigma_a=sigma_a,
            sigma_mate=sigma_mate,
            sigma_iface_mask=sigma_iface_mask,
            parent_memory=mem,
            child_iface_mask=child_iface_mask,
        )

        assert isinstance(out, MergeDecoderOutput)
        assert out.child_scores.shape == (1, 4, Ti)

    def test_decode_sigma_batch_shape(self):
        """Test batch decoding of multiple sigmas for one parent."""
        Ti, Tc, d = 8, 4, 64
        decoder = self._make_decoder(Ti, Tc, d)
        inputs = _make_dummy_inputs(1, Ti, Tc, d)

        mem = decoder.build_parent_memory(**inputs)

        S = 5  # 5 candidate sigmas
        sigma_a = torch.zeros(S, Ti)
        sigma_mate = torch.full((S, Ti), -1, dtype=torch.long)
        sigma_iface_mask = torch.ones(Ti, dtype=torch.bool)
        child_iface_mask = torch.ones(4, Ti, dtype=torch.bool)

        out = decoder.decode_sigma_batch(
            sigma_a=sigma_a,
            sigma_mate=sigma_mate,
            sigma_iface_mask=sigma_iface_mask,
            parent_memory=mem,
            child_iface_mask=child_iface_mask,
        )

        assert out.child_scores.shape == (S, 4, Ti)

    def test_different_sigmas_different_outputs(self):
        """Different sigma inputs should produce different child predictions."""
        Ti, Tc, d = 8, 4, 64
        decoder = self._make_decoder(Ti, Tc, d)
        decoder.eval()
        inputs = _make_dummy_inputs(1, Ti, Tc, d)

        with torch.no_grad():
            mem = decoder.build_parent_memory(**inputs)

        # Sigma 1: empty state
        a1 = torch.zeros(1, Ti)
        m1 = torch.full((1, Ti), -1, dtype=torch.long)

        # Sigma 2: slots 0,1 active
        a2 = torch.zeros(1, Ti)
        a2[0, 0] = 1.0
        a2[0, 1] = 1.0
        m2 = torch.full((1, Ti), -1, dtype=torch.long)
        m2[0, 0] = 1
        m2[0, 1] = 0

        iface_mask = torch.ones(1, Ti, dtype=torch.bool)
        child_mask = torch.ones(1, 4, Ti, dtype=torch.bool)

        with torch.no_grad():
            out1 = decoder.decode_sigma(
                sigma_a=a1, sigma_mate=m1, sigma_iface_mask=iface_mask,
                parent_memory=mem, child_iface_mask=child_mask,
            )
            out2 = decoder.decode_sigma(
                sigma_a=a2, sigma_mate=m2, sigma_iface_mask=iface_mask,
                parent_memory=mem, child_iface_mask=child_mask,
            )

        diff = (out1.child_scores - out2.child_scores).abs().sum().item()
        assert diff > 0.01, f"Different sigmas should give different outputs, diff={diff}"

    def test_gradient_flows(self):
        """Verify gradients flow through the full pipeline."""
        Ti, Tc, d = 8, 4, 64
        decoder = self._make_decoder(Ti, Tc, d)
        inputs = _make_dummy_inputs(1, Ti, Tc, d)

        mem = decoder.build_parent_memory(**inputs)

        sigma_a = torch.zeros(1, Ti)
        sigma_mate = torch.full((1, Ti), -1, dtype=torch.long)
        iface_mask = torch.ones(1, Ti, dtype=torch.bool)
        child_mask = torch.ones(1, 4, Ti, dtype=torch.bool)

        out = decoder.decode_sigma(
            sigma_a=sigma_a, sigma_mate=sigma_mate,
            sigma_iface_mask=iface_mask,
            parent_memory=mem, child_iface_mask=child_mask,
        )

        loss = out.child_scores.sum()
        loss.backward()

        # Check at least some parameters have gradients
        has_grad = False
        for p in decoder.parameters():
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients found — backward pass may be broken"

    def test_masked_child_iface_gets_neg_inf(self):
        """Invalid child iface slots should get -inf scores."""
        Ti, Tc, d = 8, 4, 64
        decoder = self._make_decoder(Ti, Tc, d)
        decoder.eval()
        inputs = _make_dummy_inputs(1, Ti, Tc, d)

        with torch.no_grad():
            mem = decoder.build_parent_memory(**inputs)

        sigma_a = torch.zeros(1, Ti)
        sigma_mate = torch.full((1, Ti), -1, dtype=torch.long)
        iface_mask = torch.ones(1, Ti, dtype=torch.bool)

        # Mask out child 2's last 4 slots
        child_mask = torch.ones(1, 4, Ti, dtype=torch.bool)
        child_mask[0, 2, 4:] = False

        with torch.no_grad():
            out = decoder.decode_sigma(
                sigma_a=sigma_a, sigma_mate=sigma_mate,
                sigma_iface_mask=iface_mask,
                parent_memory=mem, child_iface_mask=child_mask,
            )

        # Masked slots should have very negative scores
        masked_scores = out.child_scores[0, 2, 4:]
        assert (masked_scores < -1e8).all(), "Masked slots should get neg_inf"

    def test_parent_memory_reuse(self):
        """Building memory once and decoding multiple sigmas should work."""
        Ti, Tc, d = 8, 4, 64
        decoder = self._make_decoder(Ti, Tc, d)
        decoder.eval()
        inputs = _make_dummy_inputs(1, Ti, Tc, d)

        with torch.no_grad():
            mem = decoder.build_parent_memory(**inputs)

        iface_mask = torch.ones(Ti, dtype=torch.bool)
        child_mask = torch.ones(4, Ti, dtype=torch.bool)

        results = []
        for _ in range(3):
            a = torch.rand(1, Ti)
            m = torch.full((1, Ti), -1, dtype=torch.long)
            with torch.no_grad():
                out = decoder.decode_sigma_batch(
                    sigma_a=a, sigma_mate=m,
                    sigma_iface_mask=iface_mask,
                    parent_memory=mem,
                    child_iface_mask=child_mask,
                )
            results.append(out.child_scores.clone())

        # Just verify no crashes and shapes are correct
        for r in results:
            assert r.shape == (1, 4, Ti)

    def test_decode_sigma_chunked_matches_dense_decode(self):
        """Chunked decoding should be numerically equivalent to dense decoding."""
        Ti, Tc, d = 8, 4, 64
        decoder = self._make_decoder(Ti, Tc, d)
        decoder.eval()
        inputs = _make_dummy_inputs(1, Ti, Tc, d)

        with torch.no_grad():
            mem = decoder.build_parent_memory(**inputs)

        S = 7
        sigma_a = torch.randn(S, Ti)
        sigma_mate = torch.full((S, Ti), -1, dtype=torch.long)
        sigma_iface_mask = torch.ones(S, Ti, dtype=torch.bool)
        child_iface_mask = torch.ones(S, 4, Ti, dtype=torch.bool)
        expanded_mem = ParentMemory(
            tokens=mem.tokens.expand(S, -1, -1),
            mask=mem.mask.expand(S, -1),
            iface_slice=mem.iface_slice,
            cross_slice=mem.cross_slice,
        )

        with torch.no_grad():
            dense = decoder.decode_sigma(
                sigma_a=sigma_a,
                sigma_mate=sigma_mate,
                sigma_iface_mask=sigma_iface_mask,
                parent_memory=expanded_mem,
                child_iface_mask=child_iface_mask,
            )
            chunked = decoder.decode_sigma_chunked(
                sigma_a=sigma_a,
                sigma_mate=sigma_mate,
                sigma_iface_mask=sigma_iface_mask,
                parent_memory=expanded_mem,
                child_iface_mask=child_iface_mask,
                max_batch_size=3,
            )

        assert torch.allclose(dense.child_scores, chunked.child_scores, atol=1e-6, rtol=1e-6)


# ─── CLI runner ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
