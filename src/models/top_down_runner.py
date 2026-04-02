# src/models/top_down_runner.py
# -*- coding: utf-8 -*-
"""
Top-down runner (boundary-condition logits DP).

State passed from parent to child:
  bc_iface_logit[v, i]  (shape [Ti]) for each node v, where i indexes v's padded interfaces.

For each reachable node v at depth d, we call decoder with:
  - bc_in_iface_logit[v]                         : [B,Ti]
  - v-local tokens (iface/cross)                 : [B,Ti,*], [B,Tc,*]
  - v bottom-up latent z[v]                      : [B,d]
  - children latents z[child_q]                  : [B,4,d]
  - children local iface tokens (stable order)   : [B,4,Ti,*]
and write back:
  - iface_logit[v], cross_logit[v]
  - bc_iface_logit[child_q] <- child_iface_logit[v,q]

This runner assumes:
  - tokens are packed by NodeTokenPacker.pack_batch()
  - tokens.root_id is [B] and points to each graph root (parent_index < 0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor

try:
    from src.models.node_token_packer import PackedBatch, PackedNodeTokens
    from src.models.top_down_decoder import TopDownDecoder, TopDownDecoderOutput
    from src.models.bc_state_catalog import state_logits_to_expected_iface_usage
except Exception:  # pragma: no cover
    from .node_token_packer import PackedBatch, PackedNodeTokens
    from .top_down_decoder import TopDownDecoder, TopDownDecoderOutput
    from .bc_state_catalog import state_logits_to_expected_iface_usage


_NEG_INF = -1.0e9


@dataclass
class TopDownResult:
    """
    Global-packed outputs over the whole batch.

    Shapes:
      iface_logit:     [total_M, Ti]
      cross_logit:     [total_M, Tc]
      bc_iface_logit:  [total_M, Ti]    (top-down DP state used as decoder input per node)
      root_ids:        [B]
      node_ptr:        [B+1]
      aux:             diagnostics
    """
    iface_logit: Tensor
    cross_logit: Tensor
    bc_iface_logit: Tensor
    bc_state_logit: Optional[Tensor]
    root_ids: Tensor
    node_ptr: Tensor
    aux: Dict[str, Tensor]


class TopDownTreeRunner:
    def __init__(
        self,
        *,
        validate_reachability: bool = True,
        max_nodes_per_chunk: Optional[int] = None,
    ) -> None:
        self.validate_reachability = bool(validate_reachability)
        self.max_nodes_per_chunk = (None if max_nodes_per_chunk is None else int(max_nodes_per_chunk))

    @staticmethod
    def _iter_chunks(nids: Tensor, chunk_size: Optional[int]):
        if chunk_size is None or int(chunk_size) <= 0 or int(nids.numel()) <= int(chunk_size):
            yield nids
            return
        chunk_size = int(chunk_size)
        total = int(nids.numel())
        for start in range(0, total, chunk_size):
            yield nids[start:start + chunk_size]

    def run_batch(
        self,
        *,
        packed: PackedBatch,
        z: Tensor,                 # [total_M, d_model]
        decoder: TopDownDecoder,
    ) -> TopDownResult:
        tokens = packed.tokens
        node_ptr = packed.node_ptr

        if node_ptr.dim() != 1 or node_ptr.numel() < 2:
            raise ValueError("packed.node_ptr must be 1D with length B+1.")
        B = int(node_ptr.numel() - 1)

        root_ids = tokens.root_id.reshape(-1).to(dtype=torch.long)
        if root_ids.numel() != B:
            raise ValueError(f"Expected root_id shape [B]={B}, got {tuple(root_ids.shape)}")
        if (tokens.tree_parent_index[root_ids] >= 0).any().item():
            bad = torch.nonzero(tokens.tree_parent_index[root_ids] >= 0, as_tuple=False).view(-1)[:10].tolist()
            raise ValueError(f"Some root_ids are not roots (up to 10 indices): {bad}")

        return self._run(
            tokens=tokens,
            z=z,
            decoder=decoder,
            node_ptr=node_ptr,
            root_ids=root_ids,
            state_catalog=getattr(packed, "state_catalog", None),
        )

    def run_single(
        self,
        *,
        tokens: PackedNodeTokens,
        z: Tensor,
        decoder: TopDownDecoder,
        node_ptr: Optional[Tensor] = None,
    ) -> TopDownResult:
        """
        Optional single-graph entry. Prefer run_batch(pack_batch([data]), ...).
        Requires tokens.root_id to exist and have length 1.
        """
        root_ids = tokens.root_id.reshape(-1).to(dtype=torch.long)
        if root_ids.numel() != 1:
            raise ValueError(f"run_single expects tokens.root_id length 1, got {tuple(root_ids.shape)}")

        M = int(tokens.tree_node_depth.numel())
        if node_ptr is None:
            node_ptr = torch.tensor([0, M], device=tokens.tree_node_depth.device, dtype=torch.long)

        if tokens.tree_parent_index[root_ids[0]].item() >= 0:
            raise ValueError("tokens.root_id is not a root (tree_parent_index[root_id] must be < 0).")

        return self._run(tokens=tokens, z=z, decoder=decoder, node_ptr=node_ptr, root_ids=root_ids, state_catalog=None)

    def _run(
        self,
        *,
        tokens: PackedNodeTokens,
        z: Tensor,                      # [total_M, d_model]
        decoder: TopDownDecoder,
        node_ptr: Tensor,               # [B+1]
        root_ids: Tensor,               # [B]
        state_catalog: Optional[object],
    ) -> TopDownResult:
        if z.dim() != 2:
            raise ValueError("z must be 2D [total_M, d_model].")
        total_M = int(tokens.tree_node_depth.numel())
        if z.size(0) != total_M:
            raise ValueError(f"z.size(0)={z.size(0)} must equal total_M={total_M}.")

        device = z.device
        dtype = z.dtype

        if tokens.iface_mask.dim() != 2:
            raise ValueError("tokens.iface_mask must be [total_M, Ti].")
        if tokens.cross_mask.dim() != 2:
            raise ValueError("tokens.cross_mask must be [total_M, Tc].")

        Ti = int(tokens.iface_mask.size(1))
        Tc = int(tokens.cross_mask.size(1))

        # outputs
        iface_logit = torch.full((total_M, Ti), _NEG_INF, device=device, dtype=dtype)
        cross_logit = torch.full((total_M, Tc), _NEG_INF, device=device, dtype=dtype)

        # DP state: bc on interfaces
        bc_iface_logit = torch.full((total_M, Ti), _NEG_INF, device=device, dtype=dtype)
        bc_state_logit: Optional[Tensor] = None

        use_matching = str(getattr(decoder, "state_mode", "iface")) == "matching"
        state_used_iface: Optional[Tensor] = None
        if use_matching:
            if getattr(tokens, "state_mask", None) is None or state_catalog is None:
                raise ValueError("matching mode requires packed state catalog and per-node state_mask.")
            state_used_iface = state_catalog.used_iface.to(device=device)
            S = int(state_used_iface.shape[0])
            bc_state_logit = torch.full((total_M, S), _NEG_INF, device=device, dtype=dtype)

        # init root bc
        root_iface_mask = tokens.iface_mask[root_ids].bool()  # [B,Ti]
        if use_matching:
            assert bc_state_logit is not None
            empty_index = int(state_catalog.empty_index)
            root_state_mask = tokens.state_mask[root_ids].bool()
            bc_root_state = torch.full((root_ids.numel(), S), _NEG_INF, device=device, dtype=dtype)
            bc_root_state[:, empty_index] = 0.0
            bc_root_state = torch.where(root_state_mask, bc_root_state, torch.full_like(bc_root_state, _NEG_INF))
            bc_state_logit = bc_state_logit.index_copy(0, root_ids, bc_root_state)
            bc_root_iface = state_logits_to_expected_iface_usage(
                state_logit=bc_root_state,
                state_mask=root_state_mask,
                state_used_iface=state_used_iface,
            )
            bc_root_iface = torch.where(root_iface_mask, bc_root_iface, torch.full_like(bc_root_iface, _NEG_INF))
            bc_iface_logit = bc_iface_logit.index_copy(0, root_ids, bc_root_iface)
        else:
            bc_root = torch.full((root_ids.numel(), Ti), _NEG_INF, device=device, dtype=dtype)
            bc_root = bc_root.masked_fill(root_iface_mask, 0.0)
            bc_iface_logit = bc_iface_logit.index_copy(0, root_ids, bc_root)

        # reachability
        reached = torch.zeros((total_M,), device=device, dtype=torch.bool)
        reached[root_ids] = True

        max_depth = int(tokens.tree_node_depth.max().item()) if total_M > 0 else 0

        for d in range(max_depth + 1):
            node_ids = torch.nonzero((tokens.tree_node_depth == d) & reached, as_tuple=False).view(-1)
            if node_ids.numel() == 0:
                continue
            for node_chunk in self._iter_chunks(node_ids, self.max_nodes_per_chunk):
                # ---- gather parent node inputs ----
                node_feat_rel = tokens.tree_node_feat_rel[node_chunk]     # [B,4]
                node_depth = tokens.tree_node_depth[node_chunk]           # [B]
                z_node = z[node_chunk]                                    # [B,d]

                bc_in = bc_iface_logit[node_chunk]                        # [B,Ti]
                state_mask = tokens.state_mask[node_chunk].bool() if use_matching else None
                bc_in_state = bc_state_logit[node_chunk] if use_matching and bc_state_logit is not None else None

                iface_feat6 = tokens.iface_feat6[node_chunk]              # [B,Ti,6]
                iface_mask = tokens.iface_mask[node_chunk].bool()         # [B,Ti]
                iface_boundary_dir = tokens.iface_boundary_dir[node_chunk]
                iface_inside_endpoint = tokens.iface_inside_endpoint[node_chunk]
                iface_inside_quadrant = tokens.iface_inside_quadrant[node_chunk]

                cross_feat6 = tokens.cross_feat6[node_chunk]              # [B,Tc,6]
                cross_mask = tokens.cross_mask[node_chunk].bool()         # [B,Tc]
                cross_child_pair = tokens.cross_child_pair[node_chunk]    # [B,Tc,2]
                cross_is_leaf_internal = tokens.cross_is_leaf_internal[node_chunk]  # [B,Tc]

                # ---- gather children inputs ----
                children = tokens.tree_children_index[node_chunk].long()  # [B,4]
                child_exists = (children >= 0)                            # [B,4]
                children_clamped = children.clamp_min(0)

                child_z = z[children_clamped]                             # [B,4,d]
                child_z = child_z * child_exists.unsqueeze(-1).to(dtype=dtype)

                child_node_feat_rel = tokens.tree_node_feat_rel[children_clamped]  # [B,4,4]
                child_node_feat_rel = child_node_feat_rel * child_exists.unsqueeze(-1).to(dtype=dtype)

                child_node_depth = tokens.tree_node_depth[children_clamped]        # [B,4]
                child_node_depth = child_node_depth * child_exists.to(dtype=child_node_depth.dtype)

                # child iface tokens (stable order per child is guaranteed by packer)
                child_iface_feat6 = tokens.iface_feat6[children_clamped]           # [B,4,Ti,6]
                child_iface_mask = tokens.iface_mask[children_clamped].bool()      # [B,4,Ti]
                child_iface_mask = child_iface_mask & child_exists.unsqueeze(-1)   # AND exist
                child_iface_boundary_dir = tokens.iface_boundary_dir[children_clamped]
                child_iface_inside_endpoint = tokens.iface_inside_endpoint[children_clamped]
                child_iface_inside_quadrant = tokens.iface_inside_quadrant[children_clamped]
                child_state_mask = None
                if use_matching:
                    child_state_mask = tokens.state_mask[children_clamped].bool()
                    child_state_mask = child_state_mask & child_exists.unsqueeze(-1)

                out: TopDownDecoderOutput = decoder(
                    node_feat_rel=node_feat_rel,
                    node_depth=node_depth,
                    z_node=z_node,
                    iface_feat6=iface_feat6,
                    iface_mask=iface_mask,
                    iface_boundary_dir=iface_boundary_dir,
                    iface_inside_endpoint=iface_inside_endpoint,
                    iface_inside_quadrant=iface_inside_quadrant,
                    cross_feat6=cross_feat6,
                    cross_mask=cross_mask,
                    cross_child_pair=cross_child_pair,
                    cross_is_leaf_internal=cross_is_leaf_internal,
                    bc_in_iface_logit=bc_in,
                    child_z=child_z,
                    child_exists_mask=child_exists,
                    child_node_feat_rel=child_node_feat_rel,
                    child_node_depth=child_node_depth,
                    child_iface_feat6=child_iface_feat6,
                    child_iface_mask=child_iface_mask,
                    child_iface_boundary_dir=child_iface_boundary_dir,
                    child_iface_inside_endpoint=child_iface_inside_endpoint,
                    child_iface_inside_quadrant=child_iface_inside_quadrant,
                    bc_in_state_logit=bc_in_state,
                    state_mask=state_mask,
                    state_used_iface=state_used_iface,
                    child_state_mask=child_state_mask,
                )

                # write back local logits (differentiable)
                iface_logit = iface_logit.index_copy(0, node_chunk, out.iface_logit)
                cross_logit = cross_logit.index_copy(0, node_chunk, out.cross_logit)

                # propagate child bc logits (differentiable)
                flat_children = children.view(-1)
                flat_valid = flat_children >= 0
                if flat_valid.any().item():
                    flat_cids = flat_children[flat_valid].long()                    # [K]
                    if use_matching:
                        if out.child_state_logit is None or child_state_mask is None or bc_state_logit is None or state_used_iface is None:
                            raise RuntimeError("matching mode decoder must return child_state_logit.")
                        flat_child_state = out.child_state_logit.view(-1, out.child_state_logit.shape[-1])[flat_valid]
                        flat_child_state_mask = child_state_mask.view(-1, child_state_mask.shape[-1])[flat_valid]
                        bc_state_logit = bc_state_logit.index_copy(0, flat_cids, flat_child_state)
                        flat_child_bc = state_logits_to_expected_iface_usage(
                            state_logit=flat_child_state,
                            state_mask=flat_child_state_mask,
                            state_used_iface=state_used_iface,
                        )
                        child_iface_valid = tokens.iface_mask[flat_cids].bool()
                        flat_child_bc = torch.where(child_iface_valid, flat_child_bc, torch.full_like(flat_child_bc, _NEG_INF))
                        bc_iface_logit = bc_iface_logit.index_copy(0, flat_cids, flat_child_bc)
                    else:
                        flat_child_bc = out.child_iface_logit.view(-1, Ti)[flat_valid]  # [K,Ti]
                        bc_iface_logit = bc_iface_logit.index_copy(0, flat_cids, flat_child_bc)
                    reached[flat_cids] = True

        if self.validate_reachability and (not reached.all().item()):
            missing = torch.nonzero(~reached, as_tuple=False).view(-1)[:50].tolist()
            raise RuntimeError(f"Top-down reachability failed (up to 50 missing node ids): {missing}")

        aux: Dict[str, Tensor] = {
            "num_nodes_reached": reached.sum().to(dtype=torch.long),
        }

        return TopDownResult(
            iface_logit=iface_logit,
            cross_logit=cross_logit,
            bc_iface_logit=bc_iface_logit,
            bc_state_logit=bc_state_logit,
            root_ids=root_ids,
            node_ptr=node_ptr,
            aux=aux,
        )


__all__ = ["TopDownTreeRunner", "TopDownResult"]
