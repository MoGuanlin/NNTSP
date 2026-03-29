# src/models/labeler.py
# -*- coding: utf-8 -*-

from __future__ import annotations

"""Pseudo label generation for Neural Rao'98 DP.

Outputs:
  - y_cross/m_cross: token-level labels for crossing tokens
  - y_iface/m_iface: token-level labels for interface tokens
  - y_child_iface/m_child_iface: per-node labels supervising the boundary
    conditions passed from each node to its 4 children.

The current teacher is a validated Hamiltonian cycle found directly on the same
spanner graph used by the algorithm. We refuse to store supervision unless the
teacher is a true spanner-only cycle.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

try:
    from src.models.bc_state_catalog import project_iface_usage_to_state_index, project_matching_to_state_index
    from src.models.teacher_solver import TEACHER_LABEL_VERSION, solve_spanner_tour_exact, solve_spanner_tour_lkh
    from src.utils.lkh_solver import resolve_lkh_executable
except Exception:  # pragma: no cover
    from .bc_state_catalog import project_iface_usage_to_state_index, project_matching_to_state_index
    from .teacher_solver import TEACHER_LABEL_VERSION, solve_spanner_tour_exact, solve_spanner_tour_lkh
    from ..utils.lkh_solver import resolve_lkh_executable


@dataclass
class TokenLabels:
    """Token-level pseudo labels aligned to PackedNodeTokens fields.

    Shapes:
      y_cross: [M,Tc] float in {0,1}
      m_cross: [M,Tc] bool (valid tokens)
      y_iface: [M,Ti] float in {0,1}
      m_iface: [M,Ti] bool
      y_child_iface: [M,4,Ti] float in {0,1} (targets for parent->child BC)
      m_child_iface: [M,4,Ti] bool
      target_state_idx: [M] long (matching-state target id, or -1 if unavailable)
      m_state: [M] bool
      m_state_exact: [M] bool (True only when target_state_idx is an exact match)
      stats: dict of diagnostics
    """

    y_cross: Tensor
    m_cross: Tensor
    y_iface: Tensor
    m_iface: Tensor
    y_child_iface: Tensor
    m_child_iface: Tensor
    target_state_idx: Tensor
    m_state: Tensor
    m_state_exact: Tensor
    stats: Dict[str, Tensor]


class InfeasibleTeacherGraphError(RuntimeError):
    """Raised when the alive teacher graph cannot admit a Hamiltonian cycle."""


def _make_child_iface_targets_local(
    *,
    y_iface: Tensor,            # [M,Ti]
    iface_mask: Tensor,         # [M,Ti] bool
    tree_children_index: Tensor # [M,4] long
) -> Tuple[Tensor, Tensor]:
    """Derive y_child_iface/m_child_iface from per-node y_iface.

    This function assumes children indices are LOCAL within the same tensor,
    i.e., child id in [0, M). If this does not hold, the caller should compute
    child targets at the global packed level (see label_batch).
    """
    if y_iface.dim() != 2:
        raise ValueError("y_iface must be [M,Ti].")
    if iface_mask.shape != y_iface.shape:
        raise ValueError("iface_mask must match y_iface shape.")
    if tree_children_index.dim() != 2 or tree_children_index.shape[1] != 4:
        raise ValueError("tree_children_index must be [M,4].")

    M, Ti = int(y_iface.shape[0]), int(y_iface.shape[1])
    ch = tree_children_index.to(device=y_iface.device, dtype=torch.long)
    exists = ch >= 0

    # If the slice uses global indices, we cannot safely compute local child targets.
    if exists.any().item():
        mx = int(ch[exists].max().item())
        if mx >= M:
            # return empty (caller should compute globally)
            y_child = torch.zeros((M, 4, Ti), device=y_iface.device, dtype=y_iface.dtype)
            m_child = torch.zeros((M, 4, Ti), device=y_iface.device, dtype=torch.bool)
            return y_child, m_child

    ch0 = ch.clamp_min(0)
    y_child = y_iface[ch0]  # [M,4,Ti]
    m_child = iface_mask.bool()[ch0] & exists.unsqueeze(-1)
    y_child = y_child * m_child.to(dtype=y_child.dtype)
    return y_child, m_child


class PseudoLabeler:
    """Pseudo labels for pretraining and top-down BC supervision.

    For each graph, we obtain a teacher Hamiltonian cycle directly on the same
    sparse spanner graph used by the algorithm. The resulting selected spanner
    edges produce:
      - cross token labels: y_cross
      - iface token labels: y_iface
      - child BC labels: y_child_iface (derived from y_iface + tree children)

    Notes on batching:
      - In a PackedBatch, eids and node indices are globally offset.
      - label_batch() handles global offsets correctly and computes child targets
        on the global packed tensors.
    """

    def __init__(
        self,
        *,
        two_opt_passes: int = 50,
        use_lkh: bool = False,
        lkh_exe: str = "LKH",
        prefer_cpu: bool = True,
        teacher_mode: str = "spanner_lkh",
        teacher_lkh_runs: int = 1,
        teacher_lkh_timeout: Optional[float] = None,
        teacher_exact_timeout: float = 30.0,
        teacher_exact_max_nodes: int = 100,
    ) -> None:
        self.two_opt_passes = int(two_opt_passes)
        self.use_lkh = bool(use_lkh)
        self.lkh_exe = resolve_lkh_executable(str(lkh_exe))
        self.prefer_cpu = bool(prefer_cpu)
        self.teacher_mode = str(teacher_mode)
        self.teacher_label_version = TEACHER_LABEL_VERSION
        self.teacher_lkh_runs = int(teacher_lkh_runs)
        self.teacher_lkh_timeout = None if teacher_lkh_timeout is None else float(teacher_lkh_timeout)
        self.teacher_exact_timeout = float(teacher_exact_timeout)
        self.teacher_exact_max_nodes = int(teacher_exact_max_nodes)
        if self.teacher_mode != "spanner_lkh":
            raise ValueError(f"Unsupported teacher_mode: {self.teacher_mode}")

    def label_signature(self) -> str:
        return (
            f"{TEACHER_LABEL_VERSION}"
            f"|mode={self.teacher_mode}"
            f"|runs={self.teacher_lkh_runs}"
            f"|exact_timeout={self.teacher_exact_timeout:g}"
            f"|exact_max_nodes={self.teacher_exact_max_nodes}"
        )

    def data_has_compatible_teacher(self, data: Any) -> bool:
        if not hasattr(data, "target_edges") or getattr(data, "target_edges") is None:
            return False
        if not hasattr(data, "tour_len"):
            return False
        if not hasattr(data, "teacher_order"):
            return False
        sig = getattr(data, "teacher_label_signature", None)
        return isinstance(sig, str) and sig == self.label_signature()

    @staticmethod
    def _edge_key(u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u < v else (v, u)

    @staticmethod
    def _resolve_alive_edge_ids(data: Any, total_edges: int) -> Tensor:
        """Return original-eid indices for the alive subgraph used by tokens/DP."""
        alive_from_mask: Optional[Tensor] = None
        alive_from_ids: Optional[Tensor] = None

        if hasattr(data, "edge_alive_mask") and getattr(data, "edge_alive_mask") is not None:
            alive_mask = torch.as_tensor(getattr(data, "edge_alive_mask"), dtype=torch.bool).detach().cpu().view(-1)
            if int(alive_mask.numel()) != int(total_edges):
                raise ValueError(
                    f"edge_alive_mask has length {int(alive_mask.numel())}, expected {int(total_edges)}"
                )
            alive_from_mask = torch.nonzero(alive_mask, as_tuple=False).view(-1).to(dtype=torch.long)

        if hasattr(data, "alive_edge_id") and getattr(data, "alive_edge_id") is not None:
            alive_ids = torch.as_tensor(getattr(data, "alive_edge_id"), dtype=torch.long).detach().cpu().view(-1)
            if alive_ids.numel() > 0:
                if int(alive_ids.min().item()) < 0 or int(alive_ids.max().item()) >= int(total_edges):
                    raise ValueError("alive_edge_id contains out-of-range edge ids.")
                alive_ids = torch.unique(alive_ids, sorted=True)
            alive_from_ids = alive_ids

        if alive_from_mask is not None and alive_from_ids is not None:
            if alive_from_mask.numel() != alive_from_ids.numel() or not torch.equal(alive_from_mask, alive_from_ids):
                raise ValueError("edge_alive_mask and alive_edge_id disagree.")
            return alive_from_mask
        if alive_from_mask is not None:
            return alive_from_mask
        if alive_from_ids is not None:
            return alive_from_ids
        return torch.arange(int(total_edges), dtype=torch.long)

    def _extract_teacher_graph(self, data: Any) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor]:
        """Build the exact edge set the DP/token pipeline can actually see."""
        pos = torch.as_tensor(getattr(data, "pos"), dtype=torch.float64).detach().cpu()
        edge_index = torch.as_tensor(getattr(data, "spanner_edge_index"), dtype=torch.long).detach().cpu()
        if edge_index.dim() != 2 or int(edge_index.shape[0]) != 2:
            raise ValueError(f"spanner_edge_index must be [2,E], got {tuple(edge_index.shape)}")
        edge_attr_raw = getattr(data, "spanner_edge_attr", None)
        edge_attr = None if edge_attr_raw is None else torch.as_tensor(edge_attr_raw, dtype=torch.float64).detach().cpu().view(-1)
        E = int(edge_index.shape[1])
        if edge_attr is not None and int(edge_attr.numel()) != E:
            edge_attr = None

        alive_eids = self._resolve_alive_edge_ids(data, E)
        if alive_eids.numel() == 0:
            raise RuntimeError("Alive subgraph has no edges; cannot build teacher supervision.")

        alive_edge_index = edge_index[:, alive_eids]
        alive_edge_attr = None if edge_attr is None else edge_attr[alive_eids]
        return pos, alive_edge_index, alive_edge_attr, alive_eids

    @staticmethod
    def _teacher_graph_basic_failure_reason(edge_index: Tensor, num_nodes: int) -> Optional[str]:
        """Cheap necessary-condition check before running the teacher solver."""
        if num_nodes <= 0:
            return "empty_graph"

        edge_index = torch.as_tensor(edge_index, dtype=torch.long).detach().cpu()
        if edge_index.dim() != 2 or int(edge_index.shape[0]) != 2:
            return f"bad_edge_index_shape={tuple(edge_index.shape)}"

        non_loop = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, non_loop]
        deg = torch.zeros((int(num_nodes),), dtype=torch.long)
        if edge_index.numel() > 0:
            ones = torch.ones((int(edge_index.shape[1]),), dtype=torch.long)
            deg.scatter_add_(0, edge_index[0], ones)
            deg.scatter_add_(0, edge_index[1], ones)

        bad_deg = torch.nonzero(deg < 2, as_tuple=False).view(-1)
        if bad_deg.numel() > 0:
            preview = ",".join(str(int(x)) for x in bad_deg[:10].tolist())
            return f"degree_lt_2:{preview}"

        adj: List[List[int]] = [[] for _ in range(int(num_nodes))]
        for eid in range(int(edge_index.shape[1])):
            a = int(edge_index[0, eid].item())
            b = int(edge_index[1, eid].item())
            adj[a].append(b)
            adj[b].append(a)

        seen = [False] * int(num_nodes)
        stack = [0]
        seen[0] = True
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)

        disconnected = [idx for idx, ok in enumerate(seen) if not ok]
        if disconnected:
            preview = ",".join(str(int(x)) for x in disconnected[:10])
            return f"disconnected:{preview}"

        return None

    def validate_teacher_labels(self, data: Any) -> Tuple[bool, str]:
        """Validate stored teacher labels without rerunning LKH."""
        if not self.data_has_compatible_teacher(data):
            return False, "missing_or_stale_teacher"
        if not hasattr(data, "pos") or not hasattr(data, "spanner_edge_index"):
            return False, "missing_graph_fields"

        pos, edge_index, edge_attr, alive_eids = self._extract_teacher_graph(data)
        teacher_order = torch.as_tensor(getattr(data, "teacher_order"), dtype=torch.long).detach().cpu().view(-1)
        target_edges = torch.as_tensor(getattr(data, "target_edges"), dtype=torch.long).detach().cpu().view(-1)
        N = int(pos.shape[0])
        E_alive = int(edge_index.shape[1])

        if teacher_order.numel() != N:
            return False, f"teacher_order_len={teacher_order.numel()} expected={N}"
        if target_edges.numel() != N:
            return False, f"target_edges_len={target_edges.numel()} expected={N}"
        if teacher_order.numel() == 0:
            return False, "empty_teacher_order"
        if target_edges.numel() == 0:
            return False, "empty_target_edges"

        order_list = [int(x) for x in teacher_order.tolist()]
        if min(order_list) < 0 or max(order_list) >= N:
            return False, "teacher_order_out_of_range"
        if len(set(order_list)) != N:
            return False, "teacher_order_not_permutation"

        alive_set = set(int(x) for x in alive_eids.tolist())
        for eid in target_edges.tolist():
            if int(eid) not in alive_set:
                return False, f"target_edge_not_alive={int(eid)}"

        attr_flat = None if edge_attr is None else edge_attr.view(-1)
        if attr_flat is not None and int(attr_flat.numel()) != E_alive:
            attr_flat = None

        eid_map: Dict[Tuple[int, int], int] = {}
        length_map: Dict[Tuple[int, int], float] = {}
        for sub_eid in range(E_alive):
            a = int(edge_index[0, sub_eid])
            b = int(edge_index[1, sub_eid])
            key = self._edge_key(a, b)
            if key not in eid_map:
                eid_map[key] = sub_eid
            if attr_flat is not None:
                length_map[key] = float(attr_flat[sub_eid].item())
            else:
                dx = float(pos[a, 0].item() - pos[b, 0].item())
                dy = float(pos[a, 1].item() - pos[b, 1].item())
                length_map[key] = (dx * dx + dy * dy) ** 0.5

        induced_eids: List[int] = []
        induced_len = 0.0
        for i in range(N):
            a = order_list[i]
            b = order_list[(i + 1) % N]
            key = self._edge_key(a, b)
            if key not in eid_map:
                return False, f"off_alive_spanner_edge={a}-{b}"
            induced_eids.append(int(alive_eids[int(eid_map[key])].item()))
            induced_len += float(length_map[key])

        stored_edges = [int(x) for x in target_edges.tolist()]
        if len(set(induced_eids)) != N:
            return False, "teacher_cycle_reuses_edge"
        if stored_edges != induced_eids:
            return False, "target_edges_mismatch_teacher_order"

        stored_len = float(torch.as_tensor(getattr(data, "tour_len"), dtype=torch.float64).item())
        tol = max(1e-4, 1e-5 * max(1.0, abs(induced_len)))
        if abs(stored_len - induced_len) > tol:
            return False, f"tour_len_mismatch={stored_len:.6f} expected={induced_len:.6f}"

        return True, "ok"

    def attach_teacher_labels(
        self,
        *,
        data: Any,
        target_edges: Any,
        tour_len: float,
        teacher_order: Any,
        teacher_stats: Optional[Dict[str, int]] = None,
    ) -> None:
        teacher_stats = teacher_stats or {}
        setattr(data, "target_edges", torch.as_tensor(target_edges, dtype=torch.long))
        setattr(data, "tour_len", torch.tensor(float(tour_len), dtype=torch.float32))
        setattr(data, "teacher_order", torch.as_tensor(teacher_order, dtype=torch.long))
        setattr(data, "teacher_label_signature", self.label_signature())
        setattr(data, "teacher_label_version", TEACHER_LABEL_VERSION)
        setattr(data, "teacher_mode", self.teacher_mode)
        setattr(data, "teacher_num_direct", int(teacher_stats.get("num_direct", 0)))
        setattr(data, "teacher_num_projected", int(teacher_stats.get("num_projected", 0)))
        setattr(data, "teacher_num_unreachable", int(teacher_stats.get("num_unreachable", 0)))
        setattr(data, "teacher_num_not_alive_direct", int(teacher_stats.get("num_not_alive_direct", 0)))

    @staticmethod
    def simplify_data_for_ipc(data: Any) -> Dict[str, Any]:
        """Convert torch.Data tensors to numpy for safe/efficient IPC."""
        payload = {
            "pos": data.pos.detach().cpu().numpy(),
            "spanner_edge_index": data.spanner_edge_index.detach().cpu().numpy(),
            "spanner_edge_attr": data.spanner_edge_attr.detach().cpu().numpy(),
        }
        if hasattr(data, "edge_alive_mask") and getattr(data, "edge_alive_mask") is not None:
            payload["edge_alive_mask"] = getattr(data, "edge_alive_mask").detach().cpu().numpy()
        if hasattr(data, "alive_edge_id") and getattr(data, "alive_edge_id") is not None:
            payload["alive_edge_id"] = getattr(data, "alive_edge_id").detach().cpu().numpy()
        return payload

    def extract_teacher_supervision(
        self,
        data: Any,
    ) -> Tuple[Any, float, Any, Dict[str, int]]:
        """Compute validated teacher supervision on the sparse spanner graph."""
        import numpy as np

        if isinstance(data, dict):
            data_obj = type("TeacherGraphPayload", (), {})()
            for key, value in data.items():
                setattr(data_obj, key, value)
            data = data_obj

        pos_t, edge_index_t, edge_attr_t, alive_eids_t = self._extract_teacher_graph(data)
        basic_failure = self._teacher_graph_basic_failure_reason(edge_index_t, int(pos_t.shape[0]))
        if basic_failure is not None:
            raise InfeasibleTeacherGraphError(f"alive_teacher_graph_infeasible:{basic_failure}")
        pos_np = pos_t.numpy()
        edge_index_np = edge_index_t.numpy()
        edge_attr_np = None if edge_attr_t is None else edge_attr_t.numpy()
        alive_eids_np = alive_eids_t.numpy()

        fallback_used = False
        try:
            tour = solve_spanner_tour_lkh(
                pos=pos_np,
                spanner_edge_index=edge_index_np,
                spanner_edge_attr=edge_attr_np,
                executable=self.lkh_exe,
                runs=self.teacher_lkh_runs,
                timeout=self.teacher_lkh_timeout,
            )
        except RuntimeError as exc:
            msg = str(exc)
            if (
                pos_np.shape[0] <= self.teacher_exact_max_nodes
                and (
                    "LKH failed" in msg
                    or "off-spanner" in msg
                )
            ):
                try:
                    tour = solve_spanner_tour_exact(
                        pos=pos_np,
                        spanner_edge_index=edge_index_np,
                        spanner_edge_attr=edge_attr_np,
                        time_limit=self.teacher_exact_timeout,
                        length_weight=1.0,
                    )
                    fallback_used = True
                except RuntimeError as exact_exc:
                    raise InfeasibleTeacherGraphError(
                        f"both_lkh_and_exact_failed:lkh={msg}|exact={exact_exc}"
                    ) from exact_exc
            else:
                raise
        mapped_edge_ids = alive_eids_np[np.asarray(tour.edge_ids, dtype=np.int64)]
        stats = {
            "num_direct": int(len(mapped_edge_ids)),
            "num_projected": 0,
            "num_unreachable": 0,
            "num_not_alive_direct": 0,
            "used_exact_fallback": int(fallback_used),
        }
        return (
            np.asarray(mapped_edge_ids, dtype=np.int64),
            float(tour.length),
            np.asarray(tour.order, dtype=np.int64),
            stats,
        )

    def extract_target_edges(
        self,
        data: Any,
    ) -> Tuple[Any, float]:
        """Backward-compatible wrapper returning only edge ids and length."""
        te, tlen, _, _ = self.extract_teacher_supervision(data)
        return te, tlen

    def label_one(
        self,
        *,
        data: Any,
        tokens_slice: Any,
        device: torch.device,
        eid_offset: int = 0,
    ) -> TokenLabels:
        """Label a single graph slice.

        tokens_slice is expected to have at least:
          cross_eid, cross_mask, iface_eid, iface_mask

        If tokens_slice additionally has local tree_children_index, child BC
        labels are derived locally; otherwise child labels are returned as all
        zeros and should be computed by label_batch.
        """
        eid_offset = int(eid_offset)

        # ---- gather required fields ----
        pos = getattr(data, "pos")
        sp_edge_index = getattr(data, "spanner_edge_index")
        sp_edge_attr = getattr(data, "spanner_edge_attr")

        if self.prefer_cpu:
            pos = pos.detach().cpu()
            sp_edge_index = sp_edge_index.detach().cpu()
            sp_edge_attr = sp_edge_attr.detach().cpu()

        if sp_edge_attr.dim() == 2 and sp_edge_attr.shape[1] == 1:
            sp_w = sp_edge_attr[:, 0].tolist()
        else:
            sp_w = sp_edge_attr.view(-1).tolist()

        N = int(pos.shape[0])
        E = int(sp_edge_index.shape[1])

        # ---- token eid/mask (potentially global eids) ----
        cross_eid_g = getattr(tokens_slice, "cross_eid")
        cross_mask = getattr(tokens_slice, "cross_mask").bool()
        iface_eid_g = getattr(tokens_slice, "iface_eid")
        iface_mask = getattr(tokens_slice, "iface_mask").bool()

        if self.prefer_cpu:
            cross_eid_g = cross_eid_g.detach().cpu()
            cross_mask = cross_mask.detach().cpu()
            iface_eid_g = iface_eid_g.detach().cpu()
            iface_mask = iface_mask.detach().cpu()

        # global->local eid mapping for this graph
        cross_eid_l = torch.where(cross_eid_g >= 0, cross_eid_g - eid_offset, cross_eid_g)
        iface_eid_l = torch.where(iface_eid_g >= 0, iface_eid_g - eid_offset, iface_eid_g)

        alive_m = cross_mask & (cross_eid_l >= 0) & (cross_eid_l < E)
        if alive_m.any().item():
            alive_eids_local = torch.unique(cross_eid_l[alive_m].long()).tolist()
        else:
            alive_eids_local = []
        alive_set_local = set(int(x) for x in alive_eids_local)

        # ---- teacher tour on the same sparse spanner graph ----
        if self.data_has_compatible_teacher(data):
            selected_local_eids = set(int(x) for x in data.target_edges.tolist())
            tour_len_val = getattr(data, "tour_len", torch.tensor(0.0)).detach().cpu()
            num_direct = int(getattr(data, "teacher_num_direct", len(selected_local_eids)))
            num_projected = int(getattr(data, "teacher_num_projected", 0))
            num_unreachable = int(getattr(data, "teacher_num_unreachable", 0))
            num_not_alive_direct = int(getattr(data, "teacher_num_not_alive_direct", 0))
        else:
            te_np, tlen, order_np, teacher_stats = self.extract_teacher_supervision(data)
            self.attach_teacher_labels(
                data=data,
                target_edges=te_np,
                tour_len=tlen,
                teacher_order=order_np,
                teacher_stats=teacher_stats,
            )
            selected_local_eids = set(int(x) for x in te_np.tolist())
            tour_len_val = torch.tensor(float(tlen), dtype=torch.float32)
            num_direct = int(teacher_stats["num_direct"])
            num_projected = int(teacher_stats["num_projected"])
            num_unreachable = int(teacher_stats["num_unreachable"])
            num_not_alive_direct = int(teacher_stats["num_not_alive_direct"])

        # ---- vectorized label construction in LOCAL eid space ----
        eid_table = torch.zeros((E,), dtype=torch.bool)
        for le in selected_local_eids:
            if 0 <= le < E:
                eid_table[le] = True

        # cross labels
        y_cross = torch.zeros_like(cross_eid_g, dtype=torch.float32)
        valid_cross = cross_mask & (cross_eid_l >= 0) & (cross_eid_l < E)
        if valid_cross.any().item():
            idx = cross_eid_l[valid_cross].long()
            y_cross[valid_cross] = eid_table[idx].to(dtype=torch.float32)

        # iface labels
        y_iface = torch.zeros_like(iface_eid_g, dtype=torch.float32)
        valid_iface = iface_mask & (iface_eid_l >= 0) & (iface_eid_l < E)
        if valid_iface.any().item():
            idx2 = iface_eid_l[valid_iface].long()
            y_iface[valid_iface] = eid_table[idx2].to(dtype=torch.float32)

        # ---- optional child BC labels (only if local child indices are usable) ----
        if hasattr(tokens_slice, "tree_children_index"):
            y_child_iface, m_child_iface = _make_child_iface_targets_local(
                y_iface=y_iface,
                iface_mask=iface_mask,
                tree_children_index=getattr(tokens_slice, "tree_children_index"),
            )
        else:
            M = int(y_iface.shape[0])
            Ti = int(y_iface.shape[1])
            y_child_iface = torch.zeros((M, 4, Ti), dtype=torch.float32)
            m_child_iface = torch.zeros((M, 4, Ti), dtype=torch.bool)

        stats = {
            "tour_len": tour_len_val,
            "alive_edges": torch.tensor([len(alive_set_local)], dtype=torch.long),
            "selected_alive_edges": torch.tensor([len(selected_local_eids)], dtype=torch.long),
            "num_direct": torch.tensor([num_direct], dtype=torch.long),
            "num_projected": torch.tensor([num_projected], dtype=torch.long),
            "num_unreachable": torch.tensor([num_unreachable], dtype=torch.long),
            "num_not_alive_direct": torch.tensor([num_not_alive_direct], dtype=torch.long),
        }

        # move to target device
        return TokenLabels(
            y_cross=y_cross.to(device),
            m_cross=cross_mask.to(device),
            y_iface=y_iface.to(device),
            m_iface=iface_mask.to(device),
            y_child_iface=y_child_iface.to(device),
            m_child_iface=m_child_iface.to(device),
            target_state_idx=torch.full((int(y_iface.shape[0]),), -1, dtype=torch.long, device=device),
            m_state=torch.zeros((int(y_iface.shape[0]),), dtype=torch.bool, device=device),
            m_state_exact=torch.zeros((int(y_iface.shape[0]),), dtype=torch.bool, device=device),
            stats={k: v.to(device) for k, v in stats.items()},
        )

    def label_batch(
        self,
        *,
        datas: List[Any],
        packed: Any,
        device: torch.device,
    ) -> TokenLabels:
        """Label a PackedBatch (global packed tensors).

        Requirements:
          packed.node_ptr: [B+1]
          packed.edge_ptr: [B+1]
          packed.tokens: has (cross_eid,cross_mask,iface_eid,iface_mask,tree_children_index)

        Returns labels on `device` with shapes matching packed.tokens.
        """
        if not hasattr(packed, "node_ptr") or not hasattr(packed, "edge_ptr") or not hasattr(packed, "tokens"):
            raise ValueError("packed must have fields: node_ptr, edge_ptr, tokens")

        node_ptr: Tensor = packed.node_ptr
        edge_ptr: Tensor = packed.edge_ptr
        tokens = packed.tokens

        if node_ptr.dim() != 1 or node_ptr.numel() < 2:
            raise ValueError("packed.node_ptr must be 1D with length B+1.")
        if edge_ptr.dim() != 1 or edge_ptr.numel() != node_ptr.numel():
            raise ValueError("packed.edge_ptr must be 1D with the same length as node_ptr.")

        B = int(node_ptr.numel() - 1)
        if len(datas) != B:
            raise ValueError(f"len(datas)={len(datas)} must equal batch size B={B}.")

        total_M = int(node_ptr[-1].item())
        Ti = int(tokens.iface_mask.shape[1])
        Tc = int(tokens.cross_mask.shape[1])

        # ---- Ensure every sample has a compatible sparse-spanner teacher ----
        for b in range(B):
            if not self.data_has_compatible_teacher(datas[b]):
                te, tl, order, teacher_stats = self.extract_teacher_supervision(datas[b])
                self.attach_teacher_labels(
                    data=datas[b],
                    target_edges=te,
                    tour_len=tl,
                    teacher_order=order,
                    teacher_stats=teacher_stats,
                )

        # ---- Vectorized Bitmask Construction ----
        # 1. Gather all global target eids
        all_global_targets = []
        for b in range(B):
            targets = datas[b].target_edges.to(device)
            all_global_targets.append(targets + edge_ptr[b])
        
        targets_g = torch.cat(all_global_targets) if all_global_targets else torch.tensor([], dtype=torch.long, device=device)

        m_cross_all = tokens.cross_mask.bool().to(device)
        m_iface_all = tokens.iface_mask.bool().to(device)

        # 2. Optimized bitmask search for targets_g (much faster than torch.isin on CPU)
        max_eid = int(edge_ptr[-1].item())
        mask_table = torch.zeros(max_eid + 1, dtype=torch.bool, device=device)
        if targets_g.numel() > 0:
            mask_table[targets_g] = True
        
        # Map cross_eid and iface_eid (handling -1 padding)
        def map_labels(eids):
            # [M, T]
            out = torch.zeros_like(eids, dtype=torch.float32)
            valid = eids >= 0
            if valid.any():
                out[valid] = mask_table[eids[valid]].to(dtype=torch.float32)
            return out

        y_cross_all = map_labels(tokens.cross_eid)
        y_iface_all = map_labels(tokens.iface_eid)

        target_state_idx = torch.full((total_M,), -1, dtype=torch.long, device=device)
        m_state = torch.zeros((total_M,), dtype=torch.bool, device=device)
        m_state_exact = torch.zeros((total_M,), dtype=torch.bool, device=device)
        num_state_exact = 0
        num_state_fallback = 0
        if getattr(tokens, "state_mask", None) is not None and getattr(packed, "state_catalog", None) is not None:
            state_mask_all = tokens.state_mask.bool().to(device)
            state_used_iface = packed.state_catalog.used_iface.to(device)
            state_mate = packed.state_catalog.mate.to(device)
            max_used = int(getattr(packed.state_catalog, "max_used", int(state_used_iface.sum(dim=1).max().item())))
            m_state = state_mask_all.any(dim=1)

            for b in range(B):
                m0 = int(node_ptr[b].item())
                m1 = int(node_ptr[b + 1].item())
                e0 = int(edge_ptr[b].item())
                selected_local_eids = set(int(x) for x in getattr(datas[b], "target_edges", torch.empty((0,), dtype=torch.long)).detach().cpu().tolist())
                node_points = _compute_node_point_sets(datas[b])
                sp_edge_index = getattr(datas[b], "spanner_edge_index").detach().cpu()
                sp_u = sp_edge_index[0].tolist()
                sp_v = sp_edge_index[1].tolist()

                iface_eid_local = tokens.iface_eid[m0:m1].to(device) - e0
                iface_mask_local = m_iface_all[m0:m1]
                iface_inside_ep_local = tokens.iface_inside_endpoint[m0:m1].to(device)
                state_mask_local = state_mask_all[m0:m1]

                for local_nid in range(m1 - m0):
                    mid = m0 + local_nid
                    if not bool(m_state[mid].item()):
                        continue

                    state_idx, exact_used = _build_matching_target_for_node(
                        local_node_id=local_nid,
                        points_in_node=node_points[local_nid],
                        selected_local_eids=selected_local_eids,
                        sp_u=sp_u,
                        sp_v=sp_v,
                        iface_eid_row=iface_eid_local[local_nid],
                        iface_mask_row=iface_mask_local[local_nid],
                        iface_inside_ep_row=iface_inside_ep_local[local_nid],
                        state_mask_row=state_mask_local[local_nid],
                        state_used_iface=state_used_iface,
                        state_mate=state_mate,
                        max_used=max_used,
                    )
                    target_state_idx[mid] = int(state_idx)
                    if exact_used:
                        m_state_exact[mid] = True
                        num_state_exact += 1
                    else:
                        num_state_fallback += 1

        num_direct_sum = sum(int(getattr(d, "teacher_num_direct", len(getattr(d, "target_edges", [])))) for d in datas)
        num_projected_sum = sum(int(getattr(d, "teacher_num_projected", 0)) for d in datas)
        num_unreachable_sum = sum(int(getattr(d, "teacher_num_unreachable", 0)) for d in datas)

        # 3. Vectorized stats aggregation (using tour_len from datas)
        t_lens = [torch.as_tensor(getattr(d, "tour_len", 0.0), device=device).view(-1) for d in datas]
        tour_len_cat = torch.cat(t_lens, dim=0) if t_lens else torch.tensor([], device=device)
        
        # Combined stats dictionary for returning in TokenLabels
        stats = {
            "tour_len": tour_len_cat,
            "tour_len_mean": tour_len_cat.mean() if tour_len_cat.numel() > 0 else torch.tensor(0.0, device=device),
            "num_direct_sum": torch.tensor([num_direct_sum], device=device),
            "num_projected_sum": torch.tensor([num_projected_sum], device=device),
            "num_unreachable_sum": torch.tensor([num_unreachable_sum], device=device),
            "num_state_exact_sum": torch.tensor([num_state_exact], device=device),
            "num_state_fallback_sum": torch.tensor([num_state_fallback], device=device),
        }

        # ---- child BC labels (GLOBAL packed indices) ----
        ch = tokens.tree_children_index.long().to(device)  # [total_M,4]
        exists = ch >= 0
        ch0 = ch.clamp_min(0)
        y_child_iface = y_iface_all[ch0]  # [M,4,Ti]
        m_child_iface = m_iface_all[ch0] & exists.unsqueeze(-1)
        y_child_iface = y_child_iface * m_child_iface.to(dtype=y_child_iface.dtype)

        return TokenLabels(
            y_cross=y_cross_all,
            m_cross=m_cross_all,
            y_iface=y_iface_all,
            m_iface=m_iface_all,
            y_child_iface=y_child_iface,
            m_child_iface=m_child_iface,
            target_state_idx=target_state_idx,
            m_state=m_state,
            m_state_exact=m_state_exact,
            stats=stats,
        )


def _compute_node_point_sets(data: Any) -> List[set[int]]:
    cached = getattr(data, "_cached_node_point_sets", None)
    if cached is not None:
        return cached

    tree_parent_index = getattr(data, "tree_parent_index").detach().cpu().tolist()
    num_nodes = len(tree_parent_index)
    node_points: List[set[int]] = [set() for _ in range(num_nodes)]
    if hasattr(data, "point_to_leaf"):
        point_to_leaf = getattr(data, "point_to_leaf").detach().cpu().tolist()
        for pid, leaf_id in enumerate(point_to_leaf):
            nid = int(leaf_id)
            while nid >= 0:
                node_points[nid].add(int(pid))
                nid = int(tree_parent_index[nid])
    else:
        leaf_ids = getattr(data, "leaf_ids").detach().cpu().tolist()
        leaf_ptr = getattr(data, "leaf_ptr").detach().cpu().tolist()
        leaf_points = getattr(data, "leaf_points").detach().cpu().tolist()
        for li, leaf_nid in enumerate(leaf_ids):
            pts = leaf_points[leaf_ptr[li]: leaf_ptr[li + 1]]
            nid = int(leaf_nid)
            while nid >= 0:
                node_points[nid].update(int(pid) for pid in pts)
                nid = int(tree_parent_index[nid])

    try:
        setattr(data, "_cached_node_point_sets", node_points)
    except Exception:
        pass
    return node_points


def _infer_inside_point_for_interface(
    *,
    a: int,
    b: int,
    inside_ep_attr: int,
    points_in_node: set[int],
) -> Optional[int]:
    if inside_ep_attr == 0 and a in points_in_node:
        return a
    if inside_ep_attr == 1 and b in points_in_node:
        return b

    in_a = a in points_in_node
    in_b = b in points_in_node
    if in_a and not in_b:
        return a
    if in_b and not in_a:
        return b
    if in_a and in_b:
        return a if inside_ep_attr == 0 else b
    return None


def _build_matching_target_for_node(
    *,
    local_node_id: int,
    points_in_node: set[int],
    selected_local_eids: set[int],
    sp_u: List[int],
    sp_v: List[int],
    iface_eid_row: Tensor,               # [Ti] local eid
    iface_mask_row: Tensor,              # [Ti] bool
    iface_inside_ep_row: Tensor,         # [Ti] long
    state_mask_row: Tensor,              # [S] bool
    state_used_iface: Tensor,            # [S,Ti] bool
    state_mate: Tensor,                  # [S,Ti] long
    max_used: int,
) -> tuple[int, bool]:
    del local_node_id  # reserved for future debugging / stats hooks

    Ti = int(iface_mask_row.numel())
    target_used = torch.zeros((Ti,), dtype=torch.bool, device=iface_mask_row.device)
    target_mate = torch.full((Ti,), -1, dtype=torch.long, device=iface_mask_row.device)

    stub_points: Dict[int, int] = {}
    fallback_needed = False

    for i in range(Ti):
        if not bool(iface_mask_row[i].item()):
            continue
        eid = int(iface_eid_row[i].item())
        if eid < 0 or eid not in selected_local_eids:
            continue

        a = int(sp_u[eid])
        b = int(sp_v[eid])
        inside_point = _infer_inside_point_for_interface(
            a=a,
            b=b,
            inside_ep_attr=int(iface_inside_ep_row[i].item()),
            points_in_node=points_in_node,
        )
        if inside_point is None:
            fallback_needed = True
            continue

        target_used[i] = True
        stub_points[i] = inside_point

    used_slots = [slot for slot, used in enumerate(target_used.tolist()) if used]
    if len(used_slots) == 0:
        return project_matching_to_state_index(
            iface_used=target_used,
            iface_mate=target_mate,
            iface_mask=iface_mask_row,
            state_mask=state_mask_row,
            state_used_iface=state_used_iface,
            state_mate=state_mate,
        ), True

    if len(used_slots) > int(max_used):
        fallback_needed = True

    if len(used_slots) % 2 != 0:
        fallback_needed = True

    touched_points = set(stub_points.values())
    internal_edges: List[Tuple[int, int]] = []
    for eid in selected_local_eids:
        a = int(sp_u[eid])
        b = int(sp_v[eid])
        if a in points_in_node and b in points_in_node:
            internal_edges.append((a, b))
            touched_points.add(a)
            touched_points.add(b)

    parent: Dict[int, int] = {p: p for p in touched_points}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in internal_edges:
        union(a, b)

    comp_to_slots: Dict[int, List[int]] = {}
    for slot, point in stub_points.items():
        root = find(point)
        comp_to_slots.setdefault(root, []).append(slot)

    for slots in comp_to_slots.values():
        if len(slots) != 2:
            fallback_needed = True
            continue
        i, j = int(slots[0]), int(slots[1])
        target_mate[i] = j
        target_mate[j] = i

    exact_usable = (not fallback_needed) and all(int(target_mate[i].item()) >= 0 for i in used_slots)
    if exact_usable:
        return project_matching_to_state_index(
            iface_used=target_used,
            iface_mate=target_mate,
            iface_mask=iface_mask_row,
            state_mask=state_mask_row,
            state_used_iface=state_used_iface,
            state_mate=state_mate,
        ), True

    return project_iface_usage_to_state_index(
        iface_target=target_used.to(dtype=torch.float32),
        iface_mask=iface_mask_row,
        state_mask=state_mask_row,
        state_used_iface=state_used_iface,
    ), False


__all__ = ["PseudoLabeler", "TokenLabels"]
