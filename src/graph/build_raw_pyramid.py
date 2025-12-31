# src/graph/build_raw_pyramid.py
import torch
import numpy as np
import argparse
import os
from tqdm import tqdm
import math
from typing import List, Tuple, Dict, Optional

# 尝试导入 PyG Data，如果没有则使用 Mock
try:
    from torch_geometric.data import Data
except ImportError:
    class Data:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def __repr__(self):
            return f"Data(num_nodes={getattr(self, 'num_nodes', 'N/A')})"


# Quadrant order (fixed):
TL, TR, BL, BR = 0, 1, 2, 3


class QuadtreeNode:
    """
    四叉树节点容器（构建阶段临时结构）。

    改造要点：
    - children 固定按 TL/TR/BL/BR 存储（便于 MergeEncoder 的 positional encoding）
    - 存储 quad_in_parent（便于在 LCA 处确定 crossing 两端属于哪个子象限）
    - interface_records 预留 inside_endpoint、boundary_dir 等 DP 必备元数据
    """
    def __init__(
        self,
        node_id: int,
        x: float,
        y: float,
        w: float,
        h: float,
        depth: int,
        parent_id: int,
        quad_in_parent: int
    ):
        self.node_id = node_id
        self.x = float(x)
        self.y = float(y)
        self.w = float(w)
        self.h = float(h)
        self.depth = int(depth)

        self.parent_id = int(parent_id)
        self.quad_in_parent = int(quad_in_parent)

        self.children = [-1, -1, -1, -1]  # TL/TR/BL/BR
        self.is_leaf = True
        self.point_indices: List[int] = []

        # crossing_records: list of (eid, feat6, quad_u, quad_v, is_leaf_internal)
        self.crossing_records: List[Tuple[int, List[float], int, int, bool]] = []
        # interface_records: list of (eid, feat6_v2, inside_endpoint, boundary_dir, inter_rel_xy, inside_quad)
        #   feat6_v2 := [inside_rel_x, inside_rel_y, inter_rel_x, inter_rel_y, norm_len, angle]
        # IMPORTANT (scale invariance contract): all *_rel_* are CENTER-relative in [-1,1] (up to eps),
        # not llwh-relative in [0,1].
        self.interface_records: List[Tuple[int, List[float], int, int, List[float], int]] = []


class RawPyramidBuilder:
    """
    构建全量四叉树图 (Raw Pyramid Graph)。
    不做 r-light 剪枝，只负责：
    - 构建 quadtree
    - 注册 crossing/interface
    - 扁平化为 PyG Data，并补齐 DP/LeafEncoder 需要的结构化字段
    """
    def __init__(self, max_points_per_leaf=4, max_depth=100):
        self.max_points = max_points_per_leaf
        self.max_depth = max_depth

    # ---------------------------
    # Geometry feature encodings
    # ---------------------------
    def _edge_feat6_center_relative_endpoints(self, node: QuadtreeNode, u_pos, v_pos) -> List[float]:
        """6D 几何特征（强契约版，**中心坐标系**）。

        feat6 := [ux_rel, uy_rel, vx_rel, vy_rel, norm_len, angle]

        - (ux_rel, uy_rel), (vx_rel, vy_rel) 为相对 node bbox **中心**的归一化坐标：
            ux_rel = (ux - cx) / (w/2),  uy_rel = (uy - cy) / (h/2)
          理想范围约为 [-1,1]（允许 eps 越界）。
        - norm_len = |uv| / diag(bbox)
        - angle = atan2(dy, dx) / pi

        这样可从源头消除“ll-relative in [0,1]” 与 “center-relative in [-1,1]” 的混用。
        """
        cx = node.x + node.w * 0.5
        cy = node.y + node.h * 0.5
        hw = node.w * 0.5
        hh = node.h * 0.5

        ux = float(u_pos[0])
        uy = float(u_pos[1])
        vx = float(v_pos[0])
        vy = float(v_pos[1])

        ux_rel = (ux - cx) / (hw + 1e-12)
        uy_rel = (uy - cy) / (hh + 1e-12)
        vx_rel = (vx - cx) / (hw + 1e-12)
        vy_rel = (vy - cy) / (hh + 1e-12)

        dx = vx - ux
        dy = vy - uy
        length = math.sqrt(dx * dx + dy * dy)
        diag = math.sqrt(node.w * node.w + node.h * node.h) + 1e-12
        norm_len = length / diag
        angle = math.atan2(dy, dx) / math.pi
        return [ux_rel, uy_rel, vx_rel, vy_rel, norm_len, angle]

    def _point_in_node(self, node: QuadtreeNode, p) -> bool:
        x0, y0, x1, y1 = node.x, node.y, node.x + node.w, node.y + node.h
        return (x0 <= float(p[0]) <= x1) and (y0 <= float(p[1]) <= y1)

    def _boundary_dir_and_intersection(self, node: QuadtreeNode, inside, outside) -> Tuple[int, Tuple[float, float]]:
        """返回 (boundary_dir, intersection_xy)。

        boundary_dir encoding:
          0: Left, 1: Right, 2: Bottom, 3: Top

        规则：以 inside->outside 的射线与矩形边界的**最先**交点为准。
        """
        x0, y0, x1, y1 = node.x, node.y, node.x + node.w, node.y + node.h
        ix, iy = float(inside[0]), float(inside[1])
        ox, oy = float(outside[0]), float(outside[1])
        dx, dy = ox - ix, oy - iy

        candidates: List[Tuple[float, int, float, float]] = []  # (t, bdir, xi, yi)

        # Left x=x0
        if abs(dx) > 1e-12:
            t = (x0 - ix) / dx
            if 0.0 <= t <= 1.0:
                y = iy + t * dy
                if y0 - 1e-9 <= y <= y1 + 1e-9:
                    candidates.append((t, 0, x0, y))
        # Right x=x1
        if abs(dx) > 1e-12:
            t = (x1 - ix) / dx
            if 0.0 <= t <= 1.0:
                y = iy + t * dy
                if y0 - 1e-9 <= y <= y1 + 1e-9:
                    candidates.append((t, 1, x1, y))
        # Bottom y=y0
        if abs(dy) > 1e-12:
            t = (y0 - iy) / dy
            if 0.0 <= t <= 1.0:
                x = ix + t * dx
                if x0 - 1e-9 <= x <= x1 + 1e-9:
                    candidates.append((t, 2, x, y0))
        # Top y=y1
        if abs(dy) > 1e-12:
            t = (y1 - iy) / dy
            if 0.0 <= t <= 1.0:
                x = ix + t * dx
                if x0 - 1e-9 <= x <= x1 + 1e-9:
                    candidates.append((t, 3, x, y1))

        if not candidates:
            # fallback: choose boundary by outside position, then compute intersection with that boundary.
            if ox < x0:
                bdir = 0
            elif ox > x1:
                bdir = 1
            elif oy < y0:
                bdir = 2
            else:
                bdir = 3

            # compute intersection for the chosen boundary (consistent with prune_pyramid.py).
            if bdir == 0:  # left x=x0
                t = (x0 - ix) / (dx + 1e-12)
                yi2 = iy + t * dy
                return bdir, (x0, yi2)
            if bdir == 1:  # right x=x1
                t = (x1 - ix) / (dx + 1e-12)
                yi2 = iy + t * dy
                return bdir, (x1, yi2)
            if bdir == 2:  # bottom y=y0
                t = (y0 - iy) / (dy + 1e-12)
                xi2 = ix + t * dx
                return bdir, (xi2, y0)
            # top y=y1
            t = (y1 - iy) / (dy + 1e-12)
            xi2 = ix + t * dx
            return bdir, (xi2, y1)

        candidates.sort(key=lambda t_bdir_xy: t_bdir_xy[0])
        _, bdir, xi, yi = candidates[0]
        return bdir, (xi, yi)

    def _inside_quadrant(self, node: QuadtreeNode, px: float, py: float) -> int:
        """相对于 node 中心的象限编号（TL/TR/BL/BR = 0/1/2/3）。"""
        cx = node.x + node.w * 0.5
        cy = node.y + node.h * 0.5
        if px < cx:
            return TL if py >= cy else BL
        return TR if py >= cy else BR

    def _xy_center_rel(self, node: QuadtreeNode, x: float, y: float) -> Tuple[float, float]:
        """将绝对坐标 (x,y) 映射为 node 中心相对坐标。"""
        cx = node.x + node.w * 0.5
        cy = node.y + node.h * 0.5
        hw = node.w * 0.5
        hh = node.h * 0.5
        return (x - cx) / (hw + 1e-12), (y - cy) / (hh + 1e-12)

    def _interface_feat6_v2(self, node: QuadtreeNode, inside, outside) -> Tuple[List[float], List[float], int, int]:
        """Interface token（强契约 v2）：

        - feat6_v2 := [inside_rel_x, inside_rel_y, inter_rel_x, inter_rel_y, norm_len, angle]
                - inside_quad 直接返回，供 DP/positional encoding 使用
        """
        bdir, (xi, yi) = self._boundary_dir_and_intersection(node, inside, outside)

        ix, iy = float(inside[0]), float(inside[1])
        inside_rel = self._xy_center_rel(node, ix, iy)
        inter_rel = self._xy_center_rel(node, xi, yi)

        dx = xi - ix
        dy = yi - iy
        seg_len = math.sqrt(dx * dx + dy * dy)
        diag = math.sqrt(node.w * node.w + node.h * node.h) + 1e-12
        norm_len = seg_len / diag
        angle = math.atan2(dy, dx) / math.pi

        inside_quad = self._inside_quadrant(node, ix, iy)
        feat6 = [inside_rel[0], inside_rel[1], inter_rel[0], inter_rel[1], norm_len, angle]
        return feat6, [inter_rel[0], inter_rel[1]], inside_quad, bdir

    # ---------------------------
    # Main per-sample processing
    # ---------------------------
    def process_sample(self, points: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Data:
        """
        处理单个样本，返回 PyG Data 对象。

        输入 edge_index 为无向边的 canonical 表示 (u < v)，每条边只出现一次。
        """
        # Ensure float32 coordinates (preserve integer-grid semantics).
        points = points.to(torch.float32)
        N = points.shape[0]
        points_np = points.detach().cpu().numpy()
        edge_index_np = edge_index.detach().cpu().numpy()

        # --- 1. 构建四叉树 ---
        min_xy = points_np.min(axis=0)
        max_xy = points_np.max(axis=0)
        center = (min_xy + max_xy) / 2.0
        size = float((max_xy - min_xy).max() * 1.01)  # slightly expand

        root_x = float(center[0] - size / 2.0)
        root_y = float(center[1] - size / 2.0)

        self.node_storage: Dict[int, QuadtreeNode] = {}
        self.node_counter = 0
        self.point_to_leaf_old: Dict[int, int] = {}  # point_idx -> old leaf node id

        root_old = self._build_recursive(
            x=root_x, y=root_y, w=size, h=size, depth=0,
            point_indices=list(range(N)), all_points=points_np,
            parent_id=-1, quad_in_parent=-1
        )
        assert root_old == 0, "Root id expected to be 0 in this construction."

        # --- 2. 边注册 (crossing/interface) ---
        num_edges = edge_index_np.shape[1]

        for e_idx in range(num_edges):
            u = int(edge_index_np[0, e_idx])
            v = int(edge_index_np[1, e_idx])

            leaf_u = self.point_to_leaf_old[u]
            leaf_v = self.point_to_leaf_old[v]

            if leaf_u == leaf_v:
                # leaf internal edge
                node_leaf = self.node_storage[leaf_u]
                feat6 = self._edge_feat6_center_relative_endpoints(node_leaf, points_np[u], points_np[v])
                node_leaf.crossing_records.append((e_idx, feat6, -1, -1, True))
                continue

            # different leaves -> crossing at LCA + interface along paths
            lca_id, path_u, path_v = self._find_lca_and_paths(leaf_u, leaf_v)

            # determine which immediate-subtree/quadrant at LCA for each endpoint:
            # path_u/path_v do NOT include LCA. The node closest to LCA is the last one in the segment.
            child_u = path_u[-1] if len(path_u) > 0 else leaf_u
            child_v = path_v[-1] if len(path_v) > 0 else leaf_v
            quad_u = self.node_storage[child_u].quad_in_parent
            quad_v = self.node_storage[child_v].quad_in_parent

            # A) crossing at LCA
            node_lca = self.node_storage[lca_id]
            feat6_lca = self._edge_feat6_center_relative_endpoints(node_lca, points_np[u], points_np[v])
            node_lca.crossing_records.append((e_idx, feat6_lca, quad_u, quad_v, False))

            # B) interface along paths below LCA (each node sees this edge crossing its boundary)
            # For nodes on u-side path, u is inside; for v-side path, v is inside.
            for nid in path_u:
                node = self.node_storage[nid]
                inside_endpoint = 0  # u
                inside_p = points_np[u]
                outside_p = points_np[v]
                # robust check (optional)
                if not self._point_in_node(node, inside_p) and self._point_in_node(node, outside_p):
                    inside_endpoint = 1
                    inside_p, outside_p = outside_p, inside_p

                feat6, inter_rel, inside_quad, bdir = self._interface_feat6_v2(node, inside_p, outside_p)
                node.interface_records.append((e_idx, feat6, inside_endpoint, bdir, inter_rel, inside_quad))

            for nid in path_v:
                node = self.node_storage[nid]
                inside_endpoint = 1  # v
                inside_p = points_np[v]
                outside_p = points_np[u]
                if not self._point_in_node(node, inside_p) and self._point_in_node(node, outside_p):
                    inside_endpoint = 0
                    inside_p, outside_p = outside_p, inside_p

                feat6, inter_rel, inside_quad, bdir = self._interface_feat6_v2(node, inside_p, outside_p)
                node.interface_records.append((e_idx, feat6, inside_endpoint, bdir, inter_rel, inside_quad))

        # --- 3. 扁平化树结构与 leaf CSR ---
        node_ids_sorted = sorted(self.node_storage.keys())
        old_to_new_id = {old: i for i, old in enumerate(node_ids_sorted)}

        node_feats = []
        node_depths = []
        is_leaf_mask = []
        tree_edge_index = []
        tree_parent_index = []
        tree_children_index = []
        node_quadrant_in_parent = []

        for old_id in node_ids_sorted:
            node = self.node_storage[old_id]
            node_feats.append([node.x, node.y, node.w, node.h])
            node_depths.append(node.depth)
            is_leaf_mask.append(node.is_leaf)

            tree_parent_index.append(old_to_new_id[node.parent_id] if node.parent_id != -1 else -1)
            node_quadrant_in_parent.append(node.quad_in_parent)

        # fill children indices
        for old_id in node_ids_sorted:
            node = self.node_storage[old_id]
            children_new = [-1, -1, -1, -1]
            for q in range(4):
                cid_old = node.children[q]
                if cid_old != -1:
                    children_new[q] = old_to_new_id[cid_old]
                    tree_edge_index.append([old_to_new_id[old_id], old_to_new_id[cid_old]])
            tree_children_index.append(children_new)

        if len(tree_edge_index) > 0:
            tree_edge_index = torch.tensor(tree_edge_index, dtype=torch.long).t().contiguous()  # [2, M-1]
        else:
            tree_edge_index = torch.empty((2, 0), dtype=torch.long)

        # leaf CSR build
        leaf_old_ids = [nid for nid in node_ids_sorted if self.node_storage[nid].is_leaf]
        leaf_ids_new = [old_to_new_id[nid] for nid in leaf_old_ids]
        leaf_points_list = []
        leaf_ptr = [0]
        point_to_leaf_new = torch.empty((N,), dtype=torch.long)

        for leaf_old, leaf_new in zip(leaf_old_ids, leaf_ids_new):
            pts = self.node_storage[leaf_old].point_indices
            leaf_points_list.extend(pts)
            leaf_ptr.append(len(leaf_points_list))
            for p_idx in pts:
                point_to_leaf_new[p_idx] = leaf_new

        leaf_points = torch.tensor(leaf_points_list, dtype=torch.long)

        # crossing/interface flatten
        crossing_assigns = []
        crossing_feats = []
        crossing_child_pair = []
        crossing_is_leaf_internal = []

        interface_assigns = []
        interface_feats = []
        interface_inside_endpoint = []
        interface_boundary_dir = []
        interface_inter_rel = []
        interface_inside_quadrant = []

        for old_id in node_ids_sorted:
            node = self.node_storage[old_id]
            new_id = old_to_new_id[old_id]

            for (eid, feat6, qu, qv, is_leaf_internal) in node.crossing_records:
                crossing_assigns.append([new_id, eid])
                crossing_feats.append(feat6)
                crossing_child_pair.append([qu, qv])
                crossing_is_leaf_internal.append(is_leaf_internal)

            for (eid, feat6, inside_ep, bdir, inter_rel, inside_quad) in node.interface_records:
                interface_assigns.append([new_id, eid])
                interface_feats.append(feat6)
                interface_inside_endpoint.append(inside_ep)
                interface_boundary_dir.append(bdir)
                interface_inter_rel.append(inter_rel)
                interface_inside_quadrant.append(inside_quad)

        # --- 4. 封装 PyG Data ---
        data = Data()
        # NOTE:
        # - `num_nodes` is kept for backward compatibility and refers to the number of TSP points.
        # - Tree nodes live in a different index space; we store `num_tree_nodes` explicitly.
        data.num_nodes = N
        data.num_points = N

        data.pos = points  # [N, 2] float32

        # Root square bbox for per-instance normalization: (x, y, w, h).
        data.root_bbox = torch.tensor([root_x, root_y, size, size], dtype=torch.float32)

        # Contract version tag for downstream modules: avoid auto-detect / implicit assumptions.
        # v2 means:
        #   - tree_node_feat is llwh absolute
        #   - crossing_edge_attr is center-relative endpoints + (norm_len, angle)
        #   - interface_edge_attr is interface v2 token (inside_rel, inter_rel, norm_len, angle)
        data.coord_contract_version = torch.tensor([2], dtype=torch.long)
        data.tree_node_feat_box_mode = torch.tensor([0], dtype=torch.long)  # 0 := llwh(abs)

        # Scale-invariant coordinates in the root box: pos_norm in [0, 1] (approximately).
        root_xy = torch.tensor([root_x, root_y], dtype=torch.float32)
        data.pos_norm = (points - root_xy) / float(size)

        # original spanner (absolute lengths). Also provide per-instance normalized length for NN features.
        data.spanner_edge_index = edge_index
        data.spanner_edge_attr = edge_attr
        root_diag = float(size) * math.sqrt(2.0) + 1e-12
        data.spanner_edge_attr_norm = edge_attr / root_diag

        # tree node tensors
        data.tree_node_feat = torch.tensor(node_feats, dtype=torch.float)
        data.tree_node_depth = torch.tensor(node_depths, dtype=torch.long)
        data.is_leaf = torch.tensor(is_leaf_mask, dtype=torch.bool)
        data.num_tree_nodes = int(data.tree_node_feat.shape[0])

        data.tree_edge_index = tree_edge_index
        data.tree_parent_index = torch.tensor(tree_parent_index, dtype=torch.long)
        data.tree_children_index = torch.tensor(tree_children_index, dtype=torch.long)
        data.node_quadrant_in_parent = torch.tensor(node_quadrant_in_parent, dtype=torch.long)

        # leaf CSR
        data.leaf_ids = torch.tensor(leaf_ids_new, dtype=torch.long)           # [L]
        data.leaf_ptr = torch.tensor(leaf_ptr, dtype=torch.long)               # [L+1]
        data.leaf_points = leaf_points                                         # [sum |leaf|]
        data.point_to_leaf = point_to_leaf_new                                 # [N]

        # crossing tensors (keep original names for compatibility)
        if len(crossing_assigns) > 0:
            data.crossing_assign_index = torch.tensor(crossing_assigns, dtype=torch.long).t()  # [2, C]
            data.crossing_node_index = data.crossing_assign_index[0].contiguous()              # [C]
            data.crossing_eid_index = data.crossing_assign_index[1].contiguous()               # [C]
            data.crossing_edge_attr = torch.tensor(crossing_feats, dtype=torch.float)          # [C, 6]
            data.crossing_child_pair = torch.tensor(crossing_child_pair, dtype=torch.long)     # [C, 2]
            data.crossing_is_leaf_internal = torch.tensor(crossing_is_leaf_internal, dtype=torch.bool)  # [C]
        else:
            data.crossing_assign_index = torch.empty((2, 0), dtype=torch.long)
            data.crossing_node_index = torch.empty((0,), dtype=torch.long)
            data.crossing_eid_index = torch.empty((0,), dtype=torch.long)
            data.crossing_edge_attr = torch.empty((0, 6), dtype=torch.float)
            data.crossing_child_pair = torch.empty((0, 2), dtype=torch.long)
            data.crossing_is_leaf_internal = torch.empty((0,), dtype=torch.bool)

        # interface tensors (raw, unpruned)
        # keep interface_assign_index for compatibility, but now also store split indices
        if len(interface_assigns) > 0:
            data.interface_assign_index = torch.tensor(interface_assigns, dtype=torch.long).t()     # [2, I]
            data.interface_node_index = data.interface_assign_index[0].contiguous()                # [I]
            data.interface_eid_index = data.interface_assign_index[1].contiguous()                 # [I]
            data.interface_edge_attr = torch.tensor(interface_feats, dtype=torch.float)             # [I, 6]
            data.interface_inside_endpoint = torch.tensor(interface_inside_endpoint, dtype=torch.long)  # [I]
            data.interface_boundary_dir = torch.tensor(interface_boundary_dir, dtype=torch.long)    # [I]
            # Deprecated but kept for now (used by existing packer/tests): center-relative intersection coords.
            data.interface_inside_quadrant = torch.tensor(interface_inside_quadrant, dtype=torch.long)   # [I]
        else:
            data.interface_assign_index = torch.empty((2, 0), dtype=torch.long)
            data.interface_node_index = torch.empty((0,), dtype=torch.long)
            data.interface_eid_index = torch.empty((0,), dtype=torch.long)
            data.interface_edge_attr = torch.empty((0, 6), dtype=torch.float)
            data.interface_inside_endpoint = torch.empty((0,), dtype=torch.long)
            data.interface_boundary_dir = torch.empty((0,), dtype=torch.long)
            data.interface_inside_quadrant = torch.empty((0,), dtype=torch.long)

        return data

    # ---------------------------
    # Quadtree recursive build
    # ---------------------------
    def _build_recursive(
        self,
        x: float, y: float, w: float, h: float,
        depth: int,
        point_indices: List[int],
        all_points: np.ndarray,
        parent_id: int,
        quad_in_parent: int
    ) -> int:
        node_id = self.node_counter
        self.node_counter += 1

        node = QuadtreeNode(
            node_id=node_id,
            x=x, y=y, w=w, h=h,
            depth=depth,
            parent_id=parent_id,
            quad_in_parent=quad_in_parent
        )
        node.point_indices = point_indices
        self.node_storage[node_id] = node

        # stop condition
        if depth >= self.max_depth or len(point_indices) <= self.max_points:
            node.is_leaf = True
            for p_idx in point_indices:
                self.point_to_leaf_old[p_idx] = node_id
            return node_id

        # split
        node.is_leaf = False
        mid_x = x + w / 2.0
        mid_y = y + h / 2.0
        half_w = w / 2.0
        half_h = h / 2.0

        quadrants = {TL: [], TR: [], BL: [], BR: []}
        for p_idx in point_indices:
            px, py = float(all_points[p_idx, 0]), float(all_points[p_idx, 1])
            if px < mid_x:
                if py >= mid_y:
                    quadrants[TL].append(p_idx)
                else:
                    quadrants[BL].append(p_idx)
            else:
                if py >= mid_y:
                    quadrants[TR].append(p_idx)
                else:
                    quadrants[BR].append(p_idx)

        child_params = [
            (x,     mid_y, half_w, half_h),  # TL
            (mid_x, mid_y, half_w, half_h),  # TR
            (x,     y,     half_w, half_h),  # BL
            (mid_x, y,     half_w, half_h),  # BR
        ]

        for q in range(4):
            p_idxs = quadrants[q]
            if len(p_idxs) == 0:
                continue
            cx, cy, cw, ch = child_params[q]
            child_id = self._build_recursive(
                x=cx, y=cy, w=cw, h=ch,
                depth=depth + 1,
                point_indices=p_idxs,
                all_points=all_points,
                parent_id=node_id,
                quad_in_parent=q
            )
            node.children[q] = child_id

        return node_id

    # ---------------------------
    # LCA and path utilities
    # ---------------------------
    def _find_lca_and_paths(self, leaf_u: int, leaf_v: int) -> Tuple[int, List[int], List[int]]:
        """
        Returns:
            lca_id
            path_u: nodes on the path from leaf_u up to (excluding) lca
            path_v: nodes on the path from leaf_v up to (excluding) lca
        """
        # climb to root
        path_u_full = []
        cur = leaf_u
        while cur != -1:
            path_u_full.append(cur)
            cur = self.node_storage[cur].parent_id

        path_v_full = []
        cur = leaf_v
        while cur != -1:
            path_v_full.append(cur)
            cur = self.node_storage[cur].parent_id

        path_u_rev = path_u_full[::-1]
        path_v_rev = path_v_full[::-1]

        lca = -1
        min_len = min(len(path_u_rev), len(path_v_rev))
        for i in range(min_len):
            if path_u_rev[i] == path_v_rev[i]:
                lca = path_u_rev[i]
            else:
                break

        if lca == -1:
            raise ValueError("Tree disconnected")

        # segments excluding lca
        path_u_segment = []
        cur = leaf_u
        while cur != lca:
            path_u_segment.append(cur)
            cur = self.node_storage[cur].parent_id

        path_v_segment = []
        cur = leaf_v
        while cur != lca:
            path_v_segment.append(cur)
            cur = self.node_storage[cur].parent_id

        return lca, path_u_segment, path_v_segment


def build_raw_dataset(input_path, output_path, max_points, max_depth):
    print(f"[Step 1] Loading Spanner Graph from {input_path}...")
    try:
        data_dict = torch.load(input_path)
    except Exception:
        data_dict = torch.load(input_path, weights_only=False)

    all_points = data_dict['points']         # [B, N, 2]
    all_edges = data_dict['edge_index']      # [2, E_total] (undirected, disjoint union)
    all_edge_attrs = data_dict['edge_attr']  # [E_total, 1]
    batch_idx = data_dict['batch_idx']       # [E_total]

    num_samples = all_points.shape[0]
    N = all_points.shape[1]
    processed_list = []

    print(f"Configuring Builder: max_points={max_points}, max_depth={max_depth}")
    builder = RawPyramidBuilder(max_points_per_leaf=max_points, max_depth=max_depth)

    for b in tqdm(range(num_samples)):
        mask = (batch_idx == b)
        points = all_points[b].to(torch.float32)
        edges = all_edges[:, mask] - (b * N)  # shift back to 0-based
        attrs = all_edge_attrs[mask]

        data = builder.process_sample(points, edges, attrs)
        processed_list.append(data)

    print(f"Saving Raw Pyramid Dataset to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(processed_list, output_path)
    print("Done. Next step: Pruning.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input *_spanner.pt")
    parser.add_argument("--output", type=str, required=True, help="Output *_raw_pyramid.pt")
    parser.add_argument("--max_points", type=int, default=20, help="Max points per leaf node (Termination condition)")
    parser.add_argument("--max_depth", type=int, default=20, help="Max tree depth (Termination condition)")
    args = parser.parse_args()

    build_raw_dataset(args.input, args.output, args.max_points, args.max_depth)
