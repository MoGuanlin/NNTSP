import torch
import numpy as np
import argparse
import os
from scipy.spatial import Delaunay
from time import time
from typing import Tuple, List
from torch.utils.data import Dataset, DataLoader

class SpannerDataset(Dataset):
    def __init__(self, points_np: np.ndarray, mode: str, build_fn):
        self.points_np = points_np
        self.mode = mode
        self.build_fn = build_fn
        self.N = points_np.shape[1]

    def __len__(self):
        return self.points_np.shape[0]

    def __getitem__(self, b: int):
        p = self.points_np[b]
        if self.mode == 'delaunay':
            edges = self.build_fn(p)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Local results
        offset = b * self.N
        edges_offset = (edges + offset).copy()
        num_edges = edges.shape[1]
        b_ids = np.full((num_edges,), b, dtype=np.int64)
        # Return as numpy to avoid PyTorch shared memory / mmap limits in some environments (e.g. Docker)
        return edges_offset, b_ids

class SpannerBuilder:
    """
    将点云数据转换为几何稀疏图 (Geometric Spanner)。
    默认使用 Delaunay Triangulation。

    IMPORTANT:
    - 本版本输出“无向边集合”的一种规范表示：每条边只出现一次，且满足 u < v。
    - 若后续 GNN 需要双向消息传递，请在模型 forward 里临时扩展为双向边
      （例如 torch_geometric.utils.to_undirected / 或手动 concat 反向边）。
    """
    def __init__(self, mode: str = 'delaunay'):
        self.mode = mode

    def build_batch(self, points: torch.Tensor, num_workers: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        处理整个 Batch 的点集。

        Args:
            points: [B, N, 2] 坐标张量 (Float or Int)
            num_workers: 并行核心数

        Returns:
            edge_index: [2, E_total] 无向边索引 (LongTensor, u < v).
                        注意：这是 Disjoint Union 格式，Batch i 的索引偏移了 i*N.
            edge_attr:  [E_total, 1] 边的欧氏距离 (FloatTensor).
            batch_idx:  [E_total] 指示每条边属于哪个样本 (LongTensor).
        """
        B, N, _ = points.shape
        device = points.device

        # Scipy 不支持 GPU，转到 CPU numpy
        points_np = points.detach().cpu().float().numpy()

        print(f"Building {self.mode} spanner for {B} samples (N={N}) [UNDIRECTED] (workers={num_workers}) ...")
        t0 = time()

        # Use a moderate batch_size
        dataset = SpannerDataset(points_np, self.mode, self._build_delaunay_topology)
        loader = DataLoader(
            dataset, 
            batch_size=min(128, max(1, B // (num_workers * 2))) if num_workers > 1 else B,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=lambda x: x  # list of tuples
        )

        edge_lists = []
        batch_ids = []
        
        for batch in loader:
            for e_off_np, b_idx_np in batch:
                edge_lists.append(torch.from_numpy(e_off_np))
                batch_ids.append(torch.from_numpy(b_idx_np))

        print(f"Topology build time: {time() - t0:.4f}s")

        # concat
        edge_index = torch.cat(edge_lists, dim=1).to(device)
        batch_idx = torch.cat(batch_ids, dim=0).to(device)

        # compute edge_attr
        points_flat = points.view(-1, 2).float()
        src_coords = points_flat[edge_index[0]]
        dst_coords = points_flat[edge_index[1]]
        edge_attr = torch.norm(src_coords - dst_coords, p=2, dim=-1, keepdim=True)

        return edge_index, edge_attr, batch_idx

    def _build_delaunay_topology(self, points: np.ndarray) -> np.ndarray:
        """
        对单个点集进行 Delaunay 三角剖分，返回无向图边索引（每条边只保留一次，u < v）。
        """
        n = points.shape[0]

        # 点数太少：退化为 KNN
        if n < 4:
            return self._build_knn_topology(points, k=max(1, n - 1))

        try:
            tri = Delaunay(points)
        except Exception:
            # 可能完全共线/Qhull 失败：降级为 KNN
            return self._build_knn_topology(points, k=min(5, max(1, n - 1)))

        # tri.simplices: [num_tri, 3]
        edges_list = [
            tri.simplices[:, [0, 1]],
            tri.simplices[:, [1, 2]],
            tri.simplices[:, [2, 0]]
        ]
        edges = np.concatenate(edges_list, axis=0)  # [3*num_tri, 2]

        # canonicalize: sort endpoints so u < v, then unique
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)  # [E, 2]

        # transpose to [2, E]
        return edges.T.astype(np.int64)

    def _build_knn_topology(self, points: np.ndarray, k: int) -> np.ndarray:
        """
        Backup: 当 Delaunay 失败时的备选方案。
        输出无向边集合（每条边只保留一次，u < v）。
        """
        from sklearn.neighbors import NearestNeighbors

        n = points.shape[0]
        if n <= 1:
            return np.empty((2, 0), dtype=np.int64)

        k = int(min(max(1, k), n - 1))

        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
        _, indices = nbrs.kneighbors(points)

        src = np.repeat(np.arange(n), k)
        dst = indices[:, 1:].reshape(-1)  # 去掉自身

        edges = np.stack([src, dst], axis=1)  # [n*k, 2]
        edges = np.sort(edges, axis=1)        # canonical u < v

        # unique undirected
        edges = np.unique(edges, axis=0)

        return edges.T.astype(np.int64)

def process_dataset(input_path, output_path, num_workers: int = 1):
    print(f"Loading data from {input_path}...")
    try:
        data = torch.load(input_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Data shape: {data.shape}")  # [B, N, 2]

    builder = SpannerBuilder(mode='delaunay')
    edge_index, edge_attr, batch_idx = builder.build_batch(data, num_workers=num_workers)

    # stats (UNDIRECTED)
    num_samples = data.shape[0]
    num_nodes = data.shape[1]
    total_edges = edge_index.shape[1]
    avg_degree = (2.0 * total_edges) / (num_samples * num_nodes)

    print(f"--- Spanner Statistics (UNDIRECTED) ---")
    print(f"Total Undirected Edges: {total_edges}")
    print(f"Average Degree: {avg_degree:.2f} (Delaunay planar expected ~6.0)")
    print(f"Max Edge Length: {edge_attr.max().item():.2f}")

    save_dict = {
        "points": data,             # [B, N, 2]
        "edge_index": edge_index,   # [2, E_total] undirected u<v
        "edge_attr": edge_attr,     # [E_total, 1]
        "batch_idx": batch_idx      # [E_total]
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(save_dict, output_path)
    print(f"Spanner graph data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct Spanner Graphs from Points")
    parser.add_argument("--input", type=str, required=True, help="Path to input .pt file (from data_generator)")
    parser.add_argument("--output", type=str, required=True, help="Path to output .pt file")
    parser.add_argument("--num_workers", type=int, default=1)

    args = parser.parse_args()
    process_dataset(args.input, args.output, num_workers=args.num_workers)
