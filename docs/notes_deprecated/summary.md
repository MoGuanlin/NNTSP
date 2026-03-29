# mgl (NNTSP) 项目总结

该仓库是一个基于**深度学习和计算几何**的研究项目，旨在通过**层级神经网络（Hierarchical Neural Network）**高效解决**旅行商问题（TSP）**。其核心逻辑是通过分治和层级化的思想，将大规模的 TSP 问题分解为更简单的子问题进行预测。

## 核心功能模块总结

### 1. 层级数据表示 (`src/graph/` & `src/utils/`)
*   **空间剖分 (Quadtree)**: 使用 `src/graph/build_raw_pyramid.py` 将点集构建成四叉树结构的“金字塔”。
*   **图简化 (Spanner)**: 通过 `src/graph/spanner.py` 在点集上构建生成子图，通过边的稀疏化降低模型负担。
*   **层级剪枝 (r-light Pruning)**: 核心算法组件，利用 `src/graph/prune_pyramid.py` 实现 `r-light` 剪枝，确保模型处理的节点和边保持在可控范围内。

### 2. 模型架构 (`src/models/`)
模型采用了**自底向上（Bottom-Up）编码**和**自顶向下（Top-Down）解码**的对称结构：
*   **编码器 (`LeafEncoder`, `MergeEncoder`)**: 负责从底层的叶节点开始，逐步向上提取空间分布特征。
*   **解码器 (`TopDownDecoder`, `TopDownTreeRunner`)**: 在高层语义特征的基础上，逐步向下回归出边的决策。
*   **核心组件**: 包含基于 `Set Transformer` 的特征提取，以及专门设计的 `NodeTokenPacker` 用于高效的批处理。

### 3. 训练与教师学习 (`train.py`, `src/models/labeler.py`)
*   **伪标签学习 (Pseudo-Labeling)**: 使用 `PseudoLabeler`（内置了 2-opt 等启发式算法）生成教师标签来监督神经网络的学习。
*   **多任务损失**: 包含边预测的 BCE 损失以及层级间的一致性损失（BC Loss）。

### 4. 工作流与工具 (`Makefile`, `src/visualization/`)
*   **自动化流水线**: `Makefile` 定义了从数据生成 (`make data`)、图构建 (`make spanner`) 到金字塔构建 (`make rlight`) 的全过程。
*   **可视化**: `src/visualization/visualize_pyramid.py` 提供了一个强大的调试工具，可以渲染出不同层级的节点、边以及剪枝效果。

## 总结
该项目的技术特点在于：它不是简单地用 GNN 处理全连通图，而是利用**计算几何的先验知识**（四叉树、Spanner）显著降低了问题的搜索空间，并通过层级化的神经网络架构实现了对大规模 TSP 问题的建模，非常适合处理海量点集的路径优化任务。
