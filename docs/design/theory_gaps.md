# 理论差距分析：当前实现 vs Rao'98 PTAS (1+ε)-近似

**Date**: 2026-03-27
**Status**: Analysis

## 1. 概述

本文档分析当前 NNTSP 实现与 Rao'98 PTAS 理论算法之间的差距。Rao'98 对欧氏 TSP 给出了 (1+ε)-近似保证，其核心依赖一系列严格的理论构件。当前实现是论文算法的**神经网络近似版本**，在多个环节用工程启发式替代了理论构件，因此不具备严格的 (1+ε)-近似保证。

## 2. 差距总览

| # | 组件 | 理论要求 | 当前实现 | 影响程度 |
|---|------|----------|----------|----------|
| 1 | Spanner | (1+ε)-stretch | Delaunay 三角剖分 | 高 |
| 2 | 四叉树 | 随机偏移 | 固定对齐 | 高 |
| 3 | Portal | 离散穿越点 | 连续坐标 | 高 |
| 4 | 状态空间 | O(1/ε) 增长 | 固定 max_used=4 | 高 |
| 5 | 叶求解 | 精确 | ≤6 点枚举 | 中 |
| 6 | PARSE | 精确查表 | 启发式舍入 | 中 |
| 7 | Patching | Structure Theorem | 无 | 高 |

## 3. 逐项分析

### 3.1 Spanner：Delaunay 三角剖分 vs (1+ε)-stretch

**理论要求：** Rao'98 假设在 spanner 图上做 DP，该 spanner 必须满足 stretch 约束——对任意两点 u, v，spanner 中 u→v 最短路长度 ≤ (1+ε)·‖u-v‖₂。常用构造方法包括 MST-based spanner 或 ε-net。

**当前实现：** `src/graph/spanner.py` 使用 Delaunay 三角剖分，失败时回退到 k-NN。

**问题：**
- Delaunay 三角剖分**不保证** bounded stretch。存在点集配置使得 Delaunay 中某些点对之间的路径远长于欧氏距离。
- 限制在 spanner 边上的最优 tour 可能与真正的欧氏最优 tour 有任意大的差距。
- k-NN 回退同样不提供 stretch 保证。

**实际影响：** Delaunay 在随机/均匀分布点集上通常表现良好（实际 stretch 很小），属于"理论不行但工程可用"。对结构化/对抗性输入可能失效。

**修复方向：** 替换为显式 (1+ε)-spanner 构造（如 Θ-graph 或 WSPD-based spanner），或在 Delaunay 基础上添加额外边以保证 stretch。

### 3.2 四叉树：固定对齐 vs 随机偏移

**理论要求：** Rao'98 使用**随机偏移四叉树**（randomly shifted quadtree）。随机选择偏移量 (a, b) ∈ [0, L)²，使得四叉树边界以均匀概率穿过点集。关键性质：对于最优 tour 的任意一条长度为 ℓ 的边，它被第 i 层边界切割的概率 ≤ O(ℓ / cell_size_i)。对所有层求和后，期望额外代价 ≤ ε·OPT。

**当前实现：** `src/graph/build_raw_pyramid.py` 中四叉树固定对齐原点，无随机偏移参数。

**问题：**
- 没有概率保证：对抗性输入可以让所有长边都被边界切割，导致额外代价不可控。
- Rao'98 需要对 O(1) 个不同偏移分别运行算法，取最优结果。当前只运行一次。

**修复方向：**
1. 在 `build_raw_pyramid` 中添加随机偏移参数 `(shift_x, shift_y)`。
2. 运行多次（或使用理论确定的 O(1/ε²) 个偏移），取最优。

### 3.3 Portal 离散化：连续穿越 vs 离散 Portal

**理论要求：** Rao'98 在每条边界上等间距放置 **portal**（穿越点），间距为 cell_size / m，其中 m = O(1/ε)。tour 只能在 portal 处穿越边界。这将每条边的穿越位置限制在 O(m) 个离散点上，使得 DP 状态空间有限。

**当前实现：** Interface token 记录 spanner 边与边界的**实际交点坐标**（连续值），存储在 `iface_feat6` 中。没有任何离散化到 portal 格点的操作。

**问题：**
- 穿越位置是连续的 → 理论上有无穷多种穿越模式。
- DP 状态只编码"哪些 interface slot 活跃 + 配对"，不区分同一 slot 内的不同穿越位置。
- 代价分解 C_B[σ] = Σ C_{B_i}[τ_i] 的正确性依赖于 portal 结构——穿越位置必须在父子节点间精确对应。没有 portal，这种对应是近似的。

**修复方向：**
1. 在 pyramid 构建时将穿越位置 snap 到最近 portal。
2. 修改 interface token 使得每个 token 对应一个 portal（而非一条 spanner 边的交点）。
3. 状态枚举基于 portal 而非 interface slot。

### 3.4 状态空间：固定 max_used vs ε-dependent

**理论要求：** Rao'98 的 DP 状态空间大小为 |Ω(B)| = m^{O(m)}，其中 m = O(1/ε)。当 ε 减小时，状态空间增大，允许更精确的近似。这是 PTAS 的核心——通过增大计算量换取任意精度。

**当前实现：** `src/models/bc_state_catalog.py` 中 `max_used` 固定为 4（或用户指定的常数），与 ε 无关。

**问题：**
- `max_used=4` 意味着每条边界最多 4 个活跃穿越。如果最优 tour 需要 >4 次穿越，DP 永远找不到它。
- 没有 ε 参数——用户无法通过增大 `max_used`/`r` 来提升近似精度。
- 实际上 `r=4` 对应 Ti=16 个 interface slot，但 `max_used=4` 只允许其中 4 个活跃，这是一个非常强的限制。

**修复方向：**
1. 将 `max_used` 设为 `r` 或 `2r`，使其随精度需求增长。
2. 文档化 `max_used` 与 ε 的关系：ε ≈ C / max_used。
3. 提供 "quality vs speed" 的参数调节接口。

### 3.5 叶节点精确求解

**理论要求：** 叶节点内的受约束 TSP（访问所有内部点 + 匹配边界状态端点）必须**精确求解**。Rao'98 中叶节点包含 O(1) 个点（由四叉树深度控制），所以精确求解的开销是常数。

**当前实现：** `src/models/dp_core.py` 的叶节点求解器对 ≤6 个点使用全排列枚举，>6 个点直接标记为 infeasible（`costs[si] = +inf`）。

**问题：**
- 叶节点可能包含 >6 个点（取决于点密度和四叉树深度），此时 DP 无法找到可行解。
- 没有 Steiner 点处理——理论上路径可以经过非输入点以缩短长度。
- 枚举复杂度 O(k!) 限制了可处理的点数。

**修复方向：**
1. 使用更高效的精确 TSP 求解器（如 Held-Karp DP，O(2^k · k²)，可处理 ~20 个点）。
2. 控制四叉树深度使叶节点点数保持 O(1)。
3. 对于大叶节点，使用 Christofides 或 LKH 作为近似回退。

### 3.6 PARSE：启发式 vs 精确离散化

**理论要求：** 给定连续预测分数，需要找到最优的合法离散状态。在理论算法中，这一步是精确的——直接在有限 portal 集上枚举。

**当前实现：** `src/models/dp_core.py` 的 `parse_continuous()` 使用：
- 阈值舍入（默认 0.5）
- 每侧预算限制（≤r 个活跃）
- 奇偶校正（翻转最低分 slot）
- 共享边界平均
- Top-K fallback（尝试 7 个不同阈值）

**问题：**
- 启发式可能产生次优离散化——不保证找到最接近连续预测的合法状态。
- Top-K fallback 是 ad-hoc 的，没有覆盖率保证。
- 如果启发式失败，回退到精确枚举（≤10000 组合），但枚举范围有限。

**修复方向：**
1. 在 portal 化后，离散化变成查表操作，PARSE 的复杂度可控。
2. 或使用 ILP 求解精确最优离散化。

### 3.7 Patching Lemma / Structure Theorem

**理论要求：** Rao'98 的核心定理（Structure Theorem）：对于最优 tour OPT，存在一个代价 ≤ (1+ε)·OPT 的 tour，使得每条边界最多被穿越 O(r) 次，且穿越只发生在 portal 上。证明方法是 **patching**——将 OPT 中多余的边界穿越"修补"掉，代价增加 ≤ ε·OPT。

**当前实现：** 没有 patching 步骤。依赖 r-light 剪枝（`src/graph/prune_pyramid.py`）间接控制穿越数量——通过删除过长的 spanner 边来限制每条边界的 interface 数量。

**问题：**
- r-light 剪枝是对 spanner 边的启发式操作，不等价于对 tour 的 patching。
- 删除边可能移除最优 tour 必需的边。
- 没有理论保证剪枝后的图仍能找到 (1+ε)-近似 tour。
- Patching 是对 tour 的操作（重新路由），不是对图的操作（删边）。

**修复方向：**
1. 实现显式 patching：给定一个 tour，修改其中穿越过多的边界段。
2. 或证明 r-light 剪枝 + 有限状态枚举等价于 patching 的效果（可能不成立）。

## 4. 差距之间的依赖关系

```
随机偏移四叉树 ──→ Portal 离散化 ──→ 有限状态空间 ──→ 精确叶求解
     │                   │                   │              │
     └─── Patching ──────┘                   │              │
                                             ↓              ↓
                                    PARSE 变为查表    代价分解正确
                                             │              │
                                             └──── (1+ε) ───┘
```

- Portal 离散化是核心：它使状态空间有限、PARSE 变为精确操作、代价分解严格成立。
- 随机偏移 + Patching 共同保证额外代价 ≤ ε·OPT。
- Spanner stretch 保证限制在 spanner 上的最优解与欧氏最优解的差距。

## 5. 实践意义

尽管存在上述理论差距，当前实现在实践中仍然有效：

1. **Delaunay spanner**：在随机/半随机点集上 stretch 通常很小（经验上 <1.5），足以找到高质量 tour。
2. **固定四叉树**：对于随机点集，固定对齐与随机偏移的差异很小。
3. **max_used=4**：经验上大多数边界穿越 ≤4 次，尤其在 r=4 的情况下。
4. **PARSE 启发式**：配合 top-k fallback 和精确枚举回退，实际成功率很高。

理论差距主要影响：
- **最坏情况保证**（对抗性输入）
- **可调精度**（通过 ε 控制近似质量）
- **形式化正确性**（证明算法输出满足近似比）

## 6. 如果要实现严格 (1+ε)-近似

按优先级排序的修复路线：

1. **Portal 离散化**（最核心）→ 使 DP 状态空间严格定义
2. **随机偏移四叉树** → 提供概率保证
3. **ε-dependent 状态空间** → max_used = f(1/ε)
4. **(1+ε)-stretch spanner** → 替换 Delaunay
5. **Patching lemma 实现** → 控制额外代价
6. **叶节点精确求解扩展** → Held-Karp DP
7. **PARSE 精确化** → portal 化后自然解决
