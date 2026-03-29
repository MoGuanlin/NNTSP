# NNTSP（Neural Rao'98 TSP）理论算法交接文档（新版）

**代码快照日期**：2025-12-31  
**适用范围**：本项目当前实现的“Rao'98 风格 TSP Neural DP”（含数据预处理、bottom-up 神经 DP、top-down 边界条件传播、teacher 伪标签、训练与解码评估）。  
**目标读者**：需要在不重新通读所有代码的情况下，快速理解本项目算法思想、状态定义、训练信号，以及“完全规模泛化性（scale invariance）”来自哪里。  

---

## 0. 一页纸大框架（浅显易懂版）

把 Rao'98 的动态规划（DP）想象成对一棵 quadtree 的“**自底向上算值函数** + **自顶向下回溯边界条件**”。

- **数据结构**：把平面点集构成 quadtree（每个 box 是一个树节点）。我们不在连续几何上直接做 DP，而是把“可能参与解的几何边”限制到一个稀疏候选集（spanner）上，并对每个 box 只保留常数个“接口（interface）”候选与“跨子盒连边（crossing）”候选（r-light 剪枝）。
- **DP 状态（本项目的关键抽象）**：对每个 box 的每个 interface，我们用一个 **logit/score** 表示“该 interface 是否在最终 tour 中被使用”。  
  - 这一步是“**状态表示的规模无关化**”：每个 box 的 interface 数量被限制为常数 `Ti`，所以状态向量长度不随点数 N 增长。
- **bottom-up（神经值函数）**：叶子节点与内部节点各自用一个 encoder 产生 `z_v`（latent），它应当概括“在该 box 内部如何连线”的最优代价/结构信息（近似 Rao'98 的 DP 表条目）。
- **top-down（神经回溯/策略）**：对内部节点 v，decoder 读入：
  1) 上层传来的边界条件分数 `bc_in_iface_logit(v)`（长度 `Ti`）  
  2) v 自己的局部 token（v 的 interface/crossing 几何特征）  
  3) v 与孩子的 bottom-up latent（`z_v` 和 `z_child[q]`）  
  并输出 **四个孩子的边界条件分数** `child_iface_logit[v, q, i]`（q=0..3）。  
  这正对应“父 box 决定子 box 的边界条件”，与 Rao'98 的 DP 回溯一致。
- **teacher / 训练信号**：用启发式 TSP tour（NN+2opt）当 teacher；把 teacher 边投影到“alive spanner”上；据此对 token 打标签（cross/iface 是否被 teacher 选中），并把“孩子的 y_iface”搬运成“父→子 BC 监督”。
- **预测/解码**：模型输出 crossing logits → 聚合成 edge logits → 用贪心+patching 解码成 Hamiltonian cycle，用于评估。

> 当前实现的核心是“**接口使用与否**”的边界条件打分；“接口两两配对（matching）”尚未纳入状态/输出（这是理论上更完整的 Rao/Arora DP 方向，也是后续最重要扩展点之一）。

---

## 1. 本项目与 Rao'98 的对应关系

### 1.1 Rao'98 里我们“严格保留”的骨架
- **Spanner**：先把几何图限制到稀疏候选边（本代码用 Delaunay triangulation 得到平面稀疏图）。
- **Quadtree box 递归**：DP 在树上进行，box 按 4 叉分裂，孩子顺序固定（TL/TR/BL/BR）。
- **边在树上的“归属”**：
  - 如果一条边 (u,v) 的两端落在同一 leaf：它是 leaf 内部的“局部连接”。
  - 否则，在 LCA box 处产生“crossing”，并在 LCA 到端点 leaf 的路径上产生 interface（表示边穿过每层 box 的边界）。

### 1.2 Rao'98 里我们“工程化替代”的部分
- **r-light 化**：理论上 r-light 是关于最优解存在性的结构性引理，未必直接“删边”。  
  本项目为了实现 **token 数常数上界**，在 `prune_pyramid.py` 里做了工程版：对每个 `(node, boundary_dir)` 只保留最短的前 r 条 interface 记录，并据此“kill”全局边。  
  这是一个关键取舍：它换来强 scale invariance，但严格的 (1+ε) 近似证明需要额外论证“删边对近似比的影响”。

---

## 2. 状态定义：Interface / Crossing 与边界条件（BC）

### 2.1 Interface token（box 边界上的“穿越点/接口”）
每个 interface token 对应一条 spanner 边在某个 box 边界处的穿越事件；其核心字段是：

- `interface_assign_index = [2, I]`：第 0 行是 `node_id`，第 1 行是 `eid`（spanner edge id）。
- `interface_edge_attr = [I, 6]`（**contract v2**）：
  - inside endpoint 的相对坐标 `(inside_rel_x, inside_rel_y)`
  - 穿越点在边界上的相对坐标 `(inter_rel_x, inter_rel_y)`
  - 归一化长度 `norm_len`
  - 方向角 `angle`
- 离散属性：
  - `interface_boundary_dir`：该 interface 落在哪条边界（0/1/2/3 表示 L/R/B/T）
  - `interface_inside_endpoint`：inside 是边的哪个端点（u 还是 v）
  - `interface_inside_quadrant`：inside 端点在当前 box 的哪个孩子象限（0..3）

**边界条件（BC）的当前表示**：  
对每个 node v，我们维护 `bc_iface_logit[v, i]`（i=0..Ti-1），表示“v 的第 i 个 interface 被使用”的打分。  
这就是 top-down DP 传播的状态向量。

### 2.2 Crossing token（父节点处的“跨孩子连接候选”）
每个 crossing token 对应一条 spanner 边在其 LCA box 处连接两个孩子象限（或者 leaf 内部边）。

- `crossing_assign_index = [2, C]`：第 0 行是 `node_id`，第 1 行是 `eid`
- `crossing_edge_attr = [C, 6]`：两端点相对 box 中心的归一化坐标 + 归一化长度 + angle
- `crossing_child_pair = [C, 2]`：两端分别落在哪两个孩子象限
- `crossing_is_leaf_internal`：是否为 leaf 内部边

crossing 更像“父节点负责协调孩子之间连接”的证据：它可以被 decoder 作为“强约束信号”读取，并最终被 edge-aggregation 用于解码整条 tour。

---

## 3. 规模泛化性（Scale Invariance）从哪里来？

本项目“完全规模泛化性”的含义是：**神经网络的输入/输出维度不随 N（点数）增长**。代码中体现在以下设计点：

1. **每个节点 token 数有常数上界**
   - `Ti`：每个节点 interface token 上界
     - 若 packer 指定 `max_iface_per_node`，则 `Ti = max_iface_per_node`
     - 否则若指定 r，则 `Ti = 4r`
   - `Tc`：每个节点 crossing token 上界
     - 若 packer 指定 `max_cross_per_node`，则 `Tc = max_cross_per_node`
     - 否则若指定 r，则 `Tc = 8r`
2. **leaf 点集 token 数有常数上界**
   - 每个 leaf 至多保留 `P=max_points_per_leaf` 个点（不足 padding），并提供 mask。
3. **坐标/长度全部归一化**
   - token 连续特征都是 box 相对坐标（围绕 box 中心）与相对尺度（除以 box 宽/高或 root 尺度），避免 N 扩大导致数值尺度漂移。
4. **深度 embedding clip**
   - `NodeTokenizer` 的 depth embedding 以 `max_depth` 截断，避免树深随 N 增长导致 embedding 维度/参数随规模增长。

---

## 4. Neural DP：Bottom-up 与 Top-down 的数学语义

### 4.1 Bottom-up：从孩子到父的“值函数/摘要”
- leaf encoder 学一个函数：`z_v = f_leaf(tokens(v), points(v))`
- merge encoder 学一个函数：`z_v = f_merge(tokens(v), z_child[0..3])`

直觉上，`z_v` 近似 Rao DP 表中“在给定边界条件下最优内部连线”的压缩表示。当前实现没有显式枚举所有边界条件，而是让 `z_v` 作为可微“隐式 DP 状态”。

### 4.2 Top-down：父→子边界条件传播（你最近确立的核心语义）
对内部节点 v，decoder 计算：

- 输入：
  - `bc_in_iface_logit(v)`：上层传来的 v 的 BC 打分（长度 Ti）
  - `tokens(v)`：v 的 interface/crossing token
  - `z_v` 与 `z_child[q]`
  - `child_iface_tokens(q)`：孩子接口 token（作为 query）
- 输出：
  - `child_iface_logit[v,q,i]`：父节点给第 q 个孩子的第 i 个 interface 的打分

这一步是“回溯”：在 Rao/Arora 的 DP 中，父节点选定一个状态会诱导孩子节点状态；这里由神经网络实现该转移。

> 重要：本项目目前只预测“interface 是否使用”的分数。理论 DP 里还会有“接口如何配对/匹配”的离散结构；这是后续需要补齐的部分。

---

## 5. Teacher 伪标签与训练目标（当前版本）

### 5.1 Teacher tour
对每个样本的点集，teacher 用：
- 最近邻（Nearest Neighbor）构造初始回路
- 多次 2-opt 改进（`two_opt_passes`）

得到一个近似 TSP tour，并取其边集合。

### 5.2 从 teacher tour 到 token 标签
teacher tour 的每条几何边 (a,b)：
- 若它就是 spanner 的一条边且该边在 alive 集合里：直接选中该 eid
- 否则在 “alive spanner 子图” 上做 Dijkstra，把 (a,b) 投影成一条 spanner 边路径，选中路径上的 eids

然后：
- 若某个 token 的 `eid` 在“选中 eids”里，则 `y_* = 1`，否则 0
- mask 为 token 的有效位（padding 为 False）

### 5.3 Top-down 边界条件监督（关键）
两种等价监督方式：

- **方案 A（当前 train.py 使用）**：监督每个非 root 节点的 `bc_iface_logit[v]` 去匹配 `y_iface[v]`  
  解释：`bc_iface_logit[v]` 是父节点写入的“v 的边界条件预测”，所以这是直接的 parent→child 监督。
- **显式方案（可选）**：监督父节点输出的 `child_iface_logit[v,q]` 去匹配 `y_iface[child_q(v)]`

代码里 labeler 也提供了 `y_child_iface`（把孩子的 y_iface 搬运到父的 [4,Ti] 维度）以支持后者。

---

## 6. 解码与评估：从 logits 到 tour

- 模型输出 per-node crossing logits `cross_logit[v,t]`
- 用 `cross_eid` 聚合到 per-edge logits（amax reduce）
- 用 `edge_decode` 把 edge logits 解码为 Hamiltonian cycle：
  - 优先选 spanner 边；必要时可允许 off-spanner patching（评估用途）
- 评估指标：
  - 与 teacher 长度对比的 gap
  - 解码可行率（是否得到单连通、所有点度为 2 的回路）
  - off-spanner 边数量、初始连通分量数等

---

## 7. 与 NN-Steiner 的差异（你需要强调的点）

- **状态维度来源不同**  
  NN-Steiner（以及很多 Arora-style neuralization）常用 portal 数 m=O(log n) 或与 n 相关的离散化，导致网络输入/输出维度随规模增长。
- **本项目的核心取舍**  
  通过 r-light 剪枝，把每个 box 的“可能穿越边界的接口数”压到常数，从而让：
  - token 数常数
  - BC 向量长度 Ti 常数
  - encoder/decoder 完全不依赖 N

因此本项目的“规模泛化性”更强、更工程可落地，但也更依赖“删边/剪枝”对近似比的可控性（理论工作空间）。

---

## 8. 当前版本的已知缺口与下一步理论工作

1. **matching / pairing 结构尚未建模**  
   真实 DP 状态通常包含“接口两两配对”的组合结构。建议扩展：
   - 让 decoder 预测 pairing（例如生成匹配矩阵或指针网络）
   - 或拆成“使用概率 + 局部配对概率”，并在解码时做约束满足
2. **r-light 工程化删边对近似比的影响**  
   需要从理论上解释：删掉较长 interface 是否仍能保证近似（可能引入新的“允许误差随层级增长”的分析）。
3. **top-down 的“可达性”与错误传播分析**  
   当前为可微训练便利，root BC 初始化为 0；更严格的做法可引入显式 root BC 或将 root 也纳入自监督/一致性约束。
4. **teacher 投影的偏差**  
   teacher 基于启发式 tour，且投影受 alive spanner 限制；这会影响监督信号上界。后续可考虑更强 teacher 或多 teacher ensembling。

---

## 9. 快速术语表

- **box / node / quadtree node**：树节点，对应一个方形区域
- **interface**：边在 box 边界上的穿越事件（用于边界条件）
- **crossing**：边在某个 LCA box 处跨越两个孩子象限的事件（用于父层协调）
- **BC（boundary condition）**：每个 node 的 interface 使用/结构状态（当前版本是“使用分数”）
- **bottom-up latent `z_v`**：神经值函数摘要
- **top-down decoder**：神经回溯器，父→子传 BC
