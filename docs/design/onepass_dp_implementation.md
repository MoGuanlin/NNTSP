# 1-Pass DP 完整实现文档

**Date**: 2026-03-27
**Status**: Implemented (训练 + 推理 + 评估)

## 1. 概述

本文档描述了 1-pass 底层遍历 DP 管线的完整实现。该管线与论文 Section 3 / Figure 2 的理论算法对齐，同时作为**独立可切换路径**存在，不修改任何现有 2-pass 代码。

### 1.1 与 2-pass 的区别

| 维度 | 2-pass（现有） | 1-pass（新增） |
|------|---------------|---------------|
| **编码** | Bottom-up 全部编码完成后，Top-down 全部解码 | 每个节点编码后立即解码 |
| **解码** | TopDownDecoder：父→子 broadcast | MergeDecoder：σ-conditioned cross-attention |
| **DP** | 无显式 DP，logit → greedy | 显式 cost table + backpointer + traceback |
| **训练** | BCE(cross_logit, y_cross) + BCE(child_iface, y_child_iface) | BCE(decode_sigma(teacher_σ), y_child_iface) |
| **推理** | BU → TD → edge_logit → greedy/exact | BU+DP → cost_table → traceback → hard_logit → greedy |

### 1.2 与论文的对应关系

```
论文 Figure 2                          代码实现
─────────────────────                  ─────────────────────
h_B = Encoder(Geom(B), {h_{B_i}})  →  LeafEncoder / MergeEncoder（共用）
                                       → z_storage[nid]

parent_memory = SelfAttn(tokens)    →  MergeDecoder.build_parent_memory()
                                       → ParentMemory (cached, built once per node)

τ̃ = Decoder(σ, parent_memory)      →  MergeDecoder.decode_sigma(σ_a, σ_mate, ...)
                                       → child_scores [4, Ti]

τ = PARSE(τ̃)                       →  dp_core.parse_continuous()
                                       → child_a [4, Ti], child_mate [4, Ti]

VERIFYTUPLE(σ, τ)                   →  dp_core.verify_tuple()
                                       → bool

C_B[σ] = Σ C_{B_i}[τ_i]           →  dp_runner._lookup_child_costs()
                                       → float (total cost)

σ* = argmin C_root                  →  dp_runner: valid_costs.argmin()
Traceback                           →  dp_runner._traceback() → leaf_states
Tour reconstruct                    →  tour_reconstruct.reconstruct_tour_direct()
                                       → Hamiltonian cycle / length
```

## 2. 模块架构

### 2.1 文件一览

```
src/models/
├── merge_decoder.py      # σ-conditioned 解码器（Phase 1）
├── dp_runner.py           # 推理时 1-pass DP runner（Phase 2）
├── tour_reconstruct.py    # DP 结果 → edge logits → tour（Phase 3）
├── onepass_trainer.py     # 训练时 1-pass runner + loss（Phase 4）
├── dp_core.py             # 基础设施：PARSE, VERIFY, correspondence maps（已有）
└── bc_state_catalog.py    # 边界状态枚举 Ω(B)（已有）

src/cli/
├── train_onepass.py       # 1-pass 训练入口
├── eval_onepass.py        # 1-pass 评估入口
└── evaluate.py            # 统一评估入口（已添加 --benchmark onepass）

tests/
├── test_merge_decoder.py      # 10 个测试
├── test_dp_runner.py          # 4 个测试
├── test_tour_reconstruct.py   # 6 个测试
└── test_onepass_trainer.py    # 5 个测试
```

### 2.2 MergeDecoder（merge_decoder.py）

σ-conditioned 解码器，论文 Section 3.1 的核心。

**架构：**
```
输入:
  - 父节点 tokens: CLS + IFACE + CROSS + CHILD_LATENT
  - 父节点 embedding z_node
  - 边界状态 σ = (a, mate)

Parent Memory（build 一次，复用多个 σ）:
  tokens = NodeTokenizer(Geom(B), iface, cross, child_z)
  tokens[CLS] += z_node_proj(z_node)
  tokens = SelfAttn × L_parent (tokens)       # L_parent = parent_num_layers

Sigma Query（per-σ）:
  σ_emb = SigmaEncoder(a, mate, mask)          # [d_model]
  q = CrossAttn × L_cross (σ_emb, parent_memory)   # L_cross = cross_num_layers

Output:
  child_scores = ChildHead(q) → [4, Ti]        # 4 children × Ti interface slots
```

**关键类：**
- `SigmaEncoder`：将离散状态 (a, mate) 编码为 d_model 向量
  - 每个 slot: activation (1) + mate_onehot (Ti) → MLP → d_model
  - Masked mean pooling over valid slots
- `ParentMemory`：缓存的 self-attended token 序列
- `MergeDecoder`：完整解码器
  - `build_parent_memory(...)` → ParentMemory（per-node，一次）
  - `decode_sigma(σ, memory)` → child_scores [B, 4, Ti]（per-σ）
  - `decode_sigma_batch(σ_batch, memory)` → child_scores [S, 4, Ti]（批量σ）

### 2.3 OnePassDPRunner（dp_runner.py）

推理时的 1-pass DP，完整实现论文 Algorithm 1。

**流程：**
```python
for depth d = max_depth → 0:
    # 叶节点：精确求解
    for leaf in leaves_at_d:
        costs = leaf_exact_solve(leaf)
        z[leaf] = LeafEncoder(leaf)

    # 内部节点：编码 + 枚举 σ + 神经解码 + DP
    for node in internal_at_d:
        z[node] = MergeEncoder(children_z)
        parent_memory = MergeDecoder.build_parent_memory(z[node], children_z)

        for σ in valid_states(node):
            # 当前主流程:
            # 1. catalog_enum + neural ranking
            # 2. 若 child catalog cap 截断后仍失败，则回退到 uncapped exact enumeration

            child_scores = decode_sigma_batch(all_σ)
            for each σ:
                τ = parse_by_catalog_enum(sigmoid(child_scores[σ]))
                if τ is feasible:
                    C[node][σ] = Σ C[child][τ_i]
                else:
                    try topk_parse → exact_fallback

# Traceback
σ* = argmin C[root]
leaf_states = traceback(root, σ*, backpointers)
```

**数据结构：**
- `CostTableEntry`：per-node cost table
  - `costs: [S] float` — 每个状态的最优代价（+inf = 不可行）
  - `backptr: Dict[int, Tuple[int,int,int,int]]` — σ_idx → 子状态索引
- `OnePassDPResult`：完整 DP 结果
  - `tour_cost, root_sigma, leaf_states, cost_tables, stats`

### 2.4 Tour Reconstruct（tour_reconstruct.py）

将 DP 离散解转为 edge logits，复用现有 greedy decoder。

```
dp_result_to_logits():
  1. 从 traceback 收集每个节点的 state_index
  2. 查 catalog: active slots → iface_logit = +10, inactive → -10
  3. 查 children 共享边界: both active → cross_logit = +10

dp_result_to_edge_scores():
  logits → aggregate_logits_to_edges() → EdgeScores
  → decode_tour_from_edge_logits() → Tour
```

### 2.5 OnePassTrainRunner（onepass_trainer.py）

训练时的 1-pass runner，用 teacher σ 做单次前向（有梯度）。

**与推理的区别：**
- 推理：枚举所有 candidate σ，no_grad
- 训练：只用 teacher σ（来自 labeler 的 target_state_idx），有 grad

**流程：**
```python
for depth d = max_depth → 0:
    # 叶节点：编码（可微 index_copy）
    z = z.index_copy(0, leaf_nids, LeafEncoder(leaves))

    # 内部节点：编码 + 用 teacher σ 解码
    z = z.index_copy(0, internal_nids, MergeEncoder(children_z))
    for node with valid target_state_idx:
        σ_teacher = catalog[target_state_idx[node]]
        parent_memory = build_parent_memory(z[node], children_z)
        child_scores[node] = decode_sigma(σ_teacher, parent_memory)  # [4, Ti]
```

**Loss：**
```python
loss = onepass_loss(
    child_scores=result.child_scores,   # [M, 4, Ti]
    decode_mask=result.decode_mask,      # [M] bool
    y_child_iface=labels.y_child_iface,  # [M, 4, Ti] from PseudoLabeler
    m_child_iface=labels.m_child_iface,  # [M, 4, Ti]
)
# → masked BCE: 只在 decode_mask=True 且 m_child_iface=True 的 entry 上计算
```

**梯度流经验证：**
- LeafEncoder ✓ → 通过 z[children] → parent_memory.child_z
- MergeEncoder ✓ → 通过 z[node] → parent_memory.z_node
- MergeDecoder ✓ → 直接输出 child_scores

## 3. 训练

### 3.1 训练入口

```bash
python -m src.cli.train_onepass --train_pt <data_path> [options]
```

**关键参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train_pt` | (必填) | 训练数据 .pt 路径 |
| `--r` | 4 | pyramid level，决定 Ti = 4r |
| `--d_model` | 128 | 模型维度 |
| `--matching_max_used` | 4 | 边界状态最大活跃 slot 数 |
| `--parent_num_layers` | 3 | parent memory self-attention 层数 |
| `--cross_num_layers` | 2 | sigma query cross-attention 层数 |
| `--batch_size` | 8 | 批大小 |
| `--epochs` | 5 | 训练轮数 |
| `--lr` | 1e-4 | 学习率 |
| `--wd` | 1e-4 | weight decay |
| `--grad_clip` | 1.0 | 梯度裁剪 |
| `--pos_weight` | 0.0 | BCE pos_weight（0=不启用） |
| `--device` | cuda | 设备 |

### 3.2 数据要求

- 使用 `*_r_light_pyramid.pt` 或 `.fast.pt` 格式
- **必须** 有 teacher labels（target_edges + tour_len），首次运行会自动生成
- 内部强制 `state_mode="matching"`，因为需要 target_state_idx

### 3.3 Checkpoint 格式

```python
{
    "step": int,
    "epoch": int,
    "args": dict,                    # 所有命令行参数
    "leaf_encoder": state_dict,
    "merge_encoder": state_dict,
    "merge_decoder": state_dict,     # MergeDecoder（不是 TopDownDecoder）
    "opt": state_dict,
}
```

## 4. 推理与评估

### 4.1 评估入口

```bash
# 独立入口
python -m src.cli.eval_onepass --ckpt <path> --data_pt <path> [options]

# 统一入口
python -m src.cli.evaluate --benchmark onepass --ckpt <path> --data_pt <path>
```

**推理流程：**
```
数据加载 → NodeTokenPacker.pack_batch()
         → OnePassDPRunner.run_single()
           → dp_result_to_edge_scores()
             → edge_logit [E]
               → greedy / exact / LKH 解码
                 → Tour
                   → gap = (length / teacher_length - 1) × 100%
```

**评估参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dp_max_used` | 4 | 状态枚举 max_used |
| `--dp_max_sigma` | 0 | 每个节点最大枚举 σ 数；`0` 表示不截断，`>0` 为启发式截断 |
| `--dp_child_catalog_cap` | 0 | `catalog_enum` 中每个 child 在 C1 过滤和排序后保留的最大 state 数；`0` 表示不截断 |
| `--dp_fallback_exact` | True | 是否启用精确枚举回退 |
| `--settings` | greedy | 解码后端（greedy, exact, guided_lkh, pure_lkh） |

### 4.2 输出指标

评估脚本输出表格：

```
Method               Avg Length      Avg Gap    Total Time
----------------------------------------------------------------------
DP (cost table)        xxx.xxxx    x.xxxx%          N/A
Greedy                 xxx.xxxx    x.xxxx%      x.xxxs
Exact Sparse           xxx.xxxx    x.xxxx%      x.xxxs
```

- **DP (cost table)**：直接从 cost table 读取的最优 σ 代价（理论下界）
- **Greedy/Exact/LKH**：将 edge_logit 输入不同解码器后的实际 tour

## 5. 测试

共 25 个自动化测试，覆盖所有模块。

```bash
# 运行所有测试
python tests/test_merge_decoder.py      # 10 tests: shape, gradient, masking, reuse
python tests/test_dp_runner.py          # 4 tests: pipeline, cost tables, traceback, corr maps
python tests/test_tour_reconstruct.py   # 6 tests: logit shape/values, edge scores, e2e
python tests/test_onepass_trainer.py    # 5 tests: shape, gradient flow, loss
```

**关键验证点：**
- MergeDecoder: 不同 σ → 不同输出 ✓，masked slots → -inf ✓
- DP Runner: 合成树上完整 DP 运行 ✓，cost table + traceback 正确 ✓
- Tour Reconstruct: 端到端 DP → logits → greedy → tour ✓
- Training: 梯度流经 leaf_enc / merge_enc / merge_dec 三个组件 ✓

## 6. 与 2-pass 的共存

### 6.1 共享模块

| 模块 | 说明 |
|------|------|
| `LeafEncoder` | 叶节点编码器，两条管线完全共享 |
| `MergeEncoder` | 内部节点编码器，两条管线完全共享 |
| `NodeTokenPacker` | 数据打包，两条管线共享 |
| `PseudoLabeler` | Teacher 标签生成，两条管线共享 |
| `BoundaryStateCatalog` | 状态枚举，两条管线共享 |
| `edge_aggregation` | edge logit 聚合，两条管线共享 |
| `edge_decode` | Greedy decoder，两条管线共享 |
| `dp_core` | PARSE/VERIFY/correspondence maps，两条管线共享 |

### 6.2 独立模块

| 1-pass 专有 | 2-pass 专有 |
|-------------|-------------|
| `merge_decoder.py` | `top_down_decoder.py` |
| `dp_runner.py` | `top_down_runner.py` |
| `tour_reconstruct.py` | — |
| `onepass_trainer.py` | — |
| `train_onepass.py` | `train.py` |
| `eval_onepass.py` | `eval_and_vis.py` |

### 6.3 不修改的文件

以下文件在 1-pass 实现中**完全未修改**：
- `src/models/bottom_up_runner.py`
- `src/models/top_down_runner.py`
- `src/models/top_down_decoder.py`
- `src/models/losses.py`
- `src/models/edge_decode.py`
- `src/models/edge_aggregation.py`
- `src/cli/train.py`
- `src/cli/eval_and_vis.py`

唯一修改的已有文件：`src/cli/evaluate.py`（添加 `--benchmark onepass` 路由，3 行）。
