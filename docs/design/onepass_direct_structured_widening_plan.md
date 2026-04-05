# 1-Pass 去 `max_used` 方案：Direct Structured Supervision + Full-Semantics Widening

**Date**: 2026-04-04  
**Status**: Proposed

## 1. 目标

本文档给出一套具体改造方案，目标是让当前 `1-pass` 路线在**训练**和**推理/测试**两侧都彻底摆脱 `max_used`：

1. 训练侧不再依赖 capped catalog，也不再用 `target_state_idx` 作为主监督。
2. 推理侧不再依赖 `matching_max_used` 限制状态空间，而是用 full `4r` 语义下的结构化 widening 搜索。
3. 保留论文中“`decoder -> parse -> verify -> fallback`”的主结构。
4. 对 reviewer 能清楚说明：
   - 训练目标是直接针对状态结构 `sigma=(a,mate)`；
   - 推理阶段的近似只存在于搜索调度，不存在于语义状态空间；
   - 必要时 exact fallback 仍覆盖 full `4r` 状态空间。

一句话概括：

> 训练从“学 catalog 编号”改成“学边界结构本身”；  
> 推理由“capped catalog enum”改成“full-structure factorized widening + exact fallback”。

## 2. 为什么不能只把 `matching_max_used` 调大

当前代码里，`max_used` 不只是推理参数，而是训练定义的一部分。

### 2.1 当前训练依赖 capped catalog

当前 one-pass 训练流程中：

- [src/models/node_token_packer.py](/remote-home/MoGuanlin/NNTSP/src/models/node_token_packer.py) 用 `matching_max_used` 构造 matching catalog。
- [src/models/labeler.py](/remote-home/MoGuanlin/NNTSP/src/models/labeler.py) 把 teacher tour 投影到这个 catalog，生成：
  - `target_state_idx`
  - `m_state`
  - `m_state_exact`
- [src/models/onepass_trainer.py](/remote-home/MoGuanlin/NNTSP/src/models/onepass_trainer.py) 只对 `m_state_exact=True` 的节点做 teacher-sigma decode。

也就是说，当前训练在学的是：

> “真实边界状态在 capped catalog 里对应哪个 index”

而不是：

> “真实边界状态本身长什么样”

### 2.2 把 `max_used` 提到 full `4r` 会直接击穿当前实现

以常用配置 `r=4` 为例，`Ti=4r=16`。

状态数随 `max_used` 增长如下：

- `max_used=4`：`3761`
- `max_used=8`：`223981`
- `max_used=16`：`853467`

这会同时击穿三处：

1. `state_mask_from_iface_mask(...)` 的 `[M, S, Ti]` 广播中间量  
见 [src/models/bc_state_catalog.py](/remote-home/MoGuanlin/NNTSP/src/models/bc_state_catalog.py)

2. labeler 的 state projection 与 exact-match 检查  
见 [src/models/labeler.py](/remote-home/MoGuanlin/NNTSP/src/models/labeler.py)

3. DP cost table 的 dense `[S]` 存储方式  
见 [src/models/dp_runner.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_runner.py)

因此：

> 不能通过“把 `matching_max_used` 设成 16”来实现去 cap。  
> 必须改训练标签形式和推理状态表示。

## 3. 新方案的核心思想

新方案分成两半：

### 3.1 训练：Direct Structured Supervision

训练标签不再是 `target_state_idx`，而是直接监督：

- `parent_used_iface`
- `parent_mate`
- `child_used_iface`
- `child_mate`

即直接训练结构 `sigma=(a,mate)`。

### 3.2 推理：Full-Semantics Factorized Widening

推理不再对 full matching state 做 flat catalog enum，而改成：

- 结构化状态表示
- 稀疏 cost table
- 每个 child 做 factorized candidate generation
- tuple 层 widening + lower bound
- 必要时 full structured exact fallback

也就是说：

> 不裁状态空间；只裁搜索顺序和搜索宽度，且必须可回退。

## 4. 训练改造：Direct Structured Supervision

## 4.1 当前训练的问题

当前训练使用的主监督对象是：

- `target_state_idx`
- `y_child_iface`
- 可选 `child mate`，但仍依赖 child 的 catalog state

这意味着：

- 父节点 teacher sigma 必须先落到 capped catalog 里；
- 如果真实 teacher 需要超过 `max_used` 个 active slots，就拿不到 exact parent sigma；
- 训练样本覆盖率直接受 `max_used` 影响。

## 4.2 新训练标签应该长什么样

在新的 direct structured 训练下，labeler 直接输出以下字段：

### 父节点标签

- `parent_sigma_used: [M, Ti] bool`
- `parent_sigma_mate: [M, Ti] long`
- `m_parent_sigma_structured: [M] bool`

含义：

- `parent_sigma_used[mid]`：节点 `mid` 的真实边界开口
- `parent_sigma_mate[mid]`：节点 `mid` 的真实 matching
- `m_parent_sigma_structured[mid]`：该节点的结构标签是否足够可信，可用于 full structured supervision

### 子节点标签

- `child_sigma_used: [M, 4, Ti] bool`
- `child_sigma_mate: [M, 4, Ti] long`
- `m_child_sigma_structured: [M, 4, Ti] bool`

含义：

- 对每个 internal node，直接给出其 4 个 child 的真实结构标签
- 若某个 child slot 未激活或该 child 结构不可信，则 mask 掉对应 mate loss

## 4.3 labeler 应如何生成这些标签

当前 [src/models/labeler.py](/remote-home/MoGuanlin/NNTSP/src/models/labeler.py) 里已有一套从 teacher 边集推回 `(target_used, target_mate)` 的逻辑，只是最后又把它投影回了 catalog index。

关键 helper 是：

- `_build_matching_target_for_node(...)`

当前它的结尾是：

- exact usable 时：`project_matching_to_state_index(...)`
- 否则：`project_iface_usage_to_state_index(...)`

### 新设计

把这段逻辑拆成两层：

1. `_build_matching_target_for_node_structured(...)`
   - 直接返回：
     - `used_iface [Ti]`
     - `mate [Ti]`
     - `structured_ok bool`

2. 旧的 `_build_matching_target_for_node(...)`
   - 作为 legacy catalog-index path 保留
   - 只在旧训练/旧推理中使用

### 结构标签的质量分级

不是所有节点都应该强行做 mate supervision。建议分三类：

1. `structured_exact`
   - `used` 和 `mate` 都可信
   - 参与 `iface` + `mate` loss

2. `iface_only`
   - `used` 基本可信，但 `mate` 不足够稳定
   - 只参与 `iface` loss

3. `skip`
   - 连 `used` 都不可信
   - 完全跳过

这比当前的 `m_state_exact / m_state` 更贴合 direct supervision 需求。

## 4.4 trainer 如何修改

### 当前 trainer

当前 [src/models/onepass_trainer.py](/remote-home/MoGuanlin/NNTSP/src/models/onepass_trainer.py) 使用：

- `target_state_idx`
- `m_state`

来决定是否对节点做 decode。

### 新 trainer

新的 `OnePassTrainRunner.run_batch(...)` 应改为接收：

- `parent_sigma_used`
- `parent_sigma_mate`
- `m_parent_sigma_structured`

也就是说：

1. 对 `m_parent_sigma_structured=True` 的 internal node
2. 直接用 `sigma_a=parent_sigma_used`、`sigma_mate=parent_sigma_mate`
3. 调用 `merge_decoder.decode_sigma(...)`

不再依赖 `target_state_idx -> catalog[state_idx]`。

## 4.5 decoder 输出与 loss

建议保留当前已经接好的 `iface_mate` decoder 设计：

- `child_iface_logits [M, 4, Ti]`
- `child_mate_logits [M, 4, Ti, Ti]`

训练 loss 分为三部分：

### `L_iface`

和当前类似，slot 级 BCE：

- active slot 为正样本
- inactive slot 为负样本

### `L_mate`

仅对 active 且结构可信的 slot 计算：

- 输入：`child_mate_logits[q, s, :]`
- 目标：正确 mate `t`
- loss：cross-entropy

### `L_consistency`（可选）

这是 direct structured 路线里建议新增的弱约束项，用于降低不合法预测：

- 如果 `s` 预测 `t`，则 `t` 也应该倾向于预测 `s`
- inactive slots 不应有强 mate 偏好

推荐第一阶段先不开，第二阶段再加。

### 总 loss

第一阶段建议：

`L = L_iface + alpha_mate * L_mate`

第二阶段可扩展为：

`L = L_iface + alpha_mate * L_mate + alpha_cons * L_consistency`

## 4.6 direct structured 训练过程

新的训练流程应为：

1. `NodeTokenPacker` 打包几何/树结构，但训练路径不再强依赖 `state_catalog`
2. `Labeler` 从 teacher 边集直接提取每个节点的：
   - parent structured sigma
   - child structured sigma
3. `LeafEncoder / MergeEncoder` 自底向上编码
4. `MergeDecoder` 条件在真实 `parent_sigma` 上解码
5. 用真实 child `(used, mate)` 结构直接监督输出
6. 反向传播

### 关键变化

训练中：

- 不再需要 `target_state_idx`
- 不再需要 `m_state_exact`
- 不再需要训练期 `matching_max_used`

因此：

> 训练侧应完全移除 `matching_max_used` 作为主语义参数。  
> `r` 成为唯一的语义上界。

## 5. 推理改造：Full-Semantics Widening

## 5.1 为什么不能继续用 full flat catalog

即使训练不再依赖 catalog，如果推理还保留 full flat catalog：

- full `Ti=16` matching state 数仍是 `853467`
- dense `[S]` cost table 仍然巨大
- per-node state mask 和 child-state ranking 仍然昂贵

因此，推理不能只把 `max_used` 删掉，还必须改状态表示。

## 5.2 新的状态表示：StructuredBoundaryState

建议新增一个轻量状态 key，而不是全局 catalog index。

例如在新文件：

- [src/models/boundary_state_structured.py](/remote-home/MoGuanlin/NNTSP/src/models/boundary_state_structured.py)

定义：

- `used_mask: int`
- `mate_tuple: tuple[int, ...]`

或等价的 canonical 编码。

要求：

- 可哈希
- 可做相等比较
- 可快速转回 `(used_iface, mate)`

这样 DP table 从：

- dense `Tensor[S]`

改成：

- `Dict[StructuredBoundaryState, float]`

## 5.3 leaf cost table 改成 sparse structured map

当前 leaf solver 输出的是对 catalog states 的 dense cost table。  
新的 leaf solver 应输出：

- `Dict[state_key, cost]`
- `Dict[state_key, witness]`

这意味着：

- 只存真正可行的 leaf structured states
- 不再依赖 full global catalog

建议新建：

- [src/models/dp_leaf_solver_structured.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_leaf_solver_structured.py)

旧的 [src/models/dp_leaf_solver.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_leaf_solver.py) 保留作为 legacy path。

## 5.4 internal parse 改成 factorized widening

建议新增 parse mode：

- `factorized_widening`

新文件建议：

- [src/models/dp_parse_factorized.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_parse_factorized.py)

它替代当前的 `catalog_enum`，流程如下。

### Step 1：对子节点做结构化候选生成

给定：

- `child_iface_logits [4, Ti]`
- `child_mate_logits [4, Ti, Ti]`
- C1 约束
- child 自己的 sparse cost table

对每个 child，生成 top-`M` 个候选 structured states。

### 候选生成不能靠 flat class 排序

应改成因子化生成：

1. 枚举/搜索 activation subset
2. 对每个 subset，在 active slots 上做 non-crossing matching DP
3. 结合：
   - neural score
   - child subtree exact cost
4. 取 top-`M`

注意：

- `Ti=16`，subset 空间 `2^16=65536`，在强约束和剪枝下是可管理的
- matching 层可用 interval DP 求最佳/次优 non-crossing matching

### Step 2：tuple widening

拿到 4 个 child 的 top-`M` 列表后：

- round 1：`M=8`
- round 2：`M=16`
- round 3：`M=32`
- round 4：`M=64`
- ...

在每一轮上做：

- 共享边界一致性剪枝
- `verify_tuple`
- exact child cost lookup

### Step 3：lower bound 决定是否继续扩张

不能因为“当前 `top-M` 里已经找到可行解”就停。  
需要对未探索空间做 lower bound。

建议最小 lower bound：

`LB = current_partial_cost + sum(remaining_child_min_possible_costs)`

若 `LB >= incumbent`，则这一轮可以安全停止；  
否则继续 widening。

### Step 4：full structured exact fallback

当 widening 仍无法证明当前 incumbent 最优时，进入 full exact fallback。

注意：

> 这个 fallback 必须是 full structured fallback，  
> 不能再调用旧的 capped catalog fallback。

## 5.5 dp_runner 如何修改

当前 [src/models/dp_runner.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_runner.py) 假设：

- `CostTableEntry.costs` 是 dense tensor
- `backptr` 的键是 sigma index

新的 runner 需要改成两条并行实现：

### legacy path

保留：

- `catalog_enum`
- `catalog_enum_iface_mate`

用于兼容已有 checkpoint 和旧实验。

### new path

新增：

- `factorized_widening`

配套数据结构：

- `StructuredCostTableEntry`
  - `costs: Dict[state_key, float]`
  - `backptr: Dict[state_key, tuple[state_key, ...]]`

### root 选最优

根节点不再做 dense `argmin(costs_tensor)`，而改成：

- 在 sparse state map 中扫描 root-feasible states
- 取最小 cost 的 `state_key`

## 6. 理论对应关系

## 6.1 direct structured supervision 如何对应理论状态

理论里节点状态本来就是：

- `sigma = (a, mate)`

因此：

- 当前 catalog-index supervision 学的是“`sigma` 的编号”
- direct structured supervision 学的是“`sigma` 本身”

从理论对象对应性上，direct structured supervision 更直接。

## 6.2 widening 如何对应论文中的 parse + fallback

论文结构是：

- decoder 输出连续 4-tuple
- parse 把它离散化
- 若误差不满足条件，则 fallback exact

新的 factorized widening 对应方式是：

- decoder 仍输出连续结构偏好
- parse 不再是 one-shot rounding，而是 widening search
- widening 是 practical search schedule
- 若 widening 不足以证明最优，则 full structured exact fallback

也就是说：

> 近似只存在于“先搜哪些候选、搜多宽”；  
> 真正语义空间和最终 fallback 仍然是 full `4r`。

## 7. 可能出现的问题与解决方式

## 7.1 直接监督 `mate` 仍可能学出不合法结构

问题：

- slot 级 CE 不会自动保证 whole matching 合法

解决：

- 推理时仍保留 exact parse / verify
- 第二阶段再加 `L_consistency`

## 7.2 有些节点的 structured label 不稳定

问题：

- 某些 node 的边界 stub 无法稳定还原完整 matching

解决：

- 增加 `structured_exact / iface_only / skip` 三级 mask
- 允许一部分样本只训 `iface`

## 7.3 推理搜索仍然可能很慢

问题：

- 去掉 `max_used` 后，full state space 仍然很大

解决：

- 用 factorized child generation，而不是 flat state catalog
- widening + lower bound
- full exact fallback 只用于 hard nodes

## 7.4 训练仍然只看 teacher 路径上的 parent sigma

问题：

- 即使 direct structured supervision 落地，训练目标仍不是 full DP table

解决：

- 先把去 `max_used` 路线跑通
- 第二阶段再加 multi-sigma local-oracle supervision

## 8. 具体文件改动清单

## 8.1 训练相关

- [src/models/labeler.py](/remote-home/MoGuanlin/NNTSP/src/models/labeler.py)
  - 新增 direct structured label 生成
  - 拆出 `_build_matching_target_for_node_structured(...)`
  - 保留旧 catalog-index path 作为 legacy

- [src/models/onepass_trainer.py](/remote-home/MoGuanlin/NNTSP/src/models/onepass_trainer.py)
  - `run_batch(...)` 改成接收 `parent_sigma_used/mate`
  - loss 改成 direct structured supervision

- [src/cli/train_onepass.py](/remote-home/MoGuanlin/NNTSP/src/cli/train_onepass.py)
  - 新增 `--supervision_mode direct_structured`
  - 在该模式下不再要求训练期 `matching_max_used`

- [src/models/node_token_packer.py](/remote-home/MoGuanlin/NNTSP/src/models/node_token_packer.py)
  - 训练路径允许不构建 `state_catalog`
  - legacy path 保留 `matching_max_used`

## 8.2 推理相关

- [src/models/boundary_state_structured.py](/remote-home/MoGuanlin/NNTSP/src/models/boundary_state_structured.py)
  - 新增 structured state key 与编解码工具

- [src/models/dp_leaf_solver_structured.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_leaf_solver_structured.py)
  - leaf sparse structured cost table

- [src/models/dp_parse_factorized.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_parse_factorized.py)
  - factorized child generation
  - tuple widening
  - lower bound

- [src/models/dp_runner.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_runner.py)
  - 新增 `factorized_widening` parse mode
  - 新增 sparse structured cost-table path

- [src/cli/eval_onepass.py](/remote-home/MoGuanlin/NNTSP/src/cli/eval_onepass.py)
  - 新增 `--dp_parse_mode factorized_widening`
  - legacy `dp_child_catalog_cap` 只对旧 parse mode 生效

## 9. 测试改造方案

## 9.1 应新增的测试

- `tests/test_labeler_direct_structured.py`
  - 验证 labeler 在 `used_slots > 4` 时仍能输出 structured labels

- `tests/test_onepass_trainer_direct_structured.py`
  - 验证 trainer 不依赖 `target_state_idx`
  - 验证 `iface + mate` loss mask 行为

- `tests/test_dp_parse_factorized.py`
  - 小规模手工例子上验证 factorized child generation 的合法性和排序

- `tests/test_dp_runner_factorized_widening.py`
  - 验证 widening 能在不使用 `max_used` 的情况下找到正确 tuple

- `tests/test_full_exact_fallback_structured.py`
  - 验证 hard node 上 widening 失败时能进入 full structured fallback

## 9.2 应保留但转为 legacy 的测试

- [tests/test_child_catalog_cap.py](/remote-home/MoGuanlin/NNTSP/tests/test_child_catalog_cap.py)
  - 继续覆盖 legacy `catalog_enum`
  - 不再作为主路径测试

- [tests/test_dp_runner.py](/remote-home/MoGuanlin/NNTSP/tests/test_dp_runner.py)
  - 追加 `factorized_widening` 子集测试
  - 原有 catalog path 继续保留

## 10. 推荐落地顺序

## Phase 1：先改训练

目标：

- 让 one-pass 训练摆脱 `max_used`
- 先不动 legacy 推理主线

完成标志：

- `train_onepass --supervision_mode direct_structured` 可运行
- 训练中不再要求 `matching_max_used`

## Phase 2：新增 factorized widening 推理

目标：

- 引入 full-semantics `factorized_widening`
- 先与 legacy 路线并存

完成标志：

- `eval_onepass --dp_parse_mode factorized_widening` 可运行
- 小规模样本上与 exact full search 一致

## Phase 3：删掉主线中的 `max_used`

目标：

- 让主训练/主推理路径都不再依赖 `max_used`
- `matching_max_used` 仅保留给 legacy baseline

完成标志：

- 主文档、CLI、测试都把 no-`max_used` 路线当默认推荐

## 11. 最终结论

如果目标是：

- 训练彻底不受 `max_used` 限制
- 推理也不再靠 capped catalog
- 并且整个 one-pass 路线仍然保留论文式 `decoder -> parse -> verify -> fallback`

那么必须同时做三件事：

1. 训练改成 direct structured supervision  
2. 推理状态表示改成 sparse structured state，而不是 flat catalog index  
3. parse 改成 full-semantics factorized widening + exact fallback

只做其中一件都不够：

- 只改训练，不改推理：推理仍然被 capped catalog 卡住
- 只改 widening，不改训练：训练仍然被 `max_used` 筛掉大量样本
- 只把 `matching_max_used` 调大：当前实现会在 packer / labeler / dense cost table 处失控

因此，这不是一个单点 patch，而是一条新的 one-pass 主线设计。
