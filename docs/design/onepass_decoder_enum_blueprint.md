# 1-Pass Decoder + Enum 精简方案与实现蓝图

**Date**: 2026-04-04  
**Status**: Proposed

## 1. 这份文档解决什么问题

本文档整理当前 `1-pass` 路线的一份**精简但可落地**的改造方案，目标是同时满足下面三点：

1. 尽量贴近论文中“`decoder -> parse -> exact evaluate -> fallback`”的设计。
2. 不再把 `max_used=4` 当作主算法的语义裁剪，因为这会显著损害质量。
3. 不在第一阶段引入新的大模块（例如 tuple reranker），而是在当前 `enum parse` 基础上做最小但关键的改造。

这份方案的中心思想是：

- **保留**当前 `catalog_enum parse` 的精确离散搜索骨架；
- **不改**“exact verify + exact child-cost lookup + backpointer + traceback”这条正确性主线；
- **先不加** tuple reranker；
- **先把 decoder 从只学 iface activation，升级为学 `iface + mate` proposal scoring**；
- **把固定截断改成 progressive widening + lower-bound checked search**。

一句话概括：

> 不推翻当前的 exact enum parse，只让它看到更好的候选，并且不要再把硬截断当成语义的一部分。

## 2. 当前实现的问题

当前 `1-pass` 路线里，最核心的错位是：

- 训练时主要监督的是 `y_child_iface`，也就是 child 边界 slot 的开关；
- 但推理时真正需要决定的是 child 的**完整离散 state**，即 `(used_iface, mate)`；
- 在当前 `catalog_enum` 路径中，child state 的排序几乎只由 `used_iface` 与 decoder 输出的 activation 分数一致性决定；
- 当多个 child state 具有相同的 `used_iface`、但 `mate` 不同时，模型几乎无法区分它们；
- 因此 exact enum 虽然本身是“精确”的，但经常是在一个被错误排序、且被截断过的候选集上精确搜索。

这会导致两个直接后果：

1. `max_child_catalog_states` 一旦截断，真正最优的 child state 可能根本进不了枚举器。
2. 即使 enum parse 本身严格，最终解质量依然会因为 proposal 排序不好而下降。

## 3. 本方案的核心决策

## 3.1 保留什么

下面这些部件视为当前实现中最有价值的“精确骨架”，**不在第一阶段重写**：

- [src/models/dp_parse_catalog.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_parse_catalog.py)：基于离散 catalog 的 child tuple 枚举
- [src/models/dp_verify.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_verify.py)：合法性检查
- [src/models/dp_fallback.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_fallback.py)：exact fallback
- [src/models/dp_runner.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_runner.py)：cost table / backpointer / traceback
- [src/models/tour_reconstruct.py](/remote-home/MoGuanlin/NNTSP/src/models/tour_reconstruct.py)：direct reconstruction

也就是说，**当前 one-pass 的 exact DP 壳子仍然保留**。

## 3.2 改什么

第一阶段只做四个变化：

1. `MergeDecoder` 不再只输出 child `iface` logits，而是输出：
   - `iface_logit [4, Ti]`
   - `mate_logit [4, Ti, Ti]`
2. child state 排序从“只看 activation”改成“看 activation + mate compatibility”。
3. `max_child_catalog_states` 不再作为固定一次性截断，而是改成 widening schedule。
4. 训练从“只有 child iface BCE”改成“child iface + child mate”的双监督。

## 3.3 暂时不做什么

为了控制实现复杂度，第一阶段**不做**下面两件事：

- 不新增独立的 tuple reranker 模块
- 不把 decoder 改成 full child-state catalog classification

原因：

- tuple reranker 当然有价值，但不是最小改动；
- full catalog classification 在不使用 `max_used` 的情况下状态空间过大，不适合作为第一阶段方案。

## 4. 目标中的“无 max_used”到底是什么意思

本方案里需要明确区分两种“上界”。

### 4.1 语义上界

语义上界来自论文/DP 定义本身，即 `r-light` 通行量。

- `Ti = 4r` 代表每个节点边界最多有 `4r` 个 interface slots；
- 这部分是算法语义的一部分。

### 4.2 搜索宽度

搜索宽度是工程加速器，不属于算法语义。

例如：

- 每个 child 当前先只看前 `8` 个 state；
- 如果不够，再扩到 `16`、`32`、`64`；
- 必要时最终回到 full exact fallback。

这类 widening 只是**先少看，再可回退地补看**。

因此，本方案坚持：

> 不再用 `max_used` 做状态空间语义裁剪；  
> 可以用 widening 做搜索调度，但 widening 不能变成永久 hard cap。

## 5. 方案总览

新的 `1-pass` 推理流程如下：

1. `MergeDecoder` 对给定父状态 `sigma_parent` 输出：
   - child `iface_logit`
   - child `mate_logit`
2. `catalog_enum parse` 仍然枚举离散 child tuple；
3. 但 child state 的候选排序，改用 `iface + mate` 的联合分数；
4. 先在小宽度上搜索，尽快找到一个低 cost incumbent；
5. 用 lower bound 判断是否需要继续 widening；
6. 若 widening 仍不足以证明当前 incumbent 最优，则进入更完整的 exact fallback。

新的训练流程如下：

1. 仍然使用 `teacher parent sigma` 进行单次条件解码；
2. 从 child 的 teacher state 提取两类标签：
   - `iface` 是否激活
   - 激活 slot 的正确 `mate`
3. 训练 decoder 学会：
   - 哪些 slot 应该被激活
   - 激活后这些 slot 应如何配对

## 6. 详细实现蓝图

## 6.1 MergeDecoder 输出修改

### 现状

当前 [src/models/merge_decoder.py](/remote-home/MoGuanlin/NNTSP/src/models/merge_decoder.py) 的核心输出是：

- `child_scores [B_sigma, 4, Ti]`

它只对应 child activation logits。

### 目标输出

新增第二个 head，输出：

- `child_iface_logits [B_sigma, 4, Ti]`
- `child_mate_logits [B_sigma, 4, Ti, Ti]`

其中：

- `child_iface_logits[q, s]` 表示 child `q` 的 slot `s` 应否激活；
- `child_mate_logits[q, s, t]` 表示如果 slot `s` 被激活，它与 slot `t` 配对的相容程度。

### 推荐实现方式

保留现有 `sigma -> q -> parent_memory cross-attn` 主体不变，只修改输出头。

建议改成：

1. 先得到 `q_out [B_sigma, d_model]`
2. `iface_head(q_out) -> [B_sigma, 4 * Ti]`
3. `mate_head(q_out) -> [B_sigma, 4 * Ti * Ti]`
4. reshape 成上述两个 tensor

### 为什么不用 full child-state logits

因为在 `Ti = 4r = 16` 且不使用 `max_used` 时，full matching state 数会非常大，不适合作为 flat 分类空间。  
而 `iface + mate` 是一种分解表示，输出规模仍然可控。

## 6.2 新的 decoder 输出如何喂给 enum parse

### 现状

[src/models/dp_parse_catalog.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_parse_catalog.py) 当前对子状态排序时，主要依据：

- child state 的 `used_iface`
- decoder 给出的 activation 分数

因此，matching 几乎没有进入排序。

### 新的 child-state score

对任意一个 child catalog state `s_child = (used, mate)`，定义：

`state_score = iface_term + lambda_mate * mate_term`

其中：

#### `iface_term`

用于度量这个 child state 的 active slots 与 `child_iface_logits` 的一致性。

推荐形式：

- active slot 使用 `logsigmoid(logit)`
- inactive slot 使用 `logsigmoid(-logit)`
- 只在该 child 有效的 `iface_mask` 上计算

#### `mate_term`

用于度量 child state 内部 matching 与 `child_mate_logits` 的一致性。

推荐形式：

- 对每个 active slot `s`，取它在该 state 中的 `mate[s] = t`
- 读取 `child_mate_logits[q, s, t]`
- 把这些值求和或求平均
- 只对 active slot 计算

### 注意事项

- 若一个 slot 未激活，则它的 mate 项不参与打分；
- 为避免重复计数，可只统计 `s < mate[s]` 的配对边；
- 也可保留双向项，但需统一缩放。

## 6.3 dp_parse_catalog 的改造点

主要改 [src/models/dp_parse_catalog.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_parse_catalog.py) 里的 `_rank_child_catalog_states_for_parse(...)`。

### 新接口建议

将其输入从：

- `scores_q`

扩展为：

- `iface_logits_q`
- `mate_logits_q`

### 新流程

1. 先做现有 exact filter：
   - `finite_mask`
   - `C1` 约束过滤
2. 对剩余 child states 计算新的 `state_score`
3. 按 `state_score` 降序排列
4. 后续 enum 流程保持不变

也就是说：

> 不改 enum 本身，只改“谁先被 enum 看见”。

## 6.4 widening schedule

### 现状

当前实现中 `max_child_catalog_states` 更接近一次性截断。

### 目标

改成“分轮扩张”：

- round 1: per-child top `8`
- round 2: per-child top `16`
- round 3: per-child top `32`
- round 4: per-child top `64`
- ...
- 最后必要时 full exact fallback

### 推理逻辑

对某个 `(node, parent_sigma)`：

1. 用 `M=8` 生成每个 child 的候选 state 列表；
2. 在 `8^4` 范围内做 exact tuple enum；
3. 如果找到合法 tuple，记当前 best cost 为 `U`；
4. 计算未探索空间的 lower bound；
5. 若 lower bound 已不优于 `U`，则停止；
6. 否则扩大 `M`，继续搜索。

### 为什么不能“找到可行解就停”

因为：

- top-`M` 中找到可行解，不代表未探索空间没有更优解；
- 只有当未探索空间的 lower bound 已经不可能击败当前 incumbent，才能安全停。

## 6.5 lower bound 设计

### 最小版本

对尚未展开的 child，预先知道其候选 state 中的 exact child cost 最小值：

`LB = current_partial_cost + sum(remaining_child_min_costs)`

这个界虽然偏松，但简单且精确。

### 工程使用原则

- practical 模式：lower bound 可作为“是否继续 widening”的参考；
- audit / rebuttal 模式：只有当 lower bound 证明当前 incumbent 足够好，才提前停；否则继续 widening 或 full fallback。

## 6.6 训练标签生成

## 6.6.1 第一期：只用现有 teacher path 生成标签

这是最小实现版本，也是推荐第一步。

### 可直接复用的已有信息

当前 one-pass 训练已经有：

- teacher parent state：`target_state_idx`
- child 的 teacher `iface` 监督：`y_child_iface`

还缺的是 child 的 `mate` 监督。

### 新增的标签

对每个 internal node：

1. 取其四个 child 的 teacher state index；
2. 查 catalog 中对应 child state 的：
   - `used_iface`
   - `mate`
3. 构造：
   - `y_child_iface [M, 4, Ti]`
   - `y_child_mate [M, 4, Ti]`
   - `m_child_mate [M, 4, Ti]`

其中：

- `y_child_mate[q, s] = t` 表示 child `q` 的 slot `s` 正确配对到 `t`
- 若 slot `s` 未激活，则 `m_child_mate[q, s] = False`，该位置不计入 mate loss

## 6.6.2 第二期：加入局部 exact oracle 标签

当第一期训练稳定后，可进一步提升监督质量。

做法：

1. 对部分 `(node, parent_sigma)` 跑当前 full enum parse；
2. 得到 exact 最优 child tuple；
3. 将这个 tuple 对应的 child states 作为更贴近论文局部目标的监督。

建议只对下列 case 做：

- teacher traceback 路径上的节点
- 少量随机 parent sigma
- 当前模型 fallback 频率高的 hard cases

这样可以把额外标注成本控制在可接受范围内。

## 6.7 训练 loss 设计

### `L_iface`

延续当前的 child iface BCE。

目标：

- 学会 proposal 的 activation prior

### `L_mate`

新增 child mate 分类 loss。

推荐形式：

- 仅在 active slots 上计算
- 对每个 active slot，做 `Ti` 分类的 cross-entropy

即：

- 输入：`child_mate_logits[q, s, :]`
- 目标：正确的 `mate[s]`

### 总 loss

第一阶段建议：

`L = L_iface + alpha_mate * L_mate`

其中 `alpha_mate` 初始可从 `0.5` 或 `1.0` 开始调。

### 为什么第一阶段先不加 ranking loss

因为当前目标是：

- 先让 decoder 学会区分 matching
- 先把 enum 排序质量明显提升

在这一步之前就上 tuple-level ranking loss，会显著增加系统复杂度。

## 6.8 一次训练 step 的完整过程

下面给出推荐的第一阶段训练流程。

1. DataLoader 取一个 batch；
2. `NodeTokenPacker` 打包成 `PackedBatch`；
3. 从 teacher labels 中取 parent `target_state_idx`；
4. 从 child 的 teacher state 生成：
   - `y_child_iface`
   - `y_child_mate`
5. `OnePassTrainRunner` 前向：
   - bottom-up encode
   - teacher sigma-conditioned decode
   - 输出 `iface_logits + mate_logits`
6. 计算：
   - `L_iface`
   - `L_mate`
7. 反向传播更新：
   - `LeafEncoder`
   - `MergeEncoder`
   - `MergeDecoder`

### 关键点

训练过程中仍然只对 teacher sigma 做条件解码，保持当前 one-pass trainer 的基本结构不变。  
第一阶段不需要把 full enum 插入训练循环。

## 6.9 一次推理 merge 的完整过程

对一个内部节点 `B` 和一个父状态 `sigma_parent`：

1. decoder 输出：
   - `iface_logits`
   - `mate_logits`
2. 对每个 child：
   - exact filter：`iface_mask`、`C1`、finite child cost
   - 用新 `state_score` 给 child states 排序
3. 采用 widening：
   - top `8`
   - top `16`
   - top `32`
   - ...
4. 在当前宽度下做 exact tuple enum：
   - 共享边界一致性检查
   - `verify_tuple`
   - exact child cost lookup
5. 记录 best feasible tuple 和其 cost
6. 计算未探索空间 lower bound
7. 若不能证明当前 incumbent 足够好，则继续 widening
8. 必要时进入 full exact fallback

## 7. 逐节点精度如何衡量

本方案建议把节点级“精度”定义成**局部 regret**，而不是 logits 精度。

对某个 `(B, sigma)`，定义：

`delta(B, sigma) = (C_hat(B, sigma) - C_star(B, sigma)) / ell(B)`

其中：

- `C_hat(B, sigma)`：当前 proposal + enum + fallback 最终选出的 child tuple 的 exact cost
- `C_star(B, sigma)`：该 `(B, sigma)` 下 exact 最优 child tuple cost
- `ell(B)`：box 边长

### 可报告的统计

- mean / median `delta`
- 95% / 99% 分位数 `delta`
- per-depth `delta`
- fallback 率
- widening 最终停在第几轮
- oracle-best child tuple 是否进入 top-`M`

这些统计比单纯的 BCE accuracy 更贴近论文中的局部目标。

## 8. 代码改动清单

## 8.1 需要修改的文件

- [src/models/merge_decoder.py](/remote-home/MoGuanlin/NNTSP/src/models/merge_decoder.py)
  - 新增 `mate_head`
  - 输出 `child_iface_logits + child_mate_logits`

- [src/models/onepass_trainer.py](/remote-home/MoGuanlin/NNTSP/src/models/onepass_trainer.py)
  - 扩展 `OnePassTrainResult`
  - 新增 `mate loss`
  - 支持 `y_child_mate / m_child_mate`

- [src/cli/train_onepass.py](/remote-home/MoGuanlin/NNTSP/src/cli/train_onepass.py)
  - 从现有 labels 构造 mate 监督
  - 记录新的训练指标

- [src/models/dp_parse_catalog.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_parse_catalog.py)
  - child state ranking 改成 `iface + mate`
  - 支持 widening

- [src/models/dp_runner.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_runner.py)
  - 将一次性 `max_child_catalog_states` 调用改成 widening loop
  - 加入 lower-bound 检查

## 8.2 暂不修改的文件

- [src/models/dp_verify.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_verify.py)
- [src/models/dp_fallback.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_fallback.py)
- [src/models/tour_reconstruct.py](/remote-home/MoGuanlin/NNTSP/src/models/tour_reconstruct.py)

## 9. 分阶段落地计划

### Phase 1：最小可运行版本

- decoder 新增 mate head
- 训练加入 mate supervision
- child state ranking 加入 mate term
- 不加 widening，只替换排序

目标：

- 验证在相同 enum 宽度下，质量是否比当前 iface-only 排序更高

### Phase 2：加 widening + bound-check

- 把固定 cap 改成 widening schedule
- 加 lower-bound 停止条件

目标：

- 减少对固定 hard cap 的依赖
- 在 practical 模式下提高质量 / 速度折中

### Phase 3：局部 oracle 审计

- 对小中规模数据跑 exact local oracle
- 报告节点级 `delta`

目标：

- 直接回应“学习目标与论文局部目标对不上”的质疑

## 10. 这份方案的边界

这份方案不是“最终完美 one-pass”。

它仍然有一些限制：

- 第一阶段仍然主要依赖 teacher path sigma，而不是 full DP state supervision；
- 仍然没有 tuple reranker；
- 仍然没有把论文中的 `delta <= epsilon / n` 变成一个可直接优化的理论训练目标。

但它有三个很现实的优点：

1. 与当前实现最兼容；
2. 不再把 `max_used=4` 当成必须依赖的语义裁剪；
3. 能把“网络学的是什么”从单纯 iface 开关，推进到真正的离散 child state 结构。

## 11. 推荐结论

如果只允许做一版**最小但有效**的 one-pass 改造，本方案的推荐路线是：

1. 保留 exact enum parse；
2. 不加 tuple reranker；
3. decoder 升级为 `iface + mate` proposal scorer；
4. 训练使用 `iface + mate` 双监督；
5. 推理使用 `iface + mate` 联合 child-state 排序；
6. 逐步引入 widening 和 lower-bound 审计。

这条路线最适合作为：

- 当前工程主线的下一个版本；
- 回应 reviewer 质疑时的“精简、清晰、可落地”方案。
