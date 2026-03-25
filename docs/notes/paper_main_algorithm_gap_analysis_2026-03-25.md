# NNTSP: 论文主算法与当前工程实现差距分析

日期：2026-03-25

目的：把论文里的三条线彻底拆开，避免后续开发时把它们混成一件事。

- 线 A：论文中的 certified main algorithm
- 线 B：论文中的 practical two-pass variant
- 线 C：当前仓库默认工程实现

这份文档的核心结论是：当前仓库已经较完整地实现了线 B，并在若干位置进一步做了工程近似；它还没有实现线 A。后续如果目标是“更贴近论文主算法”，主任务不是继续打磨 greedy/LKH 后处理，而是把离散 boundary state、matching/parsing、exact merge evaluation 重新拉回主线。

## 1. 先把论文三条线分清楚

### 1.1 线 A：论文主算法（certified main algorithm）

论文第 2 节和第 3.1/3.2 节描述的主算法，本质上仍然是 Rao/Arora 风格的 DP。

- 输入是 quadtree `T` 和经过 patching 后的 `r-light` 稀疏图 `S'`
- 每个 box `B` 都有一个离散边界状态空间 `Omega(B)`
- 一个状态不仅包含“哪些 crossing/interface 被激活”，还包含这些边界端点之间的 connectivity / non-crossing pairing
- bottom-up 的对象是显式 DP cost table `C_B(σ)`
- 学习模块只负责 merge primitive：
  - 编码子表和几何
  - 对每个父状态 `σ` 预测连续 child state
  - 经过 `PARSE`
  - 做 feasibility check
  - 用精确 child-table evaluation 得到该 entry 的 cost/backpointer
- 如果 `PARSE` 失败，对该 entry 做 exact fallback
- 理论保证建立在“每个 merge call 对 normalized DP entry 是 δ-accurate”之上

换句话说，论文主算法不是“神经网络直接产出边分数然后解码 tour”，而是“神经网络给 DP merge 提建议，但合法性、可行性和代价评估仍由离散 DP 接管”。

### 1.2 线 B：论文 practical two-pass variant

论文第 3.3 节和 Appendix A.4 明确说了，为了工程加速，practical variant 去掉了主算法里剩下的 per-state loop。

- 不再显式物化全 DP tables
- 不再对每个父状态 `σ` 调 decoder
- 不再走 `PARSE -> feasibility check -> exact entry evaluation -> fallback`
- 改成：
  - 一次 bottom-up，缓存每个 box 的 latent `h_B`
  - 一次 top-down，每个 box 只在单个连续条件状态下访问一次
  - 输出连续 boundary/crossing logits
  - 聚合成 spanner edge scores
  - 再交给 greedy / 2-opt / LKH 这类 heuristic downstream

这条线在论文里被明确定位为 heuristic acceleration，不带 worst-case guarantee。

### 1.3 线 C：当前仓库默认工程实现

当前仓库默认路径总体上属于线 B，但又比论文的 practical variant 更工程化了一步。

主要体现在：

- `r-light` 不是通过 Rao-Smith patching 后“保证存在一条合法 tour 于 `S'` 中”，而是直接在候选边集上做全局删边
- top-down 默认状态仍然是 `iface usage logits`，不是完整离散 boundary state
- matching 虽然已经开始做原型，但还没有成为主训练/主推理路径
- 训练监督主要来自 projected teacher tour 的 token BCE，而不是 DP entry 级监督
- 最终输出仍是 edge score，再交给 greedy exact-on-sparse-graph 或 guided LKH，而不是 DP traceback

因此，当前仓库和论文主算法的关系应该表述为：

- “已经较好实现论文的 practical neural guidance 骨架”
- “尚未实现论文的 certified neural merge DP 主算法”

## 2. 当前仓库已经和论文对齐的部分

这些部分不是主要矛盾，后面可以尽量复用。

### 2.1 尺度无关的局部表示

仓库在这个方向上做得很到位。

- quadtree 节点的 token 数被 `r` 和 pack cap 控住
- 几何特征都做了 box-relative normalization
- encoder/decoder 的输入输出 shape 与 `N` 无关
- bottom-up / top-down 都是模块复用，不依赖实例规模

这和论文“size-independent interface”的核心思想是一致的。

### 2.2 两遍式执行框架

当前实现的整体调度，与论文 Appendix A.4 的 practical instantiation 基本一致。

- `BottomUpTreeRunner` 做整棵树的 latent 汇总
- `TopDownTreeRunner` 从 root 单次向下传播条件状态
- decoder 一次访问一个 box，并输出 child condition 与 token logits

这一层不需要推倒重来。

### 2.3 稀疏图上的 edge guidance

当前工程已经把局部预测稳定地汇总到了 edge level。

- crossing -> edge aggregation
- greedy decode
- exact decode on sparse candidate graph
- guided LKH

这部分很适合作为 practical branch 保留下来。

## 3. 当前仓库偏离论文主算法的关键位置

这里是后续改造的真正重点。

### 3.1 最关键偏差：DP state 仍不是论文主状态

论文主算法里的状态是离散 `σ in Omega(B)`，至少包含：

- 哪些 boundary slots 被使用
- 这些 slots 的 connectivity / non-crossing pairing

而当前默认状态是：

- `bc_iface_logit[v, i]`
- 语义是“第 i 个 interface 是否被使用”

这只覆盖了“used-set”，没有覆盖“how they connect”。

直接后果：

- 当前 top-down 还不是对离散 DP state 的真实 traceback
- 父子状态之间的一致性只学到了局部激活，不是完整组合结构
- decoder 无法表达 Rao/Arora DP 里最关键的 pairing feasibility

### 3.2 matching 原型已经存在，但还不是主路径

仓库里其实已经出现了明显的“往论文主算法回归”的迹象。

- `bc_state_catalog.py` 开始枚举常数大小的非交叉 matching state
- `NodeTokenPacker` 已支持 `state_mode="matching"`
- `TopDownDecoder` / `TopDownTreeRunner` 已支持 state-logit 传播
- `labeler.py` 已开始构造 `target_state_idx`
- `train.py` 已支持 `--state_mode matching`

但当前它还是原型而不是主线，主要问题有：

- catalog 只是“固定槽位上的非交叉 perfect matching 子集”
- 它没有恢复论文里完整 `Omega(B)` 的语义层
- state supervision 仍然来自 projected teacher edge set 的启发式反推
- 默认训练和默认使用路径仍然是 `iface`

所以现在最有价值的方向不是再新开一套结构，而是把这一半已经搭起来的 `matching` 线升级成主线。

### 3.3 没有显式 DP table，也没有 per-state merge

这是当前实现和论文主算法之间第二大的鸿沟。

论文主算法的 learned merge 是：

- 输入 child cost tables `C_Bi`
- 对每个父状态 `σ` 输出 child continuous states
- parse 成离散 child tuple
- exact evaluation parent entry

当前 bottom-up/decoder 则是：

- child latent -> parent latent
- parent single condition -> child logits

也就是说：

- 当前 `z_B` 是隐式摘要，不是显式 `C_B`
- 当前 top-down 单次访问 box，不是“对每个父状态 entry 求值”
- 当前没有 per-entry feasibility layer
- 当前没有 fallback-to-exact-enumeration 机制

如果未来想真正“贴近论文主算法”，这里是必须补的，而不是可选增强项。

### 3.4 `r-light` 在仓库里是删边，而不是结构定理中的存在性限制

这是第三个非常关键的偏差，而且它发生在模型之前。

论文主算法和 Rao-Smith 框架需要的是：

- 通过 patching / structural restriction 得到一个 `S'`
- 在该 `S'` 上存在近优合法 tour

当前 `prune_pyramid.py` 做的是：

- 对每个 `(node, boundary_dir)` 只保留最短 `r` 条 interface
- 然后把超出的 edge 直接全局 kill

这会带来两个问题：

1. 这比论文 practical variant 还要更强的近似
2. 一旦被 kill 的边恰好是理论上需要的那条，后面的任何 decoder 都无法补救

所以如果未来要往论文主算法回归，这一层不能再作为“默认唯一候选图构造”存在，至少需要和“理论 branch / practical branch”分开。

### 3.5 当前训练目标是 token supervision，不是 DP entry supervision

论文主算法的监督对象，若严格对齐，应更接近：

- merge entry 的 child-state prediction quality
- parse 后是否得到正确 child tuple
- normalized DP entry 误差

当前仓库的主要监督是：

- crossing BCE
- iface BCE
- BC BCE / CE

这对训练 practical guidance 很合理，但它学到的是“局部边/接口选择偏好”，不是“DP merge correctness”。

因此如果继续沿现在的 loss 打磨，收益大概率主要落在线 B，而不是线 A。

### 3.6 当前解码是 heuristic post-processing，不是 DP traceback

论文主算法输出 tour 的方式应该是：

- 从 root table 选最优 root state
- 沿 backpointer traceback
- 组合成全局合法解

当前仓库输出 tour 的方式是：

- aggregate edge logits
- greedy select / sparse exact decode / guided LKH

这些都更接近论文 practical variant 的 downstream solver，而不是主算法的最终还原。

## 4. 一个更精确的“三层对照表”

### 4.1 状态表示

- 论文主算法：离散 boundary state，含 active crossings + pairing/connectivity
- 论文 practical：连续 boundary state，用作 guidance
- 当前仓库默认：连续 iface-usage state
- 当前仓库原型：离散 matching catalog + state logits，但未主线化

### 4.2 bottom-up

- 论文主算法：操作对象是 child DP tables
- 论文 practical：操作对象是 child latents
- 当前仓库：child latents -> parent latent

### 4.3 top-down

- 论文主算法：对每个父状态 `σ` 做 decode / parse / exact eval
- 论文 practical：每个 box 单次 free-running decode
- 当前仓库：每个 box 单次 free-running decode

### 4.4 可行性保证

- 论文主算法：有显式 feasibility check 和 entry fallback
- 论文 practical：无显式 DP-level feasibility，依赖 downstream heuristics
- 当前仓库：无 DP-level feasibility，依赖 greedy/sparse exact/LKH

### 4.5 输出对象

- 论文主算法：DP table + backpointer + traceback
- 论文 practical：crossing logits / edge scores / heuristic warm start
- 当前仓库：cross/iface logits -> edge score -> heuristic decoder

### 4.6 `r-light`

- 论文主算法：来自 patching 后的结构受限图 `S'`
- 论文 practical：仍继承这条设定
- 当前仓库：直接删候选边实现常数 token 上界

## 5. 当前仓库里最值得保留的“回归支点”

虽然当前默认实现不是论文主算法，但仓库里已经有几块非常适合作为回归支点。

### 5.1 `matching` catalog 方向值得继续，不建议推翻

`bc_state_catalog.py` 的价值不在于“已经够完整”，而在于它把状态从单独的 binary interface 拉回到了“组合结构”。

后续更合理的做法是：

- 继续扩大 state 语义
- 把 `matching` 变成主训练路径
- 再让 `iface` 模式退化成 ablation / practical fast mode

### 5.2 top-down state propagation 框架可以复用

即使未来引入更完整离散状态，现有：

- `NodeTokenPacker`
- `TopDownDecoder`
- `TopDownTreeRunner`

仍然可以保留为“状态传播与局部条件查询”的骨架，只需要把传播对象从 `iface logits` 升级为更真实的 discrete-state / state-conditioned decode。

### 5.3 practical decode 分支应保留，但应降级为辅助分支

当前的：

- greedy edge decode
- sparse exact decode
- guided LKH

都不该删除。它们很适合作为：

- practical benchmark branch
- debugging branch
- ablation branch

但不应再继续扮演“主算法最终输出”的角色。

## 6. 建议的改造优先级

如果目标是“工程版本更贴近论文主算法，而不是后面的工程近似算法”，推荐按下面顺序推进。

### Phase 0：先把分支边界写清楚

先明确仓库里以后至少维护两条主路径：

- `certified_dp_branch`
- `practical_guidance_branch`

否则每次改动都会继续把二者混在一起。

最重要的不是立刻写大量代码，而是先在 CLI、文档、命名上承认这两条路径是不同目标。

### Phase 1：让 `matching` 成为默认状态，而不是实验选项

这是最先该做的。

目标：

- 默认 `state_mode` 从 `iface` 切到 `matching`
- 训练日志、checkpoint、评测都以 state prediction 为主
- `iface` 只作为辅助头或 ablation

原因：

- 这一步不需要先物化完整 DP table
- 但它能先把“父子状态传播的语义”拉回到论文主算法最关键的方向

### Phase 2：补齐 `Omega(B)` 的语义，而不只是 catalog 原型

当前 matching catalog 还是简化版，后面需要逐步补齐：

- per-side budget
- active interface subset 的合法性
- non-crossing connectivity
- 可能的空状态 / partial state / parity constraints
- 与当前固定 interface ordering 的一致定义

这一步做完，才能说“状态空间开始接近论文里的 `Omega(B)`”。

### Phase 3：把 top-down 从“单次 free-running”扩展回“per-state merge”

这是回归论文主算法的决定性一步。

需要新增的东西包括：

- 显式 parent state enumeration
- 对每个 parent state 的 child-state decode
- deterministic parse
- feasibility check
- exact child tuple evaluation
- per-entry fallback
- backpointer 存储

做完这一层后，整个系统才真正开始接近论文 3.1/3.2 的 learned merge DP。

### Phase 4：把 bottom-up latent 扩展成显式 table-aware merge

论文主算法 bottom-up merge 依赖的是 child tables，而不是单向 latent 摘要。

可以考虑分两步：

1. 先做 table-lite 版本
   - 对每个 node 存常数大小 state table
   - 每个 state entry 对应一个 cost/logit/value
2. 再做真正的 state-conditioned learned merge

这一步难度高，但如果不做，系统仍会停留在 practical branch。

### Phase 5：把 `r-light` 删边改成“结构限制分支”和“工程删边分支”并存

建议拆成两种输入图构造：

- `graph_mode=patched_structural`
- `graph_mode=pruned_engineering`

哪怕第一阶段还不能完整复现 Rao-Smith patching，也要先把“理论目标图”和“删边加速图”分开，这样后面的主算法实验才不会被候选图先验锁死。

### Phase 6：补真实 traceback，而不是继续强化 edge decode

等 state table 和 backpointer 到位后，应新增：

- root optimal state selection
- recursive traceback
- 从 child states 重建全局 tour / partial structure

这条链路打通后，才算真正有了“论文主算法的 end-to-end 工程版本”。

## 7. 我对后续实现策略的建议

这里给出一个比较稳妥的策略，避免一次性重写全部系统。

### 7.1 不要直接废弃当前 practical branch

它已经很有价值：

- 可以继续作为 benchmark
- 可以继续提供 teacher / warm start / sanity check
- 可以和未来的 certified branch 做对照

正确做法是“并行维护”，不是“推倒重来”。

### 7.2 优先做“半步回归”

所谓半步回归，是指：

- 先把状态从 `iface` 升级到 `matching`
- 再把 loss 从 token BCE 扩展到 state-aware supervision
- 再把 decode 从 edge score 扩展到 state traceback

这样每一步都能单独验证，不容易把现有工程打崩。

### 7.3 让论文主算法拥有单独的最小可运行闭环

建议最终形成一个最小实验闭环：

1. 小规模图
2. matching state catalog
3. per-state merge
4. parse + exact entry eval
5. traceback
6. 与 exact sparse solver 对照

先在小规模把主算法跑通，再往大规模 practical branch 回接。

## 8. 一个务实的开发路线图

### 8.1 第一批最值得做的代码任务

1. 把 `matching` 路径提为一级功能，而不是隐藏选项
2. 明确 `state_mode=iface` 仅代表 practical fast mode
3. 为 matching state 增加更严格的合法性测试
4. 在训练/验证日志里单独汇报 state accuracy / fallback ratio
5. 新增“per-node state traceback debug dump”

### 8.2 第二批任务

1. 新建 discrete parse 模块
2. 新建 per-state merge evaluator
3. 新建 backpointer 数据结构
4. 新建 root-to-leaf traceback 解码器

### 8.3 第三批任务

1. 重做 `r-light` 输入图构造分支
2. 把理论分支和 practical 分支的评测协议分开
3. 单独报告：
   - certified branch 的 DP correctness 指标
   - practical branch 的 runtime / LKH speedup 指标

## 9. 结论

一句话总结当前状态：

当前仓库已经是一个比较成熟的“size-independent practical neural guidance system”，但还不是“论文主算法的工程实现”。

如果后续目标是更贴近论文主算法，那么最优先的三件事是：

1. 把默认状态从 `iface` 升级到 `matching`
2. 把单次 free-running top-down 扩展回 per-state merge / parse / exact eval
3. 把当前的工程删边 `r-light` 与理论结构限制分支拆开

只有这样，工程主线才会从“论文 3.3 practical variant 的加强版”逐步回到“论文 3.1/3.2 certified main algorithm”。
