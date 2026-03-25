# NNTSP（Neural Rao'98 TSP）工程实现交接文档（新版）

**代码快照日期**：2025-12-31  
**适用范围**：你当前上传的全部 Python 源码（数据生成/预处理、tokenization/packer、bottom-up、top-down、teacher/labeler、loss/metrics、decode、train）。  
**目标读者**：接手工程实现、继续重构/加新任务头、或在新会话中继续推导/实现配对(matching)与更完整 DP。  

---

## 0. 一页纸大框架（浅显易懂版）

整个工程就是一条“**Data → PackedBatch → bottom-up z → top-down logits → teacher labels/loss → 解码评估**”的闭环。

1. **预处理**把点集变成：
   - spanner（稀疏候选边）
   - quadtree/pyramid（树结构）
   - per-node 的 interface/crossing 事件（候选 token）
   - r-light 剪枝后每个 node token 数常数上界
2. **packer**把每个样本变成固定形状 token 张量，并把 batch 拼接成全局索引空间（node_ptr/edge_ptr）。
3. **bottom-up**在树上从 leaf 往 root 计算每个 node 的 latent `z`（全局表）。
4. **top-down**从 root 往 leaf 传播边界条件 logits，并输出 token logits 与 BC logits。
5. **labeler**生成 teacher 标签，**losses** 组合 token loss + BC loss。
6. **edge_aggregation + edge_decode**把 logits 变成 tour，用于评估与可视化。

---

## 1. 代码模块总览与职责边界（按文件）

### 1.1 数据与图结构（预处理侧）
- `data_generator.py`：生成 2D 点集数据（训练/验证/测试）。
- `spanner.py`：Delaunay triangulation spanner（undirected，规范化 u<v；支持 disjoint union batch）。
- `build_raw_pyramid.py`：构建 quadtree/pyramid，注册 interface/crossing 事件与几何特征（contract v2）。
- `prune_pyramid.py`：r-light 工程化剪枝（按 `(node, dir)` 保留最短 r 条 interface），过滤 interface/crossing。

### 1.2 Packing / Tokenization（规模无关输入的工程落点）
- `node_token_packer.py`：核心打包器 `NodeTokenPacker`，输出 `PackedBatch`。
- `tokenization.py`：`NodeTokenizer` 将离散/连续 token 映射为 `d_model` embedding；同时定义 token memory layout。
- `set_transformer_block.py`：SetTransformer-style masked self-attention blocks（encoder/decoder 内部使用）。

### 1.3 Bottom-up（神经 DP 值函数）
- `leaf_encoder.py`：`LeafEncoder`：leaf 节点编码器（token + 点集 → latent）。
- `merge_encoder.py`：`MergeEncoderModule`：内部节点编码器（token + 4 子 latent → latent）。
- `bottom_up_runner.py`：`BottomUpTreeRunner`：按 depth 自底向上调度 leaf/merge encoder，产出全局 `z`。

### 1.4 Top-down（边界条件传播 / 回溯）
- `top_down_decoder.py`：`TopDownDecoderModule`：对单个节点做一次 “parent BC + local tokens + latents → child BC + token logits”。
- `top_down_runner.py`：`TopDownTreeRunner`：按 depth 从 root 往下调度 decoder，并把 child BC 写入全局 `bc_iface_logit`。

### 1.5 Teacher / Loss / Metrics / Decode / Train
- `tour_solver.py`：启发式 TSP tour（NN + 2-opt）与工具函数。
- `labeler.py`：`PseudoLabeler`：teacher 投影到 alive spanner，生成 token-level 与 BC 监督标签。
- `losses.py`：`dp_token_losses`、`masked_bce_with_logits`、以及 top-down loss 组合。
- `edge_aggregation.py`：cross logits → edge logits 聚合（amax reduce）。
- `edge_decode.py`：edge logits → 解码 tour（优先 spanner；可 patching）。
- `metrics.py`：edge-level 的 P@k/R@k 等指标封装。
- `train.py`：训练入口、验证、checkpoint。

---

## 2. Data（预处理产物）的字段契约（最重要）

以下字段由 `build_raw_pyramid.py`/`prune_pyramid.py` 保证存在，`NodeTokenPacker` 会强依赖：

### 2.1 spanner 与点
- `pos: [N,2]`
- `spanner_edge_index: [2,E]`（local，u<v）
- `spanner_edge_attr: [E,1]`（欧氏长度）

### 2.2 树结构（全局 node id 为 0..M-1）
- `tree_node_feat: [M,4]`（box llwh abs；`tree_node_feat_box_mode=0`）
- `tree_node_depth: [M]`
- `tree_parent_index: [M]`（root 的 parent <0）
- `tree_children_index: [M,4]`（孩子顺序固定 TL/TR/BL/BR）
- `is_leaf: [M] bool`
- `num_tree_nodes = M`

### 2.3 leaf 点集 CSR
- `leaf_ids: [L]`（leaf node id）
- `leaf_ptr: [L+1]`
- `leaf_points: [num_leaf_points]`（point id 列表）
- `point_to_leaf: [N]`（每点归属 leaf）

### 2.4 interface（flat records）
- `interface_assign_index: [2,I]`（node_id, eid）
- `interface_edge_attr: [I,6]`（contract v2）
- `interface_boundary_dir: [I]`
- `interface_inside_endpoint: [I]`
- `interface_inside_quadrant: [I]`

### 2.5 crossing（flat records）
- `crossing_assign_index: [2,C]`（node_id, eid）
- `crossing_edge_attr: [C,6]`
- `crossing_child_pair: [C,2]`
- `crossing_is_leaf_internal: [C] bool`

---

## 3. PackedBatch：batch 级张量契约（所有训练/推理都围绕它）

### 3.1 关键 dataclass（在 `node_token_packer.py`）
- `PackedLeafPoints`
  - `leaf_node_id: [L_total]`
  - `point_idx: [L_total, P]`（-1 padding）
  - `point_mask: [L_total, P] bool`
  - `point_xy: [L_total, P, 2]`（leaf-cell-relative，已归一化）
- `PackedNodeTokens`（所有 node 维度均为 `total_M`）
  - `tree_node_feat_rel: [total_M,4]`（root-normalized center-box）
  - `tree_node_depth: [total_M]`
  - `tree_parent_index: [total_M]`（batch-global node id）
  - `tree_children_index: [total_M,4]`（batch-global node id，-1 表示不存在）
  - `root_id: [B]`（每图 root 的 batch-global node id）
  - Interface tokens（均为 `[total_M, Ti, ...]`）：
    - `iface_eid: [total_M,Ti]`（batch-global eid；pad = -1）
    - `iface_mask: [total_M,Ti] bool`
    - `iface_feat6: [total_M,Ti,6] float`
    - `iface_boundary_dir / iface_inside_endpoint / iface_inside_quadrant: [total_M,Ti] long`
  - Crossing tokens（均为 `[total_M, Tc, ...]`）：
    - `cross_eid: [total_M,Tc]`
    - `cross_mask: [total_M,Tc] bool`
    - `cross_feat6: [total_M,Tc,6]`
    - `cross_child_pair: [total_M,Tc,2]`
    - `cross_is_leaf_internal: [total_M,Tc] bool`
- `PackedBatch`
  - `node_ptr: [B+1]`（每图 node 段）
  - `leaf_ptr: [B+1]`（每图 leaf 段）
  - `edge_ptr: [B+1]`（每图 edge 段，用于 global eid offset）
  - `graph_id_for_node: [total_M]`
  - `tokens: PackedNodeTokens`
  - `leaves: PackedLeafPoints`

### 3.2 global/local 索引规则（非常关键）
- node id：pack 后统一为 **batch-global**（由 `node_ptr` 给切片）
- edge eid：pack 后统一为 **batch-global**（每图 `eid_offset = edge_ptr[b]`）
  - 因此 tokens 的 `*_eid` 在 pack 时会加 offset
  - labeler / edge_aggregation / decode 必须遵守这个规则

---

## 4. Tokenization：NodeTokenizer 的输出 layout

`NodeTokenizer` 把“节点上下文 + interface/crossing +（可选）child slot token”组合成一个 token memory：

- 输入（对单个 node batch B）：
  - `node_feat_rel: [B,4]`
  - `node_depth: [B]`
  - `iface_*: [B,Ti,...]`
  - `cross_*: [B,Tc,...]`
  - （可选）`child_z: [B,4,d]` 与 `child_mask: [B,4]`
- 输出：`TokenizedNodeMemory`
  - `tokens: [B, T_total, d_model]`
  - `mask: [B, T_total] bool`
  - `cls_index` 与 `iface_slice/cross_slice/child_slice`（用于 runner/decoder 定位）

工程上最常用的两个入口：
- `tokenize_node(...)`：构造完整 memory（含 iface/cross/child slot）
- `embed_iface_only(...)`：只嵌入 iface tokens（用于 top-down 的 child query embedding）

---

## 5. Bottom-up：LeafEncoder / MergeEncoder / BottomUpTreeRunner

### 5.1 LeafEncoder
- 输入（按 leaf node batch B_leaf）：
  - node 的 token（iface/cross）+ leaf 点集 `[P,2]`
- 内部结构：
  - `NodeTokenizer` 生成 token memory
  - 点集 MLP 得到 point tokens
  - SetTransformer encoder 交互
  - CLS pooling 得 `z_leaf: [B_leaf, d_model]`

### 5.2 MergeEncoderModule
- 输入（内部 node batch B_int）：
  - node token（iface/cross）
  - `child_z: [B_int,4,d_model]` + mask
- 输出：`z_parent: [B_int, d_model]`

### 5.3 BottomUpTreeRunner
- 输入：`PackedBatch` + encoders
- 输出：`BottomUpResult`
  - `z: [total_M, d_model]`
  - `computed: [total_M] bool`（安全检查）
  - `root_ids: [B]`
  - `aux`：统计信息
- 关键不变量：
  - 每图恰好 1 个 root（`parent<0`）
  - internal node 计算前其 4 个孩子必须已 computed
  - 写回 `z` 用 `index_copy`，避免 autograd 问题

---

## 6. Top-down：TopDownDecoderModule / TopDownTreeRunner

### 6.1 TopDownDecoderModule（单节点转移）
**输入（B 个 node）**：
- `bc_in_iface_logit: [B,Ti]`（上层传给当前 node 的 BC）
- 当前 node token：
  - `iface_*: [B,Ti,...]`
  - `cross_*: [B,Tc,...]`
- bottom-up latent：
  - `z_node: [B,d]`
  - `child_z: [B,4,d]` + `child_exists_mask: [B,4]`
- 孩子 iface tokens（用于直接输出孩子接口打分）：
  - `child_iface_feat6: [B,4,Ti,6]` 等 + mask

**输出** `TopDownDecoderOutput`：
- `iface_logit: [B,Ti]`（本 node 的 iface token logits；常作辅助监督）
- `cross_logit: [B,Tc]`（本 node 的 crossing token logits；用于最终 edge 聚合）
- `child_iface_logit: [B,4,Ti]`（父→子 BC 打分，核心输出）

**两种模式**：
- `mode="two_stage"`：child iface tokens 作为 query，通过 Cross-Attn 从 parent memory 读取信息，再打分。
- `mode="one_stage"`：不做 cross-attn，直接对 child token embedding 做 MLP head 打分（更轻量）。

### 6.2 TopDownTreeRunner（树上调度）
- 维护全局 `bc_iface_logit: [total_M,Ti]`：
  - 初始化为 `-inf`，root 设为 0（logit=0 表示“未定/中性”）
  - 对每个 internal node v：
    - 取 `bc_in = bc_iface_logit[v]` 作为 decoder 输入
    - decoder 输出 `child_iface_logit[v,q]`
    - 写回：`bc_iface_logit[child_q] = child_iface_logit[v,q]`（只对存在孩子）
- 同时收集全局 token logits：
  - `iface_logit: [total_M,Ti]`
  - `cross_logit: [total_M,Tc]`

**输出** `TopDownResult`：
- `iface_logit, cross_logit, bc_iface_logit, root_ids, node_ptr, aux`

---

## 7. Teacher / Labeler：PseudoLabeler 与索引陷阱

### 7.1 PseudoLabeler 输出
`TokenLabels`（与 PackedBatch 对齐）：
- `y_cross, m_cross: [total_M,Tc]`
- `y_iface, m_iface: [total_M,Ti]`
- `y_child_iface, m_child_iface: [total_M,4,Ti]`（显式父→子监督用）
- `stats`: direct/projected/unreachable 等统计

### 7.2 global/local eid 处理（已经修复的关键点）
- `label_one` 内部用 **local eid 空间**建 `eid_table`（长度=E），避免 batch global offset 导致 table 爆炸。
- batch 监督建议使用 `label_batch`：
  - 它用 `packed.edge_ptr[b]` 作为 eid_offset 调 `label_one`
  - 并在 batch-global node 索引空间里构造 `y_child_iface`

---

## 8. Losses：如何把监督信号拼起来

### 8.1 token-level（辅助/主信号之一）
- `dp_token_losses(cross_logit, y_cross, m_cross, iface_logit, y_iface, m_iface, ...)`
  - cross BCE 是主要项
  - iface BCE 可用较小权重作为辅助项（`w_iface`）

### 8.2 BC loss（当前训练的核心之一）
`train.py` 当前实现：对每个 **non-root node**：
- 预测：`out_td.bc_iface_logit[v]`（父写入的 BC）
- teacher：`labels.y_iface[v]`
- mask：`labels.m_iface[v] & (~is_root)`

这等价于监督 “父→子边界条件” 且实现最简洁。

---

## 9. Edge aggregation / Decode / Metrics

### 9.1 edge aggregation
- `aggregate_cross_logits_to_edges(tokens, cross_logit) -> EdgeScores`
  - 利用 `tokens.cross_eid` 将 `[M,Tc]` 聚合到 `[E_total]`
  - 用 `amax` 规避重复记录

### 9.2 decode
- `decode_tour_from_edge_logits(pos, spanner_edge_index, edge_logit, ...) -> TourDecodeResult`
  - 贪心选边 + 度约束 + 连通性修补
  - 可选 `allow_off_spanner_patch`

### 9.3 metrics
- `edge_topk_precision_recall(...)` 等，为训练/验证日志提供 P@k/R@k。

---

## 10. train.py：训练入口与运行方式

### 10.1 主要 CLI 参数
- 数据：`--train_pt`、`--val_pt`
- 模型：`--r`（影响 Ti/Tc 默认）、`--td_mode one_stage/two_stage`
- 优化：`--lr --wd --grad_clip --batch_size --epochs`
- teacher：`--two_opt_passes`
- loss：`--w_token --w_iface_aux --w_bc`
- checkpoint：`--ckpt_dir --save_interval`

### 10.2 单步训练数据流
1. `packed = packer.pack_batch(datas)`
2. `out_bu = bu_runner.run_batch(...)` 得 `z`
3. `out_td = td_runner.run_batch(packed=packed, z=z, decoder=decoder)`
4. `labels = labeler.label_batch(...)`
5. `loss = w_token * token_loss + w_bc * bc_loss`
6. 反传 + clip + AdamW step

---

## 11. 工程不变量 / 常见坑（必读）

1. **孩子顺序固定**（TL/TR/BL/BR）  
   任意破坏将导致 top-down 的 `child_iface_logit[...,i]` 语义漂移。
2. **interface token 顺序稳定**  
   packer 内部做 stable sort（依赖边界方向与边界参数量化）；否则 child query 的位置语义不稳定。
3. **global / local eid 必须严格区分**  
   - pack 后 tokens 存的是 global eid  
   - 单图 decode/teacher 投影需要 local eid；必须减去 `edge_ptr[b]`
4. **padding 必须用 mask 屏蔽，且 logits 对 padding 置 -inf**  
   否则 BCE 与 decode 会被 padding 干扰。
5. **禁止对 requires_grad Tensor 做原地切片赋值**  
   runner 已用 `index_copy`；继续保持该模式。
6. **root 的 BC 初始化**  
   当前 root BC = 0 是“中性”；若改成别的初始化需同步改 teacher/损失与 reachability 检查。

---

## 12. 扩展点建议（面向后续工作）

- **加入 matching/pairing 头**：把 BC 从“使用分数”扩展到“配对结构”（图匹配/指针网络/流式解码）。
- **更强 teacher**：更高质量 tour 或多 teacher，降低投影偏差。
- **更强 decode**：把 degree=2 与连通性约束写成显式 ILP/最小割修补，作为评估上界或训练时的后处理。
- **训练目标一致性**：加入 parent/child BC 一致性正则（例如 KL/对称 BCE）与跨层约束（crossing 对 child BC 的一致性）。

---

## 13. 快速“接口字典”（接手时最常用）

- `NodeTokenPacker.pack_batch(datas) -> PackedBatch`
- `BottomUpTreeRunner.run_batch(batch=packed, ...) -> BottomUpResult(z, ...)`
- `TopDownTreeRunner.run_batch(packed=packed, z=z, decoder=...) -> TopDownResult(iface_logit, cross_logit, bc_iface_logit, ...)`
- `PseudoLabeler.label_batch(datas, packed, device) -> TokenLabels(y_cross/y_iface/y_child_iface, masks, stats)`
- `dp_token_losses(...) -> LossOut(loss, parts)`
- `aggregate_cross_logits_to_edges(tokens, cross_logit) -> EdgeScores(edge_logit, edge_mask)`
- `decode_tour_from_edge_logits(...) -> TourDecodeResult`
