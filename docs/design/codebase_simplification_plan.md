# 工程精简与重构计划

**Date**: 2026-03-28
**Status**: Proposed
**See also**: [工程精简重构执行清单](./codebase_simplification_checklist.md)

## 1. 背景

当前仓库中实际并存的是两条**代码路径**，而不是两条独立的 git 分支：

- **1-pass 路径**：更贴近论文主算法，核心是 `sigma-conditioned merge -> cost table -> traceback`。
- **2-pass 路径**：更偏向 practical，核心是 `bottom-up latent cache -> top-down single decode -> edge guidance -> greedy/exact/LKH`。

这两条路径同时存在本身没有问题。真正导致代码臃肿的原因是：

1. 共享骨架没有抽出来。
2. 同样的树遍历、张量取数、attention memory、CLI 装配逻辑被多次实现。
3. 若干历史兼容模式、legacy 路径和死代码仍留在主流程中。

本计划的目标不是“强行把 1-pass 和 2-pass 写成同一个类”，而是把仓库整理成三层：

- **共享骨架层**：packer、encoder、bottom-up encode、decode backend、CLI/runtime helper。
- **论文主算法层**：`merge_decoder + DP + traceback`。
- **practical 层**：`top_down_decoder + edge guidance + greedy/exact/LKH`。

## 2. 重构目标

### 2.1 主要目标

- 显著减少重复代码和巨石文件。
- 保持 1-pass / 2-pass 两条路径语义独立，但共享底层骨架。
- 让目录结构直接反映“共享层 / 1-pass / 2-pass”的边界。
- 降低后续继续补论文算法或做 practical 实验时的改动半径。

### 2.2 非目标

- 不在第一阶段追求完全统一 1-pass 和 2-pass 的算法接口。
- 不在第一阶段修改 checkpoint key、实验产物目录、CLI 参数名。
- 不在第一阶段追求“删到最少文件数”，而是优先降低维护复杂度。

## 3. 当前冗余来源

### 3.1 CLI / runtime helper 重复

以下文件中存在重复的 `set_seed`、`resolve_device`、`parse_bool_arg`、`move_data_tensors_to_device`、数据集加载与日志辅助逻辑：

- [src/cli/train.py](/remote-home/MoGuanlin/NNTSP/src/cli/train.py)
- [src/cli/train_onepass.py](/remote-home/MoGuanlin/NNTSP/src/cli/train_onepass.py)
- [src/cli/eval_and_vis.py](/remote-home/MoGuanlin/NNTSP/src/cli/eval_and_vis.py)
- [src/cli/evaluate_tsplib.py](/remote-home/MoGuanlin/NNTSP/src/cli/evaluate_tsplib.py)

此外，[src/cli/eval_and_vis.py:283](/remote-home/MoGuanlin/NNTSP/src/cli/eval_and_vis.py#L283) 和 [src/cli/evaluate_tsplib.py:495](/remote-home/MoGuanlin/NNTSP/src/cli/evaluate_tsplib.py#L495) 也几乎重复了一套 checkpoint 读取与模型构造流程。

### 3.2 bottom-up 遍历重复

- [src/models/bottom_up_runner.py:175](/remote-home/MoGuanlin/NNTSP/src/models/bottom_up_runner.py#L175) 实现了标准 bottom-up 编码。
- [src/models/onepass_trainer.py:140](/remote-home/MoGuanlin/NNTSP/src/models/onepass_trainer.py#L140) 又重复维护了一套相似的 bottom-up 编码。

这意味着 1-pass / 2-pass 虽然共用了 `LeafEncoder` / `MergeEncoder`，但并没有真正共享遍历骨架。

### 3.3 encoder 骨架重复

- [src/models/leaf_encoder.py:90](/remote-home/MoGuanlin/NNTSP/src/models/leaf_encoder.py#L90)
- [src/models/merge_encoder.py:93](/remote-home/MoGuanlin/NNTSP/src/models/merge_encoder.py#L93)

二者都在做：

- `NodeTokenizer`
- `node_ctx`
- `type_embed`
- `SetTransformer`
- `CLS pool`

差异主要只是 leaf 多了 point token，merge 多了 child token。

### 3.4 decoder attention 基建重复

- [src/models/top_down_decoder.py:65](/remote-home/MoGuanlin/NNTSP/src/models/top_down_decoder.py#L65)
- [src/models/merge_decoder.py:126](/remote-home/MoGuanlin/NNTSP/src/models/merge_decoder.py#L126)

分别维护了一套 `CrossAttentionBlock`，并且都在构建 parent memory，再让 query 去读这块 memory。

### 3.5 1-pass 巨石文件

- [src/models/dp_core.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_core.py)：1800+ 行，混合了 correspondence、verify、heuristic parse、catalog parse、leaf exact solver、Held-Karp/NN witness。
- [src/models/dp_runner.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_runner.py)：1400+ 行，混合了 orchestration、decode batching、fallback、stats、traceback。
- [src/models/tour_reconstruct.py](/remote-home/MoGuanlin/NNTSP/src/models/tour_reconstruct.py)：同时承担 direct reconstruction 与 legacy logit projection。

### 3.6 legacy / 死代码仍在主流程附近

- [src/models/dp_runner.py:692](/remote-home/MoGuanlin/NNTSP/src/models/dp_runner.py#L692) 的 `_process_internal_node` 当前无主流程调用，应视为待删除候选。
- [src/models/edge_decode.py:363](/remote-home/MoGuanlin/NNTSP/src/models/edge_decode.py#L363) 的 `DecodingDataset` 与 [src/models/decode_backend.py:46](/remote-home/MoGuanlin/NNTSP/src/models/decode_backend.py#L46) 重复，CLI 实际已统一依赖后者。
- [src/models/top_down_decoder.py:22](/remote-home/MoGuanlin/NNTSP/src/models/top_down_decoder.py#L22) 的 `one_stage` / `two_stage` 双模式更像研究期 ablation 开关。
- `dp_parse_mode=heuristic` 更适合作为 legacy 路径，而不是长期并列主流程。

## 4. 重构原则

### 4.1 先共享骨架，再拆算法头

优先把 1-pass / 2-pass 都需要的东西抽成共享层：

- CLI helper
- model factory
- bottom-up encode
- tree helper
- attention block
- parent memory builder

### 4.2 先做兼容重构，再做物理删减

对于巨石文件和稳定对外接口：

- 先抽新模块。
- 原文件先变为门面 / re-export。
- 测试稳定后再瘦身原文件。

### 4.3 第一阶段不破 checkpoint / CLI / 输出目录

前 3 个 PR 约束：

- 不改 checkpoint key。
- 不改命令行参数名。
- 不改 `outputs/`、`checkpoints/` 的现有组织方式。

### 4.4 低风险改动优先

优先级应为：

1. 删死代码与重复小件。
2. 抽 CLI / runtime 共用层。
3. 抽模型共享骨架。
4. 最后再拆 `labeler` / `packer` 这类高耦合核心。

## 5. 分阶段重构计划

## 5.1 PR1：抽出 CLI / runtime 公共层

### 目标

减少所有 CLI 入口中的重复 helper 与模型装配代码。

### 新增文件

- [src/cli/common.py](/remote-home/MoGuanlin/NNTSP/src/cli/common.py)
- [src/cli/model_factory.py](/remote-home/MoGuanlin/NNTSP/src/cli/model_factory.py)

### 迁移内容

从以下文件迁出公共函数：

- [src/cli/train.py](/remote-home/MoGuanlin/NNTSP/src/cli/train.py)
- [src/cli/train_onepass.py](/remote-home/MoGuanlin/NNTSP/src/cli/train_onepass.py)
- [src/cli/eval_and_vis.py](/remote-home/MoGuanlin/NNTSP/src/cli/eval_and_vis.py)
- [src/cli/evaluate_tsplib.py](/remote-home/MoGuanlin/NNTSP/src/cli/evaluate_tsplib.py)

建议统一的公共函数：

- `set_seed`
- `resolve_device`
- `parse_bool_arg`
- `move_data_tensors_to_device`
- `smart_load_dataset` 包装
- `log_progress`

### 额外收口

把以下两段几乎重复的模型恢复逻辑统一进 `model_factory.py`：

- [src/cli/eval_and_vis.py:283](/remote-home/MoGuanlin/NNTSP/src/cli/eval_and_vis.py#L283)
- [src/cli/evaluate_tsplib.py:495](/remote-home/MoGuanlin/NNTSP/src/cli/evaluate_tsplib.py#L495)

### 风险

低。

### 验收

- 训练与评测命令行行为不变。
- 只减少代码重复，不引入算法行为变化。

## 5.2 PR2：让 1-pass 复用 2-pass 的 bottom-up encode 骨架

### 目标

让 1-pass 与 2-pass 真正共享 bottom-up traversal，而不是只共享 encoder 类。

### 主要改动

- [src/models/onepass_trainer.py](/remote-home/MoGuanlin/NNTSP/src/models/onepass_trainer.py) 改为先调用 [src/models/bottom_up_runner.py](/remote-home/MoGuanlin/NNTSP/src/models/bottom_up_runner.py) 得到 `z`，再做 teacher-sigma decode。
- [src/models/dp_runner.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_runner.py) 也先复用 shared bottom-up encode，再做 DP merge 与 traceback。

### 预期收益

- 删除 [src/models/onepass_trainer.py:157](/remote-home/MoGuanlin/NNTSP/src/models/onepass_trainer.py#L157) 后的大量重复 traversal 逻辑。
- 统一 bottom-up 的正确性约束、debug 行为和未来优化入口。

### 风险

中等。

### 验收

- [tests/test_bottom_up_process.py](/remote-home/MoGuanlin/NNTSP/tests/test_bottom_up_process.py)
- [tests/test_onepass_trainer.py](/remote-home/MoGuanlin/NNTSP/tests/test_onepass_trainer.py)
- [tests/test_dp_runner.py](/remote-home/MoGuanlin/NNTSP/tests/test_dp_runner.py)

## 5.3 PR3：抽 tree helper

### 目标

收口多个 runner / reconstruct 文件中的重复小工具。

### 新增文件

- [src/models/shared_tree.py](/remote-home/MoGuanlin/NNTSP/src/models/shared_tree.py)

### 迁移内容

- `_build_leaf_row_for_node`
- `gather_node_fields`
- `_extract_z`
- 可能还包括 root id / depth 扫描等通用逻辑

### 当前重复位置

- [src/models/bottom_up_runner.py:85](/remote-home/MoGuanlin/NNTSP/src/models/bottom_up_runner.py#L85)
- [src/models/tour_reconstruct.py:187](/remote-home/MoGuanlin/NNTSP/src/models/tour_reconstruct.py#L187)
- [src/models/bottom_up_runner.py:194](/remote-home/MoGuanlin/NNTSP/src/models/bottom_up_runner.py#L194)
- [src/models/onepass_trainer.py:289](/remote-home/MoGuanlin/NNTSP/src/models/onepass_trainer.py#L289)

### 风险

低。

### 验收

- runner 与 reconstruct 测试全绿。

## 5.4 PR4：抽共享 decoder 基建

### 目标

把 1-pass / 2-pass decoder 共享的 attention 基础设施抽掉。

### 新增文件

- [src/models/shared_attention.py](/remote-home/MoGuanlin/NNTSP/src/models/shared_attention.py)
- [src/models/parent_memory.py](/remote-home/MoGuanlin/NNTSP/src/models/parent_memory.py)

### 迁移内容

- 合并两个 `CrossAttentionBlock`
- 合并 `NodeTokenizer -> inject z_node -> self-attn blocks -> ParentMemory`

### 影响文件

- [src/models/top_down_decoder.py](/remote-home/MoGuanlin/NNTSP/src/models/top_down_decoder.py)
- [src/models/merge_decoder.py](/remote-home/MoGuanlin/NNTSP/src/models/merge_decoder.py)

### 风险

中等。

### 验收

- [tests/test_top_down_process.py](/remote-home/MoGuanlin/NNTSP/tests/test_top_down_process.py)
- [tests/test_merge_decoder.py](/remote-home/MoGuanlin/NNTSP/tests/test_merge_decoder.py)

## 5.5 PR5：收敛 leaf / merge encoder 到共同 backbone

### 目标

把 `LeafEncoder` 与 `MergeEncoder` 的共享骨架从“复制”改为“复用”。

### 新增文件

- [src/models/base_node_encoder.py](/remote-home/MoGuanlin/NNTSP/src/models/base_node_encoder.py)

### 设计

`BaseNodeEncoder` 统一负责：

- `NodeTokenizer`
- `node_ctx`
- `type_embed`
- `SetTransformer`
- `CLS pool`

具体 token 差异由外层提供：

- `LeafEncoder` 只负责 point token。
- `MergeEncoder` 只负责 child token。

### 风险

中等。

### 验收

- 2-pass 训练和评测输出不变。
- 1-pass encode 相关测试不退化。

## 5.6 PR6：拆分 [src/models/dp_core.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_core.py)

### 目标

按职责拆开 1-pass DP 的核心算法库。

### 拆分方案

- [src/models/dp_correspondence.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_correspondence.py)
  - `CorrespondenceMaps`
  - `build_correspondence_maps`
  - `propagate_c1_constraints`
- [src/models/dp_verify.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_verify.py)
  - `verify_tuple`
  - `batch_check_c1c2`
- [src/models/dp_parse_heuristic.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_parse_heuristic.py)
  - `parse_continuous`
  - `parse_continuous_topk`
  - `parse_activation_batch`
- [src/models/dp_parse_catalog.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_parse_catalog.py)
  - `parse_by_catalog_enum`
  - `_rank_child_catalog_states_for_parse`
- [src/models/dp_leaf_solver.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_leaf_solver.py)
  - Held-Karp / NN helper
  - `leaf_solve_state`
  - `leaf_exact_solve`

### 兼容策略

第一阶段保留 [src/models/dp_core.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_core.py) 作为门面文件，负责 re-export。

### 风险

中等偏高。

### 验收

- [tests/test_dp_core.py](/remote-home/MoGuanlin/NNTSP/tests/test_dp_core.py)
- [tests/test_tour_reconstruct.py](/remote-home/MoGuanlin/NNTSP/tests/test_tour_reconstruct.py)
- [tests/test_dp_runner.py](/remote-home/MoGuanlin/NNTSP/tests/test_dp_runner.py)

## 5.7 PR7：瘦身 [src/models/dp_runner.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_runner.py)

### 目标

让 `OnePassDPRunner` 只负责 orchestration，而不是持有所有 parse / fallback / stats / traceback 细节。

### 先做的事情

- 删除 [src/models/dp_runner.py:692](/remote-home/MoGuanlin/NNTSP/src/models/dp_runner.py#L692) `_process_internal_node` 这条旧路径。

### 再拆的模块

- `dp_types.py`
- `dp_fallback.py`
- `dp_traceback.py`
- `dp_stats.py`

### 风险

中等。

### 验收

- [tests/test_dp_runner.py](/remote-home/MoGuanlin/NNTSP/tests/test_dp_runner.py)
- [tests/test_depth_fallback_stats.py](/remote-home/MoGuanlin/NNTSP/tests/test_depth_fallback_stats.py)
- [tests/test_child_catalog_cap.py](/remote-home/MoGuanlin/NNTSP/tests/test_child_catalog_cap.py)
- [tests/test_sigma_cap.py](/remote-home/MoGuanlin/NNTSP/tests/test_sigma_cap.py)

## 5.8 PR8：decode / reconstruct 收口

### 目标

明确 1-pass 的主输出路径，并把 legacy 兼容路径降级。

### 主路径

- [src/models/tour_reconstruct.py:503](/remote-home/MoGuanlin/NNTSP/src/models/tour_reconstruct.py#L503) 的 direct reconstruction 作为 1-pass 主输出。

### 兼容路径

将以下兼容逻辑移至单独 legacy 文件：

- [src/models/tour_reconstruct.py:598](/remote-home/MoGuanlin/NNTSP/src/models/tour_reconstruct.py#L598)
- [src/models/tour_reconstruct.py:737](/remote-home/MoGuanlin/NNTSP/src/models/tour_reconstruct.py#L737)

建议新文件：

- [src/models/tour_reconstruct_legacy.py](/remote-home/MoGuanlin/NNTSP/src/models/tour_reconstruct_legacy.py)

### 同步清理

- 删除 [src/models/edge_decode.py:363](/remote-home/MoGuanlin/NNTSP/src/models/edge_decode.py#L363) 的重复 `DecodingDataset`。
- 删除 [src/models/tour_reconstruct.py:647](/remote-home/MoGuanlin/NNTSP/src/models/tour_reconstruct.py#L647) 那个无消费返回值的 `build_correspondence_maps(...)` 调用。

### 风险

低到中等。

### 验收

- [tests/test_decode_tour.py](/remote-home/MoGuanlin/NNTSP/tests/test_decode_tour.py)
- [tests/test_exact_decode.py](/remote-home/MoGuanlin/NNTSP/tests/test_exact_decode.py)
- [tests/test_tour_reconstruct.py](/remote-home/MoGuanlin/NNTSP/tests/test_tour_reconstruct.py)

## 5.9 PR9：最后拆 labeler / packer

### 目标

把两个高耦合核心文件拆成 facade + 子模块结构。

### labeler 拆分建议

- `teacher_solver.py`
- `edge_projection.py`
- `matching_targets.py`
- `batch_labeler.py`

对外仍保留 `PseudoLabeler`。

### packer 拆分建议

- `pack_contracts.py`
- `pack_single.py`
- `pack_batch.py`

对外仍保留 `NodeTokenPacker`。

### 风险

高。

### 验收

- 2-pass 训练流程可以完整跑通。
- 1-pass 训练流程可以完整跑通。
- pack / label 相关测试全部通过。

## 6. 可直接删除或降级到 legacy 的候选

### 可直接删除

- [src/models/dp_runner.py:692](/remote-home/MoGuanlin/NNTSP/src/models/dp_runner.py#L692) `_process_internal_node`
- [src/models/edge_decode.py:363](/remote-home/MoGuanlin/NNTSP/src/models/edge_decode.py#L363) `DecodingDataset`

### 先降级，不立刻删除

- `TopDownDecoder.one_stage`
- `dp_parse_mode=heuristic`
- 部分 `return_aux`

### 处理原则

- 先确认 checkpoint `args`、脚本默认值、测试中是否仍依赖。
- 先在文档和代码中标记 deprecated。
- 一个稳定周期后再彻底删除。

## 7. 推荐执行顺序

### 推荐顺序

1. PR1：CLI / runtime 公共层
2. PR2：bottom-up encode 共享
3. PR4：decoder 基建共享
4. PR6：拆 `dp_core`
5. PR7：瘦身 `dp_runner`
6. PR8：decode / reconstruct 收口
7. PR9：拆 `labeler` / `packer`

### 更保守的顺序

1. PR1
2. PR3
3. PR4
4. PR8
5. PR2
6. PR6
7. PR7
8. PR9

## 8. 第一批建议落地范围

如果目标是先快速把工程“瘦下来”，建议第一批只做：

- PR1：抽 CLI / runtime helper
- PR2：让 `onepass_trainer` / `dp_runner` 复用 bottom-up encode
- 删除 [src/models/dp_runner.py:692](/remote-home/MoGuanlin/NNTSP/src/models/dp_runner.py#L692)
- 删除 [src/models/edge_decode.py:363](/remote-home/MoGuanlin/NNTSP/src/models/edge_decode.py#L363)

这批改动的特点是：

- 收益高
- 风险相对可控
- 不会直接碰论文算法语义
- 能马上降低认知负担

## 9. 测试与验收矩阵

### 共享骨架阶段

- [tests/test_bottom_up_process.py](/remote-home/MoGuanlin/NNTSP/tests/test_bottom_up_process.py)
- [tests/test_top_down_process.py](/remote-home/MoGuanlin/NNTSP/tests/test_top_down_process.py)
- [tests/test_train_step.py](/remote-home/MoGuanlin/NNTSP/tests/test_train_step.py)

### 1-pass 阶段

- [tests/test_merge_decoder.py](/remote-home/MoGuanlin/NNTSP/tests/test_merge_decoder.py)
- [tests/test_dp_core.py](/remote-home/MoGuanlin/NNTSP/tests/test_dp_core.py)
- [tests/test_dp_runner.py](/remote-home/MoGuanlin/NNTSP/tests/test_dp_runner.py)
- [tests/test_onepass_trainer.py](/remote-home/MoGuanlin/NNTSP/tests/test_onepass_trainer.py)
- [tests/test_tour_reconstruct.py](/remote-home/MoGuanlin/NNTSP/tests/test_tour_reconstruct.py)

### fallback / catalog / cap 相关

- [tests/test_depth_fallback_stats.py](/remote-home/MoGuanlin/NNTSP/tests/test_depth_fallback_stats.py)
- [tests/test_child_catalog_cap.py](/remote-home/MoGuanlin/NNTSP/tests/test_child_catalog_cap.py)
- [tests/test_sigma_cap.py](/remote-home/MoGuanlin/NNTSP/tests/test_sigma_cap.py)
- [tests/test_teacher_max_used_stats.py](/remote-home/MoGuanlin/NNTSP/tests/test_teacher_max_used_stats.py)

### decode / exact / reconstruct

- [tests/test_decode_tour.py](/remote-home/MoGuanlin/NNTSP/tests/test_decode_tour.py)
- [tests/test_exact_decode.py](/remote-home/MoGuanlin/NNTSP/tests/test_exact_decode.py)

## 10. 最终期望目录形态

一个更清晰的目标形态如下：

```text
src/
├── cli/
│   ├── common.py
│   ├── model_factory.py
│   ├── train.py
│   ├── train_onepass.py
│   ├── eval_and_vis.py
│   ├── eval_onepass.py
│   └── evaluate_tsplib.py
├── models/
│   ├── shared_tree.py
│   ├── shared_attention.py
│   ├── parent_memory.py
│   ├── base_node_encoder.py
│   ├── leaf_encoder.py
│   ├── merge_encoder.py
│   ├── top_down_decoder.py
│   ├── top_down_runner.py
│   ├── merge_decoder.py
│   ├── dp_correspondence.py
│   ├── dp_verify.py
│   ├── dp_parse_catalog.py
│   ├── dp_parse_heuristic.py
│   ├── dp_leaf_solver.py
│   ├── dp_runner.py
│   ├── tour_reconstruct.py
│   ├── tour_reconstruct_legacy.py
│   ├── decode_backend.py
│   ├── labeler/
│   │   ├── teacher_solver.py
│   │   ├── edge_projection.py
│   │   ├── matching_targets.py
│   │   └── batch_labeler.py
│   └── packer/
│       ├── pack_contracts.py
│       ├── pack_single.py
│       └── pack_batch.py
```

## 11. 建议的起始动作

开始执行时，建议先在一个单独分支上完成以下最小集合：

1. 新建 `src/cli/common.py`
2. 新建 `src/cli/model_factory.py`
3. 让 `train_onepass.py` 与 `eval_onepass.py` 改用共享 helper
4. 让 `onepass_trainer.py` 复用 `bottom_up_runner.py`
5. 删除 `dp_runner._process_internal_node`
6. 删除 `edge_decode.DecodingDataset`

这一步做完后，再继续进入 `dp_core` / `dp_runner` 的深拆分会更稳。
