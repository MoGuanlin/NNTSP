# 工程精简重构执行清单

**Date**: 2026-03-28
**Status**: Ready to Execute
**Related**: [工程精简与重构计划](./codebase_simplification_plan.md)

## 1. 使用方式

这份文档的目标不是重复解释“为什么要重构”，而是把总方案拆成可以逐项勾选的执行清单。

建议执行方式：

- 一次只推进一个 PR。
- 每个 PR 合并前都保证主测试集和最小 smoke 检查通过。
- 前 3 个 PR 不改 checkpoint key、不改 CLI 参数名、不改输出目录结构。
- 每个 PR 的描述里都附上“本次实际执行了哪些命令、哪些测试通过、哪些风险仍存在”。

## 2. 全局完成标准

只有满足以下条件，相关 PR 才算真正完成：

- [ ] 代码能通过基础编译检查：`python -m compileall src tests`
- [ ] 对应 PR 的目标测试全部通过
- [ ] 没有引入新的重复定义
- [ ] 没有意外修改 CLI 参数名或默认行为
- [ ] 文档同步更新到当前实现状态

## 3. 开工前基线记录

这一步建议在开始任何代码改动前完成一次，并把结果贴到第一个 PR 描述里。

### 3.1 基线命令

```bash
python -m pytest \
  tests/test_dp_core.py \
  tests/test_dp_runner.py \
  tests/test_onepass_trainer.py \
  tests/test_merge_decoder.py \
  tests/test_tour_reconstruct.py \
  tests/test_decode_tour.py \
  tests/test_exact_decode.py \
  tests/test_child_catalog_cap.py \
  tests/test_sigma_cap.py \
  tests/test_depth_fallback_stats.py \
  tests/test_node_token_packer.py -q
```

```bash
python -m compileall src tests
```

```bash
wc -l \
  src/models/dp_core.py \
  src/models/dp_runner.py \
  src/models/tour_reconstruct.py \
  src/models/labeler.py \
  src/models/node_token_packer.py
```

```bash
rg -n "def set_seed|def resolve_device|def parse_bool_arg|def move_data_tensors_to_device" src/cli
```

```bash
rg -n "class CrossAttentionBlock" src/models
```

```bash
rg -n "class DecodingDataset" src/models
```

```bash
rg -n "def _process_internal_node" src/models/dp_runner.py
```

```bash
python src/cli/train.py --help >/tmp/nntsp_train.help
python src/cli/train_onepass.py --help >/tmp/nntsp_train_onepass.help
python src/cli/eval_and_vis.py --help >/tmp/nntsp_eval_and_vis.help
python src/cli/evaluate_tsplib.py --help >/tmp/nntsp_evaluate_tsplib.help
```

### 3.2 基线 checklist

- [ ] 保存主测试集当前通过状态
- [ ] 记录巨石文件当前行数
- [ ] 记录 CLI `--help` 输出
- [ ] 记录重复定义当前 grep 结果

## 4. PR0：低风险删除项预清理

这个 PR 是可选的，但如果想先快速减重，建议先做。

### 4.1 目标

- 删除确认无主流程依赖的死代码
- 删除明显重复的小模块
- 不碰算法语义

### 4.2 主要文件

- `src/models/dp_runner.py`
- `src/models/edge_decode.py`
- `src/models/decode_backend.py`
- `src/models/tour_reconstruct.py`

### 4.3 Checklist

- [ ] 确认 `src/models/dp_runner.py` 中旧版 `_process_internal_node` 无调用方
- [ ] 删除 `src/models/dp_runner.py` 中旧版 `_process_internal_node`
- [ ] 确认 `src/models/edge_decode.py` 中 `DecodingDataset` 无主流程 import
- [ ] 删除 `src/models/edge_decode.py` 中重复的 `DecodingDataset`
- [ ] 检查 `src/models/tour_reconstruct.py` 中无消费结果的辅助调用
- [ ] 删除 `src/models/tour_reconstruct.py` 中无实际作用的 `build_correspondence_maps(...)` 调用
- [ ] 运行 grep，确认重复定义只剩 1 处

### 4.4 验证命令

```bash
rg -n "class DecodingDataset" src/models
```

```bash
rg -n "def _process_internal_node" src/models/dp_runner.py
```

```bash
python -m pytest \
  tests/test_dp_runner.py \
  tests/test_tour_reconstruct.py \
  tests/test_decode_tour.py \
  tests/test_exact_decode.py -q
```

### 4.5 完成标准

- [ ] `class DecodingDataset` 只保留 1 份
- [ ] `dp_runner.py` 中不再保留旧 internal-node 路径
- [ ] 相关测试通过

## 5. PR1：CLI / runtime 公共层

### 5.1 目标

把重复的 CLI helper 和模型恢复逻辑收口到公共模块。

### 5.2 主要文件

- `src/cli/common.py`（新增）
- `src/cli/model_factory.py`（新增）
- `src/cli/train.py`
- `src/cli/train_onepass.py`
- `src/cli/eval_and_vis.py`
- `src/cli/evaluate_tsplib.py`
- `src/cli/eval_onepass.py`

### 5.3 Checklist

- [ ] 新建 `src/cli/common.py`
- [ ] 在 `src/cli/common.py` 中放入 `set_seed`
- [ ] 在 `src/cli/common.py` 中放入 `resolve_device`
- [ ] 在 `src/cli/common.py` 中放入 `parse_bool_arg`
- [ ] 在 `src/cli/common.py` 中放入 `move_data_tensors_to_device`
- [ ] 在 `src/cli/common.py` 中封装数据加载与简单日志 helper
- [ ] 新建 `src/cli/model_factory.py`
- [ ] 在 `src/cli/model_factory.py` 中封装 checkpoint 加载
- [ ] 在 `src/cli/model_factory.py` 中封装 `d_model` / `matching_max_used` 等推断逻辑
- [ ] 在 `src/cli/model_factory.py` 中提供 2-pass 模型恢复入口
- [ ] 在 `src/cli/model_factory.py` 中提供 1-pass 模型恢复入口
- [ ] 更新 `src/cli/train.py` 使用公共 helper
- [ ] 更新 `src/cli/train_onepass.py` 使用公共 helper
- [ ] 更新 `src/cli/eval_and_vis.py` 使用公共 helper
- [ ] 更新 `src/cli/evaluate_tsplib.py` 使用公共 helper
- [ ] 审查 `src/cli/eval_onepass.py`，能收口的 helper 一并迁移
- [ ] 保证 CLI 参数名、默认值和 `--help` 文案不发生意外变化

### 5.4 验证命令

```bash
python src/cli/train.py --help >/tmp/nntsp_train.help.new
python src/cli/train_onepass.py --help >/tmp/nntsp_train_onepass.help.new
python src/cli/eval_and_vis.py --help >/tmp/nntsp_eval_and_vis.help.new
python src/cli/evaluate_tsplib.py --help >/tmp/nntsp_evaluate_tsplib.help.new
```

```bash
python -m compileall src/cli
```

```bash
rg -n "def set_seed|def resolve_device|def parse_bool_arg|def move_data_tensors_to_device" src/cli
```

### 5.5 完成标准

- [ ] CLI helper 在公共层只有 1 份定义
- [ ] 相关脚本仍可正常打印 `--help`
- [ ] 1-pass / 2-pass 模型恢复逻辑不再分别复制大段代码

## 6. PR2：1-pass 复用共享 bottom-up encode

### 6.1 目标

让 1-pass 的训练和推理先复用统一的 bottom-up 编码结果，再做自己的上层逻辑。

### 6.2 主要文件

- `src/models/bottom_up_runner.py`
- `src/models/onepass_trainer.py`
- `src/models/dp_runner.py`

### 6.3 Checklist

- [ ] 明确 `BottomUpResult` 里哪些字段可直接复用给 1-pass
- [ ] 如有必要，为 `BottomUpTreeRunner` 增加适合 1-pass 的只编码入口
- [ ] 保持 `BottomUpTreeRunner` 现有对外接口兼容
- [ ] 在 `src/models/onepass_trainer.py` 中移除手写 bottom-up traversal 主循环
- [ ] 让 `src/models/onepass_trainer.py` 先拿到共享 `z`，再执行 teacher-sigma decode
- [ ] 在 `src/models/onepass_trainer.py` 中保留 `decode_mask` / `child_scores` 行为不变
- [ ] 在 `src/models/dp_runner.py` 中把“节点编码”和“DP 求解”逻辑拆开
- [ ] 让 `src/models/dp_runner.py` 先消费共享 `z`
- [ ] 确保 `dp_runner.py` 中对子节点 latent 的 gather 语义与旧实现一致
- [ ] 核查 `dtype`、`device`、`computed mask` 的行为不发生回归
- [ ] 删除迁移后不再需要的本地 leaf-row / gather 重复逻辑

### 6.4 验证命令

```bash
python -m pytest \
  tests/test_bottom_up_process.py \
  tests/test_onepass_trainer.py \
  tests/test_dp_runner.py -q
```

```bash
python -m compileall src/models
```

### 6.5 完成标准

- [ ] 1-pass 不再维护第二套完整 bottom-up 遍历
- [ ] `onepass_trainer.py` 和 `dp_runner.py` 的编码阶段都复用共享骨架
- [ ] 相关单测通过

## 7. PR3：共享 tree helper

### 7.1 目标

把分散在多个文件里的树结构辅助函数收口成一个小模块。

### 7.2 主要文件

- `src/models/shared_tree.py`（新增）
- `src/models/bottom_up_runner.py`
- `src/models/onepass_trainer.py`
- `src/models/tour_reconstruct.py`
- `src/models/dp_runner.py`

### 7.3 Checklist

- [ ] 新建 `src/models/shared_tree.py`
- [ ] 迁移 `_build_leaf_row_for_node`
- [ ] 迁移 `gather_node_fields`
- [ ] 迁移 `_extract_z`
- [ ] 如果 `dp_runner.py` 有相同取数字典逻辑，也一并接到共享 helper
- [ ] 删除各文件中的私有重复实现
- [ ] 确认 helper 的输入输出命名不引入二义性
- [ ] 给新 helper 补最少量注释，说明它服务于哪些 runner

### 7.4 验证命令

```bash
rg -n "def _build_leaf_row_for_node|def gather_node_fields|def _extract_z" src/models
```

```bash
python -m pytest \
  tests/test_bottom_up_process.py \
  tests/test_onepass_trainer.py \
  tests/test_dp_runner.py \
  tests/test_tour_reconstruct.py -q
```

### 7.5 完成标准

- [ ] tree helper 只保留共享实现
- [ ] 相关 runner / reconstruct 行为不变

## 8. PR4：共享 attention / parent memory 基建

### 8.1 目标

把 `TopDownDecoder` 和 `MergeDecoder` 里重复的 attention 与 memory build 逻辑下沉。

### 8.2 主要文件

- `src/models/shared_attention.py`（新增）
- `src/models/parent_memory.py`（新增）
- `src/models/top_down_decoder.py`
- `src/models/merge_decoder.py`

### 8.3 Checklist

- [ ] 新建 `src/models/shared_attention.py`
- [ ] 抽出公共 `CrossAttentionBlock`
- [ ] 新建 `src/models/parent_memory.py`
- [ ] 抽出 parent memory build 所需 tokenization + self-attn 骨架
- [ ] 让 `TopDownDecoder` 调用共享 parent memory builder
- [ ] 让 `MergeDecoder` 调用共享 parent memory builder
- [ ] 尽量保持已有子模块命名，减少 checkpoint key 漂移
- [ ] 如果 state dict key 发生变化，增加兼容加载逻辑或映射层
- [ ] 保留各自 decoder 的 query / output head 差异，不做过度统一

### 8.4 验证命令

```bash
rg -n "class CrossAttentionBlock" src/models
```

```bash
python -m pytest \
  tests/test_top_down_process.py \
  tests/test_merge_decoder.py -q
```

### 8.5 完成标准

- [ ] `CrossAttentionBlock` 只保留共享实现
- [ ] 两个 decoder 都复用同一套 memory 基建
- [ ] 现有测试通过

## 9. PR5：合并 encoder 公共 backbone

### 9.1 目标

让 `LeafEncoder` / `MergeEncoder` 共享除输入 token 差异以外的大部分骨架。

### 9.2 主要文件

- `src/models/base_node_encoder.py`（新增）
- `src/models/leaf_encoder.py`
- `src/models/merge_encoder.py`

### 9.3 Checklist

- [ ] 新建 `src/models/base_node_encoder.py`
- [ ] 抽出 `NodeTokenizer -> node_ctx -> type_embed -> SetTransformer -> CLS pool` 主骨架
- [ ] 保留 `LeafEncoder` 自己的 leaf point token 生成逻辑
- [ ] 保留 `MergeEncoder` 自己的 child token 接入逻辑
- [ ] 核对 `forward` 签名，避免破坏上层调用
- [ ] 对比重构前后 `state_dict().keys()`，避免 checkpoint 不兼容
- [ ] 如果 `return_aux` 没有真实消费方，标记为 deprecated 或迁移到 debug 路径
- [ ] 确认 `d_model`、mask 形状和 CLS pool 语义没有变化

### 9.4 验证命令

```bash
python -m pytest \
  tests/test_bottom_up_process.py \
  tests/test_onepass_trainer.py \
  tests/test_dp_runner.py -q
```

```bash
python -m compileall src/models
```

### 9.5 完成标准

- [ ] `leaf_encoder.py` / `merge_encoder.py` 不再复制大段骨架代码
- [ ] checkpoint 兼容性有明确保证

## 10. PR6：拆分 `dp_core.py`

### 10.1 目标

把当前巨石文件拆成职责明确的多个模块，但暂时保留 `dp_core.py` 作为兼容门面。

### 10.2 主要文件

- `src/models/dp_core.py`
- `src/models/dp_correspondence.py`（新增）
- `src/models/dp_verify.py`（新增）
- `src/models/dp_parse_heuristic.py`（新增）
- `src/models/dp_parse_catalog.py`（新增）
- `src/models/dp_leaf_solver.py`（新增）

### 10.3 Checklist

- [ ] 新建 `src/models/dp_correspondence.py`
- [ ] 迁移 `CorrespondenceMaps`
- [ ] 迁移 `build_correspondence_maps`
- [ ] 迁移 `propagate_c1_constraints`
- [ ] 新建 `src/models/dp_verify.py`
- [ ] 迁移 `verify_tuple`
- [ ] 迁移 `batch_check_c1c2`
- [ ] 新建 `src/models/dp_parse_heuristic.py`
- [ ] 迁移 `parse_continuous`
- [ ] 迁移 `parse_continuous_topk`
- [ ] 迁移 `parse_activation_batch`
- [ ] 新建 `src/models/dp_parse_catalog.py`
- [ ] 迁移 `parse_by_catalog_enum`
- [ ] 迁移 `_rank_child_catalog_states_for_parse`
- [ ] 新建 `src/models/dp_leaf_solver.py`
- [ ] 迁移 Held-Karp / NN / `leaf_exact_solve` 相关逻辑
- [ ] 让 `src/models/dp_core.py` 暂时只做兼容 re-export
- [ ] 保持现有 `from src.models.dp_core import ...` 仍然可用
- [ ] 补齐新模块的 `__all__`

### 10.4 验证命令

```bash
python -m pytest \
  tests/test_dp_core.py \
  tests/test_dp_runner.py \
  tests/test_child_catalog_cap.py \
  tests/test_sigma_cap.py -q
```

```bash
python -m compileall src/models
```

### 10.5 完成标准

- [ ] `dp_core.py` 不再承载主要实现细节
- [ ] 外部 import 暂时不需要跟着改
- [ ] DP 核心测试通过

## 11. PR7：瘦身 `dp_runner.py`

### 11.1 目标

把 `dp_runner.py` 收敛为 orchestration 层，细节逻辑拆到独立模块。

### 11.2 主要文件

- `src/models/dp_runner.py`
- `src/models/dp_types.py`（新增）
- `src/models/dp_traceback.py`（新增）
- `src/models/dp_stats.py`（新增）
- `src/models/dp_fallback.py`（新增）

### 11.3 Checklist

- [ ] 抽出结果 dataclass 和公共类型定义到 `dp_types.py`
- [ ] 抽出 traceback 逻辑到 `dp_traceback.py`
- [ ] 抽出深度统计与 fallback 统计到 `dp_stats.py`
- [ ] 抽出 exact fallback 相关流程到 `dp_fallback.py`
- [ ] 删除已经确认废弃的旧 internal-node 路径
- [ ] 让 `dp_runner.py` 聚焦于“遍历节点 + 调用策略 + 组装结果”
- [ ] 将 `parse_mode` 分支逐步改为策略对象或小函数映射
- [ ] 保持 `OnePassDPRunner` 的对外构造参数兼容

### 11.4 验证命令

```bash
python -m pytest \
  tests/test_dp_runner.py \
  tests/test_depth_fallback_stats.py \
  tests/test_child_catalog_cap.py \
  tests/test_sigma_cap.py \
  tests/test_tour_reconstruct.py -q
```

### 11.5 完成标准

- [ ] `dp_runner.py` 主文件显著变短
- [ ] stats / traceback / fallback 不再混在同一文件里
- [ ] 所有 1-pass runner 相关测试通过

## 12. PR8：decode / reconstruct 收口

### 12.1 目标

把 1-pass 主 reconstruction 路径和 legacy 兼容路径分开，减少主文件认知负担。

### 12.2 主要文件

- `src/models/tour_reconstruct.py`
- `src/models/tour_reconstruct_legacy.py`（新增）
- `src/models/decode_backend.py`
- `src/models/edge_decode.py`

### 12.3 Checklist

- [ ] 明确 `tour_reconstruct.py` 中 direct reconstruction 是主路径
- [ ] 把 `dp_result_to_logits` 迁移到 `tour_reconstruct_legacy.py`
- [ ] 把 `dp_result_to_edge_scores` 迁移到 `tour_reconstruct_legacy.py`
- [ ] 在主路径调用点中只保留 direct reconstruction
- [ ] 兼容 import 仍可使用 legacy 路径
- [ ] 删除 `edge_decode.py` 中重复的 dataset / decode 辅助实现
- [ ] 确认 `decode_backend.py` 成为唯一 dataset 入口
- [ ] 清理无消费的中间调用和重复注释

### 12.4 验证命令

```bash
rg -n "class DecodingDataset" src/models
```

```bash
python -m pytest \
  tests/test_decode_tour.py \
  tests/test_exact_decode.py \
  tests/test_tour_reconstruct.py -q
```

### 12.5 完成标准

- [ ] 主 reconstruction 文件只保留主路径
- [ ] legacy 逻辑被显式隔离
- [ ] decode 侧重复类只剩 1 份

## 13. PR9：拆分 `labeler.py` / `node_token_packer.py`

### 13.1 目标

最后处理高耦合核心，把复杂实现拆开，但对外 facade 保持稳定。

### 13.2 主要文件

- `src/models/labeler.py`
- `src/models/teacher_solver.py`（新增）
- `src/models/edge_projection.py`（新增）
- `src/models/matching_targets.py`（新增）
- `src/models/batch_labeler.py`（新增）
- `src/models/node_token_packer.py`
- `src/models/pack_contracts.py`（新增）
- `src/models/pack_single.py`（新增）
- `src/models/pack_batch.py`（新增）

### 13.3 Checklist

- [ ] 为 `PseudoLabeler` 拆出 teacher solve 逻辑
- [ ] 为 `PseudoLabeler` 拆出 edge projection 逻辑
- [ ] 为 `PseudoLabeler` 拆出 matching target 构造逻辑
- [ ] 为 `PseudoLabeler` 拆出 batch labeling 流程
- [ ] 让 `labeler.py` 保留 facade 层和兼容 import
- [ ] 为 `NodeTokenPacker` 拆出 contract / validation 逻辑
- [ ] 为 `NodeTokenPacker` 拆出 single-graph pack 逻辑
- [ ] 为 `NodeTokenPacker` 拆出 batch pack 逻辑
- [ ] 让 `node_token_packer.py` 保留 facade 层和兼容 import
- [ ] 确认外部训练与评测入口无需感知内部拆分

### 13.4 验证命令

```bash
python -m pytest tests/test_node_token_packer.py -q
```

```bash
python -m compileall src/models
```

如果本地有可用的小样本数据，再额外做一次训练 smoke：

```bash
python tests/test_train_step.py \
  --data_pt data/N50/train_r_light_pyramid.pt \
  --device cpu \
  --batch_size 2
```

### 13.5 完成标准

- [ ] `labeler.py` 与 `node_token_packer.py` 不再是巨石文件
- [ ] 对外类名与 import 路径保持稳定
- [ ] 如果样本数据可用，训练入口至少完成一次 smoke 检查

## 14. 推荐执行顺序

如果优先追求低风险和稳定推进，建议顺序如下：

1. PR0
2. PR1
3. PR2
4. PR3
5. PR4
6. PR6
7. PR7
8. PR8
9. PR5
10. PR9

如果优先追求“尽快瘦身体感”，建议顺序如下：

1. PR0
2. PR1
3. PR2
4. PR6
5. PR8
6. 其余 PR

## 15. 第一批最值得马上开工的任务

如果只想先做一轮低风险精简，建议从以下清单开始：

- [ ] 做 PR0，先删死代码和重复 dataset
- [ ] 做 PR1，统一 CLI / runtime helper
- [ ] 做 PR2，让 1-pass 复用 bottom-up encode
- [ ] 做 PR6 的第一半，把 `dp_core.py` 先拆成 re-export 门面

这 4 步做完后，代码体积、重复度和理解成本都会明显下降，但算法行为基本不会被改写。
