# Rebuttal 实验计划

更新日期：2026-03-30

这份文档用于把当前论文、代码实现、审稿意见和可执行实验项对齐，作为后续 rebuttal 补实验的工作清单。

## 1. 当前论文与代码的对应关系

当前仓库里实际上有两条主线：

1. `two-pass practical variant`
   - 论文 Section 3.3 / Appendix A.4 对应的工程版。
   - 流程是 `BottomUpTreeRunner -> TopDownTreeRunner -> edge aggregation -> greedy / exact / guided LKH`。
   - 对应脚本：
     - `src/cli/train.py`
     - `src/cli/eval_and_vis.py`
     - `src/cli/evaluate_tsplib.py`
     - `src/cli/eval_twopass_timing.py`
     - `src/cli/eval_training_cost.py`
   - 这条线是论文大规模结果的主来源。
   - 这条线在论文里已经明确写成 heuristic practical acceleration，不带 end-to-end guarantee。

2. `one-pass certified-DP-style variant`
   - 论文 Section 3.1 / Figure 2 的离散 DP merge 主线的实现原型。
   - 流程是 `OnePassDPRunner -> cost tables/backpointers -> traceback -> tour reconstruction`。
   - 对应脚本：
     - `src/cli/train_onepass.py`
     - `src/cli/eval_onepass.py`
     - `src/cli/evaluate_tsplib_compare.py`
     - `src/cli/evaluate_tsplib_onepass_models.py`
   - 对应核心模块：
     - `src/models/dp_runner.py`
     - `src/models/merge_decoder.py`
     - `src/models/dp_core.py`
     - `src/models/dp_fallback.py`

这意味着 reviewer 关心的“certified 版本是否真的实现”和“practical 版本到底测了什么”是可以分开回答的，不需要混成一件事。

## 2. 审稿问题分型

### 2.1 主要靠实验补强

- Reviewer qTM9 Q1
  - 训练成本：训练时长、teacher 生成成本、与推理加速的对比。
- Reviewer si9p Q1
  - certified one-pass 是否实现、能跑到多大、fallback 频率是多少。
- Reviewer si9p Q4
  - practical 部署里实际用的 spanner 类型、patching / pruning 细节、`|E|` 和 degree 统计。
- Reviewer si9p Weakness
  - 预处理和推理各阶段时间占比。
- Reviewer si9p Weakness
  - hierarchy 的独立贡献，需要和 `spanner_uniform_lkh`、`pure_lkh`、`guided_lkh` 做消融。
- Reviewer x5Nh Q3
  - 如果时间允许，可做 `train on n=100` 的扩展泛化实验。

### 2.2 主要靠文字澄清

- Reviewer H8Zv
  - `delta = epsilon / n` 的动机、Remark 3.4 的解释、Lemma 3.2 的 error 定义、Lines 349-351、training/inference connection、dataset size。
- Reviewer qTM9 Q2
  - 对其他 DP 问题的可推广性。
- Reviewer x5Nh Q1/Q2/Q4
  - 相关工作差异、CVRP 可扩展性、轻量 attention 模块解释。
- Reviewer si9p Q2/Q3
  - BCE 训练目标和 `delta` 假设的关系、`error <= O(epsilon)L` 到乘法近似的正式说明。

### 2.3 需要“诚实澄清 + 轻量实验”，不适合硬证明

- Reviewer si9p Q2
  - 当前训练目标是 masked BCE，不存在一个已经写出来的“BCE -> entrywise delta-accuracy”严格证明。
  - 更合理的 rebuttal 策略是：
    - 明确承认 theorem 使用的是 sufficient worst-case assumption；
    - practical 训练目标是 engineering surrogate；
    - 补 empirical proxy，而不是暗示已有严格桥接证明。

## 3. 已有脚本与 reviewer concern 的映射

### 3.1 训练成本

现成脚本：

- `src/cli/eval_training_cost.py`
- 统一入口：`src/cli/evaluate.py --benchmark training_cost`

已知仓库里已有 smoke 输出：

- `outputs/eval_training_cost_smoke/smoke_summary.json`

可直接回答：

- 模型参数量
- train log 解析出的 wall-clock training time
- GPU-hours
- teacher generation / dataset preprocessing 的 sample-based 投影时间
- 收敛曲线

建议优先级：最高。

### 3.2 two-pass 端到端时间拆解

现成脚本：

- `src/cli/eval_twopass_timing.py`
- 统一入口：`src/cli/evaluate.py --benchmark twopass_timing`

已知仓库里已有 smoke 输出：

- `outputs/eval_twopass_timing_smoke/..._summary.json`

可直接回答：

- spanner construction
- quadtree building
- patching / pruning
- neural inference
- score aggregation / warm start / candidate construction
- LKH search
- total wall-clock

建议优先级：最高。

### 3.3 hierarchy / guidance 的独立贡献

现成脚本：

- `src/cli/eval_and_vis.py`
- `src/cli/evaluate_tsplib.py`

现成 setting：

- `spanner_uniform_lkh`
- `guided_lkh`
- `pure_lkh`（synthetic）
- `paper_lkh`（TSPLIB paper table reference，不是新跑的 baseline）

可直接做的比较：

- `spanner_uniform_lkh` vs `guided_lkh`
  - 回答“只是 LKH 强，还是 hierarchy guidance 真有用”
- `guided_lkh` vs `pure_lkh`
  - synthetic 上可以直接跑真实 LKH baseline
- `greedy` vs `guided_lkh`
  - 说明 downstream LKH 的增益

建议优先级：高。

### 3.4 one-pass / certified-DP-style 实现与 fallback

现成脚本：

- `src/cli/eval_onepass.py`
- `src/cli/evaluate_tsplib_compare.py`
- `src/cli/evaluate_tsplib_onepass_models.py`

已知仓库里已有 one-pass 输出：

- `outputs/eval_onepass/.../results.jsonl`
- `outputs/eval_tsplib_compare/..._summary.json`

可直接回答：

- one-pass 代码确实已实现
- 每层 `num_sigma_total`
- `num_fallback`
- per-depth fallback rate
- one-pass 与 two-pass 的运行时间差距
- 当前 one-pass 的可行规模和质量

限制：

- reviewer 要的是“相比 exact merge enumeration 的 speedup”。
- 当前现成脚本更擅长报告 fallback 频率和总时间，不一定已经有完全对齐的 exact-merge baseline 表。
- 这一项可能需要补一个更聚焦的小脚本或补表。

建议优先级：高，但可能需要少量代码补充。

### 3.5 spanner / pruning 统计

当前现状：

- `evaluate_tsplib.py` 和 `eval_twopass_timing.py` 已经能拿到 `num_spanner_edges`、`num_alive_edges` 之类统计。
- 但 reviewer si9p Q4 想要更系统的 `|E|`、degree distribution、实例级统计表。

建议：

- 增补一个专门的统计脚本，输入 TSPLIB 实例列表，输出：
  - `N`
  - 原始 spanner 边数
  - pruning 后 alive 边数
  - 平均度 / 最大度 / degree percentiles
  - alive ratio

建议优先级：高。

## 4. 推荐补实验顺序

### 第一批：低风险、高收益、几乎不需要改代码

1. 训练成本统计
2. two-pass 时间拆解
3. `spanner_uniform_lkh` / `guided_lkh` / `pure_lkh` 消融
4. one-pass 小规模可运行性和 fallback 统计

这四项能直接回应 Reviewer qTM9 和 si9p 的大部分实验型质疑。

### 第二批：可能需要少量补代码

5. TSPLIB / synthetic 的 spanner 统计表
6. one-pass 相对 exact merge baseline 的局部对照
7. 资源口径说明表
   - CPU-only LKH
   - GPU+CPU 的 Ours+LKH
   - 补上更细的资源注释

### 第三批：高成本、视 rebuttal 时间决定

8. `train on n=100` 的扩展实验
9. 更大范围 TSPLIB 复验
10. BCE 与 DP surrogate 的额外 empirical proxy

## 5. 建议先写入 rebuttal 的核心口径

以下几条建议在写 rebuttal 时保持口径一致：

1. two-pass practical variant 和 one-pass certified-DP-style variant 明确区分。
2. practical 版的大规模实验主张的是 strong size generalization + heuristic acceleration，而不是端到端 guarantee。
3. theorem 中的 `delta = epsilon / n` 是 sufficient worst-case assumption，不要把 masked BCE 训练说成已经严格推出了这个条件。
4. 对 reviewer 质疑资源公平性时，不要硬辩；应补时间拆解，并明确 GPU/CPU 资源口径。
5. 对 reviewer 质疑 hierarchy 贡献时，优先用 `spanner_uniform_lkh` vs `guided_lkh` 的实证说话。

## 6. 当前最值得优先运行的命令

### 6.1 训练成本

```bash
source /remote-home/MoGuanlin/anaconda3/etc/profile.d/conda.sh
conda activate tsp
python -m src.cli.evaluate \
  --benchmark training_cost \
  --ckpt checkpoints/run_r4_20260325_034547/ckpt_best.pt \
  --teacher_data_pt data/N50/train_r_light_pyramid.pt \
  --output_dir outputs/rebuttal_training_cost
```

### 6.2 two-pass 时间拆解

```bash
source /remote-home/MoGuanlin/anaconda3/etc/profile.d/conda.sh
conda activate tsp
python -m src.cli.evaluate \
  --benchmark twopass_timing \
  --ckpt checkpoints/ckpt_best.pt \
  --synthetic_n 500 \
  --sample_idx 0 \
  --sample_idx_end 20 \
  --device cuda:0 \
  --output_dir outputs/rebuttal_twopass_timing_n500
```

### 6.3 hierarchy 贡献消融

```bash
source /remote-home/MoGuanlin/anaconda3/etc/profile.d/conda.sh
conda activate tsp
python -m src.cli.evaluate \
  --benchmark synthetic \
  --ckpt checkpoints/ckpt_best.pt \
  --synthetic_n 500 \
  --settings spanner_uniform_lkh,guided_lkh,pure_lkh \
  --sample_idx 0 \
  --sample_idx_end 100 \
  --device cuda:0 \
  --output_dir outputs/rebuttal_ablation_n500
```

### 6.4 one-pass vs two-pass 小规模对照

```bash
source /remote-home/MoGuanlin/anaconda3/etc/profile.d/conda.sh
conda activate tsp
python -m src.cli.evaluate_tsplib_compare \
  --onepass_ckpt checkpoints/onepass_r4_20260329_042311/ckpt_final_step_136300.pt \
  --twopass_ckpt checkpoints/run_r4_20260325_034547/ckpt_best.pt \
  --instance_preset n100_500 \
  --device cuda:0 \
  --output_dir outputs/rebuttal_onepass_vs_twopass
```

## 7. 下一步执行建议

如果按 rebuttal 性价比排序，建议从下面两项开始：

1. `training_cost`
2. `twopass_timing`

这两项最稳、最容易形成 rebuttal 表格，而且几乎不需要解释额外假设。
