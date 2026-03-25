# NNTSP

NNTSP 对应论文 **Size-Independent Neural Dynamic Programming for Euclidean TSP with (1+epsilon)-Approximation Guarantee**。这个仓库实现了一条完整的 Euclidean TSP 研究流水线：从点集生成、spanner 构建、quadtree/pyramid 预处理、`r-light` 剪枝，到神经动态规划模型训练、可视化、TSPLIB 评测，以及与 POMO、NeuroLKH、LKH-3 的对比。

项目的核心目标不是直接在全图上做黑盒神经求解，而是把计算几何先验和层级动态规划结合起来，让模型在输入规模变化时仍保持结构不变、模块可复用，并且能够与 LKH-3 这类强启发式求解器低成本集成。

## 论文摘要速览

- 论文研究的是 **Euclidean TSP**。
- 方法核心是一个 **size-independent neural DP**：问题规模变化只会影响模块复用次数，不改变每个局部模块的输入输出形态。
- 理论上保留了 Euclidean TSP 的 **`(1 + epsilon)` 近似保证**。
- 工程上加入了可落地的预处理与剪枝策略，并支持把神经预测接入 **LKH-3** 作为 guided solver。
- 论文摘要中给出的结果包括：模型只在 `n = 50` 上训练，也能泛化到 TSPLIB 中最高 `n = 85900` 的实例；接入 LKH-3 后可获得显著加速。

对应 PDF 已整理到 [docs/paper/32503_Size_Independent_Neural_.pdf](/remote-home/MoGuanlin/NNTSP/docs/paper/32503_Size_Independent_Neural_.pdf)。

## 仓库结构

```text
NNTSP/
├── README.md
├── Makefile
├── src/
│   ├── cli/              # 训练、评测、可视化、消融入口
│   ├── dataprep/         # 数据加载与 fast dataset 格式
│   ├── graph/            # spanner / raw pyramid / r-light 剪枝
│   ├── models/           # bottom-up / top-down / labeler / decode / metrics
│   ├── utils/            # 数据生成、LKH 工具、格式转换
│   └── visualization/    # 金字塔与预测结果可视化
├── src_baselines/        # 当前仓库内维护的基线训练代码
├── tests/                # 调试与结构正确性测试脚本
├── docs/
│   ├── guides/           # 使用指南（例如 LKH 安装）
│   ├── notes/            # 历史交接文档与实验记录
│   └── paper/            # 论文 PDF
├── benchmarks/
│   └── tsplib/           # TSPLIB 基准集
├── third_party/
│   └── baselines/        # 原始第三方基线代码快照
├── data/                 # 生成的数据集与 LKH 相关资源
├── checkpoints/          # 模型权重与训练日志
├── outputs/              # 评测图像与导出结果
├── logs/                 # 其他运行日志
├── artifacts/            # 样例数据等小型辅助产物
└── archive/              # 历史备份与旧实现归档
```

## 核心模块说明

### 1. 数据预处理

- [src/utils/data_generator.py](/remote-home/MoGuanlin/NNTSP/src/utils/data_generator.py)：生成二维 Euclidean 点集。
- [src/graph/spanner.py](/remote-home/MoGuanlin/NNTSP/src/graph/spanner.py)：基于 Delaunay triangulation 构建候选稀疏图。
- [src/graph/build_raw_pyramid.py](/remote-home/MoGuanlin/NNTSP/src/graph/build_raw_pyramid.py)：把点集和 spanner 变成 quadtree/pyramid 结构。
- [src/graph/prune_pyramid.py](/remote-home/MoGuanlin/NNTSP/src/graph/prune_pyramid.py)：执行 `r-light` 剪枝，得到规模无关的局部 token 上界。

### 2. 模型

- [src/models/node_token_packer.py](/remote-home/MoGuanlin/NNTSP/src/models/node_token_packer.py)：把树结构和候选边编码成固定布局的 token。
- [src/models/leaf_encoder.py](/remote-home/MoGuanlin/NNTSP/src/models/leaf_encoder.py) 与 [src/models/merge_encoder.py](/remote-home/MoGuanlin/NNTSP/src/models/merge_encoder.py)：bottom-up 编码。
- [src/models/top_down_decoder.py](/remote-home/MoGuanlin/NNTSP/src/models/top_down_decoder.py) 与 [src/models/top_down_runner.py](/remote-home/MoGuanlin/NNTSP/src/models/top_down_runner.py)：top-down 边界条件传播。
- [src/models/labeler.py](/remote-home/MoGuanlin/NNTSP/src/models/labeler.py)：教师标签生成，可用 2-opt 或 LKH-3。
- [src/models/edge_decode.py](/remote-home/MoGuanlin/NNTSP/src/models/edge_decode.py)、[src/models/exact_decode.py](/remote-home/MoGuanlin/NNTSP/src/models/exact_decode.py) 与 [src/models/lkh_decode.py](/remote-home/MoGuanlin/NNTSP/src/models/lkh_decode.py)：贪心解码、精确稀疏图解码和神经引导 LKH。

### 3. 数据加载

[src/dataprep/dataset.py](/remote-home/MoGuanlin/NNTSP/src/dataprep/dataset.py) 提供了 `smart_load_dataset`：

- 如果存在 `*.fast.pt`，优先加载合并后的高效格式。
- 如果只有原始 `List[Data]`，会自动整合并缓存成 `*.fast.pt`。
- `FastTSPDataset` 会在 `__getitem__` 时再物化单样本，从而减小 Python 对象开销。

### 4. CLI 入口

- [src/cli/train.py](/remote-home/MoGuanlin/NNTSP/src/cli/train.py)：训练与验证。
- [src/cli/eval_and_vis.py](/remote-home/MoGuanlin/NNTSP/src/cli/eval_and_vis.py)：样本级评测、对比和可视化。
- [src/cli/evaluate_tsplib.py](/remote-home/MoGuanlin/NNTSP/src/cli/evaluate_tsplib.py)：TSPLIB 大实例评测。
- [src/cli/ablation_r.py](/remote-home/MoGuanlin/NNTSP/src/cli/ablation_r.py)：不同 `r` 的消融训练。

## 数据与产物命名约定

在 `data/N20/`、`data/N50/` 这类目录下，常见文件的语义如下：

- `train.pt` / `val.pt` / `test.pt`：原始点集数据。
- `*_spanner.pt`：spanner 构图结果。
- `*_raw_pyramid.pt`：未剪枝的原始金字塔结构。
- `*_r_light_pyramid.pt` 或 `*_r4_light_pyramid.pt`：`r-light` 剪枝后的训练输入。
- `*.fast.pt`：合并缓存版，加载更快。
- `*.labeled_*.pt`：附带 teacher 标签的版本。

## 环境要求

推荐环境：

- `conda` 环境名：`tsp`
- Python 3.9+
- PyTorch 2.7.x
- `torch-geometric`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `tqdm`

进入环境后再运行仓库命令：

```bash
conda activate tsp
```

如果你要跑 guided LKH 或 TSPLIB 评测，还需要安装 **LKH-3**。安装说明见 [docs/guides/INSTALL_LKH.md](/remote-home/MoGuanlin/NNTSP/docs/guides/INSTALL_LKH.md)。

## 快速开始

### 1. 安装依赖

```bash
conda activate tsp
make deps
```

### 2. 生成训练数据

```bash
conda activate tsp
make data spanner raw rlight SIZES="20 50" R=4
```

### 3. 可视化一个样本

```bash
conda activate tsp
make vis SIZES="50" R=4
```

### 4. 训练主模型

```bash
conda activate tsp
make train TRAIN_N=50 R=4 DEVICE=cuda USE_LKH=1
```

等价的模块命令形式：

```bash
conda activate tsp
python -m src.cli.train \
  --train_pt data/N50/train_r_light_pyramid.pt \
  --val_pt data/N50/val_r_light_pyramid.pt \
  --r 4 \
  --device cuda \
  --decode_backend greedy \
  --epochs 20
```

如果你想在验证阶段直接把后处理切换为精确稀疏图解码：

```bash
conda activate tsp
python -m src.cli.train \
  --train_pt data/N50/train_r_light_pyramid.pt \
  --val_pt data/N50/val_r_light_pyramid.pt \
  --r 4 \
  --device cuda \
  --decode_backend exact \
  --exact_time_limit 30 \
  --exact_length_weight 0.0
```

这里的 `exact` 指的是“在当前 `spanner` 稀疏候选图上精确求解最大分数 Hamiltonian cycle”，不是完整 Euclidean 全图上的精确 TSP。

### 5. 验证或仅做标签预计算

```bash
conda activate tsp
make eval CKPT=checkpoints/ckpt_best.pt TRAIN_N=50 R=4
make labels TRAIN_N=50 R=4 USE_LKH=1
```

### 6. 生成可视化评测结果

```bash
conda activate tsp
make vis_pred CKPT=checkpoints/ckpt_best.pt TRAIN_N=50 R=4
make bench_eval CKPT=checkpoints/ckpt_best.pt BENCHMARK=synthetic SYNTHETIC_N=10000 R=4
make bench_eval CKPT=checkpoints/ckpt_best.pt BENCHMARK=synthetic SYNTHETIC_N=2000 R=4 SETTINGS=greedy,exact,guided_lkh
make bench_eval CKPT=checkpoints/run_r4_20260324_200302/ckpt_best.pt BENCHMARK=synthetic SYNTHETIC_N=500 R=4 SETTINGS=exact EXACT_TIME_LIMIT=3000
make bench_eval CKPT=checkpoints/run_r4_20260325_034547/ckpt_best.pt BENCHMARK=synthetic SYNTHETIC_N=2000 R=4 SETTINGS=exact EXACT_TIME_LIMIT=3000
```

### 7. TSPLIB 评测

```bash
conda activate tsp
make bench_eval CKPT=checkpoints/ckpt_best.pt BENCHMARK=tsplib NUM_INSTANCES=10 R=4
make bench_eval CKPT=checkpoints/ckpt_best.pt BENCHMARK=tsplib TSPLIB_SET=largest10 R=4 SETTINGS=ours
make bench_eval CKPT=checkpoints/run_r4_20260324_200302/ckpt_best.pt BENCHMARK=tsplib TSPLIB_SET=paper R=4 SETTINGS=exact EXACT_TIME_LIMIT=3000
make bench_eval CKPT=checkpoints/run_r4_20260325_034547/ckpt_best.pt BENCHMARK=tsplib TSPLIB_INSTANCES=rl5915,rl5934,pla7397 R=4 SETTINGS=exact EXACT_TIME_LIMIT=3000 RUN_TAG=matching_exact_subset
```

默认会读取 [benchmarks/tsplib](/remote-home/MoGuanlin/NNTSP/benchmarks/tsplib)。

统一评测入口现在是 `make bench_eval`：

- `BENCHMARK=synthetic`：使用人工标准数据集；用 `SYNTHETIC_N=500/2000/...` 指定规模，或用 `DATA_PT=...` 直接指定数据文件。
- `BENCHMARK=tsplib`：使用 TSPLIB；用 `TSPLIB_SET=largest10/paper/all/largest:25` 选择预设集合，或用 `TSPLIB_INSTANCES=rl5915,rl5934,...` 指定实例列表。

原来的 `make eval_lkh` 和 `make eval_tsplib` 仍然可用，但现在底层都走统一入口。

TSPLIB 评测现在会在每个实例结束后立刻追加保存结果，默认写到 `outputs/eval_tsplib/` 下的三份文件：

- `*.csv`：便于直接看表格或导入 Excel。
- `*.jsonl`：每行一个实例结果，适合中途中断后继续分析。
- `*_summary.json`：滚动更新的完整汇总快照。

如需自定义保存位置，可额外指定 `EVAL_OUTPUT_DIR=...` 或旧参数 `TSPLIB_SAVE_DIR=...`；如需固定文件前缀，可指定 `RUN_TAG=...` 或旧参数 `TSPLIB_RUN_TAG=...`。

`SETTINGS` 支持逗号分隔的 setting 列表，例如 `greedy,exact,guided_lkh`，也支持分组别名如 `ours`、`baselines`、`all`。不同 benchmark 支持的 setting 集合略有不同；如需查看完整支持列表与 TSPLIB preset，可运行：

```bash
python -m src.cli.evaluate --benchmark synthetic --list_settings
python -m src.cli.evaluate --benchmark tsplib --list_settings
python -m src.cli.evaluate --list_tsplib_presets
```

### 8. `r` 消融

```bash
conda activate tsp
python -m src.cli.ablation_r --gpu 0
python -m src.cli.ablation_r --r 2 4 6 8 --device cuda:0
python -m src.cli.ablation_r --r 2 4 6 8 --device cuda:0 --decode_backend exact --exact_time_limit 30
```

## 基线与第三方代码

仓库中有两类基线代码：

- [src_baselines/](/remote-home/MoGuanlin/NNTSP/src_baselines)：当前项目内直接调用和训练的基线实现，例如 POMO、NeuroLKH。
- [third_party/baselines/](/remote-home/MoGuanlin/NNTSP/third_party/baselines)：第三方原始仓库快照，用于对照、数据生成或追溯实现来源。

Makefile 中目前已经集成：

- `make train_POMO`
- `make train_neuraLKH`
- `make data_std`

其中 `make data_std` 会调用 `third_party/baselines/attention-learn-to-route/` 下的数据生成脚本。

## 目录整理原则

本次整理采用以下约定：

- `src/` 只放当前可执行、可维护的主代码。
- `docs/` 集中放论文、说明和交接材料。
- `tests/` 与主源码分离，避免测试脚本混入运行包。
- `third_party/` 单独存放外部项目快照。
- `archive/` 只保留历史备份，不再作为当前实现的一部分。

`docs/notes/` 中的历史文档仍然很有价值，但部分命令和路径可能是重构前的写法；当它们与当前目录结构冲突时，请以本 README 为准。

## 常用路径

- 论文 PDF：[docs/paper/32503_Size_Independent_Neural_.pdf](/remote-home/MoGuanlin/NNTSP/docs/paper/32503_Size_Independent_Neural_.pdf)
- LKH 安装指南：[docs/guides/INSTALL_LKH.md](/remote-home/MoGuanlin/NNTSP/docs/guides/INSTALL_LKH.md)
- 工程交接文档：[docs/notes/NNTSP_工程交接文档_新.md](/remote-home/MoGuanlin/NNTSP/docs/notes/NNTSP_工程交接文档_新.md)
- 理论交接文档：[docs/notes/NNTSP_理论算法交接文档_新.md](/remote-home/MoGuanlin/NNTSP/docs/notes/NNTSP_理论算法交接文档_新.md)

## 已知注意事项

- `data/` 和 `checkpoints/` 很大，属于实验产物目录，不建议直接纳入版本控制。
- `USE_LKH=1` 时需要显式提供可用的 `LKH` 可执行文件，或者确保它已在系统路径中。
- `FastTSPDataset` 依赖 Linux `fork` 的 Copy-On-Write 特性；大数据训练更适合在 Linux 环境中运行。
- `docs/notes/` 中有一些历史记录使用了旧路径或旧实验命名，这些文档保留用于追溯，不代表当前推荐入口。
