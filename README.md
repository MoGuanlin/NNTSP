# NNTSP

NNTSP 对应论文 **Size-Independent Neural Dynamic Programming for Euclidean TSP with (1+epsilon)-Approximation Guarantee**。仓库当前维护两条可执行主线：

- `2-pass practical path`：`BottomUpTreeRunner -> TopDownTreeRunner -> edge aggregation -> greedy/exact/guided LKH`
- `1-pass DP path`：`OnePassDPRunner -> cost table/backpointer -> traceback -> tour reconstruction`

项目说明文档已在 `2026-04-04` 做过一次清理。当前应优先参考：

- 根目录 [README.md](/remote-home/MoGuanlin/NNTSP/README.md)
- [docs/README.md](/remote-home/MoGuanlin/NNTSP/docs/README.md)
- [docs/guides/INSTALL_LKH.md](/remote-home/MoGuanlin/NNTSP/docs/guides/INSTALL_LKH.md)
- [docs/design/onepass_dp_implementation.md](/remote-home/MoGuanlin/NNTSP/docs/design/onepass_dp_implementation.md)
- [docs/design/theory_gaps.md](/remote-home/MoGuanlin/NNTSP/docs/design/theory_gaps.md)
- [docs/paper/32503_Size_Independent_Neural_.pdf](/remote-home/MoGuanlin/NNTSP/docs/paper/32503_Size_Independent_Neural_.pdf)

`docs/archive/` 下的文档只保留历史背景，不应作为当前代码和命令的依据。

## 当前代码结构

```text
NNTSP/
├── src/
│   ├── cli/              # 稳定训练、评测、可视化入口
│   ├── experiments/      # 论文实验编排脚本
│   ├── dataprep/         # 数据加载与 fast dataset 格式
│   ├── graph/            # spanner / raw pyramid / r-light 剪枝
│   ├── models/           # one-pass / two-pass / decode / labeler
│   ├── utils/            # LKH、数据转换、辅助工具
│   └── visualization/    # 金字塔与预测可视化
├── src_baselines/        # 当前仓库直接维护的基线代码
├── tests/
├── docs/
│   ├── README.md         # 当前文档索引
│   ├── design/           # 当前设计说明
│   ├── guides/           # 使用指南
│   ├── paper/            # 论文 PDF
│   └── archive/          # 历史说明、rebuttal、重构工作记录
├── benchmarks/tsplib/
├── data/
├── checkpoints/
├── outputs/
└── third_party/
```

## 当前推荐工作流

### 1. 环境

推荐使用 `conda` 环境 `tsp`：

```bash
conda activate tsp
make deps
```

如果要跑 guided LKH 或 teacher 生成，还需要先安装 LKH-3，见 [docs/guides/INSTALL_LKH.md](/remote-home/MoGuanlin/NNTSP/docs/guides/INSTALL_LKH.md)。

### 2. 数据预处理

```bash
conda activate tsp
make data spanner raw rlight SIZES="20 50" R=4
```

常见产物：

- `train.pt / val.pt / test.pt`：原始点集
- `*_spanner.pt`：稀疏候选图
- `*_raw_pyramid.pt`：原始 quadtree / pyramid
- `*_r_light_pyramid.pt`：`r-light` 剪枝后的主训练输入
- `*.fast.pt`：合并缓存版数据

### 3. 训练 2-pass 主模型

建议优先使用模块命令，而不是历史 Makefile 包装参数。

```bash
conda activate tsp
python -m src.cli.train \
  --train_pt data/N50/train_r_light_pyramid.pt \
  --val_pt data/N50/val_r_light_pyramid.pt \
  --r 4 \
  --device cuda \
  --lkh_exe LKH \
  --epochs 20
```

补充说明：

- 当前 teacher 是同一 `spanner` 图上的 validated Hamiltonian cycle。
- teacher 相关参数使用 `--teacher_lkh_runs`、`--teacher_lkh_timeout`。
- 验证后处理使用 `--decode_backend {greedy,exact}` 控制。

### 4. 统一评测入口

Synthetic：

```bash
conda activate tsp
python -m src.cli.evaluate \
  --benchmark synthetic \
  --ckpt checkpoints/ckpt_best.pt \
  --r 4 \
  --device cuda \
  --synthetic_n 500 \
  --settings greedy,guided_lkh
```

TSPLIB：

```bash
conda activate tsp
python -m src.cli.evaluate \
  --benchmark tsplib \
  --ckpt checkpoints/ckpt_best.pt \
  --r 4 \
  --device cuda \
  --tsplib_set paper \
  --settings ours
```

查看当前支持的 setting 和 preset：

```bash
conda activate tsp
python -m src.cli.evaluate --benchmark synthetic --list_settings
python -m src.cli.evaluate --benchmark tsplib --list_settings
python -m src.cli.evaluate --list_tsplib_presets
```

### 5. 论文实验脚本

实验编排脚本已经统一移到 `src/experiments/`。例如：

```bash
conda activate tsp
python -m src.experiments.evaluate_tsplib_compare --help
```

不要再使用旧的 `src.cli.evaluate_tsplib_compare` 之类路径。

## 当前实现要点

### 数据与几何

- [src/graph/spanner.py](/remote-home/MoGuanlin/NNTSP/src/graph/spanner.py)：spanner 构图
- [src/graph/build_raw_pyramid.py](/remote-home/MoGuanlin/NNTSP/src/graph/build_raw_pyramid.py)：构建 raw pyramid
- [src/graph/prune_pyramid.py](/remote-home/MoGuanlin/NNTSP/src/graph/prune_pyramid.py)：执行 `r-light` 剪枝

### 2-pass

- [src/models/leaf_encoder.py](/remote-home/MoGuanlin/NNTSP/src/models/leaf_encoder.py)
- [src/models/merge_encoder.py](/remote-home/MoGuanlin/NNTSP/src/models/merge_encoder.py)
- [src/models/top_down_decoder.py](/remote-home/MoGuanlin/NNTSP/src/models/top_down_decoder.py)
- [src/models/edge_decode.py](/remote-home/MoGuanlin/NNTSP/src/models/edge_decode.py)
- [src/models/lkh_decode.py](/remote-home/MoGuanlin/NNTSP/src/models/lkh_decode.py)

### 1-pass

- [src/models/merge_decoder.py](/remote-home/MoGuanlin/NNTSP/src/models/merge_decoder.py)
- [src/models/dp_runner.py](/remote-home/MoGuanlin/NNTSP/src/models/dp_runner.py)
- [src/models/tour_reconstruct.py](/remote-home/MoGuanlin/NNTSP/src/models/tour_reconstruct.py)

### 标签与数据加载

- [src/models/labeler.py](/remote-home/MoGuanlin/NNTSP/src/models/labeler.py)
- [src/dataprep/dataset.py](/remote-home/MoGuanlin/NNTSP/src/dataprep/dataset.py)

## 文档约定

- `docs/` 根下只保留当前应优先参考的文档。
- `docs/archive/` 中的内容是历史记录、审稿材料、重构工作日志，只用于追溯背景。
- 如果文档和代码冲突，以代码、测试和 `--help` 为准。
- 如果要给大模型做仓库上下文，优先喂 `README.md`、[docs/README.md](/remote-home/MoGuanlin/NNTSP/docs/README.md) 和当前 `src/cli/* --help` 输出，不要先喂 `docs/archive/`。
