# LKH-3 Integration Walkthrough

This document describes the implementation and usage of LKH-3 as a neural-guided and standalone solver for TSP post-processing.

## 1. Motivation
While the neural network provides fast edge predictions, a greedy decoder often misses global connectivity or makes local mistakes. LKH-3 is a state-of-the-art TSP heuristic that can:
1.  **Refine Predictions**: Use model logits as weights to find the best possible tour according to the neural network (Neural-Guided LKH).
2.  **Provide a Strong Baseline**: Compare model-guided results with pure Euclidean LKH results to quantify the "intelligence" of the network.

## 2. Implementation Overview

### A. LKH Solver Utility (`src/utils/lkh_solver.py`)
A lightweight wrapper that handles:
-   **File Generation**: Converts Python data to `.tsp` (TSPLIB) and `.par` formats.
-   **Explicit Matrix Support**: Maps edge logits to integer costs for guided search.
-   **Process Execution**: Manages the `LKH` subprocess with timeouts.

### B. Parallel Decoding (`src/models/lkh_decode.py`)
Uses `torch.utils.data.DataLoader` to achieve task-level parallelism. It launches multiple LKH processes concurrently, each in a unique temporary directory to avoid race conditions.

### C. Multi-Baseline Comparison (`eval_and_vis.py`)
The evaluation script now compares three distinct baselines for a given sample:
-   **B1: Greedy + Patching**: The baseline heuristic currently in the repo.
-   **B2: LKH-Guided**: LKH-3 searching on the space defined by model-predicted costs.
-   **B3: LKH-Pure**: LKH-3 solving the raw Euclidean TSP instance.

## 3. Usage via Makefile

We have provided a dedicated target in the `Makefile` for this evaluation:

```bash
# 1. Default behavior (LKH path set in Makefile, CKPT defaults to best)
make eval_lkh

# 2. Run for a specific sample index
make eval_lkh TEST_IDX=5

# 3. Custom LKH path or specific checkpoint
make eval_lkh LKH_EXE=/path/to/LKH CKPT=checkpoints/ckpt_final.pt
```

> [!NOTE]
> In the `Makefile`, the `CKPT` variable now has target-specific defaults:
> - `train`: Defaults to `None` (starts a new training session).
> - `eval`, `vis_pred`, `eval_lkh`: Defaults to `checkpoints/ckpt_best.pt`.

### Outputs
-   **Console Logs**: Detailed lengths and Gaps vs Teacher for all three baselines.
-   **Visualizations**: A plot saved in `outputs/eval_lkh/vis_sample_X.png` showing:
    -   `Red`: Greedy Tour (B1)
    -   `Blue`: LKH-Guided Tour (B2)
    -   `Green`: LKH-Pure Tour (B3)
    -   `Dashed Gray`: Teacher Tour

## 4. Technical Detail: Logit to Cost Mapping
To guide LKH-3 with neural predictions, we convert logits into a cost matrix:
1.  **Inverse Relation**: Higher logit (more likely edge) -> Lower cost.
2.  **Scaling**: $Cost = \text{round}(10^4 \times (MaxLogit - Logit))$.
3.  **Sparsity**: Edges outside the spanner are assigned a massive penalty ($10^8$) to force the solver to prioritize spanner edges.

---
> [!TIP]
> LKH-3 is a standalone executable. You must download and compile it from [http://vectors.uoa.gr/lkh/](http://vectors.uoa.gr/lkh/) before running the `eval_lkh` target.
