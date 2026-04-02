# rl5915: Pure-LKH / Spanner Control Experiments

实验目标：在 `rl5915` 上拆分分析 “限边本身” 与 “warm start” 对 LKH 的影响。

统一设置：

- instance: `rl5915`
- graph pipeline: `delaunay + prune`
- `RUNS = 1`
- LKH executable: `data/lkh/LKH-3.0.13/LKH`
- timeout: `1200s`

边集规模：

- full spanner edges: `17728`
- effective (pruned) edges: `10702`

结果汇总：

| Setting | Candidate Edge Set | Warm Start | Time (s) | Obj | Notes |
|---|---:|---:|---:|---:|---|
| Pure LKH | full Euclidean graph | none | 316.39 | 565621 | baseline |
| Pure LKH + full spanner | 17728 | none | 382.14 | 565709 | only restrict to full spanner |
| Pure LKH + effective spanner | 10702 | none | 1200.10 | N/A | timed out, no feasible tour |
| Pure LKH + effective spanner + same warm start | 10702 | guided greedy | 135.44 | 568349 | LKH time only |

Warm-start details for the last row:

- warm-start source: the same greedy decode used by `2-pass + guided LKH`
- warm-start obj: `619128`
- warm-start generation time: `2.28s`
- model load time: `0.31s`
- inference time: `0.61s`
- total extra pre-LKH overhead: about `3.20s`
- end-to-end time including warm-start generation: about `138.65s`

对应结果文件：

- `pure LKH`
  - `outputs/eval_tsplib_hierarchy_ablation/pure_lkh_rl5915_smoke.json`
- `pure LKH + full spanner`
  - `outputs/eval_tsplib_hierarchy_ablation/pure_lkh_rl5915_full_spanner_smoke.json`
- `pure LKH + effective spanner`
  - `outputs/eval_tsplib_hierarchy_ablation/pure_lkh_rl5915_spanner_only_smoke.json`
- `pure LKH + effective spanner + same warm start`
  - `outputs/eval_tsplib_hierarchy_ablation/pure_lkh_rl5915_effective_spanner_same_warm_start_smoke.json`

可以直接支持的结论：

1. 仅仅把 `pure LKH` 的候选图限制到 `full spanner`，不会造成灾难性退化；时间和质量都只小幅变差。
2. 真正有问题的是把搜索进一步限制到 `effective/pruned spanner` 且不给 warm start；这会让 LKH 在 `rl5915` 上 20 分钟内都出不了可行解。
3. 一旦在同一个 `effective spanner` 上提供与 `guided LKH` 同源的 warm start，LKH 会重新变得可用，而且时间显著下降到 `135s`，质量也回到接近 pure-LKH 的水平。
4. 因此，在 practical pipeline 里，`warm start` 不是边缘因素；对于被 prune 后的候选图，它是让 LKH 从“几乎不可用”回到“可稳定求解”的关键组成部分。
