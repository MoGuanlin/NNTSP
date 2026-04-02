# Paper First 5: Pure LKH vs Pure+Full-Spanner vs 2-Pass Guided

说明：

- `Pure LKH` 时间是 solver-only。
- `Pure LKH + Full Spanner` 时间是 end-to-end：parse + hierarchy preprocessing + LKH。
- `2-Pass Guided` 时间是 end-to-end：parse + hierarchy preprocessing + NN inference + warm start + guided LKH。
- `Gap vs Pure` 仅在 `Pure LKH` 得到可行解时定义。

| Instance | N | Pure LKH Obj | Time (s) | Pure + Full Spanner Obj | E2E Time (s) | Gap vs Pure (%) | 2-Pass Guided Obj | E2E Time (s) | Gap vs Pure (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rl5915 | 5915 | 565621 | 314.97 | 565709 | 384.53 | 0.02 | 582162 | 6.64 | 2.92 |
| rl5934 | 5934 | 556475 | 441.60 | 556129 | 574.18 | -0.06 | 569738 | 5.91 | 2.38 |
| pla7397 | 7397 | timeout | 1200.14 (timeout) | timeout | 1202.23 (timeout) | N/A | 24071243 | 10.24 | N/A |
| rl11849 | 11849 | timeout | 1200.08 (timeout) | timeout | 1203.70 (timeout) | N/A | 939024 | 27.34 | N/A |
| usa13509 | 13509 | timeout | 1200.16 (timeout) | timeout | 1203.93 (timeout) | N/A | 20227152 | 35.59 | N/A |
| **Average (Common Feasible)** | **-** | **561048** | **378.28** | **560919** | **479.36** | **-0.02** | **575950** | **6.28** | **2.65** |

结果文件：

- `outputs/eval_tsplib_hierarchy_ablation/paper_first5_pure_lkh.csv`
- `outputs/eval_tsplib_hierarchy_ablation/paper_first5_pure_full_spanner.csv`
- `outputs/eval_tsplib_hierarchy_ablation/paper_first5_pure_fullspanner_vs_guided_vs_pure.csv`