# TSPLIB Hierarchy Ablation on Paper First 5 Instances (Patched Alive Spanner)

实验入口：

```bash
python -m src.cli.evaluate_tsplib_hierarchy_ablation \
  --ckpt checkpoints/run_r4_20260325_034547/ckpt_best.pt \
  --instances rl5915,rl5934,pla7397,rl11849,usa13509 \
  --device cuda:0 \
  --lkh_exe data/lkh/LKH-3.0.13/LKH \
  --num_workers 1 \
  --guided_top_k 20 \
  --save_dir outputs/eval_tsplib_hierarchy_ablation \
  --run_tag paper_first5_hierarchy_patched
```

结果文件：

- `outputs/eval_tsplib_hierarchy_ablation/paper_first5_hierarchy_patched.csv`
- `outputs/eval_tsplib_hierarchy_ablation/paper_first5_hierarchy_patched.jsonl`
- `outputs/eval_tsplib_hierarchy_ablation/paper_first5_hierarchy_patched_summary.json`

说明：

- 这版 `spanner_uniform_lkh` 不再使用全量 `spanner_edge_index`，而是只使用 `prune_r_light_single` 之后的 `alive_edge_id / edge_alive_mask` 对应的 patched alive edges。
- 因此，这一版和 `guided_lkh` 在候选边图上是同口径的；区别只在于 `guided` 使用神经排序，`uniform` 使用统一权重。
- 下表时间使用 `end-to-end` 口径：`parse + spanner + quadtree + pruning + method-specific decode/LKH`。
- `Guided vs Uniform` 两列里，`Obj Improve` 为正表示 `guided` 更好；`Time Delta` 为正表示 `guided` 更慢。

| Instance | N | 2-Pass+Guided LKH Obj | E2E Time (s) | Patched Spanner+Uniform+LKH Obj | E2E Time (s) | Guided vs Uniform Obj Improve (%) | Guided vs Uniform Time Delta (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| rl5915 | 5915 | 582162 | 6.64 | 671689 | 2.14 | 13.33 | 210.43 |
| rl5934 | 5934 | 569738 | 5.91 | 635808 | 2.21 | 10.39 | 167.93 |
| pla7397 | 7397 | 24071243 | 10.24 | 26150239 | 2.69 | 7.95 | 280.55 |
| rl11849 | 11849 | 939024 | 27.34 | 1065420 | 4.94 | 11.86 | 453.21 |
| usa13509 | 13509 | 20227152 | 35.59 | 22550399 | 6.02 | 10.30 | 491.23 |
| **Average** | **-** | **-** | **17.15** | **-** | **3.60** | **10.77** | **320.67** |

结论：

- 前一版 baseline 的主要问题是错误地让 `spanner_uniform` 使用了全量 spanner 边，因此它比 `guided` 多看到了很多被 r-light patching 删除的边。
- 修正后，`guided_lkh` 在前 5 个 paper 实例上全部优于 patched-uniform baseline，平均 `Obj Improve = 10.77%`。
- 代价是 runtime 明显更高，平均端到端时间约为 patched-uniform 的 `4.77x`。
