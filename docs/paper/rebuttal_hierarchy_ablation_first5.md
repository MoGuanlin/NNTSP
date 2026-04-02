# TSPLIB Hierarchy Ablation on Paper First 5 Instances

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
  --run_tag paper_first5_hierarchy
```

结果文件：

- `outputs/eval_tsplib_hierarchy_ablation/paper_first5_hierarchy.csv`
- `outputs/eval_tsplib_hierarchy_ablation/paper_first5_hierarchy.jsonl`
- `outputs/eval_tsplib_hierarchy_ablation/paper_first5_hierarchy_summary.json`

说明：

- `2-Pass+Guided LKH` 复用了主实验的 two-pass inference + guided top-k candidate + LKH 路径。
- `Spanner+Uniform+LKH` 不跑神经网络，只把 r-light spanner 作为 uniform candidate set 交给 LKH。
- 下表时间使用 `end-to-end` 口径：`parse + spanner + quadtree + pruning + method-specific decode/LKH`。
- `Guided vs Spanner` 两列里，正数表示 `guided` 更差。

| Instance | N | Paper Ref Obj | 2-Pass+Guided LKH Obj | E2E Time (s) | Spanner+Uniform+LKH Obj | E2E Time (s) | Guided vs Spanner Obj Delta (%) | Guided vs Spanner Time Delta (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rl5915 | 5915 | 572085 | 582162 | 6.53 | 568342 | 3.13 | 2.43 | 108.36 |
| rl5934 | 5934 | 559712 | 569738 | 5.84 | 558688 | 2.96 | 1.98 | 97.36 |
| pla7397 | 7397 | 23382264 | 24071243 | 10.16 | 23313157 | 3.64 | 3.25 | 179.05 |
| rl11849 | 11849 | 929001 | 939024 | 26.26 | 926815 | 7.76 | 1.32 | 238.54 |
| usa13509 | 13509 | 20133724 | 20227152 | 34.36 | 20025906 | 7.66 | 1.00 | 348.57 |
| **Average** | **-** | **-** | **9277863.8** | **16.63** | **9078581.6** | **5.03** | **2.00** | **194.37** |

补充口径：

- `Paper Ref Obj` 仅用于对齐 paper 主表中的实例集合与参考数字，不建议把它直接当成 rebuttal 主结论。
- 更关键的是同一实例上 `guided` 和 `spanner_uniform` 的直接对比：当前实现里，前 5 个实例上 `guided` 全部慢于且差于 `spanner_uniform`。
