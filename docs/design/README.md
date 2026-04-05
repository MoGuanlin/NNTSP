# 设计文档

当前目录只保留仍适合直接参考的设计说明。

## 当前文档

- [onepass_dp_implementation.md](/remote-home/MoGuanlin/NNTSP/docs/design/onepass_dp_implementation.md)
  作用：描述 1-pass DP 当前实现、主要模块和与论文图示的对应关系。

- [onepass_decoder_enum_blueprint.md](/remote-home/MoGuanlin/NNTSP/docs/design/onepass_decoder_enum_blueprint.md)
  作用：描述当前推荐的 1-pass 精简改造方案：保留 exact enum parse，不上 tuple reranker，先把 decoder 升级成 `iface + mate` proposal scorer，并配 widening / bound-check。

- [onepass_direct_structured_widening_plan.md](/remote-home/MoGuanlin/NNTSP/docs/design/onepass_direct_structured_widening_plan.md)
  作用：描述 1-pass 主线彻底摆脱 `max_used` 的完整改造方案：训练改成 direct structured supervision，推理改成 full-semantics factorized widening + exact fallback。

- [theory_gaps.md](/remote-home/MoGuanlin/NNTSP/docs/design/theory_gaps.md)
  作用：说明当前工程实现与 Rao'98 PTAS 理论构件之间的差距。

## 已归档的设计文档

以下内容已经移到 [../archive/design/](/remote-home/MoGuanlin/NNTSP/docs/archive/design)：

- 工程瘦身计划 / checklist / 基线快照
- 阶段 2 运行时保护方案
- 旧的 1-pass refactor 设计稿

这些归档文档主要用于追溯重构过程，不应作为当前实现的入口说明。
