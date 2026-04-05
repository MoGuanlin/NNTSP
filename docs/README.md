# 文档索引

这份索引用来区分“当前可信文档”和“历史归档文档”。

## 当前可信文档

这些文档描述的是当前仓库应优先参考的实现和用法：

- [README.md](/remote-home/MoGuanlin/NNTSP/README.md)：项目总览、当前推荐命令、代码结构
- [guides/INSTALL_LKH.md](/remote-home/MoGuanlin/NNTSP/docs/guides/INSTALL_LKH.md)：LKH-3 安装
- [design/README.md](/remote-home/MoGuanlin/NNTSP/docs/design/README.md)：当前设计文档索引
- [design/onepass_dp_implementation.md](/remote-home/MoGuanlin/NNTSP/docs/design/onepass_dp_implementation.md)：1-pass DP 当前实现说明
- [design/onepass_decoder_enum_blueprint.md](/remote-home/MoGuanlin/NNTSP/docs/design/onepass_decoder_enum_blueprint.md)：当前推荐的 1-pass 精简改造方案与实现蓝图
- [design/onepass_direct_structured_widening_plan.md](/remote-home/MoGuanlin/NNTSP/docs/design/onepass_direct_structured_widening_plan.md)：1-pass 主线去 `max_used` 的完整改造方案
- [design/theory_gaps.md](/remote-home/MoGuanlin/NNTSP/docs/design/theory_gaps.md)：当前实现与理论 PTAS 的差距
- [paper/README.md](/remote-home/MoGuanlin/NNTSP/docs/paper/README.md)：论文 PDF 索引

## 当前应优先使用的代码入口

- 训练：`python -m src.cli.train`
- 统一评测：`python -m src.cli.evaluate`
- TSPLIB 专用评测：`python -m src.cli.evaluate_tsplib`
- 论文实验：`python -m src.experiments.<name>`

如果需要确认当前参数面，请直接运行对应脚本的 `--help`。

## 归档文档

[archive/README.md](/remote-home/MoGuanlin/NNTSP/docs/archive/README.md) 下的内容都属于历史材料，包括：

- 旧的重构计划、checklist、阶段性运行时保护记录
- rebuttal 实验计划和审稿材料
- 旧 walkthrough、交接说明和临时分析

这些文档可能包含：

- 已删除的 CLI 路径
- 已废弃的参数名
- 当时有效、现在已经不成立的工程判断

如果你是在做开发或给大模型提供仓库上下文，请默认忽略 `docs/archive/`，除非你明确需要历史背景。
