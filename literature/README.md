# 文献笔记总览（literature）

本目录对四篇 ProteinMPNN 相关的论文进行全景分析，按「论文」组织，每篇论文拆解到五个子维度。

## 论文清单

| # | 论文 | 简称 | 年份/出处 | 核心贡献 |
|---|------|------|-----------|----------|
| P1 | Improving Protein Expression, Stability, and Function with ProteinMPNN (Sumida et al.) | **Sumida2024** | JACS 2024 | 用 ProteinMPNN 改造天然蛋白，提升表达/稳定性/功能（湿实验验证） |
| P2 | Improving Protein Sequence Design through Designability Preference Optimization (Xue et al.) | **ResiDPO / EnhancedMPNN** | arXiv 2506.00297 (preprint) | 残基级 DPO 对齐「可设计性」，设计成功率提升近 3 倍 |
| P3 | Property-Driven Protein Inverse Folding with Multi-Objective Preference Alignment (Hou et al.) | **ProtAlign / MoMPNN** | ICLR 2026 | 多目标偏好对齐，同时优化可设计性 + 溶解度 + 热稳定性 |
| P4 | Tuning ProteinMPNN to reduce protein visibility via MHC Class I through DPO (Gasser et al.) | **CAPE-MPNN** | PEDS 2025 | 用 DPO 调 ProteinMPNN 降低 MHC-I 免疫可见性（去免疫） |

## 主题主线

四篇论文共享一条清晰的线索：**以 ProteinMPNN（及其变体 LigandMPNN）为骨干逆折叠模型，用「偏好对齐 / 强化学习」技术（DPO 及其多目标、残基级扩展）把「序列恢复」这一传统训练目标，重新对齐到更贴近实际应用的目标上**。

- P1 是「零训练」的 baseline：不动模型权重，只靠固定功能位点 + 结构过滤来做稳定化。
- P2/P3/P4 是「训练式」的创新：用 DPO 微调模型，把目标从「恢复原生序列」改为「可设计性 / 可开发性 / 免疫隐形」。

分类关系：
- P1 → 属于 **baseline 思想**：ProteinMPNN 如何被 direct 应用（固定位点 + AF2 过滤）。
- P2、P3、P4 → 属于 **innovation**：三条 DPO 改进路线，可互相参照。
  - P2：**残基级解耦**（把 sentence-level DPO 拆成 residue-level 偏好项 + 约束项）。
  - P3：**多目标 + 自适应 margin + semi-online**（求解多个性质冲突）。
  - P4：**单一专用性质**（免疫可见性）+ **PWM 近似加速**（把昂贵的 netMHCpan 换成 PWM 打分用于训练）。

## 子目录说明

- `baseline/` — 各论文的全景流程管线（端到端 pipeline）。
- `innovation/` — 各论文 baseline 之上的创新方式与创新模块。
- `pattern/` — 跨论文提炼的规律、工作量分析。
- `tools/` — 各论文使用的外部工具及解决的问题。
- `phenomena/` — 常见实验现象与意外现象（含作者尝试解决的部分）。

## 快速跳转

- [baseline 总览](./baseline/README.md)
- [innovation 总览](./innovation/README.md)
- [pattern 规律与工作量](./pattern/README.md)
- [tools 外部工具](./tools/README.md)
- [phenomena 现象](./phenomena/README.md)
- [源码索引](../../source/README.md)