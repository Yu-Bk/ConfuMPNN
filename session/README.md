# session — 会话记录

本目录记录本次会话（Claude Code 对话）的概要，以及导出的文档说明。

## 会话概要（2026-08-15）

**任务**：对四篇 ProteinMPNN 相关论文做「全景分析」，并在 `E:\AIworkspace\project_try\文献\总结` 下建立结构化项目（literature + session + source）。

**输入论文**（用户上传 4 个 PDF）：

1. Improving Protein Expression, Stability, and Function with ProteinMPNN（JACS 2024）
2. Improving Protein Sequence Design through Designability Preference Optimization（arXiv 2506.00297）
3. Property-Driven Protein Inverse Folding with Multi-Objective Preference Alignment（ICLR 2026）
4. Tuning ProteinMPNN to reduce protein visibility via MHC Class I through DPO（PEDS 2025）

**执行流程**：

1. 读取 memory（用户画像：YuBK，计算药物发现，蛋白设计方向，要求模块拆解+公式+代码对照的深度技术笔记）。
2. 提取 4 个 PDF 全文，逐篇精读（含公式、超参数、benchmark、附录）。
3. 识别主题主线：**ProteinMPNN + DPO/偏好对齐**（P1=零训练 baseline；P2/P3/P4=DPO 微调的三条路线）。
4. 通过 GitHub MCP 核实各论文开源代码真实链接与状态。
5. 建立目录结构并撰写分层笔记。

**关键判断**：

- 四篇论文构成清晰谱系：P1（baseline，零训练固定位点+AF2 过滤）→ P2（残基级单目标 DPO）→ P3（多目标 semi-online DPO）→ P4（专用免疫目标 DPO）。
- 开源现状：P2 代码未释出（违反「开源优先」偏好，已标注），P3 仅 checkpoint，P4 最完整，P1 靠第三方复现。

## 产出文档清单

```
总结/
├── literature/
│   ├── README.md                      # 文献笔记总览 + 四篇论文清单 + 主题主线
│   ├── baseline/                      # 全景流程管线（end-to-end pipeline + 公式 + 过滤标准）
│   │   ├── README.md
│   │   ├── P1_Sumida2024_baseline.md
│   │   ├── P2_ResiDPO_baseline.md
│   │   ├── P3_ProtAlign_baseline.md
│   │   └── P4_CAPE-MPNN_baseline.md
│   ├── innovation/                    # 创新方式 + 创新模块
│   │   ├── README.md
│   │   ├── P1_Sumida2024_innovation.md
│   │   ├── P2_ResiDPO_innovation.md
│   │   ├── P3_ProtAlign_innovation.md
│   │   ├── P4_CAPE-MPNN_innovation.md
│   │   └── cross-paper_innovation.md  # 跨论文创新对比（演进谱系）
│   ├── pattern/README.md              # 规律提炼 + 工作量分析 + 可迁移组件
│   ├── tools/README.md                # 外部工具清单 + 解决问题
│   └── phenomena/README.md            # 常见现象 + 意外现象 + 遗留问题
├── session/
│   └── README.md                      # 本文件（会话概要）
└── source/
    └── README.md                      # 开源代码链接 + clone 命令 + GitHub 核实情况
```