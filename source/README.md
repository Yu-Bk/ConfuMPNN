# source — 开源代码与命令索引

本目录记录四篇论文的开源源码链接、可用的 clone/install 命令，以及 GitHub 实际情况核实结果（核实日期 2026-08-15）。

## 核实总览

| 论文 | 官方代码仓库 | 状态 |
|------|-------------|------|
| P1 Sumida2024 (JACS) | 无专属 repo；protocol 由 `Nicholas-Krasnow/sequence-design-guide` 复现 | ⚠️ 无原生仓库，有第三方复现 |
| P2 ResiDPO (EnhancedMPNN) | 未公开（论文称「将发布数据集」） | ❌ 代码未释出 |
| P3 ProtAlign (MoMPNN) | `Qivon7/MoMPNN` | ✅ 有（仅 checkpoint，兼容 ProteinMPNN 格式） |
| P4 CAPE-MPNN | `hcgasser/CAPE_MPNN` | ✅ 完整（含 Docker/训练/评估） |

## 详情与命令

### P1 Sumida2024

无作者官方仓库；方法完全基于公开工具。复现用第三方 protocol 仓库：

```bash
# 第三方复现（Krasnow et al. 2025，基于 Sumida 2024 protocol）
git clone https://github.com/Nicholas-Krasnow/sequence-design-guide.git
# 描述: Protocol to redesign enzyme sequences for stability and expression,
#       developed using ProteinMPNN (Dauparas 2022), based on Sumida 2024.

# 依赖的核心工具
git clone https://github.com/dauparas/ProteinMPNN.git   # 序列设计
# RoseTTAFold joint inpainting（原 repo：RFdiffusion / RoseTTAFold）
# AlphaFold2（结构预测 + 过滤）
```

**说明**：本论文的价值在于「方法 + 湿实验数据」，代码层面依赖 ProteinMPNN + RoseTTAFold + AF2 三大公开工具。

### P2 ResiDPO (EnhancedMPNN)

- **代码未公开**。论文正文贡献列出「将发布 large-scale 残基级 pLDDT 数据集」，但截至 2026-08-15，GitHub 无 EnhancedMPNN/ResiDPO 官方仓库（搜索 `EnhancedMPNN`、`ResiDPO`、`designability preference optimization protein`、作者 `Fanglei Xue` 均无果）。
- 基座模型：**LigandMPNN**（开源）。

```bash
git clone https://github.com/dauparas/LigandMPNN.git   # base model
```

### P3 ProtAlign (MoMPNN)

```bash
git clone https://github.com/Qivon7/MoMPNN.git
# 内容：ICLR 2026 官方仓库，含 mompnn_paper_checkpoints/ 全部模型变体。
# 关键特性：checkpoint 与 ProteinMPNN 格式完全兼容，
#           可直接用 LigandMPNN 的推理管线加载（无需改代码）。
```

- 推理：建议用 `dauparas/LigandMPNN` 的 README「Available models → ProteinMPNN」小节加载自定义 checkpoint。
- paper link: https://openreview.net/forum?id=m826DekCpp

### P4 CAPE-MPNN

```bash
export REPO=CAPE_MPNN
git clone https://github.com/hcgasser/${REPO}.git
# 完整代码 + Docker 环境 + 训练/评估脚本。
```

关键命令（摘自 README）：

```bash
# 1) 生成 PWM 近似预测器（训练加速，替代 netMHCpan）
MHC-I_rank_peptides.py \
  --output ${PF}/data/input/immuno/mhc_1/Mhc1PredictorPwm \
  --alleles "HLA-A*02:01+HLA-A*24:02+HLA-B*07:02+HLA-B*39:01+HLA-C*07:01+HLA-C*16:01" \
  --tasks rank+pwm+stats+agg --peptides_per_length 1000000

# 2) DPO 超参搜索（每次生成一个微调模型）
cape-mpnn.py --hyp ${PF}/configs/CAPE-MPNN/hyp/hyp_b69bb1.yaml --hyp_n 1

# 3) 评估（Docker 容器内 Jupyter）
CAPE-Eval/cape-eval_mpnn.ipynb
```

依赖：Linux + Docker + GPU；评估需 localcolabfold + DE-STRESS（含 Rosetta）。

## 依赖的工具仓库汇总（所有论文通用）

| 工具 | 仓库 |
|------|------|
| ProteinMPNN | https://github.com/dauparas/ProteinMPNN |
| LigandMPNN | https://github.com/dauparas/LigandMPNN |
| RFdiffusion | https://github.com/RosettaCommons/RFdiffusion |
| RFdiffusion2 | （RFdiffusion 内，Atom-level active site scaffolding） |
| ESM / ESMFold | https://github.com/facebookresearch/esm |
| TM-align | https://zhanggroup.org/TM-align/ |
| Protein-Sol | https://protein-sol.manchester.ac.uk/software |
| TemBERTure | https://github.com/ibmm-unibe-ch/TemBERTure |
| HyperMPNN | https://github.com/meilerlab/HyperMPNN |
| SolubleMPNN | https://github.com/dauparas/ProteinMPNN（soluble_model_weights） |
| InstructPLM | https://github.com/Eikor/InstructPLM |
| ProteinDPO | https://github.com/evo-design/protein-dpo |
| netMHCpan 4.1 | （外部生物信息工具，非 GitHub） |
| ColabFold | https://github.com/YoshitakaMo/localcolabfold |
| DE-STRESS | https://github.com/wells-wood-research/de-stress |
| dl_binder_design（IG） | https://github.com/nrbennet/dl_binder_design |

## 提醒（重要）

- P2（EnhancedMPNN）**代码与权重均未开源**，若复现只能自实现 ResiDPO 算法并基于 LigandMPNN 微调。这与用户「只考虑开源」的偏好冲突——需注意。
- P3（MoMPNN）开源了 checkpoint 但**未开源训练代码**（仓库仅 inference + 权重）。
- P4（CAPE-MPNN）是四篇中开源最完整的（训练 + 评估 + 环境 + 数据管线）。