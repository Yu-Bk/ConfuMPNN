# tools — 各论文使用的外部工具及解决问题

本目录记录每篇论文调用的**外部工具/模型/数据库**，以及它们各自解决什么问题。这是复现论文、搭 pipeline 的工具清单。

## 按论文分组

### P1 Sumida2024

| 工具 | 类型 | 解决问题 |
|------|------|----------|
| **ProteinMPNN** (dauparas/ProteinMPNN) | 逆折叠生成模型 | 根据 backbone 生成折叠该结构的序列 |
| **RoseTTAFold joint inpainting** | 骨架重塑 | 重塑低保守 loop 区骨架，进一步稳定结构 |
| **AlphaFold2**（单序列，无 MSA） | 结构预测 | 评估设计序列能否折叠回目标结构（pLDDT/Cα RMSD） |
| **UniRef30 / UniRef100** | 序列数据库 | TEV 家族保守位点识别；myoglobin 设计序列相似性分析 |
| **PROSS**（对照方法） | 传统稳定化工具 | 作为「进化信息+Rosetta 能量」路线的对照 |
| 湿实验：E. coli、IMAC、SEC、CD、UV/vis | 表征 | 表达/纯化/热稳定性/heme 功能表征 |
| 微秒 MD 模拟 | 动力学 | 解释远端突变为何提升活性（loop 刚性化） |

### P2 ResiDPO

| 工具 | 类型 | 解决问题 |
|------|------|----------|
| **LigandMPNN** | 基座逆折叠模型 | 作为 base model（因酶设计需配体上下文） |
| **AlphaFold2** | 结构预测 + 奖励信号 | 生成 per-residue pLDDT 标签；benchmark 判定设计成功 |
| **RFDiffusion2** | 骨架生成 | 生成酶活性位点骨架 benchmark + binder 骨架 |
| **RFDiffusion** | 骨架生成 | binder 设计 benchmark 的骨架生成 |
| **PDB** | 数据集 | PDB-D 训练集来源（X-ray <3.5Å 单体） |
| DPO（及 RobustDPO/RSO/KTO/NCA/SPPO 等变体） | 对齐算法 | 对照实验（Table 4） |

### P3 ProtAlign

| 工具 | 类型 | 解决问题 |
|------|------|----------|
| **ProteinMPNN** | 基座模型 | 作为 base backbone（最广泛、湿实验验证最多） |
| **ESMFold** (facebookresearch/esm) | 结构预测 | 计算 TM-score 判可设计性；pLDDT(B-factor) |
| **AlphaFold2 + Initial Guess** (dl_binder_design) | 结构预测 | pTM(IG) 作为可设计性信号（IG 比 TM 更关注折叠置信度） |
| **TM-align** (zhanggroup) | 结构比对 | 计算 TM-score |
| **ESM-2 (esm2_t33_650M_UR50D)** | 蛋白质语言模型 | 伪似然 Evolutionary Perplexity（EP，进化合理性） |
| **Protein-Sol** (manchester) | 溶解度预测器 | Sol 偏好信号 |
| **TemBERTure** | 热稳定性预测器 | Thermo 偏好信号 |
| **RFDiffusion** | 骨架生成 | de novo 骨架 benchmark + binder 骨架 |
| **CATH 4.3** | 数据集 | 训练 + 晶体重设计 test set |
| ProteinInvBench | benchmark | CATH4.3 JSONL 数据集来源 |
| **SolubleMPNN / HyperMPNN / InstructPLM / ESM-IF / ProteinDPO** | 基线模型 | 多类对照（子集训练、RL-DPO、SOTA 逆折叠） |

### P4 CAPE-MPNN

| 工具 | 类型 | 解决问题 |
|------|------|----------|
| **ProteinMPNN** | 基座模型 | foundation model，DAUPARAS 原版数据与架构 |
| **netMHCpan 4.1** | MHC-I 递呈预测 | 评估期精确判定递呈肽（rank 2% 阈值） |
| **PWM 分类器**（自构） | 近似预测器 | 训练期快速近似 netMHCpan，避免在线调用太慢 |
| **mmseqs2** | 序列聚类 | 30% 序列同一性聚类去泄漏 |
| **RFdiffusion / FoldingDiff** | 骨架生成 | 下游工作流中的模板生成 |
| **localcolabfold (ColabFold)** | 结构预测 | 评估期验证设计序列结构保真 |
| **DE-STRESS（含 Rosetta）** | 结构质量统计 | 结构质量的生物统计评估 |
| **Docker + GPU** | 环境封装 | 复现环境与超参搜索 |

## 跨论文工具汇总（去重）

**核心生成/预测模型**：ProteinMPNN、LigandMPNN、RFDiffusion、RFDiffusion2、RoseTTAFold(inpainting)、AlphaFold2、ESMFold、ColabFold、ESM-2。

**对齐算法**：DPO（核心）、以及 RobustDPO/RSO/KTO/NCA/SPPO/IPO/LiPO/SimPO（对照变体）。

**性质预测器（偏好信号源）**：pLDDT(AF2)、TM-score(ESMFold+TM-align)、pTM(IG)、EP(ESM-2 伪似然)、Protein-Sol、TemBERTure、netMHCpan/PWM。

**数据集/数据库**：PDB、CATH 4.3、UniRef30/100、ProteinMPNN 原始数据集、ProteinInvBench、AME 酶基序数据集。

**结构比对/质量**：TM-align、Kabsch、DE-STRESS(Rosetta)。

**表征手段（P1 湿实验）**：IMAC/SEC/CD/UV-vis/微秒 MD。

## 一个关键模式

所有论文的偏好信号都来自**「预测器」（结构预测器或性质预测器）而非湿实验标注**，这是蛋白质设计 DPO 对齐「无需昂贵人类反馈」的根本原因——也因此方法能大规模自动化。P4 更进一步展示了「昂贵预测器（netMHCpan）如何降级为廉价近似（PWM）」，为训练期解耦了精确性与速度。