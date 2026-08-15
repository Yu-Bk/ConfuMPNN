# P4 — CAPE-MPNN 全景流程管线

**论文**: Tuning ProteinMPNN to reduce protein visibility via MHC Class I through direct preference optimization
**作者**: Hans-Christof Gasser, Diego A. Oyarzún, Javier Antonio Alfaro, Ajitha Rajan 等（爱丁堡大学等）
**出处**: Protein Engineering, Design and Selection, 2025, 38, gzaf003 | DOI: 10.1093/protein/gzaf003

## 核心设计思想

治疗性蛋白（尤其经 mRNA/基因疗法的胞内表达蛋白）会被细胞毒性 T 淋巴细胞（CTL）经 MHC-I 通路识别，引发抗转基因免疫。本文首次把**去免疫（deimmunization）**直接融入 ProteinMPNN，用 **DPO**（LLM 对齐技术）微调，使其在保持结构的前提下生成 **MHC-I 表位更少**的序列。

## 完整流程（端到端）

```
① 定义「可见性 visibility」：序列中所有被 MHC-I 递呈的 8–10mer 的（去重）计数
            │
② 训练时加速：用 PWM 近似替代昂贵的 netMHCpan（评估时仍用 netMHCpan 4.1）
            │
③ 数据：ProteinMPNN 原始数据集（PDB 3.5Å 截止 + mmseqs2 30% 聚类去泄漏）
            │
④ DPO 对齐：偏好信号 = MHC-I 递呈预测器（扮演「人类反馈」角色），ProteinMPNN 为 foundation model
            │
⑤ 得到 CAPE-MPNNN（权重可直接替换 ProteinMPNN）
            │
⑥ 评测：结构保真（ColabFold 预测 + DE-STRESS/Rosetta）vs 免疫可见性降低的 trade-off
            │
输出：在保持折叠结构的前提下降低免疫可见性的序列设计模型
```

## 各模块原理

### ① 免疫可见性定义（Definition 1）

- **绝对可见性**：序列中所有被递呈的 k-mer（k=8,9,10）的**去重数量**（一个 k-mer 被多个等位基因递呈只计 1 次）。
- **相对可见性**：序列可见性 ÷ 模板（PDB 结构）可见性，用于跨蛋白比较。
- 示例：11 AA 序列 `GANIWGANNNV` 含 2 个 10mer、3 个 9mer、4 个 8mer；在假设患者（6 个 HLA 等位基因）与 netMHCpan 2% rank 阈值下，只有 `ANIWGANNNV`、`NIWGANNNV` 被递呈 → 绝对可见性 = 2。
- 假设患者等位基因：HLA-A*02:01, A*24:02, B*07:02, B*39:01, C*07:01, C*16:01。
- **重要区分**：免疫可见性 ≠ 免疫原性（可见是 CTL 反应的必要非充分条件）。

### ② PWM 近似（训练加速）

netMHCpan 训练时在线调用太慢，改为构造 **PWM 分类器**做近似：

- 采样 300 万随机 8–10mer，用 netMHCpan（2% rank）判定是否被 6 个等位基因递呈。
- 每个 PWM 一行一个 AA、一列一个位置，值 = 递呈肽中该 AA 出现在该位置的 log-likelihood 概率。
- 新肽的递呈判定：log-likelihood 之和超过阈值；阈值按「使 PWM 在随机集上预测的递呈比例与 netMHCpan 一致」来定。
- 6 个等位基因合并为**一个 PWM**（训练用），评估仍用 netMHCpan 4.1。

### ③ DPO 目标（公式 1）

L = −E[ log σ( β·log(πθ(yw|x)/πref(yw|x)) − β·log(πθ(yl|x)/πref(yl|x)) ) ]

- x = backbone 模板，yw/yl = 生成序列，πref=ProteinMPNN，πθ=CAPE-MPNN。
- β 控制偏离原模型的程度。
- ProteinMPNN 扮演 foundation model，MHC-I 递呈预测器扮演「人类反馈」。

### ④ 数据集（与 ProteinMPNN 原论文相同）

- PDB 3.5 Å 截止，473,062 链（106,344 独特序列）。
- mmseqs2 30% 序列同一性聚类 → 23,349 train / 1,464 val / 1,539 test 簇。
- 防泄漏：与选中链 TM-score > 0.7 的链全部 mask（不给序列信息）；每 2 个 epoch 刷新 Structure Dataset。

### ⑤ 评测管线（trade-off 分析）

- 用 ColabFold 预测设计序列结构，验证结构保真。
- 用 netMHCpan 4.1 计算免疫可见性，DE-STRESS（含 Rosetta）做结构质量统计。
- 关键：在「结构保真 vs 可见性降低」之间做 trade-off 分析，识别合适的 DPO 超参数集。

## 关键结果

- DPO 有效降低 MHC-I 可见性，且不损害蛋白结构完整性。
- 模型权重（CAPE-MPNN）与 ProteinMPNN 同架构，可直接替换进现有 RFdiffusion + ProteinMPNN 等工作流。