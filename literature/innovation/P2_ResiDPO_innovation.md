# P2 — ResiDPO 创新点

> 本文核心创新：把「可设计性」从「序列恢复」这一错误对齐的训练目标中剥离出来，用**残基级解耦的 DPO** 直接优化。

## 创新方式（思路层）

1. **把设计性低诊断为目标错位（objective misalignment）问题**：指出传统 PSD 优化 sequence recovery，但 designability（折叠成目标结构的概率）与之相关性弱。这是对问题的新框架——因此可借用 LLM 的 RLHF/DPO 对齐技术。

2. **用 pLDDT 当客观偏好信号**：蛋白比自然语言的优越性在于——不需要主观人类偏好，可用 AF2 的 pLDDT 作为**定量、客观、逐残基**的奖励信号（pLDDT 与 Cα RMSD 强相关，附录 Fig.6 佐证）。

3. **残基级解耦**：利用「固定骨架 → 定长序列」这一蛋白特性，把 sentence-level DPO 拆成「偏好学习」与「约束学习」两个残基级目标，消除标准 DPO 中两者共享单一损失的梯度冲突。

## 创新模块（实现层）

1. **Residue-level Preference Learning（RPL，公式 2-3）**：只在「优序列局部 pLDDT 显著优于劣序列（差 > α=10）」的位点上强化偏好；若 I 为空退化为序列级标准 DPO。

2. **Residue-level Constraint Learning（RCL，公式 4-5）**：在「已又高又自信（pLDDT>β=80 且 πref>γ=0.5）」的位点上做 KL 约束，防灾难性遗忘、保结构。

3. **Relative Sampling 偏好对生成**：pLDDT 差 > δ=10 的任意配对，比 Rejection/Application 采样数据更多样。

4. **pLDDT Accuracy 快捷评测指标**：用「模型输出似然 vs 实际 pLDDT」的相关性作为省去全 AF2 的消融代理指标。

## 关键数据（创新的有效性）

| 方法 | pLDDT Acc. | Seq. Recovery |
|------|-----------|---------------|
| LigandMPNN | 57.71 | 57.63 |
| DPO (Relative Sampling) | 62.11 | 57.03 |
| RPL only | 63.44 | 54.23 |
| **ResiDPO (RPL+RCL)** | **66.08** | 55.56 |

- RPL 单独用会掉序列恢复（54.23），加 RCL 后两者平衡（55.56 + 66.08），证明 RCL 防遗忘的关键作用。

## 机制解释（4.5 节）

EnhancedMPNN 倾向引入更多**带电残基**（E、K、R），把 A/Q/S/T 大幅替换；P、G 基本不变。解释：A/S/T 的在「埋藏/暴露」上语义模糊，G/P 有独特构象偏好故清晰。**本质是「降低序列-结构映射的模糊性」，从而提升 AF2 预测置信度**。

## 与其他论文的关联

- P2 是单目标（可设计性）的残基级 DPO；P3 是**多目标**（可设计性 + 溶解度 + 热稳定）的序列级 DPO，且 P3 明确把 ResiDPO 列为 related work。
- P2 与 P4 都用 DPO + 预测器当偏好信号，但 P2 用 pLDDT（残基级）、P4 用 MHC 递呈（序列级，但本质也是「避免生成某些 k-mer」）。
- P2 的「用 AF2 结构预测器当偏好信号」与 P3 的「TM/pTM + 性质预测器」是同一思想的不同粒度。