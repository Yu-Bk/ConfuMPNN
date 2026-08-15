# P2 — ResiDPO（EnhancedMPNN）全景流程管线

**论文**: Improving Protein Sequence Design through Designability Preference Optimization
**作者**: Fanglei Xue, Andrew Kubaney, David Baker 等（UTS / UW IPD）
**出处**: arXiv:2506.00297（preprint，投稿中）

## 核心设计思想

传统蛋白序列设计（PSD）模型的训练目标是**序列恢复**（sequence recovery，最大化 P(native_seq | backbone)），但这与实际目标**「可设计性」**（designability，设计序列是否真的折叠成目标结构）不一致。本文把 PSD 重新框架为**偏好优化问题**，用 AlphaFold2 的 **pLDDT** 作为客观的偏好信号，用改良的 **DPO（ResiDPO）** 直接对齐「可设计性」。

## 完整流程（端到端）

```
① 训练集构建：PDB 单体（X-ray，<3.5Å，<1000 aa）→ 结构聚类去泄漏
            │
② 采样 + 标注：LigandMPNN 每结构生成 8 条序列（T=1.0）→ AF2 预测 → 得到 per-residue pLDDT
            │
③ 偏好对生成：Relative Sampling（pLDDT 差 > δ=10 的序列对 → (yw, yl)）
            │
④ ResiDPO 训练：RPL（残基级偏好学习）+ RCL（残基级约束学习）微调 LigandMPNN
            │
⑤ 得到 EnhancedMPNN
            │
⑥ 评测：酶设计 benchmark + binder 设计 benchmark，AF2 全预测判定设计成功
            │
输出：可设计性大幅提升的序列设计模型
```

## 各模块原理

### ① 数据集 PDB-D

- 从 PDB 选 X-ray 晶体、分辨率 < 3.5 Å、长度 < 1000 aa 的单体结构。
- 数据切分：2021-09-30 之后发布的结构作为验证集（沿用前期工作），做**结构聚类**，含验证结构的簇整体归为验证簇。→ 训练 19,203 结构，验证 1,690 代表结构。
- 每结构用 LigandMPNN 生成 8 条序列（T=1.0 鼓励多样性），AF2 预测得 per-residue pLDDT 标签。

### ② 偏好对生成（三种采样策略，消融后选定 Relative Sampling）

| 策略 | 做法 | 问题 |
|------|------|------|
| Rejection Sampling | 取批次内 pLDDT 最高序列为 yw，其余随机为 yl | 基线 |
| Application Sampling | yw: pLDDT>80，yl: pLDDT<75 | 高 pLDDT 序列稀缺，数据量少 |
| **Relative Sampling**（选定） | 任意 pLDDT 差 > δ 的序列对，(高者 yw, 低者 yl) | δ=10 最优，数据量大且多样 |

- δ=10 比 δ=30 好：两者 top 分数相近，但 δ=10 数据量大得多（9,557 vs 1,283 对）。

### ③ DPO 基础目标

标准 DPO 损失（公式 1）：

L_DPO = −E[ log σ( β·log(πθ(yw|x)/πref(yw|x)) − β·log(πθ(yl|x)/πref(yl|x)) ) ]

其中 β 控制偏好的强度（相对正则化）。x=backbone，yw/yl=序列，πref=预训练 LigandMPNN，πθ=微调模型。

### ④ ResiDPO 的两个解耦项

标准 DPO 用单一 sequence-level 损失，同时平衡「偏好最大化」与「KL 正则」，会互相冲突。ResiDPO 利用**蛋白序列定长**特性，做残基级解耦：

**RPL（Residue-level Preference Learning）**：
```
L_RPL = −E[ log σ( Σ_{i∈I} (log πθ(yw_i|x) − log πθ(yl_i|x)) / |I| ) ]
I = { i | pLDDT(yw, i) − pLDDT(yl, i) > α }
```
- 只在「优序列局部结构显著优于劣序列」的残基位点上强化偏好（α=10）。
- 若 I 为空（两序列 pLDDT profile 相似），退化为整条序列的标准 DPO 逻辑。

**RCL（Residue-level Constraint Learning）**：
```
L_RCL = E[ Σ_{j∈J} πref(yw_j|x)·log( πref(yw_j|x)/πθ(yw_j|x) ) / |J| ]
J = { j | pLDDT(yw, j) > β 且 πref(yw_j|x) > γ }
```
- 在「已经预测得又好又自信」的位点上做 KL 约束，防止灾难性遗忘、保持结构完整性。

**总损失**：`L_ResiDPO = L_RPL + λ·L_RCL`（λ=0.01）。

### ⑤ 超参数

α=10, β=80, γ=0.5, λ=0.01。β=80 依据「pLDDT>80 通常视为足够可设计」的实践。Adam，lr=5e-7，100,000 iterations，3% warmup + cosine annealing，2×L40 GPU，总 batch 8，梯度累积 16。

### ⑥ 评测（两条 benchmark）

**酶设计**（RFDiffusion2 活性位点骨架，5 个酶/5 个 EC 类）：
- 每酶 1000 骨架，每骨架 8 序列，T=0.1；催化残基固定；剔除 Cys。
- 成功判定：`pLDDT > 80` 且 `Cα RMSD < 1.5 Å`。

**binder 设计**（RFdiffusion binder 基准，5 个靶标）：
- 每靶 100 骨架，每骨架 8 序列。
- 成功判定：`pLDDT > 80`、`inter-chain PAE < 10`、`Cα RMSD < 1 Å`。

## 关键结果

- 酶设计：6.56%（LigandMPNN）→ 17.57%（EnhancedMPNN），近 3 倍；骨架成功率 19.74% → 40.34%。
- binder 设计：7.07% → 16.07%（2.27 倍）。
- 验证集 pLDDT accuracy：57.71 → 66.08；数据效率：ResiDPO 用 1k 样本 ≈ DPO 用 19k 样本。