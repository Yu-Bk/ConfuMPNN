# P3 — ProtAlign 创新点

> 本文核心创新：把「可开发性（溶解度、热稳定性）」这类**不自带结构一致性**的性质，与「可设计性」一起用**多目标半在线 DPO + 自适应偏好 margin** 联合优化。

## 创新方式（思路层）

1. **多目标偏好对齐**：从「单性质微调」升级为「多性质联合对齐」，用一个框架同时满足可设计性 + 多个可开发性目标，且不需要领域专家设计子集或手动调参。

2. **半在线 DPO**：结合在线探索（rollout 用当前策略）与离线效率（rollout/评估与训练解耦，批量推理），被证明效果不输纯在线（引 Lanchantin 2025），但部署简单得多。

3. **自适应偏好 margin**：解决多目标间的**优化方向冲突**——若 yw 在辅助性质上反而更差，就自动降低该对的 margin，避免单一性质被过度强化。

## 创新模块（实现层）

1. **灵活 margin 的 DPO 损失 L_MO（公式 4）**：
   - `m_k(yw,yl) = λ·Σ_{k′≠k} w_{k′}·(r_{k′}(yw) − r_{k′}(yl))`，margin 训练前离线预计算。
   - 权重固定（IG/TM=0.6，其他=0.4），β=0.5。

2. **顺序无关模型的共享顺序 log-ratio 估计（公式 5）**：用 K 个共享随机排列 σ 近似 πθ 与 πref 的 log-ratio，显著降低方差、稳定训练。这是把 DPO 从 left-to-right LLM 迁移到 ProteinMPNN 的**关键工程创新**。

3. **semi-online 迭代训练（Algorithm 1）**：高温 rollout → 预测器标注 → 构造 D_k → 离线训练，多轮演化。rollout 阶段用高温 τ 促进多样性，评估用低温 0.1。

4. **排序配对 + δ 阈值过滤（4.4）**：第 i 名配第 (N/2+i) 名，分数差须 > δ_k，滤掉噪声注释。

5. **性质预测器组合**：TM/pTM（可设计性）+ EP(ESM 伪似然)/Protein-Sol/TemBERTure（可开发性）作为 proxy annotator。

## 关键数据（创新的有效性）

- CATH4.3：MoMPNN[Thermo+IG+EP] 达 Thermo 0.963（vs ProteinMPNN 0.769、HyperMPNN 0.929），Sol 0.723 vs 0.719；TM/pLDDT 几乎不退。
- de novo：MoMPNN[Sol+IG+EP] RMSD 6.17 优于 ProteinMPNN 6.86。
- 消融（A.2）：Weighted-score DPO（多目标合并为单分数）在 Evo. ppl 上最优，但 RMSSD/TM/pLDDT/AAR 全面退化；MoMPNN 更均衡，迭代更稳定，AAR 仅降 1%。
- 迭代趋势（Fig.5）：weighted-score DPO 的 IG 第 6 轮后开始下降（过拟合单目标），MoMPNN 稳定收敛。

## 性质可解释性（附录 A.1）

- **溶解度**：MoMPNN 在亲水残基比例、表面净电荷（-14.16 vs -4.62）、电荷分布均匀度、GRAVY（-0.736 vs -0.293）等指标全面优于 SolubleMPNN。
- **热稳定性**：MoMPNN 与 HyperMPNN 呈几乎一致的残基再分布（表面正电荷↑、核心疏水↑、柔性残基↓），说明它「继承了」热稳定模型的特征分布，同时保持表面极性/核心疏水的平衡。

## 与其他论文的关联

- 直接回应 P2 的局限：P2 只优化可设计性，无法扩展到「与可设计性冲突的可开发性」。
- 与 P4 对比：P4 是单性质（免疫可见性）的 DPO，P3 是通用多性质框架，P4 可视为 P3 框架的一个特例。P3 引用了 ProteinDPO（实验稳定性偏好 DPO）、ResiDPO 等作为 related work。

## 局限（论文自述）

1. 缺乏湿实验验证（全是 in silico 指标）。
2. 只研究单体性质；binder 复杂度性质未探索。