# P3 — ProtAlign（MoMPNN）全景流程管线

**论文**: Property-Driven Protein Inverse Folding with Multi-Objective Preference Alignment
**作者**: Xiaoyang Hou, Junqi Liu, Jian Tang 等（北大 / BioGeometry / Mila）
**出处**: ICLR 2026 | arXiv:2603.06748

## 核心设计思想

真实设计流程要求蛋白**既可设计（designable）又可开发（developable）**——可溶性、热稳定性、表达量等。现有做法（事后突变、推理时偏置、子集重训）都是**目标依赖**且需大量领域知识或仔细调参。ProtAlign 提出**多目标偏好对齐框架**，用 semi-online DPO + 自适应偏好 margin 同时优化多样性开发性质并保持结构保真度。

## 完整流程（端到端）

```
① 基础模型：ProteinMPNN（order-agnostic autoregressive），训练集 CATH 4.3
            │
② 半在线迭代（T 轮，每轮 N 个 backbone）：
      rollout（高温 τ 采样 n 条序列）
        → K 个性质预测器打分
        → 每个性质构造偏好对数据集 D_k
      training（离线优化，均匀采样各 D_k）
            │
③ 自适应 margin 的 DPO 损失（L_MO）
            │
④ 得到 MoMPNN
            │
⑤ 评测：CATH4.3 晶体重设计 / de novo 骨架设计 / binder 设计
            │
输出：溶解度/热稳定性提升、且可设计性不退化的模型
```

## 各模块原理

### ① ProteinMPNN 概率表示

ProteinMPNN 是**顺序无关的自回归模型**，序列概率按随机排列 σ 分解：

πθ(y|x,σ) = ∏_i πθ(y_{σ(i)} | x, y_{σ(<i)})

训练用交叉熵 + teacher forcing（随机排列 σ）。

### ② 多目标 + 自适应 margin 的 DPO（L_MO，公式 4）

核心创新。多目标策略目标（公式 2）：

argmax_θ L(πθ) = E[ Σ_k w_k·r_k(x,y) ] − β·D_KL(πθ∥πref)

融合 Bradley-Terry 偏好模型（公式 3）后，每条性质 k 得到**带自适应 margin 的 DPO 损失**：

```
L_MO(θ; D_k) = −E[ log σ( w_k · ( β·log(πθ(yw|x)/πref(yw|x)) − β·log(πθ(yl|x)/πref(yl|x)) − m_k(yw,yl) ) ) ]

m_k(yw, yl) = λ · Σ_{k′≠k} w_{k′} · ( r_{k′}(x,yw) − r_{k′}(x,yl) )
```

- **自适应 margin 的直觉**：若 yw 在某个**辅助性质**上反而比 yl 差，就降低这对样本所需的 margin，避免「为了优化单一性质而牺牲其他性质」的冲突。
- margin 在训练前用性质预测器**预先算好**（权重与数据集固定后即可离线算）。
- 训练时从各 D_k 属性均衡采样。

### ③ 顺序无关模型的 log-ratio 高效估计（公式 5）

ProteinMPNN 非 left-to-right，精确估计 πθ 需大量解码顺序采样。采用**共享顺序采样**（借鉴离散扩散 LLM）：

p̂θ(y|x) = (1/K)·Σ_k πθ(y|x, σ_k)，πref 用**相同的 σ_k** 采样。

共享顺序大幅降低 log-ratio 估计方差，训练更稳定。

### ④ 半在线训练（Algorithm 1）

- 每轮 t：当前策略 πt 在**高温 τ**（高于评估温度）rollout → K 个预测器打分 → 构造 D_k → 离线优化若干步 → 得 π^{t+1}。
- 好处：结合在线探索与离线效率；rollout/评估与训练解耦，利于批量计算；预测器无需改动，兼容现有方法。

### ⑤ 偏好对构造（4.4）

- N 条候选序列按性质打分排序，第 i 名与第 (N/2+i) 名配对（i≤N/2）。
- 仅当分数差 `M_k(yw) − M_k(yl) > δ_k`（性质特定阈值）才纳入，过滤噪声注释。

### ⑥ 性质分类与预测器

- **可设计性（designability）属性**：TM（ESMFold 预测结构与参考结构的 TM-score）或 pTM（AF2 + Initial Guess）。
- **可开发性（developability）属性**：
  - 通用质量：ESM-2 伪似然（Evolutionary Perplexity, EP）。
  - 定向质量：Protein-Sol（溶解度 Sol）、TemBERTure（热稳定性 Thermo）。

### ⑦ 评测与指标

三条 benchmark：
- **CATH4.3 晶体重设计**：RMSD↓、TM↑、pLDDT↑、EP↓、Sol↑、Thermo↑、AAR↑（氨基酸恢复率）。
- **de novo 骨架**（RFDiffusion 生成 50–500 aa，1,824 骨架）：同指标（无 AAR）。
- **binder 设计**（6 个靶标，PD-L1/SC2RBD/BHRF1/PD-1/CLN1-14 等）：成功判定 pLDDT>80、inter-chain PAE<10、Cα RMSD<2Å。

## 关键结果

- CATH4.3：MoMPNN 在几乎不退化的 RMSD/TM/pLDDT 下，Sol/Thermo 显著优于 ProteinMPNN，并超越专门子集训练的 SolubleMPNN/HyperMPNN。
- de novo：MoMPNN 结构一致性甚至超过 ProteinMPNN；IG 目标优于 TM。
- binder：MoMPNN[Sol+IG+EP] 序列/骨架成功率略高于 ProteinMPNN，EP 与 Sol 显著更优。
- 消融：Weighted-score DPO（把多目标合并成单分数）在单一指标上最优但在其他指标退化；MoMPNN 更均衡、迭代更稳定、AAR 下降仅 1%。