# ConfuMPNN — 完整项目计划

## 项目概述

将蛋白质在特定 pH 环境下的理化性质（净电荷、局部电荷分布）作为条件约束，整合到基于结构的蛋白序列生成流程中，形成从"骨架结构 → pH 感知条件生成 → 结构验证"的完整管道。项目仓库：**ConfuMPNN**（已创建于 GitHub）。

核心创新点：**在 LigandMPNN 这类显式建模配体/金属/核酸原子上下文的结构条件逆折叠模型上，首次加入 pH 感知的电荷条件控制**——现有工作主要集中在纯序列模型或抗体专用模型上。

设计哲学：**用户指定工作环境的 pH，模型自动学习在该 pH 下应该选什么残基。** pI 不作为直接输入——它是生成序列的推导属性（给定序列的氨基酸组成，pI 由氨基酸侧链 pKa 表唯一确定），在验证阶段用作一致性检查。

> **📎 两级计划：** 本文件是第一版（主线：pH 电荷条件生成）。**第二版拓展计划**（多目标可开发性微调——让生成序列更可设计/热稳定/可溶，并把微调模型放回本管线）见 **`index/PROJECT_EXTEND.md`**。条件嵌入 context 两版一致。

---

## 第一部分：文献调研与原创性评估

### 1.1 已有相关工作

以下按"与项目重叠程度"从高到低排列，标注来源和确认状态：

**A. 性质条件化的蛋白序列生成（技术范式成熟，但场景不同）**

| 工作 | 模型类型 | 控制的性质 | 与本项目的区别 | 来源/确认状态 |
|------|---------|-----------|--------------|-------------|
| LaMBO-2 / NOS (Gruver et al., 2023) | 离散扩散（序列空间） | 表达量、结合亲和力、developability | 纯序列模型，无结构上下文 | arXiv 2305.20009v2 ✓ |
| Guided Generation for Developable Antibodies (Zhao et al., 2025) | 离散扩散（抗体轻重链） | Developability（溶解度、稳定性等） | 抗体专用，无配体/小分子上下文 | arXiv 2507.02670v1 ✓ |
| MP2D (Kong et al., 2026) | 离散扩散 + MCTS | 多目标（抗菌肽、binder） | 多目标 Pareto 优化框架，不做逆折叠 | arXiv 2605.05829v1 ✓ |
| AntiBARTy Diffusion | 扩散模型（序列） | 溶解度（Protein-Sol） | 类别条件，无结构上下文 | 对话引用，需确认 |
| Chroma (Generate Biomedicines) | 扩散模型（结构+序列） | Conditioner API（能量函数） | 框架级工具，不做配体上下文 | 已发表，需确认 |
| MolGPT | Transformer（自回归，小分子） | 多性质（discrete control token） | 小分子领域；用离散 token 做条件化，精度受限于 token 数量 | 已发表，需确认 |
| NExT-Mol | Transformer（自回归，小分子） | 多性质（soft prompt） | 小分子领域；用 MLP→连续向量做条件化，精度无损失 | 已发表，需确认 |
| TaxDiff (Lin et al., 2024) | 扩散模型（序列） | 物种分类标签 | 控制粒度是"物种"而非"理化性质" | arXiv 2402.17156v1 ✓ |
| DiMA (Meshchaninov et al., 2024) | 连续扩散（PLM 隐空间） | 蛋白家族、motif scaffolding | 通用条件生成框架，不做理化性质 | arXiv 2403.03726v4 ✓ |
| ProteinRL | RL + ProGen2 | 多种性质（RL 优化） | RL 路线，可作为方法对比 baseline | 对话引用，需确认 |

**结论：目前没有任何工作将 pI/电荷/电荷聚集作为内嵌 conditioning 加入到 LigandMPNN 这种显式建模配体原子上下文的结构条件逆折叠模型中。** 最近的工作（UMA-Inverse, 2026年7月, arXiv 2607.07866v1 ✓）也只在结构精度上改进 LigandMPNN，未涉及性质条件化。

**B. 偏好对齐方法（ProtAlign/MoMPNN，ICLR 2026）——边界机制参考**

ProtAlign 基于 ProteinMPNN，用半在线多目标 DPO 做偏好对齐。核心创新是**灵活偏好边界（flexible preference margin，Eq.4/14）**：在优化属性 k 时，从加权奖励中扣除所有其他属性 k' 的贡献（Eq.14: `r_k = (1/w_k)·[β·log(π_θ/π_ref) - Σ_{k'≠k} w_{k'}·r_{k'}]`），若 win/lose 对在其他属性上反向，margin 自动缩小优化力度。消融实验（Appendix A.2）证明：不加 margin 的 Weighted-score DPO 会退化为单目标优化，甚至不如基础模型。

**重要概念区分：** ProtAlign 的"条件"指的是它的基础输入条件 x（PDB backbone 结构，即 ProteinMPNN 标准的 P(S|X) 中的 X），**不是**用户可编辑的性质条件向量。ProtAlign 没有条件编码器——它的"控制"是通过 DPO 隐式地让模型偏好某些性质，用户不能显式输入"我要 pI=7"。

**ProtAlign 论文中直接可迁移的三个设计：**
1. **自适应边界机制（m_k）→ Phase 2 多约束冲突检测**：论文 Eq.14 的核心洞察——单独优化属性 k 时的"等效奖励"等于总贡献减去其他属性的贡献。迁移到我们的框架：每个候选氨基酸的"等效 bias"等于它对当前约束的收益减去它对其他活跃约束的代价。优势是我们的"reward"是闭式公式（无预测误差），比 ProtAlign 的 predictor-based reward 更干净。
2. **半在线训练策略（Algorithm 1）→ Phase 2 DPO 辅助循环**：每轮用当前模型重新生成→重新打分→重新构造偏好对，避免分布漂移。
3. **多目标与 Weighted-score 的消融对比（Appendix A.2）→ Phase 2 实验设计**：带 margin vs 不带 margin（所有约束加权成一个标量）vs 单约束 vs 基础模型，四组对照。ProtAlign 的实验结果已证明不加 margin 会导致退化。

**不通用的部分：** ProtAlign 基于 ProteinMPNN 而非 LigandMPNN，权重不能复用。它的属性预测器（Protein-Sol, TemBERTure, ESMFold）打分不是闭式的，需要用训练好的神经网络——而我们项目的核心属性（pH→电荷、净电荷、局部电荷聚集）全部是闭式可计算的。ProtAlign 不做条件编码器，不让你指定目标值。

**B. pH/质子化感知的结构预测（真正的空白区）**

- 目前没有任何深度学习模型内生地理解 pH/质子化态。
- PypKa server 是对 AF2/AF3 输出做后处理（pKa 预测、质子化态分配），已对 >20 万个 PDB 和 AlphaFold DB 结构做了预计算。
- PEP-FOLD4 用经典力场做短肽（10-40 残基）的 pH 依赖结构预测，不是深度学习方法。
- 相关研究（PNAS 2026）系统性地证实了 AI 结构预测模型在放置可电离残基时违反物理化学规则的问题。

**结论：本项目的 RF3 + PypKa 验证管线是这个子领域目前唯一可行的路线。**

**C. 为什么不选用 MolGPT 而是参考 NExT-Mol 的条件注入方式**

MolGPT 把连续的性质值（如 pH=7.4）量化为离散的 control token——要么四舍五入为整数（精度大损），要么为每个值建一个 token（pH 4.0-10.0 每 0.1 要 60 个 token）。NExT-Mol 用 2 层 MLP 把连续标量直接映射为连续的 soft prompt 向量，无精度损失——对于 pH 这种精确到 0.1 甚至 0.01 的条件值，这是正确的选择。

### 1.2 项目的原创性定位

> "我把已经在抗体开发性优化中验证过的 guided/conditioned generation 范式，迁移到了配体感知的逆折叠场景（LigandMPNN），并且用 AI 结构预测模型的物理缺陷（可电离残基放置错误）作为下游验证的动机，搭建了一个'约束生成 + 结构验证'的闭环。"

**与 ProtAlign（ICLR 2026）的区分——核心差异：**

| | ProtAlign / MoMPNN | 本项目 (ConfuMPNN) |
|---|---|---|
| 范式 | 偏好对齐（DPO），P(S\|X) 整体偏好偏移 | 条件生成 P(S\|X, C)，用户显式指定目标值 |
| 控制方式 | 隐式（模型自动偏好高溶解度） | 显式（用户输入 pH=7.4 → 生成对应序列） |
| 条件编码器 | 无 | 有（Soft Prompt MLP） |
| 物理不稳定性处理 | 无 | 有（结构感知过滤器 + 金属配位分类） |
| 多目标冲突处理 | 灵活偏好边界 m_k（DPO 框架内） | 多约束冲突检测（Phase 2，迁移自 m_k，但用闭式公式替代 reward model） |
| 输入 | PDB backbone | PDB backbone |

**ProtAlign 论文中的关键实验证据（Appendix A.2）：** 不加 margin 的 Weighted-score DPO（多目标加权成一个标量）会退化为单目标优化，在某些指标上甚至不如基础模型；带 margin 的 MoMPNN 所有指标均衡提升且训练稳定。这个结果直接支持"加入条件边界能增强多目标效果"的核心论点。

故事线：

> "ProtAlign 证明了一个关键方法论——没有冲突检测边界的多目标优化会退化。ConfuMPNN 把这个洞察从 DPO 偏好对齐迁移到了条件生成框架：我们不仅是第一个在 LigandMPNN 上做显式性质条件控制的，而且通过迁移 ProtAlign 的边界机制（用闭式公式替代 reward model，更精确、零训练成本），实现了多约束之间的冲突预警和自适应调控。"

---

## 第二部分：技术可行性评估

### 2.1 外部模型获取与用途

| 组件 | 获取方式 | 用途 | 硬件需求 |
|------|---------|------|---------|
| **LigandMPNN** | GitHub 仓库（dauparas/LigandMPNN），权重自带的下载脚本获取 | **核心改造对象**：条件注入 + 引导采样 | 5080 (16GB)，262 万参数 |
| **RF3 / RosettaFold3** | GitHub 开源，本地部署（替换 AF3） | **验证管线首选**：LigandMPNN 生成序列 → RF3 回折 → backbone RMSD 验证 | 待评估（5080 需确认；不行用备选 ESMFold） |
| **ESMFold** | ESM GitHub（Meta），权重公开可下载 | **验证管线备选**：如果 RF3 本地部署难度太大或 5080 跑不动，用 ESMFold 做回折验证 | 5080 可跑 |
| **PROPKA** | `pip install propka`，纯 Python | 交叉验证质子化态（每个残基的 pKa 偏移） | 无需 GPU |
| **PypKa** | Web server (https://pypka.org) + Python API | 更现代的质子化态/滴定曲线分析 | 无需 GPU |
| **Biopython** | `pip install biopython` | 训练标签批量计算（`charge_at_pH`）+ pI 计算 + PDB 解析 | 无需 GPU |
| **溶解度预测器** | 从论文 GitHub 拉取（可选） | Phase 4 可选：多目标集成 | 5080 够用 |

**不需要下载的：** AF3 权重（被 RF3 替代），大规模蛋白语言模型预训练权重。

### 2.2 硬件资源匹配

| 硬件 | 用途 |
|------|------|
| **RTX 5080 (16GB)** | 日常开发、全流程推理、Level 1 引导采样、RF3/ESMFold 回折验证、性质预测器 |
| **2× A100 (40/80GB)** | LigandMPNN 条件微调训练（单卡足够）、批量候选序列生成 |

**结论：硬件完全够用。**

### 2.3 代码工作量估算（你主要审查，我编写）

| 阶段 | 内容 | 估算时间 |
|------|------|---------|
| Phase 0 | 环境搭建、Git 配置、跑通 LigandMPNN demo | 3-4 小时 |
| Phase 1 核心 | 可微电荷计算 + 结构感知过滤器 + 引导采样 | 10-15 小时 |
| Phase 1 调试 | 在 3-5 个示例蛋白上跑通、调参 | 3-5 小时 |
| **小计（拿到初步结果）** | **可展示的 minimum viable result** | **18-24 小时** |
| Phase 2 | 条件编码器 + 微调训练 + 超参调优 | 15-20 小时 |
| Phase 3 | 验证管线（RF3/ESMFold + PypKa） | 8-10 小时 |
| Phase 4 | 集成 + 文档 + 面试准备 | 10-15 小时 |
| **总计** | | **50-70 小时** |

### 2.4 时间线可行性

8 周是合理的——但拿到初步结果（Phase 1 跑通）可以在 2-3 周内完成。剩下的微调和验证可以慢慢迭代。

---

## 第三部分：与 Transformer 和 Diffusion 的关系

### 3.1 架构分析

LigandMPNN 编码器 = MPNN（消息传递图神经网络），解码器 = 自回归解码（GPT-style）。本项目涉及的是 MPNN + 自回归解码，条件控制机制直接来自 Transformer/Diffusion 领域的方法论。

### 3.2 条件控制机制与 AI 范式的映射

| 你要做的事 | 对应的 AI 范式 | 代表方法 |
|-----------|---------------|---------|
| 把 pH/电荷编码为 embedding 注入解码器 | Transformer 条件生成（soft prompt） | NExT-Mol |
| 用可微公式的梯度 bias logits | Diffusion 的 energy-based conditioner | Chroma |
| 推理时不训练，只改采样策略 | Classifier guidance（自回归版本） | NOS/LaMBO-2 |
| 条件微调 + 辅助 loss | 通用条件生成范式 | 适用于 Transformer 和 Diffusion |

### 3.3 时间富余时的加分项

- 在 UMA-Inverse（纯 Transformer 风格 encoder）上做条件化对照
- 在 EvoDiff（序列扩散模型）上做条件化对照

"复现你的条件化方案"的意思是：把你写的条件编码器和引导采样逻辑，不加修改地搬到这些模型上，跑一遍看看效果。这是对照组实验，不是重新发明。

---

## 第四部分：完整 Pipeline 设计

### 4.1 总体架构

```
输入：PDB 结构（骨架坐标）  ← 注意：输入是PDB，不是序列
  │
  ├─→ [Step 1: 骨架提取]
  │     PDB → LigandMPNN 原生 parse_PDB → 骨架坐标 (N, CA, C, O + 虚拟 CB)
  │     注意：如果用户只有序列没有结构 → 用 RF3/ESMFold 折叠 → 得到 PDB 结构
  │
  ├─→ [Step 2: 条件约束定义]
  │     用户指定：
  │     - 工作环境 pH（必填，如 pH=5.0 溶酶体, 7.4 血液, 8.0 肠道）
  │     - 可选：目标净电荷（None=不指定）
  │     - 可选：局部电荷聚集约束（None=默认宽松阈值 10Å 内正/负电荷 ≤6）
  │     注意：pI 不作为直接输入——它是生成序列的推导属性
  │
  ├─→ [Step 3: 条件约束序列生成]  ← 核心创新
  │     LigandMPNN 自回归解码 + pH 感知条件引导
  │     ├─ 可微净电荷 lookahead：每一步对候选氨基酸做精确的前瞻计算
  │     ├─ 结构感知过滤器：实时检测极端电荷聚集（k-NN 图上常数时间查询）
  │     │   - 空间正/负电荷聚集（10Å 内 ≥6）
  │     │   - 盐桥过密（10Å 内 K/E 或 R/D 配对 ≥5 对）
  │     │   - 核心电荷渗入（burial >0.8 且 8Å 内带电残基 ≥4）
  │     │   - 同号电荷空间聚类（4+ 同号在 8Å 连通图内）
  │     │   以上均为 logit bias 实时注入，不是事后过滤
  │     └─ 条件编码器（Level 2 微调阶段）：pH → MLP → soft prompt tokens
  │     → 生成 N 条候选序列（N=100~1000）
  │
  ├─→ [Step 4: 候选序列筛选]  ← 三层过滤
  │     Filter 1: 结构自洽性
  │       候选序列 → RF3/ESMFold 重新折叠 → backbone RMSD + pLDDT
  │     Filter 2: 性质命中度
  │       计算生成序列的 pI，验证是否与工作 pH 物理自洽
  │     Filter 3: 交叉验证
  │       用 PypKa/PROPKA 独立验证质子化态
  │
  ├─→ [Step 5: 输出]
  │     排序后的候选序列 + 验证报告
  │
  └─→ [可选 Step 6: 迭代优化]
        调整 bias 权重重新生成
```

### 4.2 条件向量设计（核心设计决策）

**设计原则：pH 主导，Mask-aware 编码。**

```python
# 用户配置（YAML 文件）
condition_config:
  pH: 7.4                 # [必填] 工作环境的pH
  net_charge: null         # [可选] null=不指定, float=目标净电荷
  local_pos_limit: 6       # [可选] null=默认宽松(6), int=10Å内正电荷数上限
  local_neg_limit: 6       # [可选] null=默认宽松(6), int=10Å内负电荷数上限

# 转换为编码器输入（mask-aware设计，避免0值歧义）
# [pH, has_charge_flag, charge_value_or_placeholder, 
#  has_pos_limit_flag, pos_limit_or_placeholder,
#  has_neg_limit_flag, neg_limit_or_placeholder] → shape [7]

# 示例：
# 只指定 pH=7.4，其他都不指定：
#   [7.4, 0, 0.0, 0, 0.0, 0, 0.0]
# 指定 pH=7.4, net_charge=0, pos_limit=8：
#   [7.4, 1, 0.0, 1, 8.0, 0, 0.0]
```

`has_X_flag` 告诉条件编码器 MLP 哪些值是真的、哪些是占位符。不参与 loss 计算的值由 flag 控制。

### 4.3 氨基酸侧链 pKa 表（不需要重构 vocabulary）

20 种标准氨基酸的侧链 pKa 是已知常数——固定查表：

```
Asp (D)   侧链-COOH    3.9    碱性→去质子化带 -1
Glu (E)   侧链-COOH    4.3    碱性→去质子化带 -1
His (H)   咪唑基       6.0    碱性→去质子化为中性（正电消失）
Cys (C)   侧链-SH      8.3    碱性→去质子化带 -1
Tyr (Y)   酚羟基       10.1   碱性→去质子化带 -1
Lys (K)   侧链-NH3+    10.5   碱性→去质子化为中性（正电消失）
Arg (R)   胍基         12.5   碱性→去质子化为中性（正电消失）
N-端      α-NH3+       ~9.7   碱性→去质子化为中性
C-端      α-COOH       ~2.3   碱性→去质子化带 -1
```

**为什么 LigandMPNN 和 RF3 不需要不同质子化状态的 token：**

LigandMPNN 的氨基酸 alphabet 是标准 20 种氨基酸。它不区分 HIE（epsilon 位质子化）/HID（delta 位质子化）/HIP（双质子化）——全部视为 His。RF3/ESMFold 同样不区分。PDB 文件中 HIE/HID/HIP 标记大多来自事后工具（PROPKA）的添加。

因此：**不需要修改 LigandMPNN 的 vocabulary。** 生成序列仍是标准 20 种氨基酸。pI/电荷约束通过条件编码器 + logit bias 实现，结合层面用游离 pKa 表计算电荷，验证阶段用 PypKa 做微环境修正。

### 4.4 训练时的多 pH 数据增强

**不对具体整数值采样**——用均匀分布随机采样连续 pH 值：

```python
for sequence in training_sequences:
    for _ in range(n_pH_samples_per_seq):
        pH = random.uniform(4.0, 10.0)  # 连续采样
        charge = net_charge_at_pH(sequence, pH)
        pI = find_pI(sequence)
        training_samples.append((backbone, sequence, pH, charge, pI))
```

这样模型在训练时看到的是 [4, 10] 范围内的连续 pH 分布，推理时输入任意 pH 值都有泛化可能。

### 4.5 条件注入机制（Level 2 微调阶段）

**条件向量标准化（来自另一方案的重要建议）：**

不同量纲的条件值（pH 4-10 vs 净电荷 -20~+20 vs 局部电荷上限 0-12）直接输入 MLP 会导致梯度不稳定。在进入条件编码器之前标准化：

```python
# 从训练集计算 μ 和 σ
C_normalized = (C - μ) / σ  # 每维度独立标准化

# 然后送入条件编码器
cond_tokens = ConditionEncoder(C_normalized)
```

标准化常量在训练前计算一次，写入 config 文件，推理时复用。

**条件编码器架构（NExT-Mol 风格 soft prompt）：**

```
条件向量 c = [pH, flag1, val1, flag2, val2, flag3, val3]  shape: [7]
     │
     ▼
  Linear(7 → 64) → GELU → Linear(64 → 128) → GELU → Linear(128 → 4×128)
     │
     ▼
  reshape → [4, 128]  # 4个soft prompt token，拼接到decoder输入前面
```

---

## 第五部分：分阶段实施计划

### Phase 0：环境搭建 + 代码阅读（约 3-4 小时）

| 任务 | 内容 |
|------|------|
| Clone LigandMPNN + 跑通示例推理 | 理解 parse_PDB / featurize / decode |
| 创建 conda 环境 + 安装依赖 | Python 3.10, PyTorch, Biopython, propka 等 |
| 安装 Biopython，验证 charge_at_pH 计算 | ground truth 用于调试可微版本 |
| 评估 RF3 开源代码安装难度和硬件要求 | 如果 5080 跑不动，启用 ESMFold 备选 |
| Git 配置 + 推送初始代码到 ConfuMPNN | `git init` → `git remote add` → `git push` |
| 写 environment.yml 确保可复现 | conda env export |

### Phase 1：Level 1 引导采样（拿到初步结果，约 18-24 小时）

| 模块 | 内容 |
|------|------|
| `differentiable_charge.py` | 可微 Henderson-Hasselbalch 电荷计算 + 单元测试 |
| `isoelectric_point.py` | pI 二分搜索（验证用，不需可微） |
| `structure_aware_filter.py` | 5 条检测规则的实时 logit bias 注入器 |
| 阈值统计 | 从 PDB 采样 1000 条蛋白，确定 99 分位默认阈值 |
| `configs/filter_presets.yaml` | 可配置预设：default / nucleic_acid_binding / membrane / acidic |
| `guided_sampler.py` | 包装 LigandMPNN decoder，整合 charge_lookahead + filter |
| 3-5 个示例蛋白上跑 baseline | 不同 pH + 不同预设场景的对比 |

**Phase 1 里程碑：不改模型代码，纯采样策略实现 pH 感知的电荷约束生成。极端不合理序列被拦截，剩下的宽松处理。**

### Phase 2：条件微调（Level 2）

- 批量构建多 pH 训练标签（连续 pH 采样，Biopython 批量计算）
- 条件向量标准化：从训练集计算每维度的 μ 和 σ，写入 config
- 实现条件编码器（Soft Prompt），注入 LigandMPNN decoder 前缀位置
- A100 上微调 LigandMPNN（全量，262 万参数）
- **复合 loss**：CE_loss + λ_c * charge_deviation + λ_l * structure_penalty + λ_dpo * DPO_aux
  - λ_dpo ≈ 0.01-0.05（极轻权重，ProtAlign 延伸：偏好信号辅助校准）
  - DPO 偏好对数据：用属性预测器对 batch 内序列打分排序，构造 win/lose 对
  - 半在线训练循环（ProtAlign Algorithm 1 迁移）：每迭代轮次用当前模型重新采样→重新打分→重新构造偏好对
- **多约束冲突检测（ProtAlign margin 迁移）**：当 ≥2 条约束规则同时活跃时，对每个候选氨基酸计算"等效 bias" = 对当前约束的收益 - 对其他活跃约束的代价。若净收益为负（冲突），降低该候选的 logit bias 权重。优势：用闭式公式替代 ProtAlign 的 predictor-based reward model，零训练成本、无预测误差。
- 对比实验：原版 LigandMPNN vs logit-bias（Level 1）vs 条件微调（Level 2）vs Level 2 + margin
- **核心消融（ProtAlign Appendix A.2 框架迁移）**：单约束（仅 pH）vs 多约束加权成一个标量（无 margin）vs 多约束带 margin vs 基础模型，四组对照。验证 margin 是否有效防止多目标退化。
- 注入方式对比：Soft Prompt vs FiLM
- **核心实验（Go/No-Go）**：同一 backbone，不同 pH 条件 → 生成不同序列 → property 按预期变化（pH↑ → 偏负电残基增多, pH↓ → 偏正电残基增多）
- 如果没有明显的 pH 依赖响应 → 回到数据/架构调整

> **📎 与第二版衔接：** Phase 2 产出的条件微调模型可直接作为第二版（`index/PROJECT_EXTEND.md`）的起点，在该模型上做多目标可开发性微调（可设计/热稳定/可溶）；第二版微调后的更优模型也会放回本 Phase 的管线中作为默认生成器。

### Phase 3：验证管线

- 搭建 RF3/ESMFold + PypKa 验证管道
- 三层筛选 + 交叉验证
- 方法一致性分析
- **Compositional Generalization 实验**：
  训练集不包含某些 property combination → 测试模型能否泛化到未见组合
- **Baseline 对比**：LigandMPNN 原版 vs SolubleMPNN（如果可接入）vs 本项目（Level 1 / Level 2 / Level 2+margin）
- **可行性边界分析**：识别什么 (X, C) 组合无法生成合理序列

### Phase 4：集成 + 文档

- 整合为 run_pipeline.py 一键运行
- README + 技术报告 + 面试故事线

---

## 第六部分：风险评估与缓解

### 6.1 工程风险

| 风险 | 缓解措施 |
|------|---------|
| RF3 本地部署难度大 / 5080 跑不动 | 备选 ESMFold，代码中做双轨适配 |
| Logit-bias 强度不好调 | 从弱 bias 开始，用 sequence recovery 监控 |
| 条件微调后结构匹配度下降 | CE loss 保留较高权重 |
| 旧代码工具环境不兼容 | Docker 封装 |
| 多 pH 数据增强训练时间翻倍 | 先用 3 个 pH 跑通全流程再扩展 |

### 6.2 物理层面的不稳定性

- **盐桥过密**：已被结构感知过滤器规则 3 捕获（10Å 内 ≥5 对）
- **疏水核心电荷渗入**：分层策略——区分聚集性错误和孤立功能性残基
- **His 的 pH 敏感性**：多 pH 训练数据明确标注每个 pH 下 His 的实际电荷
- **金属配位**：读 LigandMPNN 已有的 metal_coordinated_mask，强/弱金属分类处理
- **翻译后修饰**：边界声明，不在建模范围内

### 6.3 正负电荷交替过滤

规则 3（盐桥过密）已经捕获 K-E-K-E 模式：如果 10Å 内 ≥5 对 K/E 或 R/D 配对（N-O <5Å），触发 bias 抑制进一步的正负电荷配对。不需要额外规则。

### 6.4 你不需要手动创建的计算

以下全部由现成工具覆盖：
- 净电荷计算：Biopython `charge_at_pH`（Henderson-Hasselbalch + 游离 pKa 表）
- pKa 微环境修正：PROPKA / PypKa（自带 Poisson-Boltzmann 经验修正）
- ΔG 折叠自由能：不在本项目范围内
- 溶剂化能：不在本项目范围内

---

## 第七部分：关键决策记录

1. **为什么是 LigandMPNN 而不是 AF3/RF3**：AF3/RF3 的序列是输入不是输出。RF3 定位为验证工具。
2. **为什么自回归比扩散更适合本项目**：pI/电荷是闭式可计算的，自回归精确 lookahead 比扩散的 noise-state classifier guidance 更简单直接。
3. **为什么用 NExT-Mol 而非 MolGPT 的条件注入**：连续 pH 值需要连续向量编码（精度无损），MolGPT 的离散 token 方案精度不够。
4. **为什么不需要改 vocabulary**：LigandMPNN 和 RF3 都不区分质子化状态的 His。pI 约束通过条件编码器 + logit bias 实现。
5. **硬件策略**：5080 全流程，A100 做微调。
6. **输入是 PDB 结构而非序列**：像 LigandMPNN 原生一样，输入 PDB 结构。从序列建结构只在用户只有序列时启用。

---

## 第八部分：项目输出清单

> 本清单对应第一版主线。**第二版拓展**（多目标可开发性微调，MoMPNN 接入/自微调/集成）的输出清单见 `index/PROJECT_EXTEND.md` 第四节 Stage E0–E5。

- [ ] `confumpnn/` — 完整代码仓库（已创建于 GitHub: ConfuMPNN）
  - [ ] `src/differentiable_charge.py` — 可微 pH 感知净电荷计算器
  - [ ] `src/isoelectric_point.py` — pI 查找器（二分搜索）
  - [ ] `src/structure_aware_filter.py` — 结构感知过滤器（5 条规则，YAML 可配置 + 场景预设）
  - [ ] `src/guided_sampler.py` — Logit-bias 引导采样 wrapper
  - [ ] `src/condition_embedding.py` — pH 感知条件编码器（Soft Prompt）
  - [ ] `src/validation_pipeline.py` — RF3/ESMFold + PypKa 验证管线
  - [ ] `src/run_pipeline.py` — 一键运行入口
  - [ ] `configs/filter_presets.yaml` — 过滤器场景预设
  - [ ] `configs/condition_defaults.yaml` — 条件向量默认配置
  - [ ] `notebooks/` — 实验 notebook
  - [ ] `README.md` — 项目文档
  - [ ] `environment.yml` — 可复现环境
- [ ] `docs/technical_report.md` — 技术报告
- [ ] `docs/literature_review.md` — 文献调研
- [ ] `models/` — 微调 checkpoint
- [ ] `results/` — 实验数据