# ConfuMPNN 拓展计划 — 多目标可开发性微调（第二版）

> **第二版计划。** 第一版为 `index/PROJECT_PLAN.md`（pH 感知电荷条件生成主线）。
> （文件名说明：本文件即「拓展计划」，采用正确拼写 `PROJECT_EXTEND.md`；早期口头提到的
> `PROTECT_EXTRACT.md` / `PROTECT_PLAN.md` 为 `PROJECT_*` 的笔误。）
> 本文件是它的**拓展（第二版）**：在原计划框架不变的前提下，通过**模型微调**
> 让生成序列更偏向「**可设计（designable）**、**热稳定（thermostable）**、
> **溶解性好（soluble）**」，再把微调得到的模型**放回原计划管线**，提高序列
> 生成与可用（湿实验可验证）的成功率。
>
> 两级计划关系：
> - **第一版（主线）**：以输入条件（pH/电荷）为导向，靠条件嵌入影响序列生成 → 回答「如何生成符合 pH 约束的序列」
> - **第二版（本文件，拓展）**：把生成模型本身改得更好 → 回答「如何让生成的序列不仅符合约束，还可设计、热稳、可溶」

---

## 一、拓展动机：从「符合条件」到「可用」

第一版（PROJECT_PLAN.md）解决的核心问题是：**用户指定工作 pH，模型自动生成符合该 pH 电荷约束的序列**。管线已经跑通（Phase 1：引导采样 + 结构感知过滤器 + 电荷前瞻）。

但生成序列的**下游可用性**（能否折叠回骨架、是否热稳定、是否可溶、是否表达正常）目前主要依赖生成后的过滤与验证（Phase 3），模型本身并不知道这些目标。这带来两个问题：

1. **成功率低**：纯随机采样 + 事后过滤，大量序列在过滤环节被丢弃。
2. **与可设计性存在潜在冲突**：直接按电荷引导强拉残基，可能破坏序列可折叠性（pLDDT/折叠稳定性下降）。

**拓展目标**：在保持「pH 条件嵌入影响序列生成」这一主框架不变的前提下，**把「可设计性、热稳定性、溶解性」内化为模型的能力**——通过微调（条件嵌入 context 与原计划一致），让模型学会**同时**满足 pH 条件与可用性目标，从而把生成成功率整体推高。

---

## 二、路线选择：优先用文献现成模型（用户决策）

用户的明确偏好（2026-08-16）：

> 更倾向于**直接用文献中提到的微调好的模型**，或者**参考他们的已知可行方法**进行模型微调；如果模型没有开源、或条件嵌入差别过大无法调用，**再考虑自己训练模型参数**。

据此，按优先级给出三条路线，逐级降级：

### 路线 A（首选）：直接用文献已微调好的模型

**候选：MoMPNN（ProtAlign，ICLR 2026）** ✅ 已开源

- 仓库：https://github.com/Qivon7/MoMPNN
- 作者：Junqi Liu, Xiaoyang Hou, Chence Shi, Xin Liu, Zhi Yang, Jian Tang
- 做了什么：在 **ProteinMPNN** 上用**多目标半在线 DPO + 自适应偏好 margin**，同时优化三个目标：**可设计性**（TM/pLDDT）、**溶解度**（Protein-Sol）、**热稳定性**（TemBERTure）。
- **为什么高度契合本拓展**：ProtAlign 优化的正是「可设计 + 溶解度 + 热稳定」三目标——与本拓展的「可设计、热稳定、溶解性好」逐字对应。它能直接回答「用哪套现成方法、哪些开源属性预测器」的问题，避免从零设计奖励函数。
- **需要验证的兼容性风险（开工第一件事）**：
  1. MoMPNN 是 **ProteinMPNN** 变体，而本项目主干是 **LigandMPNN**（多了配体原子上下文）。需确认 MoMPNN 权重能否加载进 LigandMPNN（或反之，我们把它当 backbone 用）。
  2. 本项目特色是「条件嵌入影响生成」（pH 作为条件向量进模型）。MoMPNN **没有条件编码器**——它的「条件」是 DPO 隐式偏好，用户不能显式输入 pH。因此路线 A 的正确打开方式是：**MoMPNN 权重当 backbone / 生成器，pH 条件嵌入（我们已有的 `ConditionEncoder`）叠加在其解码侧**。这正是原计划「条件嵌入 context 不变，只是把基础模型换成更优的微调版」。
  3. 若 MoMPNN 是纯 backbone-only（不处理配体上下文），则对无配体蛋白（本项目目前测试的 1BC8 无配体）可直接用；有配体的场景需评估。

**备用候选**：ResiDPO / EnhancedMPNN（arXiv 2506.00297）——优化「可设计性」（pLDDT，残基级 DPO），**未找到官方代码仓库**，故仅作为方法参考而非直接可调用模型。SolubleMPNN / HyperMPNN 等单体可开发性模型同样可作为 backbone 备选（如能用）。

### 路线 B（次选）：参考文献已知可行方法，自己微调

如果路线 A 的现成权重不可用（如开源权重与 LigandMPNN 不兼容、或无配体上下文支持），则**按 ProtAlign 的整套方法在 LigandMPNN 上自行微调**——直接复用文献验证过的设计：

- **多目标半在线 DPO + 自适应偏好 margin**（ProtAlign 核心）：
  - 离线 DPO 效率 + 在线 rollout 探索折中（半在线：每轮用当前模型 rollout → 属性预测器打分 → 构造 win/lose 偏好对 → 训练 → 下一轮）。
  - 自适应 margin 解决多目标冲突：win 样本在其他属性上反而更差时自动缩小该对 margin，避免单目标过度强化（这正是原计划 Phase 2「多约束冲突检测」的理论来源，拓展中把它落地为训练目标）。
- **顺序无关 log-ratio 估计**（ProtAlign Eq.5）：DPO 从自回归 LLM 迁移到 ProteinMPNN 的关键工程技巧，多条随机重排求共享 log-ratio，降低方差。
- **属性预测器组合**：可设计性用 ESMFold TM/pLDDT（我们的 `confumpnn-esmfold` 环境已有）；溶解度用 Protein-Sol（开源）；热稳定用 TemBERTure（开源）。这些正是文献已验证的「廉价标注器」。

### 路线 C（兜底）：自己设计训练参数

仅当**模型没开源、或条件嵌入差别过大无法调用**时才走。届时在现有 Phase 2 条件编码器 + 复合损失框架内自行构造多目标损失（可设计性用 ESMFold pLDDT 作软标签/偏好对，溶解度与热稳定用开源预测器打分），训练全部参数。此路线工程量最大，优先避免。

---

## 三、与原计划的关系：条件嵌入 context 不变

按用户明确要求——**「条件嵌入 context 和原计划部分一样」**：

- **输入条件仍然通过 `code/src/condition_embedding.py` 的 `ConditionEncoder`（Soft Prompt）注入**，条件向量定义（mask-aware `[7]`：pH + 可选净电荷/局部电荷上限）与 PROJECT_PLAN.md 4.2 完全一致。
- 拓展只改变**基础模型权重**（换成微调过的版本），不改变条件注入机制、引导采样机制（`guided_sampler.py`）、电荷计算（`differentiable_charge.py`）、结构感知过滤器（`structure_aware_filter.py`）。
- **微调后的模型放回原计划管线**：原 Phase 1/2/3 的「PDB → 条件嵌入 → 引导采样 → 候选序列 → RF3/ESMFold 验证」整体不变，仅把生成器替换为更优模型，预期提高序列生成成功率和湿实验可用率。

```
第一版管线（不变）：
  PDB → 骨架/配体上下文 → [ConditionEncoder pH 条件嵌入] → 引导采样(电荷+结构过滤) → 候选序列 → RF3/ESMFold验证
                                            ↑
                                    基础模型 = 第二版微调后的模型（路线 A/B/C）
```

---

## 四、分阶段实施（拓展部分）

> 时间估算按个人经验给出，实际以执行情况为准。所有产出按 `index/FILE_MANAGEMENT.md` 分类存放。

### Stage E0：现成模型可用性调研（0.5–1 天）
- Clone MoMPNN 仓库，检查：
  - 权重格式、是否含配体上下文（`ligand_mpnn` 权重 vs 纯 backbone）；
  - 能否 `load_state_dict` 进我们的 `ProteinMPNN`/`LigandMPNN` 模型类（LigandMPNN 代码里模型类就叫 `ProteinMPNN`，权重结构相近）。
- 确认 MoMPNN 的三目标细节（属性预测器清单、β、margin 权重、训练轮数）能否复用为我们的标注管线。
- 输出：`analysis/2026-08-XX_mompnn_avail.md`，给出路线 A 可行性结论（可用 / 需适配 / 放弃）。

### Stage E1：路线 A 接入（若可行，1–2 天）
- 加载 MoMPNN 权重到 backbone；
- 在其解码侧挂上现有 `ConditionEncoder`（pH → soft prompt tokens）——**条件嵌入 context 不变**；
- 跑通 `run_guided.py --weights <MoMPNN.pt>`，对比原版 LigandMPNN 的序列恢复率与 pH 响应；
- 用 ESMFold pLDDT / Protein-Sol / TemBERTure 对生成序列打分，验证三目标是否优于原版。

### Stage E1b：验证扩展 — 多 PDB × 多 pH × 多 target 混合设计（0.5–1 天）
> E1 对照实验（1BC8，pH7.4，target=0）已验证三目标优势，但**单蛋白单条件**样本不足构成统计置信度。本阶段把验证扩展到 4 个 PDB × 3 pH × 3 target，并把主证据升级为 **TM-score 回折自洽性**。完整设计：`session/2026-08-16_e1_validation_design.md`。

- **PDB 代表性矩阵**（已标注；⚠️ 1BC8 身份修正为 SAP-1 ETS 转录因子 DNA 结合域，非普通球状蛋白）：
  | PDB | 蛋白 | 长度 | 折叠 | 代表性 |
  |------|------|------|------|--------|
  | 1BC8 | SAP-1 ETS 转录因子 DNA 结合域（人） | 93aa | winged HTH（α/β） | 核酸结合蛋白 |
  | 1CRN | Crambin（种子） | 46aa | α+β，3 二硫键 | 极小/疏水/刚性 |
  | 1UBQ | Ubiquitin（人） | 76aa | β-grasp（α+β） | 典型可溶球状 |
  | 2LZM | T4 溶菌酶 | 164aa | 全 α 为主 | 较大/经典工程模型 |
- **混合实验设计（机制 + 泛化分层）**：每个 PDB ① 基线组（生理 pH，target=自然净电荷）→ 回折 **TM-score** vs 原结构（客观结构保持）；② 条件组（pH 4/7.4/9 × target −5/0/+5）→ 电荷偏差/%sol/Tm 梯度。方案 A（同骨架变条件）答机制、方案 B（各蛋白各自条件+对比原结构）答泛化——两者分层互补，缺一归因不清、缺二无法外推。
- **主证据用 TM-score**（ESMFold 回折存结构 → us-align 算 TM），pLDDT 仅作辅助（模型自我置信度，可能被先验欺骗）。
- **位点固定对照臂**：新增 `run_guided.py --fixed_residues`；用 `pka.py` 预检「固定后剩余位点电荷可调区间」是否覆盖 target；作为功能约束再设计场景。
- **阈值防过拟合**：先验设定（pLDDT>80 / |ΔQ|≤0.3 / %sol≥native）+ **留一蛋白（leave-one-protein-out）** 稳定性检查 + 报告原始分布。
- 产出：`analysis/report/2026-08-16_e1_extended.md`；脚本 `code/tests/e1_extended.sh`。

### Stage E2：路线 B 自微调（若需，2–3 天）
- 数据：**骨架/划分用 CATH 4.2 S40（ESM-IF 同源剪枝划分，train/val/test 明确分离、同源分离防泄漏，自带序列/结构/二级结构/sasa）**；条件标签（pH/净电荷）用 `pka.py` + `differentiable_charge.py` 从结构现算；再叠加 PROJECT_PLAN.md Phase 2 的多 pH 连续采样数据增强（`pH = uniform(4.0, 10.0)`）；
- 训练：多目标半在线 DPO（ProtAlign Algorithm 1 迁移）+ 自适应 margin；
- 标注：可设计性 ESMFold pLDDT（`confumpnn-esmfold` 环境）+ Protein-Sol + TemBERTure（皆开源）；
- 注入方式对比：Soft Prompt vs FiLM（沿用原计划 Phase 2 的消融设计）；
- 硬件：A100 单卡足够（参考 P3: 8×4090/20 轮；P2: 2×L40/100k iter，我们的任务量更小）。

### Stage E3：路线 C 兜底（若需，时间未估）
- 仅在 E0 判定现成模型不可用且无法复刻时启用；
- 在 Phase 2 条件微调基础上，把三目标以偏好对/软标签形式并入复合损失。

### Stage E4：集成回原计划 + 对照实验（1–2 天）
- 把微调后模型设为 `run_guided.py` 默认生成器；
- **核心对照实验**（Go/No-Go）：
  - 原版 LigandMPNN vs MoMPNN(路线A) vs 自微调(路线B)；
  - 指标：序列恢复率、ESMFold pLDDT、TM-score、pH 电荷响应曲线（pH 4→10 净电荷单调变化）、溶解度/热稳定代理分、可用率（pLDDT>80 且电荷达标 的比例）。
- 预期结论：微调模型在**保持 pH 响应**的同时，pLDDT/可溶性/热稳定显著提升。

### Stage E5：文档与记录（0.5 天）
- 更新 `session/` 会话记录、`index/DOCUMENT_INDEX.md`、README；
- 微调权重按文件管理规范归档（不 git 跟踪大权重）。

---

## 五、风险与缓解

| 风险 | 缓解 |
|------|------|
| MoMPNN 权重与 LigandMPNN 结构不兼容（无配体上下文 / 层名差异） | E0 先做严格兼容检查；必要时走路线 B 自行微调 |
| MoMPNN 无条件编码器，pH 条件注入需叠加 | 条件嵌入 context 不变，用现有 `ConditionEncoder` 挂在解码侧（这正是原计划设计） |
| DPO 微调导致 pH 条件响应退化（微调覆盖了条件嵌入学到的映射） | 保留条件嵌入 + 电荷偏差 loss（`losses.py` 已有）；对照实验监控 pH 响应曲线 |
| 属性预测器（Protein-Sol/TemBERTure）本地不可跑 | 文献已开源（ProtAlign 用了它们），按需克隆；或退化为闭式代理（GRAVY/电荷聚集） |
| 多目标冲突导致单目标过拟合（ProtAlign 已证明加权退化） | 必须用自适应 margin，不用单标量加权 |
| 微调后序列恢复率下降（AAR 降） | AAR 不是目标（pattern 笔记：AAR 与可设计性相关性弱）；以 pLDDT/可用率为准 |
| 自微调算力不足 | 优先 A；B 用半在线减少 rollout 次数；数据量从 1k 骨架起（P2 证明 1k backbone 即有效） |

---

## 六、可迁移的文献组件（来自 literature 笔记）

本拓展计划直接受益于四篇文献笔记中已验证的组件，清单如下：

| 组件 | 来源 | 用途 |
|------|------|------|
| 多目标半在线 DPO + 自适应 margin | P3 ProtAlign | 路线 B 核心训练范式 |
| 顺序无关 log-ratio 估计 | P3 ProtAlign Eq.5 | DPO 迁移到非自回归模型的关键工程 |
| pLDDT 作可设计性奖励 / 代理 | P2 ResiDPO、P3 | 可设计性标注；P2 证明 1k backbone 数据即有效 |
| 昂贵预测器 → 廉价近似 | P4 CAPE-MPNN（PWM）、P2（pLDDT Accuracy） | 训练期用轻量打分，评估期用完整版 |
| 固定功能位点保留功能 | P1 Sumida2024 | 设计时固定活性/功能位点，避免微调破坏功能 |
| 共享顺序 + 高温 rollout | P3/P2 | 偏好对构造的多样性保障 |

> 详细方法学见 `literature/innovation/*.md` 与 `literature/pattern/README.md`。

---

## 七、与第一版的衔接总结

| 维度 | 第一版（PROJECT_PLAN.md） | 第二版（本文件） |
|------|--------------------------|------------------|
| 解决的问题 | 如何按 pH/电荷条件生成序列 | 如何让生成序列可设计/热稳/可溶 |
| 条件嵌入 | `ConditionEncoder`（Soft Prompt） | **不变**，context 相同 |
| 生成器 | 原版 LigandMPNN | 微调后模型（A/B/C 之一） |
| 引导采样/过滤器/电荷计算 | Phase 1 已实现 | **复用**，不改 |
| 验证 | RF3/ESMFold + PypKa（Phase 3） | 复用 + 增加 pLDDT/溶解度/热稳定打分 |
| 训练 | Phase 2 条件微调（CE+电荷偏差+DPO_aux） | 多目标可开发性微调（叠加或替换） |
| 成功判定 | pH 响应正确 | pH 响应正确 **且** 可用率高 |

**里程碑（拓展部分）**：拿到一个「既保持 pH 电荷响应、又显著提高 ESMFold pLDDT / 溶解度 / 热稳定打分」的生成模型，并集成回 `run_guided.py` 一键跑通。

---

## 八、决策记录（本文件）

1. **优先直接调用文献微调好的模型**（路线 A），首候选 MoMPNN（ProtAlign, ICLR 2026）——因为它优化的三目标与本拓展逐字对应，且已开源（GitHub: Qivon7/MoMPNN）。
2. **条件嵌入 context 与原计划一致**——只换基础模型权重，不换条件注入机制。
3. **现成模型不可用时才自微调**（路线 B），且严格按 ProtAlign 方法（半在线 DPO + margin），不自行发明训练目标。
4. **兜底路线 C**（自己训练参数）仅在模型未开源且无法复刻时启用。
5. 微调产物放回原计划管线，作为后续所有生成的默认生成器。
6. **E1 验证采用「机制 + 泛化」混合设计**：同骨架变条件答机制（方案 A）、各蛋白生理条件 + 回折 TM-score 对比原结构答泛化（方案 B）；**主证据用 TM-score 回折自洽性，pLDDT 仅辅助**（2026-08-16，见 `session/2026-08-16_e1_validation_design.md`）。
7. **Phase 2 训练数据优先用 CATH 4.2 S40（ESM-IF 同源分离划分）**，条件标签（pH/电荷）用自有模块从结构现算，不手工整理（2026-08-16）。
8. **阈值一律先验设定 + 留一蛋白检查，不做阈值搜索**（防过拟合，2026-08-16）。

---

*第二版计划，2026-08-16 建立。与第一版 `PROJECT_PLAN.md` 构成整体。*
