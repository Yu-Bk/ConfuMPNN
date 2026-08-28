# 会话记录 — 2026-08-25/26：项目学习、论文缺口分析与 PROJECT_LOCAL 方案制定

> 落盘：2026-08-27｜对应产出：`index/PROJECT_LOCAL.md`（v3 方案）
> 本文件记录本轮完整思考过程、对话要点与决策依据，供后续追溯（与 GitHub `session/` 体例一致）。

---

## 一、背景

1. 用户要求学习 GitHub `Yu-Bk/ConfuMPNN`（**以网页端为准**；本地 `plan/` git 对应 `AIDD_construction_learning`，与网页端不一致，弃用）。
2. 完整阅读：README / CLAUDE.md / WORKFLOW_GUIDE.md / docs(4) / index(计划+判据) / code(全部 `src/` + 2 主脚本) / analysis/report(19) / literature / session / source / weights_release / data。网页端权威内容克隆至 `%TEMP%\ConfuMPNN_github_ref`。
3. 关键事实（支撑后续决策）：
   - v9 定稿 2026-08-19，停止训练；双编码器（v7 MoMPNN / v9 LigandMPNN）
   - 泛化验证：10 蛋白 × 5 电荷臂 × n30 × ligand/protein 双模式（全部在 pH 7.4）
   - **无 AF2、无 PypKa/PROPKA 实际验证、无 SolubleMPNN 系统对照、无假设检验**（grep 全库核实）
   - 电荷根因：模型"无差别删减带电残基"→ 电荷斑块丢失 + 表面疏水化（GRAVY↑）

---

## 二、对话要点与结论（逐轮）

### 轮 1（2026-08-25）论文缺口分析

结论（6 大缺口，按"致命→加分"）：
1. **致命**：无可比 baseline；电荷物理真实性未验证（PypKa 零实现）；单一验证器 ESMFold；删减捷径副作用未处理；无统计检验
2. **重要**：缺"条件改变分布"的直接证据；泛化单一 pH；缺应用案例；消融未体系化；不可复现
3. **加分**：少量湿实验（表达+可溶+IEX）；与序列级方法轻量对比；机制 figure（删减→斑块丢失→疏水化）
4. 已拥有而未开发的两个论文素材：**方法**（可控逆折叠）+ **机制发现**（模型靠删减而非置换优化组成性质）

### 轮 2（2026-08-25）技术细节 11 问

| 问题 | 结论 |
|------|------|
| 8 个 pH 标签为何？ | 唯一维度是 pH（采样自 [4,10]）；charge/pI 均为 native 序列派生；另一作用是批内（B=8）共享编码省算力。无其他含义 |
| Uniform[1,perturb_scale] 是什么 | 30% 样本 target= native±整数(1..scale)，符号各半；教"target 偏离时如何偏移"，制造 CE vs charge 冲突；占位符样本≠扰动样本 |
| "argmax 附近的具体序列" | 训练优化 softmax 期望电荷 E[Q]，推理是低温采样得到的具体一条序列；分布尖锐时二者差大 → 过冲 2.9×；charge_temp=0.5 锐化 E[Q] 对齐，过冲 2.57→1.04 |
| KL 中 argmax 翻盘 | KL 管分布距离管不住 argmax（P(K)0.30→0.29 时 K→R），seq_keep 直接对无条件 argmax 序列做 CE |
| 权重/扰动比例/epochs 是否调？ | 做成消融而非先验改；0.5→0.3 是修"过度重写"（第十四轮记录）；30 epoch 已收敛，更有用是验证集早停+报曲线 |
| 降 seq_keep 能否缓解删电荷 | 能轻微缓解但代价大（S1/折叠）；推荐"表面添加电荷监督 L_add"独立损失，不动 λ_keep |
| λ_keep=0.5 要调吗 | 保持（用户明确固定）；抬高压缩设计空间、降低伤选择性；作为消融项（0.3/0.5/0.7） |
| 逆密度加权为何关 | 第18轮 cap=5 伤 1UBQ（0/5）；cap=2 平衡；极端正电被 v7 数据+课程学习+温度化根治后边际收益小 |
| --cond_encoder | 是训练好的编码器权重路径，必须显式给；pH/target 通过 CLI 传、编码器自动映射；不给则退回引导采样；建议加"按 backbone 自动选 v7/v9"保险 |
| 冻结 backbone 合理吗 | 合理（保护 DPO 权重+模块化+算力）；容量边界用"冻结 vs LoRA vs 全量"消融量化；v9 backbone 是原版 LigandMPNN（非 DPO） |
| 主证据加 RMSD | 合理且零成本（US-align 已输出）；TM 判拓扑、RMSD 看局部贴合、按域报告 |

### 轮 3（2026-08-26）深度设计 4 问

1. **显式"添加电荷"监督**：
   - 先例：无完全相同做法；组件有据——表面加电荷（supercharging, Lawrence 2007 JACS）、可微引导（Chroma conditioner）、多目标电荷生成（MP2D）、**自有的 Phase1 电荷 lookahead（非可微版本）**
   - 方向：需要更负→表面加 D/E；更正→表面加 K/R；**只在表面**（物理依据：埋藏电荷代价数 kcal/mol，Hendsch & Tidor 1994）
   - 损失：`L_add = |Σ w_i·p_i(D/E) − target_surface|`，`w_i=σ(k·(fracSASA−θ))`；方向需要时启用、以净电荷目标为上限
   - SASA：不在条件向量（粒度不匹配 [L] vs 7 维）；走旁路注入 h_V 或做损失权重
2. **只给 pH=8.5 会向 pI=8.5 靠近吗**：
   - 不会自动；正确输入是 pH=8.5 + target=0（pI=净电荷 0）
   - **发现训练-推理不一致**：训练恒 flag=1，推理不给 target 时 flag=0 从未见过 → 行为不可预测（读代码核实）
3. **可微正损失/融合方式/骨架影响**：
   - 不是"识别团块就跳过"，是概率层面的梯度压力（structure_penalty_loss）
   - 结构特征与条件分两通道：全局条件→prompt；逐残基物理特征→旁路 h_V；在 h_V 内融合
   - 骨架不变动+三锚（CE/KL/seq_keep）控制漂移 → 微型漂移可接受、大幅破坏不可行
4. **能量冲突/只加电荷/验证模块**：
   - 能量冲突本征存在（溶剂化 vs 核心、新增 vs 已有盐桥）→ L_add 与结构惩罚成对
   - 不会"只能加电荷"：方向性+目标上限+表面mask+结构惩罚+三锚
   - 用户提出的"编辑/验证模块"（删残基→重打分；区域设 Gly→模型补全）成立，= mutate→screen + masked redesign（与 P1 Sumida2024 一致）；`--fixed_residues` 已支持区域重设计；物理层面矛盾无法完全逃避（重设计区域仍可能与邻域成簇），验证模块需保留结构过滤+折叠+电荷复算

### 轮 4（2026-08-26/27）方案落盘

- 决策：SASA 用 fractional SASA；选"显式 target 自动补全"而非"训练加 flag=0"；确认只加表面+三条新空间约束
- 产出 `index/PROJECT_LOCAL.md`（本会话记录的配套方案）

---

## 三、决策清单（D1–D12 汇总，详见 PROJECT_LOCAL §2）

| # | 决策 |
|---|------|
| D1 | pH-only 语义用 target 自动补全（=native_charge@pH）；不用 flag=0 训练（历史漂移风险+重训成本） |
| D2 | 治删减捷径用 L_add + 条件解耦 + 结构惩罚（v10），不做全局 λ 调整 |
| D3 | 暴露度指标用 fractional SASA |
| D4 | 三条新空间约束：表面资格硬门槛、核心近邻、pH 自适应带电集合 |
| D5 | Tm/sol/designability 作评估指标，不进条件向量（与 ProtAlign 重复） |
| D6 | RMSD 加入 H1 辅助指标 |
| D7 | 逆密度加权保持默认关 |
| D8 | λ_keep 保持 0.5，作消融项 |
| D9 | 湿实验暂不做；2 条蛋白可选项 |
| D10 | 结构信息走 SASA 旁路（不进 7 维条件向量） |
| D11 | 冻结 backbone 维持；容量消融（冻结/LoRA/全量）写进论文 |
| D12 | 两层架构：生成层 + 编辑/验证层（process_seqs / score_only） |

---

## 四、未解决问题（待用户确认，见 PROJECT_LOCAL §11）

1. 投稿目标（计算/AI 向 vs 生物方法向）——影响 C7 与湿实验优先级
2. v10 定位（主方法 vs 改进型消融）
3. 数据/权重公开策略（数据子集、Docker、新 Release）
4. 湿实验是否列入里程碑（D9 默认不列）

---

## 五、下一步行动建议

1. **先批 PROJECT_LOCAL**（尤其 §11 四个遗留问题的方向）
2. 启动 **P0**：target 自动补全、RMSD 输出、PROPKA 脚本、fractional SASA、pH 自适应过滤器
3. P1 最小必要先跑：C1–C4 + C6、PROPKA 复核（H4）、AF2 子集
4. 跑通后再进入 P2（消融）与 P3（v10），期间可并行整理图表与 Methods 草稿

*本记录随项目推进按需增补。最后更新：2026-08-27。*
