# PROJECT_LOCAL — 论文导向第三版实验方案

> **版本**：v3（代号 `PROJECT_LOCAL`）｜**日期**：2026-08-26 定稿，2026-08-27 落盘｜**状态**：计划（未执行）
> **前两版**：`index/PROJECT_PLAN.md`（v1：pH/电荷条件生成主线）、`index/PROJECT_EXTEND.md`（v2：多目标可开发性微调 / MoMPNN 接入）
> **定位**：以**投稿论文**为目标，在 v7/v9 双编码器定稿（2026-08-19）基础上：解决已知问题、补齐对照、定义消融与统计口径、规划论文章节与图表。
> **依据**：v9 泛化验证（`analysis/report/2026-08-19_v9_generalization_validation.md`）、电荷边界与根因分析（`2026-08-18_model_charge_limits.md`）、`WORKFLOW_GUIDE.md`、与用户 2026-08-25/26 两轮技术讨论。
> **配套会话记录**：`session/2026-08-26_project_local_plan.md`（思考过程、对话与决策的完整记录）。

---

## 0. 现状盘点（方案执行前的资产）

| 类别 | 资产 | 位置/说明 |
|------|------|-----------|
| 模型 | v7 编码器（MoMPNN，无配体/小蛋白）；v9 编码器（LigandMPNN，配体模式） | GitHub Release `preview1.0.0`，SHA256 已核 |
| 代码 | `run_guided.py` / `train_finetune.py` / `code/src/*`（可微电荷、条件编码器、结构过滤、4 项损失） | `code/` |
| 数据 | CATH 7,886 域（v7）；配体 4,972 复合物（v9）；验证 10 + 迁移 10；SHA256 | `data/`（git 不跟踪，NAS 备份） |
| 验证 | 泛化 10×5臂×n30×双模式；电荷边界三区间；E1 四指标对照（MoMPNN vs LigandMPNN） | `analysis/report/` |
| 打分工具 | ESMFold 回折 + US-align TM、Protein-Sol、TemBERTure | 已跑通 |
| 判据 | DESIGN_CRITERIA v2（H1/H2/H3 + S1*–S4） | `index/` |

---

## 1. 已知问题清单与解决策略

| # | 问题 | 证据 | 解决策略（→章节） |
|---|------|------|-------------------|
| P1 | **删减捷径**：模型无差别删减带电残基（非精确置换）→ 电荷斑块丢失 + 表面疏水化（GRAVY↑） | `model_charge_limits §8.2`（1BJ4：105→18 带电残基） | §3.1 **v10 方案**（A 条件解耦 + B 表面添加电荷监督 + C 结构惩罚）——核心方法升级 |
| P2 | **pH-only 输入（flag=0）训练-推理不一致**：训练恒 flag=1，推理 flag=0 从未见过 → 行为不可预测 | `train_finetune.py` vs `run_guided.py` 对照 | §3.2 自动补全 target=native_charge@pH（决策 D1） |
| P3 | 极端电荷边界：v7 正电过冲（+8 仅 40%）、v9 负电欠冲（−8 仅 40%） | 泛化验证 / charge_limits | §3.1（v10 缓和）+ §6.3 边界使用协议（文档化） |
| P4 | 长蛋白（L≥470）/ 血红素类失败 | 泛化验证 | §6.3 可靠性分层表 + limitation；不承诺修复（v10 顺带改善则加分） |
| P5 | 电荷"物理真实性"未验证：全为游离 pKa + HH 自洽 | PypKa/PROPKA 代码零实现（仅文档注释） | §3.3 PROPKA/PypKa 微环境修正复核 → 新判据 **H4** |
| P6 | 单一验证器（ESMFold）；3/10 泛化蛋白 native 回折即不可靠 | 泛化验证 | §3.4 AF2 交叉回折（150–300 条） |
| P7 | 条件化后 sol/tm/designability 是否退化未系统评估 | `phase3_antidrift`（n20：%sol/Tm 真实下降） | §4 对照 C6 + §7 统计（论文"代价"章节） |
| P8 | 结构信息未进入学习；电荷斑块增删不可控、只能手动固定位点 | 无 SASA/区域条件 | §3.5（二阶段）：fractional SASA 旁路 + 区域级电荷条件 + 斑块伪标签 |
| P9 | 统计严谨性缺失（无假设检验/CI/效应量） | 全仓报告均为描述统计 | §7 统一统计口径 |
| P10 | 条件向量 flag 维度归一化 std=0（flag 恒 1 的隐含问题） | `condition_defaults.yaml` | 当前由 D1 回避；若未来加 flag=0 样本需重算 μ/σ |

---

## 2. 决策记录（2026-08-25/26 讨论定稿）

| # | 决策 | 理由 / 依据 |
|---|------|-------------|
| D1 | **pH-only 语义**：采用"显式传 target=native_charge@pH"并做成 `run_guided` 自动补全；**不**采用"训练加 flag=0 样本" | flag=0 语义含糊（保持native？均值？无监督？）；重训两编码器 + 重算归一化；第十六/十七轮"无监督占位符→负漂移→折叠失败"历史风险。flag=0 仅留作远期"部分条件化"（S3 判据） |
| D2 | 治删减捷径：**表面添加电荷监督 L_add + 条件解耦 A + 结构惩罚 C**（v10），不做全局 λ 调整 | 全局调 λ 是"用一个旋钮治一个局部问题"；L_add 直接对抗"只删不加"，且不影响"不必要不动"先验 |
| D3 | **暴露度指标用 fractional SASA**（残基 SASA / Gly-X-Gly 参考），不沿用 10Å 邻居数近似 | 邻居数近似粗糙；fractional SASA 是标准暴露度口径 |
| D4 | 三条新空间约束：①表面资格硬门槛（fracSASA≥θ 才计入 L_add）；②核心近邻约束（加强现有 `core_charge`，新增电荷不得落入埋藏残基 8Å 内）；③pH 自适应带电集合（极端 pH 把 His/Cys/Tyr 按质子化态纳入过滤器带电集） | ①防软权重拉扯；②埋藏电荷代价大（Hendsch & Tidor 1994）；③现有过滤器 pH 无关、只在强带电 K/R/D/E |
| D5 | **Tm/sol/designability**：作为**评估指标**（必需），**不加入条件向量/训练目标** | 与 ProtAlign 路线重复（MoMPNN 已内化 sol/tm）+ 预测器误差；论文叙事=未破坏 backbone 内化性质 |
| D6 | **RMSD 加入主证据辅助指标**（与 TM 联报、按域报告） | US-align 已输出 RMSD，零成本；TM 判拓扑、RMSD 看局部贴合 |
| D7 | **逆密度加权保持默认关闭** | 历史 cap=5 伤中性域（1UBQ 0/5）；cap=2 平衡；瓶颈已被 v7 数据+课程学习+温度化根治 |
| D8 | **λ_keep 保持 0.5**（用户明确固定），作为消融项验证（0.3/0.5/0.7） | 抬高→设计空间被压缩；降低→S1 重写/折叠风险 |
| D9 | **湿实验暂不做**（赶时间），主线纯计算；2 条蛋白（表达+可溶+IEX）作可选加分项 | 计算论文可投；湿实验是"有时间再补"的高杠杆选项 |
| D10 | 结构信息进入方式：**不进 7 维条件向量**（粒度不匹配），用 **SASA 旁路注入 h_V**（不动 backbone）；区域级条件为二阶段 | 条件向量是全局标量，SASA 是逐残基 [L] |
| D11 | 冻结 backbone + 只训编码器维持；论文补"冻结 vs LoRA(解码器后 K 层) vs 全量微调"消融 | 量化"控制力 vs 破坏度"权衡；adapter/LoRA 是高效微调主流 |
| D12 | 两层工具架构：生成层（v10 模型）+ **编辑/验证层**（`process_seqs` / `score_only`；人工删残基→重打分；区域设 Gly→模型重设计；`--fixed_residues` 区域重设计） | 用户提出的后处理=标准的 mutate→screen 循环 + masked redesign（与 P1 Sumida2024 一致） |

---

## 3. 核心方法升级：v10 设计（未训练，先写方案）

### 3.1 治"删减捷径"：A+B+C

**A. 条件解耦（治极端电荷根因）**
- 现状：训练 target = native ± 扰动，导致"骨架类型"与"target 电荷"强耦合（碱性骨架只见正电 target）
- 改法：对任意骨架施加**与自身 native 无关**的随机正/负 target，让"中性骨架 + 高正电"等组合进入训练分布
- 预期：扩大可靠区至约 [native−10, native+10]；仅改数据采样，成本低

**B. 表面添加电荷监督（治"只删不加"）**
- 方向性：需要更负 → 表面位点增加 D/E；需要更正 → 表面位点增加 K/R
- 损失（可微，soft-count）：

```
L_add = | Σ_i w_i · p_i(D/E) − target_surface_count_signed |   （方向需要时启用；换成 K/R 同理）
w_i = σ( k·(fracSASA_i − θ) )                                   # 埋藏位 fracSASA≈0 → w_i≈0
w_i 额外乘以 mask_core：若位置落在埋藏残基 8Å 内 → 0（决策 D4-②）
```

- 以净电荷目标为**上限**（不是无限加）；只加表面；配合 C 防成簇 → 不会"只能加电荷"
- 损失位置权重 = fractional SASA（**决策 D3**），用旁路/预计算载体（见 §3.5）

**C. 结构惩罚（保留并动态加强）**
- 保留 `structure_penalty_loss`，并在"大额添加"的扰动样本上动态加强盐桥/聚集惩罚
- 新增 **pH 自适应带电集合**（决策 D4-③）：按工作 pH 把 His(pKa 6.0)/Cys(8.3)/Tyr(10.1) 纳入过滤器带电残基集

**v10 训练协议（初版）**：沿用 v7/v9 已验证超参（λ_c=0.5, λ_kl=0.05, λ_keep=0.5, charge_temp=0.5, perturb_prob=0.3, placeholder_prob=0.15）；**新增** `--add_supervision`（B，λ_add 待扫 0.1/0.3/0.5）、`--decouple_perturb`（A）、`--ph_aware_filter`（C 增强）。冒烟（50 域）→ 正式 30 epoch → 对照实验 → 消融（§5 A7）。

### 3.2 pH-only 输入自动补全（P2 / D1）

- 改造点：`run_guided.py` 中，当 `--target_charge` 未给出时，**自动填充 `target = net_charge(native_seq, pH)`**（native 电荷本来就要算），不再走 `flag=0`
- 语义："设计一条在该 pH 下保持 native 电荷行为的序列"——完全落在训练分布（自洽样本）内，行为可预测
- 用户想"把 pI 拉到目标值"仍需显式传 `--target_charge 0`（并查电荷边界）
- 验证：新增 `--auto_target_charge` 开关的对照（A9）；回归测试 `tests/`

### 3.3 PROPKA/PypKa 微环境电荷复核（P5）

- 流程：
  1. 生成序列 → ESMFold/AF2 回折结构
  2. PROPKA3（单链）/ PypKa 对每条结构计算**微环境修正 pKa**
  3. 用修正 pKa 重算净电荷 = "物理修正电荷 Q_phys"
  4. 对比"设计电荷 Q_design（游离 pKa）"产出表格：
     `蛋白 | 臂 | target | Q_design 均值 | Q_phys 均值 | ΔQ | 命中(以物理口径)`
- **新判据 H4（物理真实性）**：`|Q_phys 均值 − target| ≤ 2.0` 的臂达标率（阈值与 H2 一致，待定稿）；在论文中明确"设计是组成层面电荷，非微环境精确 pKa 设计"的边界
- 工具接入：`propka`/`pypka` 安装进 `confumpnn` 或独立 env；脚本 `code/tests/propka_charge_check.py`
- 预期风险：部分臂物理口径下"失败"→ 这正是 limitation/边界证据，不掩盖
- （可选 oracle）**CamSol-pH**（Brief. Bioinform. 2023）：pH 依赖可溶性预测器，与 Protein-Sol 互补；可加入验证管线作"pH 环境下的可溶性复核"（见 §11.2 查新定位）

### 3.4 AF2 交叉回折（P6）

- 协议：泛化 10 蛋白 × 3 代表臂（native / −8 / +8）× n=5–10 ≈ **150–300 条**
- 目的：证明"ESMFold 结论在 AF2 下方向一致"（独立第二验证器），不完全替代 ESMFold 全量
- 指标：AF2 pLDDT + 回折结构 vs 参考骨架的 TM/RMSD
- 注意：AF2 输入不配体（apo 回折）；对血红素/金属同为局限 → 写入 Discussion
- 算力：A100 一天内可完成

### 3.5 结构信息进入学习（二阶段，不阻塞主线 P0-P4）

- **SASA 旁路**：逐残基 fractional SASA（FreeSASA/DSSP/prody 预计算，backbone 冻结 → 静态）→ 小投影层加到 h_V（不动 backbone），与 ConditionEncoder 一起训练
- **区域级电荷条件**（用户目标："某个位置形成/规避电荷斑块"）：
  - 条件向量扩展为 per-region：`[全局pH, 全局电荷, 区域1(位置+类型+强度), 区域2(...)]`
  - 保护区域→"不删电荷斑块"；目标区域→"添加/消除"；规避冲突由 `L_place` 学
- **斑块伪标签自动生成**：SASA 过滤表面位点 + Cα 距离聚类 → 每域生成"斑块位置+电荷类型"标签（纯计算，无需人工）
- **数据量估算**：2,000–5,000 域 × 每域 10–20 区域增强 ≈ 2–5 万有效样本（全局电荷不用加数据，见 P3 分析：瓶颈是目标设计非样本量）
- **天然蛋白对（R1）**：CATH superfamily 内"骨架相似、pI/电荷不同"对子作参照/验证（不创新训练主数据）

---

## 4. 对照实验（Baselines）

**统一协议**：测试集（泛化 10 + 扩展 CATH ~50 骨架）× 电荷臂（native / ±2 / ±8）× 多 pH（优先 7.4，扩展 4/5/9/10）× n≥20–30 × 固定 seed；每臂输出：H2 电荷命中（dev≤2.0）、H1 折叠（TM 中位 + 失败率 + **RMSD**）、pLDDT、%sol、Tm、GRAVY、多样性（pairwise identity / 熵）。

| # | 对比项 | 检验的 claim | 现状态 |
|---|--------|--------------|--------|
| C1 | 无条件 backbone（MoMPNN / LigandMPNN，无编码器） | 条件化绝对增益；不可控基线 | 有代码，未同协议跑 |
| C2 | 引导采样路线（Phase 1：电荷 lookahead + 结构过滤） | "推理时引导 vs 条件微调" | 有代码，未与 v7/v9 同协议 |
| C3 | **MoMPNN（DPO-only）vs MoMPNN+v7** | **显式条件 vs 隐式偏好**（核心 claim） | **未做（最高优先）** |
| C4 | LigandMPNN vs LigandMPNN+v9 | 配体模式下的条件化增益 | 有碎片，未成表 |
| C5 | SolubleMPNN / ProteinMPNN（若能接入） | 可溶性专项对照 | 未做（可选） |
| C6 | **条件化 vs 无条件 的 sol/tm/pLDDT** | 条件化的"代价"（P7，论文代价章节） | 部分历史（phase3），需 v7/v9 最终模型系统化 |
| C7 | 序列级条件化（LM-design 等 1–2 个）（投稿计算向时） | 结构条件 vs 序列条件定位 | 未做（可选） |
| C8 | 配体消融：ligand vs 去配体（同一模型） | 配体上下文贡献 | ✅ 已有（v9 §5），复用 |

---

## 5. 消融实验（Ablations）

| # | 消融项 | 具体设置 | 现状 |
|---|--------|---------|------|
| A1 | 损失项 | 全量 vs −KL vs −seq_keep vs −温度化；λ_c∈{0.1,0.3,0.5,0.7}；λ_kl on/off；λ_keep∈{0.3,0.5,0.7} | 碎片有，未规范 |
| A2 | 注入方式 | cross-attention vs FiLM vs concat | 未做 |
| A3 | 数据规模 | 999 / 2,176 / 7,208 / 7,886 域 | 中间产物可重训 |
| A4 | 训练数据策略 | 扰动比例 {0.2,0.3,0.5}；占位符 on/off；课程学习 on/off；逆加权 on/off（cap 2/5） | 碎片有，未成表 |
| A5 | 配体消融 | ligand vs 去配体（复用 C8） | ✅ 已有 |
| A6 | 条件向量维度 | 仅 pH / +charge / +区域条件 | 未做 |
| A7 | **v10 组件** | A（解耦）on/off、B（L_add）on/off、C（结构惩罚加强）on/off——三因子最小设计 | 新方案必做 |
| A8 | charge_temp | 1.0 / 0.5 / 0.3 | 有 1.0 vs 0.5 碎片 |
| A9 | target 自动补全 | on/off（验证 pH-only 语义，配 P2） | 新 |
| A10 | 容量（可选） | 冻结 vs LoRA(解码器后K层) vs 全量微调 | 新（D11） |

---

## 6. 验证与判据升级

### 6.1 H1 升级（TM + RMSD）
- H1a（主）：ESMFold 回折 TM 中位 ≥ 0.70、失败率（TM<0.5）≤ 10%
- H1b（辅，新增 D6）：**backbone RMSD 中位**（阈值待数据定，建议 < 2.5–3.0 Å），按域报告（RMSD 不做长度归一，大蛋白/多域报告需谨慎）

### 6.2 新增 H4 物理真实性（P5）
- 见 §3.3；PROPKA 复核后 Q_phys 命中率作为支撑指标；"失败"即物理边界证据（写明是 limitation 而非模型缺陷）

### 6.3 边界/可靠性分层表 + 使用协议
- 表：结构类（小分子/RNA/DNA/金属配体/血红素）× 规模（L≤312 / 313–469 / ≥470）× 电荷臂（native±2 / −8 / +8）
- 输出"可靠区 / 警告区 / 危险区"（沿用 charge_limits 三区间形式）
- 使用协议（写进 README/论文 §usage）：v7 无配体小蛋白、v9 配体模式；v9 正电到 +8、负电保守 −5；长蛋白需检查

### 6.4 多样性监控（对应 S1*）
- 生成序列两两 identity（防坍塌，<0.8）、每位置熵随 epoch 与条件的变化
- 输出图：跨条件 identity 矩阵 + 每位置氨基酸分布热图（条件改变分布的"直接证据"，P9 相关）

---

## 7. 数据集与统计口径

### 7.1 测试集扩展
- 泛化集：现 10 蛋白（保留）
- 扩展：CATH 非冗余 ~50 骨架，按序列同源性分层（ESM-IF 风格 30% 同源划分），**从训练集显式排除**（复用 `--exclude`）
- 配体测试：现 5 + 扩展（如需要）

### 7.2 多 pH 网格
- 泛化实验现全在 pH=7.4 → 扩至 pH ∈ {4, 5, 7.4, 9, 10}（部分蛋白全网格，其余至少 5/7.4/9）
- 每蛋白输出"pH×电荷臂"响应矩阵（controllability 核心图）

### 7.3 统计检验清单（P9）
| 检验 | 用途 |
|------|------|
| 达标率 ± 95% CI（bootstrap） | H2/H4 每臂 |
| Spearman / Mann-Whitney U（pH 与生成电荷） | 验证"pH 响应单调性" |
| 配对 Wilcoxon / 配对 t（条件化 vs 无条件，同蛋白同臂） | C6 代价显著性 |
| 成对比较（C3/C4：同 backbone ± 编码器） | 显式条件增量显著性 |
| 生成多样性（pairwise identity 均值±CI） | 防坍塌 |

### 7.4 防泄漏说明
- 训练/验证分离（CATH + 配体训练 vs 泛化+扩展验证），`--exclude` 机制 + 清单化写入论文 Methods

---

## 8. 分阶段执行计划

| 阶段 | 内容 | 预计 | 产出 |
|------|------|------|------|
| **P0** | 代码/环境改造：target 自动补全、RMSD 输出、PROPKA 脚本、fractional SASA 计算、pH 自适应过滤器 | 1–2 天 | 改造后 `run_guided.py`/`tests/` 通过 |
| **P1** | 缺口实验：C1–C4、C6 + PROPKA 复核（H4）+ AF2 子集 + 统计脚本 | 3–5 天（A100） | 对照表 + 物理真实性表 + AF2 交叉表 |
| **P2** | 消融 A1–A6 | ~1 周 | 消融全套表 |
| **P3** | v10 训练与验证：A/B/C 组件 + A7–A9 | 1–2 周（含重验证轮） | v10 编码器 + 组件消融 |
| **P4** | 泛化扩样本（~50 骨架）+ 多 pH 网格 + 统计 + core 图表 | ~1 周 | 主图/主表数据 |
| **P5** | 写作 + 复现（Docker/env、数据子集公开、统一评测脚本、超参/seed 表）+ 文档同步 | 1–2 周 | manuscript 初稿 + 复现包 |

**总计**：5–8 周（单人、部分并行）。硬预期按 D9：纯计算路线；湿实验（可选加分项，2 条蛋白表达+可溶+IEX）另行安排，不阻塞主线。

---

## 9. 风险与缓解

| 风险 | 缓解 |
|------|------|
| v10（A/B/C）训练不收敛或电荷控制退化 | 冒烟（50 域）先验；损失按组监控（self/mild/extreme）；不收敛回退 v7/v9 超参 |
| L_add 过冲（添加过头） | 以净电荷目标为上限 + 表面 mask + `L_guard`（带电比例上界，可选） |
| PROPKA 复核后命中率大幅下降（物理口径） | 作为物理边界证据 + limitation；阈值先验设定，不做阈值搜索 |
| ESMFold 与 AF2 结论分歧 | 报告两表；分歧蛋白单列讨论（ESMFold 对血红素已知不可靠） |
| 消融过拟合测试集 | 统一协议 + 固定 seed + 测试集在消融前冻结（阈值先验设定） |
| 算力 | 训练 A100、推理 5080；P1/P2 并行分批 |
| 数据不可完全复现（标签随机性） | 重建说明 + NAS 备份 + 关键文件 SHA256（沿用） |

---

## 10. 论文章节与图表映射（草案）

**故事线**：Controllable pH-Aware Protein Inverse Folding（显式、连续条件控制）+ 机制发现（条件逆折叠模型通过"删减而非置换"逼近电荷目标 → 电荷斑块丢失与表面疏水化的代价）。

| 图表 | 内容 | 对应实验 |
|------|------|---------|
| Fig.1 | 方法框架图（ConditionEncoder→soft prompt→注入 h_V；两条技术路线） | — |
| Fig.2 | Controllability：pH×电荷臂响应矩阵 + 跨条件 identity 热图 | P4 / §6.4 |
| Fig.3 | 泛化（H1/H2）：~50 骨架 × 5 臂 达标率 + TM/RMSD + 失败率 | P4 |
| Fig.4 | 核心对照 C3（MoMPNN vs +v7）与 C4、C6（sol/tm/pLDDT 代价） | P1 |
| Fig.5 | 机制发现：生成 vs native 的 D/E/K/R 组成、GRAVY、电荷斑块计数（1BJ4 病态示例） | P1/P4 |
| Fig.6 | 物理真实性：Q_design vs Q_phys（PROPKA）散点 | P1 |
| Fig.7 | 边界分层表（结构类×规模×臂）+ 使用协议 | P4 |
| Fig.8（可选）| v10 消融：A/B/C 三因子 + 容量消融（冻结/LoRA/全量） | P3/P2 |
| Tab.1 | 对照实验全指标表（C1–C4, C6） | P1 |
| Tab.2 | 消融总表（A1–A9） | P2/P3 |
| Tab.3 | 复现信息（数据划分、seed、超参、权重 Release） | P5 |

---

## 11. 同行查新（2026-08-28）：附近工作与区分

> 执行：`nature-academic-search` 技能；MCP 学术源未挂载 → 按技能 fallback 走 **OpenAlex**（覆盖 CrossRef / PubMed / arXiv / bioRxiv）；3 轮 30 组关键词，命中 400+、去重后筛读约 280 篇；范围 2021-2026。
> 会话记录：`session/2026-08-28_novelty_audit.md`。

### 11.1 结论

- **未发现直接冲突**：`pH-conditioned protein generation`、`charge-conditioned sequence diffusion protein` 等精确组合均 0 命中；无任何工作以"**连续 pH/净电荷为显式条件 + 结构逆折叠（LigandMPNN/ProteinMPNN 系）+ 可微 HH 电荷目标**"做序列生成。
- 核心组合在本检索范围内**未被占据**；同时全文核实 **DynamicMPNN = 多构象条件**（22 页正文 0 处 charge/pH/electrostatics，已读全文）。
- **局限提醒**：OpenAlex 对极新 2026 预印本可能滞后，不含 Google Scholar / 中文库；仍需补 arXiv/bioRxiv 页面级检索与最近邻全文精读（见 §11.4）。

### 11.2 最近邻必须区分（论文 Related Work 区分表）

| 最近邻 | 年份/出处 | 它做什么 | 与你的本质不同 |
|---|---|---|---|
| **SurfPro** | 2024, arXiv 2405.06693 | 给定目标表面 + 表面生化性质条件化生成序列（层级表面编码器 + 自回归解码） | 条件是"表面几何/生化性质场"（非 pH/净电荷标量）；非 LigandMPNN 微调；无 HH 电荷目标 |
| **SurfDesign / SurfFold / BC-Design** | 2025-2026 | 表面/生化感知逆折叠（表面法向、曲率、生化特征并入模型提精度） | 把表面生化特征当**输入特征**提精度，非"按 pH/电荷条件生成"；**需精读 BC-Design，确认其 biochemistry-aware 是否含电荷特征** |
| **Controllable protein design with LMs** | 2022, Nat. Mach. Intell. | 属性标签控制蛋白语言模型 | 序列空间 + 离散标签；无结构、无连续 pH/电荷 |
| **Regression Transformer** | 2023, Nat. Mach. Intell. | 连续性质回归作为条件序列生成（分子/蛋白 LM） | 序列空间连续性质条件化范式；不基于结构逆折叠 |
| **Integrative multiobjective (NSGA-II)** | 2024, PLoS Comput. Biol. | 演化多目标优化整合多个目标做序列设计 | 优化/筛选框架，非模型内条件编码器；无 pH/电荷特化 |
| **LaMBO-2 / Guided Discrete Diffusion** | 2023, arXiv 2305.20009 | 离散扩散 + 性质引导 | 纯序列扩散 + guidance（已列为 baseline，C 组） |
| **Electrostatics as a Guiding Principle（酶设计）** | 2024, JCTC | 静电计算引导酶设计（物理/能量法） | 物理计算引导，非学习型条件；作"静电引导"先例引用 |
| **pH-responsive filaments / antibody nanoparticles** | 2024, Nat. Nanotechnol. / NSMB | 设计**对 pH 响应**的组装/解离结构（埋藏 His 触发） | 任务是"pH 触发结构转变"（开关/传感器）；**不是**"固定工作 pH 下的电荷/组成控制"——审稿人最易混淆项，需专门区分段 |
| **CamSol pH-dependent solubility** | 2023, Brief. Bioinform. | pH 依赖可溶性的**预测器** | 非生成器；建议作为验证 oracle（已入 §3.3 可选） |
| **ABACUS-T / MapDiff / DynamicMPNN** | 2024-2026 | 多模态条件逆折叠（原子+多状态+进化）/ mask 先验扩散 / 多构象 | 条件分别是原子上下文/结构置信度/构象集合——均无理化标量条件 |

### 11.3 对方案与论文定位的影响

1. "首个 pH/电荷条件化逆折叠"**可主张，但必须加限定词**：*首个在配体感知结构逆折叠上，以连续 pH + 净电荷为显式条件、并用可微 HH 电荷目标训练的方法*——这样 SurfPro/SurfDesign（表面场条件）、RT/可控 LM（序列空间）、NSGA-II（优化框架）都无法一句话击穿。
2. **护城河 = 机制发现（删减捷径→电荷斑块丢失→表面疏水化）+ v10 表面可控制电荷设计**——检索未发现同类机制研究，其与所有条件生成论文均不重叠，是最抗打的贡献。
3. 审稿人最可能引用的混淆项 = **pH-responsive filaments / antibody nanoparticles**（2024 顶刊，高影响力）：Related Work 需专门一段讲清"pH 触发结构开关 vs 固定工作 pH 的电荷/组成控制"。

### 11.4 新增行动项

- [ ] 精读 **SurfPro / SurfDesign / BC-Design** 全文（重点：BC-Design 是否含电荷/静电输入特征），结论写入 `literature/`
- [ ] 补 arXiv/bioRxiv **页面级**关键词检索（OpenAlex 对 2026 预印本可能滞后），覆盖 `charge-conditioned / pH-conditioned inverse folding` 2025-2026
- [ ] CamSol-pH 接入验证管线（可选 oracle，§3.3）
- [ ] 按 §11.2 表格撰写论文 Related Work 段落

---

## 12. v10 泛化失败归因与 v11 修复计划（2026-08-28）

> 关联提交：`d3d2ec3`（v10 泛化验证完成 + 发现电荷控制退化）、`9df9cf7`（NaN 复检 + stats 入版本）。
> 交付脚本：`v10_repair/`（诊断脚本 v10_diag_response_curve.py、train_finetune v11 补丁、README）。
> **⚠️ 2026-08-28 诊断已跑（17 蛋白响应曲线），结论修正如下**：
> 外推假说**未坐实**——训练覆盖区内（native±12，n=17）slope 已 **1.59±0.57**；负区外 1.59±0.65，
> 正区外 **1.10±0.34**（正target正常、负target超线性过冲）。真正的机制 = **负向响应增益失控**：
> "删 K/R 捷径（P1 根因）"与 **B（L_add 表面加 D/E）同向双算 → 净效果≈2×Δ**；decouple 弱化 native
> 锚 → 自洽 native 点也过冲（7pujA01 dev 12、1A65 dev 26）；深负外推只是放大器。已确认 MoMPNN
> 训练为 **A+B+C 三组件全开**（log `log/v10_train_mompnn.log` + 管线脚本），验证报告 §5"只改
> decouple"归因**不正确**。损坏并非全局（2d3yA00/1C6O 近乎完美）→ 修 B 有望直接修复。

### 12.1 数据事实（130 臂全真实，来自 stats JSON）

- **v10-MoMPNN 崩在负目标域**：每蛋白拟合 `生成 = a·target + b`，a∈[0.9,2.4]（1A65 2.09 / 1AXW 2.11 / 1BJ4 2.37），b 全部为负（−2.4~−5.7）；负域比值 1.9~2.3、n8 命中 **0/28**、neg 2..8 命中 1/12；正域基本正常（pos 2..8 命中 3/4）；折叠未受损（TM 正常甚至略升）。
- **v10-Ligand 未崩**：native 4/10 vs v9 7/10、p2 8/10（改善）、n8 持平；GRAVY/recovery 与 v9 几乎一致（0.295/0.470 vs 0.281/0.475）。
- **关键判别**：两个模型用完全相同的 A+B+C，结果天差地别 → 组件本身不是祸首，**训练数据对 target 值域的覆盖**才是分水岭。

### 12.2 根因（按证据强度）

1. **主因（目标外推）**：CATH（MoMPNN）域 native 多集中在 [−15,+15]，±12 相对扰动的 target 密度 ≥ −27 且中心在 −10~+10；验证集大蛋白 target 落在 **−19~−35** → 编码器输入（归一化后 ≤ −3σ）超出训练区间 → GELU-MLP 外推段斜率≈2 → "生成≈2×target + 负偏置"。Ligand 训练数据天然含大而负复合物 → 覆盖该靶区 → 不外推 → 不崩。
2. **次因（B 计数语义叠加）**：L_add 目标 = "表面新增 |Δ| 个电荷"（计数），与电荷损失"净电荷=target"语义叠加，且模型既有"删减捷径"仍活跃 → 净效果≈2Δ（负向可删的 K/R 充裕；正向可删的 D/E 少 → 正向不放大）。
3. **辅因（C 横批）**：boost=1.5 在约 94% 批次（批内任一样本扰动）对全批（含自洽样本）生效，且整批共用 pH_b[0]（潜在 bug）→ 轻微推负。
4. **已排除**：推理期归一化错配（checkpoint 自带训练时 μ/σ，已核实 load_condition_encoder）；欠拟合（训练 charge 终点 2.05/2.82，训练域内实为 identity——是外推而非没学会）。

> ⚠️ 待闭环：v10 泛化报告称"MoMPNN 唯一改动是 decouple"，与 `run_v10_pipeline.sh` 阶段 3.5 的三组件命令矛盾；**需从训练机 push `log/v10_train_mompnn.log` + `_prog.json` 定论 MoMPNN 实际开了哪些 flag**，否则归因不可信。

### 12.3 v11 修复计划（三步，均带判据）

**步骤 1（零成本，先于一切）：响应曲线诊断** — `v10_repair/v10_diag_response_curve.py`
- 用现有 v10 checkpoint，在"训练域蛋白（CATH 8 个）"与"验证域蛋白（10 个）"上扫 target∈[−34,18]，输出每蛋白 slope/int/r²。
- 判据：训练域 slope≈1 且验证域 slope≈1.8~2.4 → **外推坐实**，直接上 v11 重训；训练域也≈2 → 先只开 A-fix 再逐个加 B/C；都≈1 → 停止、复查（勿重训）。

**✅ 步骤 1 已执行（2026-08-28）结果**：外推未坐实（区内 slope 1.59±0.57）；**改为 v11a B-OFF 消融优先**（见下）。

**步骤 2：train_finetune v11 补丁**（`v10_repair/train_finetune_v11_patch.md`，三处搜索-替换）
- **A-fix（主修复）**：新增 `--decouple_absolute`——绝对 target ∈ Uniform[−35, +20]，直接覆盖验证靶区，与 native 无关；保留 offset=target−native 语义供 B/分组监控。
- **B-fix**：`--add_target_scale`（建议 0.5）+ `--lambda_add 0.1`（降权）——L_add 半量，避免与删减叠加双算。
- **C-fix**：结构惩罚改**逐样本 boost + 逐样本 pH**（修横批自洽样本、共用 pH_b[0] 两个问题）。
- 流程：语法检查 → 冒烟 50 域 → **按实验矩阵 v11a(B-OFF)→v11b(A-fix)→v11c(全fix)**（一次只动一个变量）。

**步骤 3：闭环** — 用同一诊断脚本在 v11 checkpoint 上重跑（**验证域 slope 回到 ≈1、|b|<1**），再跑完整泛化验证 + 对照 C1/C3。若 slope 仍 >1.3 → **A7 三因子消融**（仅A / 仅B / A+B+C 各 30 epoch，一次只加一个组件——v10 的教训是三组件耦合无法归因）。

**验证指标补强（P1 治疗证据，当前缺失）**：泛化验证必须新增 §6.4 的 H3 成簇违规率 +"生成 vs native 的带电残基总数 / GRAVY Δ / 表面带电占比 / 电荷斑块计数"——否则无法证明 B 修好了"只删不加"、C 防住了成簇。

### 12.4 论文章节决策（并行准备）

- **止血**：主方法回退 v7/v9（已验证可用），v10 作为"失败分析 + 消融证据"章节。
- **若 v11 成功**：v11 作主方法；v10 失败仍作为机制/negative 发现写进 Discussion——"条件化逆折叠的电荷控制受训练 target 值域覆盖约束，目标区间外推会导致增益≠1 的系统标定失调（负域尤甚）"（配步骤 1 的响应曲线图），与 §10 原"机制发现"主线衔接。

---

## 13. 遗留问题（待用户确认）

1. **投稿目标**：计算/AI 向（会议或 ML 子刊）vs 生物计算方法向（Nat Commun / JACS / Protein Science / Structure 等）——决定 C7、湿实验优先级与写法。**§11 查新已确认无直接冲突**，两类可任选；核心 claim 措辞建议按 §11.3.1 限定
2. **v10 定位**：作为论文主方法（重训替换 v7/v9）vs 作为"改进型消融"（v7/v9 为主方法，v10 为机制/改进章节）
3. **公开策略**：是否公开数据子集（~500 域）与新 Release；代码是否补 Docker
4. **湿实验**：是否列入里程碑（D9 默认不列；若目标高影响刊，建议 2 条蛋白可选做）

---

## 附录 A：与 v1/v2 衔接对照

| 维度 | v1（PROJECT_PLAN） | v2（PROJECT_EXTEND） | v3（本文件） |
|------|--------------------|---------------------|-------------|
| 使命 | pH/电荷条件生成主线 | 多目标可开发性微调（MoMPNN 接入） | **论文导向**：补齐验证/对照/消融/统计 |
| 生成器 | LigandMPNN | +MoMPNN（DPO） | v7/v9 双编码器 + v10（改进） |
| 条件注入 | ConditionEncoder（soft prompt） | 不变 | 不变 + SASA 旁路（二阶段） |
| 验证 | RF3/ESMFold + PypKa（计划） | 复用 | +AF2 交叉 + PROPKA 复核（H4）+ RMSD |
| 新增机制 | — | — | v10（A 解耦 + B 表面添加监督 + C 结构惩罚）、target 自动补全、区域级条件（二阶段） |
| 成功判定 | pH 响应正确 | pH 响应正确且可用率高 | v2 + 统计显著 + 物理真实性 + 边界明确 |

---

*第三版方案，2026-08-26 讨论定稿，2026-08-27 落盘，2026-08-28 更新（同行查新 §11）。配套会话记录：`session/2026-08-26_project_local_plan.md`、`session/2026-08-28_novelty_audit.md`。与 v1/v2 共同构成 ConfuMPNN 整体规划。*
