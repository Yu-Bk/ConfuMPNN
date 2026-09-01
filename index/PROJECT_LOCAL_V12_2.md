# PROJECT_LOCAL_V12_2 — v12.2 深化训练 + 完整验证计划（2026-08-30）

> **状态**：**v12.2 完整验证链已完成并收尾（2026-08-31）**——训练→诊断（slope 1.00 达标）→组成→hold-out（40.6%）→泛化 per-protein（72%）→小样本现场标定（74%）→无泄露（44%）→Tm/Sol（S2 0/50 恶化）→无过拟合。**v9 迁移待定（用户暂缓，2026-08-31）**。详见 `analysis/report/2026-08-31_v12_2_{training,diag,tm_sol,summary}.md`。
> **关联**：v3 方案 `index/PROJECT_LOCAL.md`；v12/v12.1 训练 `session/2026-08-29_v11_ablation.md` §9-§13；闭环+泛化报告 `analysis/report/2026-08-30_v12_1_validation.md`。

---

## 0. 背景（为什么做 v12.2）

v12.1 + n50 校准已达成：
- slope 1.04（valid 区内达标）、组成健康（D/K 收敛 native）、H1 折叠 8/10 健康蛋白全过、负向 n8 8/10
- 剩余短板：① 正向电荷 p2/p8 偏弱（校准参数小样本脆弱性已用 n50 部分修复，2FEO seed2000 dev 3.75→2.03）；② 1AXW 双峰分布无法校准；③ 长蛋白 1A65 欠冲

**v12.2 目标**：训练侧深化治正向电荷弱（上界锚定），同时补齐验证体系（同分布 hold-out + 物理性质 Tm/Sol）。

---

## 1. v12.2 训练深化设计

**在 v12.1 基础上加第三个监督（单一变量）**：
- 启用预留的 `surface_charge_target_loss`（v12_losses.py 已有）：锚定**表面净电荷 = target − 核心 native 电荷**（核心锁死，表面承担全部电荷变化）
- 与 v12.1 严格单一变量：保留 frac_floor 0.5 / gravy_margin 0.4 / λ_v12 0.2，新增 λ_target 待定（0.1-0.3）
- 目的：正向电荷化时给"加 K/R 的量"一个 target 上界锚，治欠冲/过冲不对称

**训练前检查**（启动长任务前必须）：计划 / 数据（labels_balanced_v7 完整）/ 环境（GPU 空闲）/ dry-run 冒烟（50 域 1ep 确认 λ_target 生效无 NaN）。

---

## 2. 验证计划（本轮核心新增，用户批准 2026-08-30）

### 2.1 同分布 hold-out 验证集（15%，从训练集分层划出）——**最终方案（2026-08-30 用户定）**

从训练集 **7,886 域按电荷分层划 15%**（非剩余域），保证验证集与训练集均值/方差一致。

- **为何不从剩余 27,445 域抽**：实测剩余域天然偏负（charge@7.4 最高 ~+8，>+5 仅 ~1 个），训练集正电尾（+8~+45，1,211 域）来自外部碱性域——**剩余域无法匹配训练集分布**（dry-run 三次实测：等量分层 mean 差 −3.4；可达范围匹配后最高两箱候选仍为 0 → mean 差 −2.8，均不达标）→ 用户判断"全新数据方差均值差距过大"，弃剩余域方案
- **分层划分**：训练集 charge@7.4 8 分位等频箱，每箱划 15%（147 域/箱）→ **hold-out 1,176 域（14.9%）+ train85 6,710 域**
- **分布实测**：hold-out mean=0.05/std=8.33、train85 mean=0.01/std=8.30、完整 0.02/8.30——**mean 差 0.04、std 差 0.03 完美匹配**，验证集含正电尾（+45）
- **产物**：`data/cath/labels_holdout_train.npz`（1,176 域 × 8 pH）、`data/cath/labels_v12_2_train.npz`（6,710 域）
- **⚠️ 对 v12/v12.1 该验证集是"见过域"（它们训了全部 7,886）；仅对 v12.2 是未见的（只训 85%）**——这正是本轮划出的意义
- **评估**：v12.2 + 校准 → native 电荷命中（H2）、recovery、可选折叠
- 脚本：`code/tests/build_holdout_split.py`（已跑，产物已生成）；`code/tests/validate_holdout.py`（评估，待写）
- 剩余域方案脚本 `build_holdout_labels.py` 保留但**停用**（dry-run 已证不可行，作失败记录）

### 2.2 验证蛋白物理性质补充（在现有泛化方案上统一）

已有：电荷 H2、PROPKA H4、ESMFold 折叠 H1（TM/pLDDT/RMSD）、GRAVY、recovery。
**补充**（对泛化 10 蛋白的生成序列，对比 native + 无条件基线）：
- **Tm 热稳定性**：`code/tests/temberture_score.py`（已存在，E1b 用过）
- **Sol 溶解性**：Protein-Sol 工具链（E1b 时代用过，需确认脚本/工具位置）
- ESMFold / propka 已有，纳入统一流程

### 2.3 前几版补充验证（先看 v12.2 结果，再定）

若 v12.2 验证体系建立，对 **v7 / v9 / v10 / v11** 补跑 2.1+2.2（hold-out 评估 + Tm/Sol），看能否得到新结论。
- **补充直接在对应版本的脚本和实验数据分析报告中**（`code/tests/` + `analysis/report/`），不新建分支

### 2.4 H3 电荷聚集合法性（2026-09-01 采纳，方案见 `PROJECT_SUPPLEMENT_H3_REVIEW.md` §1）

**背景**：H3 是 DESIGN_CRITERIA 四大硬判据**唯一空置**的——"条件臂序列在结构过滤器规则下违规率 ≤ 基线 + 5pp"（证明电荷条件化不产生物理不可能的电荷布局，审稿人必问）。

**执行方案（复用现有规则，零新增）**：
- **脚本**：`code/tests/h3_charge_legality.py`（新写）——把 `structure_aware_filter.py` 的 4 条规则（charge_cluster / salt_bridge / core_charge / same_sign_cluster）改成**事后统计**模式（全序列已解码，只统计不干预）
- **输入**：泛化生成 PDB（`output/generalization_*/`）+ 基线① native_ref + 基线② 无条件基线（**必须用训练均值占位 net_charge=1.4243**，`None` 会 poly-G 退化）
- **方法**：坐标=采样用骨干 PDB（Cα 距离矩阵），逐规则统计"触发位置数/总长"，4 规则并集去重，按 pH 分臂（`pH_adaptive_charged_aa`），输出每蛋白每臂违规率表
- **判据**：条件臂违规率 ≤ max(native, 无条件) + 5pp → PASS（先取证后定标）
- **覆盖**：**mompnn + ligand 两条线**都要跑（配体删减捷径下 H3 尤其重要——验证"成对删"是否产生电荷聚集）
- **额外价值**：顺带回答"`--preset` 解码时 structure_filter 引导是否真约束住最终序列"（事后违规率高 → `--strength` 不足）

---

## 3. 执行顺序（今晚流程不变，仅扩展验证部分）

1. **新表泛化**（进行中，`run_v12_1_validation.sh` 已换 n50 表）→ 结果出来后分析、写报告、更新 memory/session
2. **hold-out 验证集构建**（2.1：脚本 + 分层抽 + 标签 + 评估）
3. **v12.2 训练**（1：训练前检查 → 训练 ~5h）
4. **v12.2 泛化验证**（复用 run_v12_1_validation.sh 换 checkpoint）
5. **Tm/Sol 补充**（2.2）
6. **前几版补充验证**（2.3，视结果）
7. **v9 迁移评估**（若 v12.2 效果更好 → LigandMPNN 重训 + 配体校准表）

## 4. 判据

- v12.2 slope（valid 区内）+ 校准后 ∈ [0.9, 1.15]
- 正向 p2/p8 达标率较 v12.1 提升（3/10 → ≥5/10）
- hold-out 验证：native 电荷命中率 ≥ 训练分布内基准
- 物理性质：Tm/%sol 与 native/无条件基线对比无明显恶化（S2 判据，报告绝对值）

## 5. 产物

- 计划：本文档
- hold-out 标签：`data/cath/labels_holdout.npz`
- 评估输出：`output/holdout_eval.json`
- Tm/Sol：`output/tm_sol_v12_1/`
- 报告：`analysis/report/2026-08-30_v12_2_*.md`（训练 + 验证）

## 6. 训练后自动执行流程（用户授权 2026-08-30 自动执行）

用户明确授权：**训练完成后判断模型训练正常 → 若正常自动写分析报告+项目记录对话记录 → 完成后自动开始验证和诊断**（无需逐项等确认）。

### A. 训练完成判断（收到监控通知时，先判断再汇报）
- 训练 log：epoch 30 是否完成、有无 NaN、cd self/mild/extreme 分组监控值
- 对照 v12.1 基准：total 3.66 / ce 1.90 / charge 2.52 / keep 0.90，cd mild 2.55 / extreme 3.06
- checkpoint 确认：`output/finetune_v12_2/` 存在完整 epoch 权重
- **异常（NaN / 崩溃 / 损失不收敛）→ 立即停下，汇报用户决策，不进入 B/C**

### B. 报告 + 记录（自动，训练正常时）
1. `analysis/report/2026-08-30_v12_2_training.md`：配置、收敛曲线、vs v12.1 对照、λ_target 效果
2. `session/` 追加对话记录
3. memory `confumpnn-project-status.md` 更新
4. git add + commit + push（**排除训练中文件**）

### C. 自动开始验证和诊断（自动，对齐 §3 执行顺序 + §4 判据）
1. **17 蛋白响应诊断**（valid 区 slope 判据 [0.9,1.15]）
2. **组成分析**（D/K 计数 vs native，治过度添加是否复发）
3. **hold-out 评估**（`validate_holdout.py` 待写，在 `labels_holdout_train.npz` 上 H2/recovery——对 v12.2 是真正未见数据）
4. **泛化验证**（复用 `run_v12_1_validation.sh` 换 checkpoint，ESMFold 回折 TM/pLDDT/RMSD + GRAVY + PROPKA H4）
5. **Tm/Sol 补充**（对齐 §2.2：现有泛化 10 蛋白 + native + 无条件基线）
6. **v9 迁移评估**（对齐 §3.7 + `PROJECT_V9_LIGAND_PLAN.md`）：若 v12.2 效果好 → LigandMPNN 重训 `--ligand --v12_supervision` + λ_target 配体适配 + 配体模式 SASA/组成适配 + v9 配体诊断校准表 → 泛化复验
6.5. **H3 电荷聚集合法性**（§2.4，`h3_charge_legality.py`）——**mompnn + ligand 两条线**都跑：泛化生成 PDB vs native_ref vs 无条件基线，4 规则事后统计违规率，判据 ≤ 基线+5pp
7. 全部结果汇总 → 写验证报告 → 汇报用户
8. **若结果需方向决策（不达标/新短板）→ 暂停，等用户决策**（此点不在自动授权内）

### D. 保存纪律（用户要求 2026-08-30："过程中及时保存分析报告/项目进度/对话记录/下一步计划"）
- **每完成一个阶段（A/B/C 每步）立即保存**：写/追加 `analysis/report/` 报告、`session/` 对话记录、memory `confumpnn-project-status.md`、本计划文档 → `git commit + push`
- **不攒到最后一起写**——任何中断（含重大错误停下汇报）时，已完成阶段的记录都在 git 里
- 训练/验证日志等大文件产物不 commit（`output/` 已 gitignore；`log/` 按需，训练进行中的 log 不提交）

### E. 扩大校准域 + 无泄露补跑（用户批准 2026-08-31）

**背景**：① 校准表只用 17 蛋白（204 点）拟合 global，结构域太少（用户质疑）；② per-protein 校准的泛化验证（H2 72%）对评估蛋白有"响应信息泄漏"（per-protein 来自该蛋白自身诊断）——需无泄露口径。

**方案**：
1. **批量校准建表**：训练集 `labels_v12_2_train.npz`（6,710 域，v12.2 真训练域）抽样 100 域 × 5 target（native±[8,4,0,4,8]）× n10 → 500 个 (target, 生成电荷均值) 点 → 拟合 **global 校准表 `charge_calibration_v12_2_big.json`**（数据量 204→500 点，来自 100 个不同域）。**训练域模型见过且不在评估集 → 天然无泄露**（⚠️ 不可用 `labels_balanced_v7` 全量——含 hold-out 1176 域）
2. **无泄露泛化 H2 补跑**：big 表 `--calibrate global` 重采样 10 蛋白 5 臂 n30，只算电荷（不重跑 ESMFold/TM——折叠与校准无关）
3. **三口径对比**：per-protein 72%（已标定参照）/ big-global（无泄露）/ hold-out 40.6%（完全未见域）
4. 写报告 → git 提交

**判据**：无泄露泛化 H2 应显著优于 hold-out（40.6%）；若 big-global ≈ hold-out → 确认 global 校准对未见蛋白的固有局限（论文如实报告）

**脚本**：`code/tests/build_calibration_big.py`（新建）+ `validate_generalization.py --calibrate global` 复用

---

## 7. 配体模式删减根治设计（A1+A2+keep/free 开关，2026-09-01）

> **背景**：配体泛化 H2 72% 达标但组成系统性删减（8/10 蛋白 0.53-0.65×，定向配体口袋），根因 =
> 监督逃逸×配体疏水先验×v12 微调放大（`2026-09-01_v12_2_ligand_comp_analysis.md`）。
> 口袋 fix 是推理侧补丁（钉死残基），不治本且造成 2FEO 电荷失配。本节为**训练侧根治设计**。
> **执行决策**：取决于配体 Tm/Sol 物化验证——无恶化 → 降级为"论文如实报告 + fix 缓解"；
> 恶化 → 按本节重训。设计本身不受结果影响。

### 7.1 核心思想：三块互斥残基分区（解决 pocket vs core 矛盾 bug）

**v12 现状的矛盾点**：`surface_charge_target_loss` 用 `core_mask=(~surf)` 锁死核心
（q_core 用 native one-hot 算，不可微）。而深部口袋（frac_sasa<0.25）被划入"核心"。
若 A2 直接把口袋并入表面 mask，同一残基既在 q_core（**native 值**、锁死）又在 q_surf
（**生成值**、监督）→ **双算 → 模型改了口袋残基时总电荷 drift 且监督看不见**（矛盾 bug）。

**解法：残基空间三块互斥，每残基只属一块**：

| 分区 | 定义 | 行为 |
|------|------|------|
| **core（锁死）** | frac_sasa<0.25 **且** 距配体≥8Å | 保持 v12 现状：q_core = native one-hot 锁死 |
| **pocket（温和改）** | 距配体<8Å（**无论 frac_sasa**） | 纳入监督视野：净电荷锚 + A1 双向计数 |
| **surface（温和改）** | frac_sasa≥0.25 且非口袋 | 现状表面监督 |

三块无重叠 → q_core 不再含任何口袋残基 → pocket 残基的生成电荷全部进入 q_surf 监督
→ 总电荷恒 = target（无 drift、无双算）。

### 7.2 "温和更改"的量化定义（保量不保位，≠fix）

pocket 残基**不是逐残基钉死**（那是 fix），而是**总量约束 + 放行具体位置**：

1. **净电荷**：pocket∪surface 生成净电荷锚到 `target − q_core`（`surface_charge_target_loss`
   mask 从 surface 扩展为 surface∪pocket）。
2. **总数（A1 双向计数，防成对删 + 防成对加）**：
   `pocket_count_loss = relu(N_p·floor − gen) + relu(gen − N_p·ceil)`，D/E 与 K/R 双计数。
   - floor ≈ 0.7（堵配体删减 0.53-0.65 触发）
   - ceil ≈ 1.3（防成对加——**v12 只设下限无上限 → 过度添加 1.5-2× 的教训**）
3. **具体位置**：完全自由（softmax 采样）。

### 7.3 keep/free 开关（结合 vs 疏远配体，设计意图二分）

| 模式 | 语义 | 实现 |
|------|------|------|
| `--pocket_mode keep`（默认）| 保/加强配体结合：pocket 带电总数+净电荷受保护 | 训练侧 A1 + charge 锚（7.2）|
| `--pocket_mode free`（可选）| 疏远配体：允许自由改口袋 | 一期：推理侧不传保护（模型默认倾向 = 原版配体疏水先验，**零训练成本**）；二期：训练注入 pocket 保护占位符 flag（类 S3 占位符），模型学会 keep/free 双语义 |

### 7.4 改动清单

1. `code/src/v12_losses.py`：新增 `pocket_count_loss`（双向计数，可传 mask）；`surface_charge_target_loss` mask 扩展为 surface∪pocket
2. `code/train_finetune.py`：每域算三块互斥 mask（距配体 8Å 由 PDB HETATM 在线算）；传 pocket 参数
3. 数据：pocket mask 随训练在线计算（ligand_train PDB 含配体原子），无需新数据
4. 推理：`run_guided.py` / `validate_generalization.py` 加 `--pocket_mode keep|free`
5. 复验：重训 → 配体诊断 slope → 组成分析（target 0.7-1.3×）→ 泛化复验（H2/H1/H4/Tm/Sol）

### 7.5 超参消融（防重蹈 v12 过度添加）

- 单一变量：floor（0.6/0.7/0.8）× ceil（1.2/1.3）× λ（0.1/0.2）
- **floor 与 ceil 必须同时设**（v12 教训：只设下限 → 成对加逃逸）
- 判据：组成倍率 ∈ [0.7, 1.3]、slope ∈ [0.9, 1.15]、H2 ≥ 当前 72%

### 7.6 执行前提

- 配体 Tm/Sol 物化验证结果出来后再定是否重训（见 §7 开头决策）
- 重训前 dry-run 冒烟（50 域 1ep 确认 `pocket_count_loss` 生效无 NaN）

**✅ 执行状态（2026-09-01）**：物化验证**恶化**（负电臂 Tm 9/50 Δu<−5 + H3 n8 失败）→ 用户确认走路径 B。
实现 commit `1b93e87`（`pocket_count_loss` + 三块互斥分区 + `surface_charge_target_loss` extra_mask），
dry-run 过（50 域 1ep 无 NaN）→ **v13 配体重训已启动**（GPU6，`output/finetune_ligand_v13/`，
`--pocket_mode keep --pocket_cutoff 8.0 --pocket_floor 0.7 --pocket_ceil 1.3 --lambda_pocket 0.2`）。
复验链：组成（0.7-1.3×）→ 配体 slope → 泛化 H2/H1/H4 → H3 → Tm/Sol。详见 `session/2026-09-01_v13_pocket_retrain.md`。

### 7.7 v13 复验链判据修正（2026-09-01 用户决策）

- **⚠️ 扩样本量**：H3 判定不能只看 n8 单臂（n30 有偶然性）——**v13 泛化采样 n 30→50（全 5 臂）**，
  H3 基于**全臂 × n50** 统计判定（不只 n8），Tm/Sol 复测同批序列。样本量扩大不增加代码成本
  （H3/Tm 是 CPU 轻量统计），只增加 GPU 采样时间。
- **⚠️ bias 补丁非必选、默认不加、不进复验链**（用户 2026-09-01 强调"不是一定要用的"）：
  它只是"删减根治失败时的备选方案"，**默认路径 = 不加**，仅在全部条件满足时才单独评估，
  且评估若副作用超限（拉偏 H2/降多样性）就放弃。决策框架：
  - **先看 v13 复验（扩样本 H3）结果**：删减被治、聚集消失 → **不加**（多余干扰，默认情形）；
  - 仅当**全臂扩样本后仍超标** → 才评估推理侧 bias 补丁：`conditioned_sample` 透传 `bias_callback`
    （复用 Phase 1 `StructureAwareFilter`+`make_dynamic_callback`），`run_guided/validate`
    加 `--structure_filter_strength`（0=关）。⚠️ strength 需实测平衡——过强会拉偏净电荷
    命中（H2）、降多样性、可能动 Tm/Sol；副作用超限则放弃补丁方案。
  - 训练侧 C 组件（`--ph_aware_filter`）始终保留（软惩罚）；bias 补丁只作第二道防线，非必须。
