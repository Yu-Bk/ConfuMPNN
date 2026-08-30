# PROJECT_LOCAL_V12_2 — v12.2 深化训练 + 完整验证计划（2026-08-30）

> **状态**：计划定稿（2026-08-30 晚，用户批准）。**新表（n50）泛化已验证达标（2FEO 修复、H2 29/50）**；hold-out 验证集构建进行中；v12.2 训练待用户确认后启动。
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
7. 全部结果汇总 → 写验证报告 → 汇报用户
8. **若结果需方向决策（不达标/新短板）→ 暂停，等用户决策**（此点不在自动授权内）
