# PROJECT_LOCAL_V12_2 — v12.2 深化训练 + 完整验证计划（2026-08-30）

> **状态**：计划定稿（2026-08-30 晚，用户批准）。v12.2 训练尚未启动，先等新表泛化结果。
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

### 2.1 同分布 hold-out 验证集（15%，分层抽样 + 构建标签）

- **数据源**：CATH S40 剩余未训练域 **27,445 个**（总 34,653 − 训练 7,886）——"有答案（结构→native 电荷）但没训练"的天然 hold-out
- **分层抽样**：对剩余域按 native 电荷分箱（如 8 箱，与训练标签同构），目标**均值/方差匹配训练集**（训练集：均值 1.42、std 9.44、范围 −44.6~+56.1）；**各箱数量可弹性调整**（某箱多点少点均可，只要整体分布一致——"美化数据"意图）
- **规模**：~1,200-2,000 域（15% 量级）
- **构建标签**：`net_charge` 物理计算 @ 8 pH（与训练 labels_balanced_v7 同构：domain_ids/seqs/coords/pH/charge/pI）——物理计算不需训练
- **评估**：v12.1/v12.2 + 校准 → native target 电荷命中（H2）、recovery、可选折叠
- 脚本：新建 `code/tests/build_holdout_labels.py`（分层抽 + 建标签）+ `code/tests/validate_holdout.py`（评估）

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
