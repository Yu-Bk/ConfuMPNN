# 会话记录 — 论文叙事规划 + 三缺口执行（2026-09-03 下午）

> 触发：v12.3 vs v12.2 裁决材料齐备后，用户提出论文叙事想法，要求：
> ① 措辞用"**一定程度缓解**"（非"完全修复"——1A65 仅 0/5→2/5）；
> ② 依次补齐数据缺口；③ 对照的 global 校准不影响原本 global（影响则另行保存）；
> ④ 保存输出结果、明确路径与可作图类型；⑤ 保存本次讨论。
> 关联：v12.3 定稿报告 `2026-09-03_v12_3_vs_v12_2_final.md`、v12.3/v14 验证重构 `2026-09-03_validation_standards.md`。

## 一、论文叙事（用户想法，已修正口径）

**用户原始想法**：整理 v12.2 数据，论文讲"发现长蛋白不行 → 补一定数据库后长蛋白问题被修正，但原本短蛋白受影响（v12.3）"。

**口径修正（重要，避免审稿人推翻）**：
- 不是"长蛋白不行"，是"**长×深负组合**不行"——v12.2 对中性长蛋白 1BJ4 小样本标定后 5/5 全过，"长"本身非问题；真问题是深负长蛋白 1A65（−26.9，标定救不了）。1BJ4 仅未标定 big-global 口径失败。
- 修正后表述：**模型对训练覆盖不足的"长×深负"组合外推失败 → 补负电富集长蛋白后该外推被"一定程度缓解" → 但训练分布被拉偏，覆盖内蛋白电荷控制回退**（trade-off）。
- v12.4 双向计数**是否启动，等 v14（配体 + A1 全局化）泛化结果再决定**——v14 是双向计数 global 化的验证场。

## 二、三个缺口（本记录时状态）

### 缺口①：v12.2 侧长蛋白基线（对照证据）
- 目的：给"v12.2 长×深负不行"提供 v12.2 对 13BB/1CDG（v12.3 新增验证长蛋白）的失败基线，不只靠 1A65 一个点。
- **状态**：agent（a48202ea40c37039e）后台执行中（GPU 采样，需数小时）。
- 产物路径：`output/paper_gap1_v122_long/`（generalization_small/ + generalization_bigglobal/ + 报告 `analysis/report/2026-09-03_paper_gap1_v122_long.md`）。
- 对照数据（v12.3 权威，直接引用）：1A65 small 2/5、1BJ4 5/5、13BB 1/5、1CDG 4/5；big-global 1A65 1/5、1BJ4 1/5、13BB 4/5、1CDG 3/5。
- ⚠️ 只读 `charge_calibration_v12_2_big.json`，绝不重拟合/复写 → 原 global 不受影响。

### 缺口②：in 5 覆盖内对比（trade-off 证据）✅ 完成
- 产物：`output/paper_gap2_in5_compare/in5_compare_v122_vs_v123.json` + `PLOTTING.md`
- 数据：5 蛋白 × 两口径 × 两版本逐臂。汇总 small 23/25→20/25（92%→80%）、global 17/25→15/25（68%→60%）。
- 作图：分组柱状图 A1（主图）/ 命中格子 heatmap A2 / dev 箱线 A3。

### 缺口③：训练分布 L×q 散点（覆盖证据）✅ 完成
- 产物：`output/paper_gap3_distribution/train_Lq_distribution_v122_vs_v123.json` + `PLOTTING.md`
- 关键统计：L>400 2.0%→8.8%；q≤−25 0.64%→1.38%；1BJ4 n_close 20→96、13BB 5→44（out→boundary）、1A65 4→24、1CDG 1→7（仍 out）。
- 作图：L×q 散点对比 B1（主图）/ n_close 迁移条形 B2 / q CDF B3。

## 三、数据产物总表（论文写作引用）

| 用途 | 路径 | 关键数字 |
|---|---|---|
| 覆盖内回退 | `output/paper_gap2_in5_compare/in5_compare_v122_vs_v123.json` | 92→80%、68→60% |
| 覆盖扩大（分布） | `output/paper_gap3_distribution/train_Lq_distribution_v122_vs_v123.json` | L>400 2.0→8.8% |
| v12.2 长蛋白基线 | `output/paper_gap1_v122_long/`（跑完待填） | 待 agent |
| 完整裁决 | `analysis/report/2026-09-03_v12_3_vs_v12_2_final.md` | 两口径×三组全表 |
| 方法论标准 | `analysis/report/2026-09-03_validation_standards.md` | coverage 框架 |

## 四、待办
- 缺口① agent 完成后：汇总 v12.2 长蛋白基线 → 补进论文证据链
- 配体 v14 验证链完成后：决定 v12.4 双向计数是否启动
- 后续论文图：按各 PLOTTING.md 执行（matplotlib/Python）

## 五、2026-09-03 晚：v14 配体验证集方法学修正（2E9R_X → 5O60_E，执行 agent）

- **背景**：2E9R_X（FMDV RdRp, L476, native −10.1）不具 RNA"结合蛋白"代表性（核糖体 RNA 结合蛋白天然正电；
  它是长负聚合酶，big/small 0/5 且过冲），用户判定归入"**长×大电荷变化可设计性有限**"档（与蛋白模式 1A65 同类），
  从标准验证集移除归档。5O60_E（核糖体蛋白 E, L209, native +11.18, coverage=in n_close151, held-out）替换作 RNA 代表。
- **产物**：
  - manifest 更新：`data/validation_pdbs/validation_manifest_v14_{final,in}.json`（archived/replaced 注明 2E9R_X→5O60_E；in 10 项含 5O60_E）
  - 5O60_E 小样本标定：`output/charge_calibration_v14_5O60E_only.json`（slope 1.433, LOOCV 2.73 reliable）+ 合并表 `output/charge_calibration_v14_small_v2.json`
  - 两口径采样：`output/generalization_ligand_v14_big/ligand/5O60_E/`（**4/5**，仅 p8 dev 3.34）、
    `output/generalization_ligand_v14_small/ligand/5O60_E/`（**4/5**，仅 p8 dev 2.71）
  - H2 汇总（新 in-10）：big-global **26/50=52%**、小样本 **40/50=80%**；剔 2E9R_X 未含 5O60_E 的 in-9 中间口径
    big 22/45=49% / small 36/45=80% / per 40/45=89%（与配体 agent 参考一致）；per 口径 5O60_E 无诊断条目不入（=40/45）
  - 归档报告：`analysis/report/2026-09-03_long_neg_charge_limitation.md`（1A65 + 2E9R_X 两模式证据 + 2E9R_X 数据路径）
  - 主报告 §4.2 已更新（in 蛋白表/H2 三口径/组成表去 2E9R_X 列/数据源）
  - ⚠️ 5O60_E 组成/H1/Tm-Sol 链未在本任务重跑（需 per 口径采样 + ESMFold 链，属后续）
- **计数注**：任务文本写"in(9)"，实为 2E9R_X 与 5O60_E **1:1 置换** → in 仍 10 项；"in-9 参考值"= 剔 2E9R_X、未含 5O60_E 的中间口径。
