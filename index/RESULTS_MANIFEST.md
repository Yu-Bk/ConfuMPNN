# 实验结果清单（RESULTS MANIFEST）——2026-09-01 快照

> 目的：列出全部**数值结果文件**（json/csv，绘图用），按类别索引。output/ 整体 gitignore，**这些 json 已强制 git add 上传 GitHub**（`git add -f output/*.json` 等），复现/绘图可直接从仓库取。
> 绘图建议：每个 json 是"哪种图的原料"见各节末。

---

## 一、校准表（13 个 json，`output/charge_calibration_*.json`）

| 文件 | 内容 | 用途 |
|------|------|------|
| `charge_calibration_v12_2.json` | **当前主校准表**（mompnn 17 蛋白 per-protein + global）| 校准协议论文 |
| `charge_calibration_v12_2_ligand.json` | 配体模式校准表（global 1.330）| 配体论文 |
| `charge_calibration_v12_2_big.json` | 92 训练域 big 表（global 1.459）| 无泄露对照 |
| `charge_calibration_v12_2_small.json` | 小样本现场标定（n10=50 条）| 表外蛋白协议 |
| `charge_calibration_v12_2_small_n20.json` | n20 版（对比 n10）| 弯曲蛋白研究 |
| `charge_calibration_v12_2_small_test.json` | 测试版 | — |
| `charge_calibration_v12_1.json` / `_n50.json` | v12.1 校准表 + n50 重拟合 | 版本对比 |
| `charge_calibration_v12_1.json` | v12.1 | 版本对比 |
| `charge_calibration_v10.json` / `v11a` / `v11b` / `v11c` / `v7` | 早期版本校准 | 消融对比 |

**绘图**：响应曲线（slope/intercept）、校准前后命中率对比。

## 二、诊断响应曲线（13 个 json）

| 文件 | 内容 |
|------|------|
| `v12_2_diag_response.json` / `v12_2_diag_calibrated.json` | v12.2 诊断（17 蛋白响应，校准前后）|
| `v12_2_ligand_diag_response.json` | **配体模式诊断**（17 蛋白）|
| `v12_1_diag_response.json` / `v12_1_diag_n50.json` / `v12_1_diag_calibrated.json` | v12.1 诊断 |
| `v10_diag_response.json` / `v10_calib_diag.json` | v10 诊断 |
| `v11a/v11b/v11c_diag_response.json` / `v11a_calib_diag.json` | 消融系列诊断 |
| `v7_diag_response.json` / `v12_diag_response.json` | v7 / v12 诊断 |

**绘图**：每蛋白 target→实际电荷响应曲线（拟合 slope）、校准效果图。

## 三、泛化验证统计（9 个 json）

| 文件 | 内容 |
|------|------|
| `generalization_ligand_v12_2_stats.json` | **配体模式 v12.2 泛化**（10 蛋白×5 臂，H1/H2/组成）|
| `generalization_v12_2_calib_stats.json` | mompnn v12.2 泛化（per-protein 校准）|
| `generalization_v12_2_calib_small_stats.json` | 小样本现场标定版 |
| `generalization_v12_2_calib_small_n20_stats.json` | n20 版 |
| `generalization_v12_2_calib_noleak_stats.json` | 无泄露 big-global 版 |
| `generalization_v12_1_calib_stats.json` | v12.1 泛化 |
| `generalization_v10_mompnn_stats.json` / `generalization_v10_ligand_stats.json` | v10 双 backbone |
| `generalization_v9_stats.json` | v9 泛化（早期）|

**绘图**：逐蛋白逐臂 dev 分布、H2 命中率条形图、正向/负向对比、mompnn vs ligand 对比。

## 四、组成/其他（7 个 json）

| 文件 | 内容 |
|------|------|
| `v12_2_comp.json` | mompnn v12.2 组成（D/K 倍率，诊断 5 蛋白）|
| `v12_2_ligand_comp.json` | **配体组成**（10 蛋白，0.53-0.65× 删减证据）|
| `holdout_eval_v12_2.json` | hold-out 15% 评估（28 域×8pH×n5）|
| `transfer_stats.json` / `transfer_v9_stats.json` | 迁移验证（v7/v9 对比）|
| `ph_scan_stats.json` / `ph_scan_sanity.json` | pH 扫描（3 蛋白 × pH5/7.4/9）|

**绘图**：组成倍率对比（删减证据图）、hold-out 命中率、迁移 dev。

## 五、口袋 fix 实测（`output/pocket_fix_test/`）

| 文件 | 内容 |
|------|------|
| `compare_fix_effect.json` | fix 前后组成/电荷/恢复对比（2FEO/1AXW/1C6O）|
| `deletion_transfer.json` | 分区域删减分析（5 区域 × fix 前后）|

**绘图**：fix 前后口袋删减对比条形图、区域删减热图。

## 六、Tm/Sol（`output/tm_sol_v12_2/`）

| 文件 | 内容 |
|------|------|
| `tm_sol_summary.json` | TemBERTure Tm + Protein-Sol %sol 汇总（10 蛋白×5 臂 vs native/无条件基线）|

**绘图**：Tm/%sol 前后对比、电荷臂 vs 基线 Δ。

## 七、验证链逐臂 csv（970 个）

- `output/generalization_*/**/arm_*/tm.csv`（TM-score + RMSD）
- `output/generalization_*/**/arm_*/plddt.csv`（ESMFold pLDDT）
- 每臂 31 条（30 生成 + 1 native）

**绘图**：逐臂 TM/pLDDT 分布箱线图、失败率。

## 八、训练产物（17 目录，`output/finetune_*/`）

- 每目录含 `finetune_epoch030.pt`（权重，gitignore 不上传）+ 训练日志 `log/train_progress.json`（loss 曲线）

**绘图**：训练 loss 曲线（total/ce/charge/keep）。

---

## 绘图推荐映射（快速上手）

| 论文图 | 数据源 |
|--------|--------|
| 电荷响应校准（slope 1.579）| `v12_2_diag_response.json` + `charge_calibration_v12_2.json` |
| 泛化 H2 命中率（mompnn 72% vs ligand 72%）| `generalization_v12_2_calib_stats.json` + `generalization_ligand_v12_2_stats.json` |
| 删减捷径证据（组成 0.53-0.65×）| `v12_2_ligand_comp.json` + `v12_2_comp.json` |
| 口袋 fix 效果 | `pocket_fix_test/compare_fix_effect.json` |
| 分区域删减 | `pocket_fix_test/deletion_transfer.json` |
| H4 物理复核 | `propka_ligand_v12_2/*.json`（Q_design vs Q_phys）|
| 训练 loss | `output/finetune_v12_2/log/train_progress.json` |
| Tm/Sol | `tm_sol_v12_2/tm_sol_summary.json` |
| 三口径校准 | `holdout_eval_v12_2.json`（40.6%）+ `generalization_v12_2_calib_noleak_stats.json`（44%）+ `generalization_v12_2_calib_small_stats.json`（74%）|
