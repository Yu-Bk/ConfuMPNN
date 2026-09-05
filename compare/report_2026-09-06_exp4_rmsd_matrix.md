# 报告 exp4 — RMSD 矩阵恢复/聚合（2026-09-06 完成）

> 目的：最新几版输出只存了 TM/pLDDT，"RMSD"看起来丢了。核查：逐序列 rmsd csv 在最新几版**未保存**；但各 `*_gen_stats.json` 已存 **per-arm `rmsd_median`（+tm_median/plddt_median）** → 可恢复"蛋白×电荷臂"的 RMSD 汇总矩阵。
> 脚本：`code/tests/aggregate_rmsd_matrix.py`（读任意 *_gen_stats.json → 蛋白×arm 的 rmsd/tm/plddt 矩阵）。
> 矩阵数据：`output/rmsd_matrix_v14_v13.json`。

## 结果（示例，native 臂 rmsd_median Å）
| 蛋白 | v14-clean | v13-in10 |
|---|---|---|
| 1AS2 | 1.88 | 1.79 |
| 1BJ4 | 2.14 | 2.07 |
| 1CGE | 1.02 | 1.01 |
（完整：10 蛋白 × 5 臂，见 JSON）

## 局限
- **只有中位数、无逐序列 RMSD 矩阵**（逐序列只在更早版本/部分 csv 保存）；若要逐序列热图需从已存 ESMFold folds 重算 TM-score（脚本 `code/tests/tm_score.py`）或回旧 csv。
- RMSD 与 TM 在回折一致，论文主证据仍用 TM；RMSD 作辅助。

## 数据/图路径
- 矩阵 JSON：`output/rmsd_matrix_v14_v13.json`
- 脚本：`code/tests/aggregate_rmsd_matrix.py`
- 源：`output/v14_ligand_gen_stats_clean.json`、`output/v13_ligand_gen_stats_in10.json`（可扩展其它 `*_gen_stats.json`）
