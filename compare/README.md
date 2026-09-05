# compare/ — 对比实验登记（2026-09-05 起）

> 本目录收**版本/方法间对比实验**的方案、数据与结果。消融放 `ablation/`（勿混）。

## 已产出（权威位置在 analysis/report/ 或 output/，此处登记指针）
| 对比 | 结论摘要 | 报告 |
|---|---|---|
| v13-in10 vs v14-clean（同协议） | H2 64→90%、H3 48/50→50/50、S2 11/50→0/50、1A65 dev 8.98→2.6；组成 v13 每蛋白删得更轻（RNA/DNA 0.93-0.99 vs 0.56-0.69） | `analysis/report/2026-09-04_v13_in10_validation.md` |
| 配体 v13 vs v14 更迭 | 改进集中在 RNA/DNA 扩充收益；组成未改、每蛋白更深 | `analysis/report/2026-09-05_ligand_history_v13_v14.md` |
| 蛋白模式 vs 配体模式删减根因 | 蛋白较轻/表面有下限；配体每区更狠(pocket 差+0.14-0.30) | `analysis/report/2026-09-05_protein_history_vs_ligand_deletion.md` |
| 蛋白 v12.3 vs v12.2 | in 覆盖内 v12.3 退步；长/深负外推 v12.3 有价值 | `analysis/report/2026-09-03_v12_3_vs_v12_2_final.md` |
| 校准三口径（per-protein/小样本/global） | 72/74/40-44% | CLAUDE.md 校准三口径段 |

## 09-06 新增补充实验计划（先计划后执行）
| 计划 | 类型 | 状态 |
|---|---|---|
| `plan_exp1_barebackbone_control.md` | 裸 backbone(MoMPNN/LigandMPNN) 无条件 vs 条件生成，12 组同 seed | 🔄 执行中（prot GPU2/lig GPU6） |
| `plan_exp2_bias_vs_encoder.md`(在 ablation/) | encoder vs bias-only 消融 | 🔄 并入 bundle |
| `plan_exp5_wilcoxon.md` | exp1/2 数据后 Wilcoxon 配对 | ⏳ 依赖数据 |
| `plan_exp7_pH_response.md` | pH=5/9 两 class pH 敏感性 | ⏳ 等 GPU 空 |
| （已完成）命中率 CI `code/tests/hitrate_ci.py`；RMSD 矩阵 `code/tests/aggregate_rmsd_matrix.py` | — | ✅ a6b7334 |
