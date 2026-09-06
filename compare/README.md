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
| exp5 Wilcoxon 配对（exp1/2 数据） | 逐序列命中 B(enc) vs C(bias)：C 全臂更优（两模式 15/15，p≈6e-5，med +0.49~+0.60）；A vs B 极端臂增益显著、native 臂不显著；补充 S1 v13-in10 vs v14-clean dev 配对 n=50 未显著(med −0.13 p=0.11) | `report_2026-09-06_exp5_wilcoxon.md` + `output/wilcoxon_exp15.json` |
| exp7 跨 pH(5/7.4/9) 敏感性（蛋白 v12.2 / 配体 v14） | classA 蛋白 H2 pH5 0/20(pH5 崩)/7.4 19/20/9 13/20；配体 7/15/9/15/4/15(pH9 最差)；classB 固定 target 换 pH 序列 identity 0.77-0.86 → pH 敏感真阳性但非恒 target；H1 TM 0.87-0.97+H4 PROPKA 抽查折叠完好 | `report_2026-09-06_exp7_pH_prot.md` + `report_2026-09-06_exp7_pH_lig.md`（数据 `output/exp_pH_{prot,lig}/`） |

## 09-06 新增补充实验计划（先计划后执行）
| 计划 | 类型 | 状态 |
|---|---|---|
| `plan_exp1_barebackbone_control.md` | 裸 backbone(MoMPNN/LigandMPNN) 无条件 vs 条件生成，12 组同 seed | 🔄 执行中（prot GPU2/lig GPU6） |
| `plan_exp2_bias_vs_encoder.md`(在 ablation/) | encoder vs bias-only 消融 | 🔄 并入 bundle |
| `plan_exp5_wilcoxon.md` | exp1/2 数据后 Wilcoxon 配对 | ✅ 见 `report_2026-09-06_exp5_wilcoxon.md` |
| `plan_exp7_pH_response.md` | pH=5/9 两 class pH 敏感性 | ✅ 见 `report_2026-09-06_exp7_pH_{prot,lig}.md` |
| `report_2026-09-06_exp3_hitrate_ci.md` | 命中率 95%CI(Wilson+精确) 工具与结果 | ✅ a6b7334 |
| `report_2026-09-06_exp4_rmsd_matrix.md` | RMSD 矩阵聚合(蛋白×臂, rmsd/tm/plddt) | ✅ a6b7334 |
