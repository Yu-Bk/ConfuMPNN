# ConfuMPNN — 文档定位索引（2026-09-05 全量刷新）

> 项目**所有文档的定位索引**。规则见 `FILE_MANAGEMENT.md`；分类规则 `logical_chain.md`。
> **⚠️ 新增 / 移动 / 删除任何文档后，必须同步本文件。**
> ★ = 当前"权威/最新"（论文引用首选）。

## 0. 权威入口与规则
| 文档 | 说明 | 更新 |
|---|---|---|
| ★`analysis/report/2026-09-05_repo_audit.md` | 全仓盘点+缺口审计（本索引的事实来源，含逐文件清单） | 09-05 |
| `README.md` | 项目入口/复现/目录/状态 | 09-05 刷新 |
| ★`WORKFLOW_GUIDE.md` | 唯一权威使用指南（面向新人，原理/数据流/参数/损失/命令） | 09-05 刷新 |
| ★`logical_chain.md` + `index/FILE_MANAGEMENT.md` | 文件分类唯一规则（09-05 修订：顶层 output/log/data/ablation/compare） | 09-05 |
| `CLAUDE.md`(根) | 项目说明（状态已刷到 v14/消融/核糖体） | 09-05 刷新 |
| ★`docs/MIGRATION_GIT_POLICY.md` | git / GitHub Release / NAS-网盘 三路归档唯一规则 | 09-04 |
| `index/DESIGN_CRITERIA.md` | 判据 v2（H1-H4 / S1-S4 / H2 dev≤2 等） | 08-17+ |
| `index/RESULTS_MANIFEST.md` | 论文关键结果 JSON 清单（**09-05 待随本索引刷新**） | 09-01 |

## 1. 计划 / 决策 / 方向（`index/`）
| 文档 | 主题 | 更新 |
|---|---|---|
| `PROJECT_PLAN.md`(v1) / `PROJECT_EXTEND.md`(v2) | 早期整体技术计划 | 08-16 |
| `PROJECT_LOCAL.md`(v3 论文导向) | P1–P10 / D1–D12 / v10 方法(A+B+C) / 对照 C1–C8 / 消融 A1–A10 / 判据 H4 | 08-27 |
| `PROJECT_LOCAL_P1_PLAN.md` | v3 §8 P1 对照实验细化 | 08-27 |
| ★`PROJECT_LOCAL_V12_2.md` | v12.2 深化训练+完整验证计划（§9.5 in-10、§10 验证集 15%+per-epoch 曲线，当前蛋白交付线） | 09-04 |
| `PROJECT_V9_GENERALIZATION_PLAN.md` / `PROJECT_V9_LIGAND_PLAN.md` | v9 泛化 / 配体重训计划 | 08-19 |
| `PROJECT_SUPPLEMENT_H3_REVIEW.md` | H3 电荷聚集合法性判据 | 09-01 |
| `PROJECT_LOCAL_V14_DELETION_FIX_PLAN.md` | 删减定位/fix/大样本三任务 + v13-in10 对照计划 | 09-04 |
| `PROJECT_LOCAL_V14_FINAL_EXPERIMENTS.md` | 收尾实验：受控消融 + E.coli 核糖体可设计性 | 09-05 |
| `index/v10_repair/` | v10 修复包（诊断/分段/补丁/适配） | 08-28 |

## 2. 分析报告（`analysis/report/`，59 份；★权威）
**Phase1-3 / v7（08-16~18）**：`2026-08-16_{e1_pH_response,e1_three_targets,e1_extended,e4_default_mompnn,phase1_examples,phase2_training_start,phase3_pH_response,phase3_antidrift,phase3_charge_temp,phase3_n20_antidrift}.md`、`2026-08-17_{phase3_s1_fix,phase3_v2_validation,phase3_v3_placeholder_fix,phase3_v4_balanced_data,phase3_v5_cap2_analysis}.md`、`2026-08-18_{phase3_v6_class_balance,phase3_v7_curriculum,model_charge_limits,ligand_binding_capacity,model_validation_phscan,seq_sanity_and_transfer,v9_ligand_training}.md`（★`model_charge_limits`=电荷使用边界）。
**v9 泛化（08-19）**：★`2026-08-19_v9_generalization_validation.md`。
**v10-v12（08-27~31）**：`2026-08-27_v10_ligand_training.md`、★`2026-08-28_v10_diag_response.md`、`2026-08-28_v10_validation.md`、★`2026-08-29_calibration_verify.md`、★`2026-08-29_v11b_c_compare.md`、`2026-08-29_v12_training.md`、★`2026-08-30_v12_1_validation.md`、`2026-08-31_{v12_2_training,v12_2_diag,v12_2_tm_sol}.md`、★`2026-08-31_v12_2_summary.md`(蛋白完整链+校准三口径)。
**v12.2 配体迁移/v13/v14（09-01~04）**：`2026-09-01_{v12_2_ligand_diag,v12_2_ligand_validation,v12_2_ligand_comp_analysis,v12_2_ligand_tm_sol_h3,pocket_fix_test}.md`、★`2026-09-02_v13_ligand_validation.md`、`2026-09-02_v14_rna_data_a1_global.md`、`2026-09-02_{v12_3_diag,v12_3_curvature_analysis,ligand_af3_fold_data}.md`、★`2026-09-03_validation_standards.md`(判据/口径)、`2026-09-03_v12_3_vs_v12_2_final.md`、`2026-09-03_long_neg_charge_limitation.md`(能力边界档)、`2026-09-03_paper_gap1_v122_long.md`、★`2026-09-04_v14_clean_validation.md`(v14 权威测试链)、★`2026-09-04_v13_in10_validation.md`(v13-vs-v14 同协议对照)。
**三实验/曲线/子结论（09-04）**：`2026-09-04_{v14_deletion_location,v14_fixbinding,v14_largen_search,val_loss_curves,paper_subconclusions}.md`。
**09-05（版本史/核糖体/审计）**：★`2026-09-05_protein_history_vs_ligand_deletion.md`、★`2026-09-05_ligand_history_v13_v14.md`、`2026-09-05_taskA_timeline.md`、★`2026-09-05_7k00_ribosome_design.md`、`2026-09-05_repo_audit.md`。
**消融/对比（独立目录）**：`ablation/report/2026-09-05_ablation_{prot,lig}.md`(★受控消融，顶层 `ablation/`)；`compare/README.md`(对比登记)。

## 3. 关键会话（`session/`，37 份；★决策类）
★`2026-09-04_decision_log.md`(验证集方法定稿)、`2026-09-04_v14_clean_chain_autolog.md`(clean 链检查点)、`2026-09-04_valset_build.md`(805 验证集构建)、`2026-09-04_val_loss_curve_build.md`、`2026-09-04_task{1_deletion_location,2_fixbinding,3_largen}.md`、`2026-09-04_v13_in10_chain.md`、`2026-09-05_{taskA_protein_history,taskB_ligand_history,7k00_ribosome,ablation_run}.md`、早期 `2026-08-17_validation_plan_v2.md`、`2026-09-02_v14_rna_data_a1_global.md`、`2026-09-02_v12_3_long_retrain.md` 等。

## 4. 技术/使用/部署（`docs/`，均 09-05 待随新状态刷新或已注明）
`TECH.md`(原理) / `CONFIG.md`(配置) / `USAGE.md`(使用) / `SETUP_NEW_MACHINE.md`(新机) / `MIGRATION_GIT_POLICY.md`(归档唯一规则)。`data/README.md`+`SHA256SUMS.txt`(数据说明/校验，09-05 增补)。

## 5. literature / figure / 其它
`literature/`(baseline/innovation/pattern/tools/phenomena + ★`note_2025_global_bias_local_features.md`)、`source/README.md`(源码登记)、★`figure/plan_01.md`(论文全部图 Fig1-27 计划，唯一图目录)。

## 目录状态（2026-09-05）
| 目录 | 状态 |
|---|---|
| code/ output/ log/ data/ | 完整；output 顶层论文 JSON 全入库；data 大文件→NAS 备份 |
| analysis/report/ | 59 份，权威链已标 ★；被证伪报告可移 `analysis/archieved/` |
| ablation/ compare/ figure/ | 09-05 新：消融两族完成 / 对比登记 / 图计划(无成品图待画) |
| session/ index/ weights_release/ | session 37 份；index 刷新；weights_release 待补 v12.2/v12.3/v14 |
