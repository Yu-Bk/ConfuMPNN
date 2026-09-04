# Task B 会话 — 配体版本更迭史 + v13 vs v14 对照（2026-09-05）

> 任务：梳理配体模式版本史（v9→v12.2-ligand→v13→v14），重点 v13 vs v14 数据/监督差异与改进面。
> 纯文档/数据分析，CPU，零采样。产物：`analysis/report/2026-09-05_ligand_history_v13_v14.md`（权威报告）+ 本会话。
> 未 git commit/push。

## 一、读了什么（证据链）

- 会话：`session/2026-08-31_v12_2_ligand_migration.md`、`2026-09-01_v13_pocket_retrain.md`、
  `2026-09-02_v14_rna_data_a1_global.md`、`2026-09-02_v12_3_long_retrain.md`（仅对照）
- 报告：`analysis/report/2026-09-01_v12_2_ligand_comp_analysis.md`、`2026-09-01_v12_2_ligand_validation.md`、
  `2026-09-02_v13_ligand_validation.md`、`2026-09-04_v14_clean_validation.md`、`2026-09-04_v13_in10_validation.md`、
  `2026-09-03_long_neg_charge_limitation.md`、`2026-09-04_v14_deletion_location.md`、`2026-09-04_v14_fixbinding.md`、
  `2026-09-04_v14_largen_search.md`、`2026-09-04_paper_subconclusions.md`、`2026-08-18_v9_ligand_training.md`、
  `2026-09-02_v14_rna_data_a1_global.md`
- 设计：`index/PROJECT_LOCAL_V12_2.md`（§7 A1+A2、§9.2 数据扩充、§9.3 A1 全局化）、
  `index/PROJECT_LOCAL_V14_DELETION_FIX_PLAN.md`
- 数据实物：`data/ligand_train/*.npz` 域数、`data/validation_pdbs/validation_manifest_v14_{in,boundary}.json`
- 训练日志末轮：`log/v9_train.log`(111.5min/30ep)、`log/v12_2_ligand_train.log`(992.6min/30ep)、
  `log/v13_ligand_train.log`(572min/30ep)、`log/v14_ligand_train.log`(832.8min/50ep)
- 产物存在性：四个 finetune_ligand_* ckpt 目录、`generalization_ligand_v13_in10`、`generalization_ligand_v14_clean`、
  `*_comp_*.json`、`charge_calibration_*_in10/clean.json`

## 二、实物核对结果（关键数字已坐实）

- `labels.npz` = 4957 域（v9-ligand/v12.2-ligand/v13 共用训练数据）
- `labels_v14_final.npz` = 5371 域（= 4957 + RNA/DNA 414；`labels_rna_v14_sup2.npz`=414）
- in-10 manifest 10 蛋白 = 6D2O/1AS2/2FEO/5CQH/1CGE/1BJ4/21KL_A/5O60_E/3MXB_A/9DWG_L；boundary=1A65
- v13 log 末轮 charge 3.60（30ep 仍在缓降）；v14 末轮 charge 3.08（50ep 收敛更深）

## 三、核心发现（汇报要点）

1. **口径陷阱**：v13「旧集 70% / S2 17/50」与 v14「in-10 90% / S2 0/50」**不是同一测试集**。
   严格同集同协议对照（v13-in10 vs v14-clean）才是权威：**H2 64%→90%**。
2. **v13→v14 差异四件套**：数据 +414 RNA/DNA（7.7%，sup 不 append 整体重跑合并）；
   atom_context 16→25 bug 修复；A1 keep(pocket floor0.7/λ0.2)→global(surface∪pocket floor0.8/λ0.3+normalize)；
   epochs 30→50。**v12 全套参数（frac_floor0.5/gravy0.4/λ_v12 0.2/λ_target0.2）原样未动**。
3. **改进**（in-10）：H2 32/50→45/50；H3 48/50→50/50；S2 11/50→0/50；1A65 boundary dev 8.98→2.6；
   提升集中 21KL_A/9DWG_L(1→5)、1BJ4(0→5)、1AS2(1→5)。RNA/DNA 数据扩充收益成立。
4. **未改进/倒退**：组成删减 10/10 蛋白 0.43-0.69×，**每蛋白都比 v13 深**（共享 5 蛋白 v13 0.56-0.70 vs v14 0.43-0.60）；
   RNA/DNA 成员差距最大（v13 0.93-0.99 因 OOD 未学会编辑 vs v14 0.56-0.69）；唯一 H2 倒退 = 2FEO(5/5→0/5)。
5. **为何 A1-global 没治好删减**（三点机制，§4.3 报告）：软监督被电荷目标压过（floor 0.8 区域实测 retention 仅 0.48-0.55）；
   frac_sasa<0.25 非 pocket 深埋核心仍逃逸（core 0.39、最深 SASA 箱 0.35）；删减是模型调电荷主杠杆、无「反号替换/新增带电」替代路径。
   → v14 用更深删减换更高 H2（「组成换电荷」取舍）。

## 四、产物

- 权威报告：`analysis/report/2026-09-05_ligand_history_v13_v14.md`
- 本会话：`session/2026-09-05_taskB_ligand_history.md`
- 未 commit/push（按要求）。

## 五、给主会话的注意点

- 论文若引用 v13-vs-v14，务必带测试集口径注（旧 10-集 vs in-10）。
- 2FEO 是 v14 唯一 H2 缺口（v14-clean 0/5），但大样本(n200) 2FEO 有 3 臂能捞到三达标序列 → 属「高方差/存在性受限」非「能力系统性」。
- 组成删减仍是跨版本未决硬伤（v12.2-ligand/v13/v14 三代），等待用户决策 D（v12.4/组成监督/论文口径）。
