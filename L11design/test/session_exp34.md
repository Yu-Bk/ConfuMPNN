# Session — L11 Exp3 + Exp4（现场小样本标定后完整重做 Exp2 与 Exp1，2026-09-09）

> 归属：`L11design/`。目标：L11（表外高 pI）用**现场小样本标定**后完整重跑
> Exp2（Exp3：pH7.4 × target{+6.87,+8,+10,+12} × 蛋白/配体 × n=300=2400）与
> Exp1（Exp4：2 模式 × C1 pH7.4/+6.87、C2 pH8/+6.59 × n=100=400）。
> 固定链 I `I3 I5 I9 I34 I35 I89 I124 I131 I134 I135`（native，0 错配要求）。
> 参考：`PLAN.md`「实验三&四」「方法澄清」。

## 校准拟合（探针批，仿 build_calibration_small.py，单蛋白 L11 最小实现）
- 探针：L11 骨架（含固定位，与真实采样一致）native_q(round=+7) ±[8,4,0,4,8] 5 档 × n_per=10 = 50 条/蛋白。
- 蛋白（MoMPNN+v12.2×L11.pdb）；配体（LigandMPNN atom25+v14×L11_RNA.pdb）。
- pH7.4 与 pH8 各拟合一张表（Exp3 只需 pH7.4；Exp4 各用对应 pH）。
- 拟合表 → `output/exp3/cal/L11_small_{prot,lig}_pH{74,80}.json`（per_protein 键：蛋白 L11 / 配体 L11_RNA，供 run_guided `--calibrate auto` 命中）。

| 模式 | pH | slope | intercept | LOOCV | unreliable | 探针响应(target→mean) |
|---|---|---|---|---|---|---|
| protein | 7.4 | 1.7303 | −2.3707 | 0.60 | 否 | −1→−4.08, 3→3.02, 7→9.02, 11→17.43, 15→23.32 |
| protein | 8.0 | 1.7547 | −1.8234 | 0.75 | 否 | −1→−3.56, 3→3.53, 7→9.75, 11→18.55, 15→24.03 |
| ligand  | 7.4 | 1.7558 |  0.0728 | 2.48 | 否 | −1→−0.58, 3→3.31, 7→11.99, 11→21.81, 15→25.29 |
| ligand  | 8.0 | 1.7770 |  1.1555 | 2.50 | 否 | −1→0.40, 3→5.05, 7→12.38, 11→23.36, 15→26.78 |

- global 兜底 = 原表 global（蛋白 v12_2、配体 v14_clean），即 Exp1/2 口径。

## 采样（CPU 并行，resume；CUDA_VISIBLE_DEVICES=""）
- `test/exp34_sampling.py --groups-json exp3_groups.json/exp4_groups.json`
- **修复 Bug（关键）**：初版脚本 `cal_file = L11 / cal_root / ...` 路径双重 `L11design/L11design`，
  导致校准表未找到 → run_guided 落到「校准表不可用且 yaml enabled=false，**未校准**」（日志 cond_vec 电荷=原始 target）。
  已改为 `ROOT / cal_root`，冒烟验证 cond_vec=校准后 target_eff、日志显示 per-protein(L11)。
- Exp3：8 组 × 300，seeds 3101–3204，3.0 min。
- Exp4：4 组 × 100，seeds 4101–4202，0.9 min。
- 固定位逐条校验 0 错配（build_fastas 报告）。

## 电荷落点（修正后，采样 summary）
- Exp3 dev_mean：蛋白 +0.79/+0.66/+0.55/+0.29，配体 +0.27/+0.72/+0.54/+1.07（|dev|≤2：蛋白 31–34%、配体 29–41%）。
- Exp4 dev_mean：prot_C1 +0.40、prot_C2 +1.31、lig_C1 +0.79、lig_C2 +0.75（|dev|≤2：38/41/44/39%）。

## 打分/折叠管线
- ESMFold（GPU 轮询）→ pLDDT + folds；TM/RMSD vs `input/L11_chainI.pdb`（US-align）；
  TemBERTure Tm；Protein-Sol %sol；固定位校验；回收（vs native 链 I）。
- Tm/Sol（CPU）后台 `exp34_cpu_scores.sh`；ESMFold `exp34_fold_scheduler.py`（等空卡）。

## 产物
- 采样 output/exp3|exp4/<dir>/；数据 data/exp3|exp4/<dir>/；
- 报告 test/report_exp3.md（Exp3 vs Exp2）、report_exp4.md（Exp4 vs Exp1）；
- 备份 backup/L11_exp34_<date>.tar.gz。

## 执行结果（完成 2026-09-09 ~11:35）
- Tm(Sol) 全完成：Tm exp3 9/9 目录、exp4 4/4；Protein-Sol 13/13 csv。
- ESMFold 13/13 目录（多 GPU 并行调度 exp34_fold_scheduler_mp.py，GPU1/4/6/7 时隙）；TM/RMSD 13/13（vs L11_chainI.pdb）。
- 固定位 0 错配：Exp3 2400 条 + Exp4 400 条 = 2800/2800。

### Exp3（Bsmall vs Exp2 global；n=300/组，pH7.4）
| mode | target | dev_mean Bsmall→global | 命中率 Bsmall→global |
|---|---|---|---|
| protein | 6.87/8/10/12 | +0.79/+0.66/+0.55/+0.29 → +2.49/+2.31/+3.64/+3.56 | 33/32/34/31% → 29/27/27/23% |
| ligand | 6.87/8/10/12 | +0.27/+0.72/+0.54/+1.07 → +3.15/+3.66/+3.69/+5.00 | 41/39/39/29% → 28/26/23/18% |
- 平均 Δ命中率 +10%（配体 +11~+16 pp、蛋白 +4~+8 pp）；均值 dev 收敛到 +0.3~+1.1。
- std≈4–5 不变（不治散布）；pLDDT/TM/RMSD/Tm/Sol 与 Exp2 基本一致（蛋白 TM≈0.65–0.67，配体 TM≈0.58–0.62；回收≈27%）。

### Exp4（Bsmall vs Exp1 global；n=100/组）
| grp | target | dev Bsmall→global | 命中率 Bsmall→global |
|---|---|---|---|
| prot_C1 | 6.87 | +0.40 → +1.70 | 38% → 29% |
| prot_C2 | 6.59 | +1.31 → +3.94 | 41% → 21% |
| lig_C1 | 6.87 | +0.79 → +3.32 | 44% → 24% |
| lig_C2 | 6.59 | +0.75 → +5.34 | 39% → 16% |
- 电荷 dev 从 +1.7~+5.3 压到 +0.4~+1.3；std≈3.8–4.9；pLDDT/TM/RMSD/Tm/Sol 与 Exp1 同档（配体 TM 略低 ~0.01–0.03，属抽样噪声）。

### 结论（一句话）
现场小样本标定**治增益、不治散布**：两模式各 target 均值电荷 dev 从 global 的 +2.3~+5.3 收敛到 +0.3~+1.3，
命中率 |dev|≤2 提升 +4~+16 pp（蛋白 31–34%、配体 29–44%），std≈4–5 不变，折叠/稳定/可溶未见系统性改变，固定位 100%。

## 数据路径
- output/exp3/{protein,ligand}_q{6.87,8,10,12}/；output/exp4/{prot,lig}_{C1,C2}/
- data/exp3/<dir>/{seqs.fa,plddt.csv,tm.csv,seqs.fa.tm.csv,seqs.fa-protein_sol.csv,metrics.csv}；data/exp3/_per_group_summary.csv
- data/exp4/<dir>/merged.csv；data/exp4/summary_stats.json
- 报告 test/report_exp3.md、test/report_exp4.md；脚本 test/exp3*.py、exp4*.py、exp34_*.py/sh
- 备份 backup/L11_exp34_2026-09-09.tar.gz
