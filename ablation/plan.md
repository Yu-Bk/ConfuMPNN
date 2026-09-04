# 受控减预算消融落地计划（2026-09-05，ablation/ 收口版）

> 母计划：`index/PROJECT_LOCAL_V14_FINAL_EXPERIMENTS.md` §1。本文件为执行落地副本，
> 记录两族 run 矩阵、精确配方命令、数据抽样与评估口径。
> 产物目录（用户 2026-09-05 指定）：`ablation/plan.md` `ablation/data/`
> `ablation/runs/{prot,lig}/run_<tag>/` `ablation/report/` `ablation/figure/`。
> 不写 `output/ablate_*` 与 `analysis/report/`；不 git commit/push。

## 1. 目的与铁律
- 目的：对最终配方（蛋白 v12.2 / 配体 v14）做"单模块 OFF vs 全开"受控消融，
  给出各监督模块**相对贡献/排序**，非绝对 SOTA。
- 铁律：同族内同一数据子集（25% 分层抽）、同轮次（蛋白 10 / 配体 16）、同 seed(42)、
  同基座/超参/backbone（只训 ConditionEncoder）；每个 OFF run **只允许一处差异**。

## 2. 数据子集
- 蛋白：源 `data/cath/labels_v12_3_train.npz`（6580 域，含长蛋白），分层抽 25%（保 L / 电荷
  分布）→ `ablation/data/labels_ablate_prot.npz`（~1645 域）。dompdb=`data/cath/S40/dompdb`。
- 配体：源 `data/ligand_train/labels_v14_final.npz`（5371 域），分层抽 25%（保 RNA/DNA≈420/5371
  =7.8%）→ `ablation/data/labels_ablate_lig.npz`（~1343 域）。dompdb=`data/ligand_train/all_pdb`。
- schema 同源：domain_ids(N,)/seqs(N,)/coords(N,)/pH(8N,)/charge(8N,)/pI(8N,)，每域 8 (pH,charge) 臂。

## 3. run 矩阵
| run 标签 | 蛋白族(5) | 配体族(6) | 单点差异 |
|---|---|---|---|
| FULL | v12.2 全开 | v14 全开 | 对照 |
| run_nov12comp | −v12组成 | 同 | 去掉 `--v12_supervision`（v12_comp/gravy/v12_ct 全 off） |
| run_notarget | −λ_target | 同 | 不传 `--lambda_target`(=0)，其余 v12 保留 |
| run_noA1 | — | −A1(pocket global) | `--lambda_pocket 0`（保留 pocket_mode global，不动 λ_target 分区） |
| run_noph | −ph_filter | 同 | 去掉 `--ph_aware_filter` |
| run_nokeep | −seq_keep | 同 | `--lambda_keep 0` |

蛋白族 A1 列无意义（protein 无配体 pocket=0）→ 删去 = 5 runs。

## 4. 配方（权威源已核对）
### 蛋白族 FULL（v12.2 配方，MoMPNN）
```
--device cuda:6 --epochs 10 \
--weights MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt \
--labels <prot_subset> --dompdb data/cath/S40/dompdb \
--lambda_c 0.5 --lambda_kl 0.05 --lambda_keep 0.5 \
--charge_temp 0.5 --perturb_prob 0.3 --placeholder_prob 0.15 \
--decouple_perturb --decouple_range 12.0 \
--ph_aware_filter --structure_boost 1.5 \
--v12_supervision --frac_floor 0.5 --gravy_margin 0.4 --lambda_v12 0.2 --lambda_target 0.2 --sasa_threshold 0.25 \
--pocket_mode keep --pocket_cutoff 8.0 --pocket_floor 0.7 --pocket_ceil 1.3 --lambda_pocket 0.0 \
--out_dir ablation/runs/prot/run_FULL \
--log_file ablation/runs/prot/run_FULL/train.log
```
（`--lambda_pocket 0` 显式关 pocket，与 v12.2 eval 口径 `keep,λ_pocket=0` 一致；protein 无配体
  → keep/global 分区等价。）

### 配体族 FULL（v14 配方，LigandMPNN / atom25）
```
--device cuda:6 --epochs 16 --ligand \
--weights LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
--labels <lig_subset> --dompdb data/ligand_train/all_pdb \
--lambda_c 0.5 --lambda_kl 0.05 --lambda_keep 0.5 \
--charge_temp 0.5 --perturb_prob 0.3 --placeholder_prob 0.15 \
--decouple_absolute --decouple_abs_lo=-35.0 --decouple_abs_hi=20.0 \
--ph_aware_filter --structure_boost 1.5 \
--v12_supervision --frac_floor 0.5 --gravy_margin 0.4 --lambda_v12 0.2 --lambda_target 0.2 --sasa_threshold 0.25 \
--pocket_mode global --pocket_cutoff 8.0 --pocket_floor 0.8 --pocket_ceil 1.3 --lambda_pocket 0.3 \
--out_dir ablation/runs/lig/run_FULL \
--log_file ablation/runs/lig/run_FULL/train.log
```
原子上下文：train_finetune 现版本 protein number_of_ligand_atoms=0；ligand=25（bug 修复后统一）。

## 5. 评估
- **前向 val-loss**（终末 epoch）：`code/tests/val_loss_curve.py`，**同族统一 FULL 口径 tag**
  保证横向可比——蛋白族 `--tag v12_2`、配体族 `--tag v14_ligand`；`--epoch_list <final>`
  `--ckpt_dir ablation/runs/{prot,lig}/run_<tag>`。
  - 蛋白 val 集：`data/cath/labels_holdout_train.npz`(1176) + dompdb
    `data/cath/S40/dompdb_pdb`；supp `data/cath/labels_v12_3_valsupp.npz`(23) + supp_dompdb
    `data/cath/S40/dompdb_valsupp` → 合计 1199。
  - 配体 val 集：`data/ligand_train/labels_v14_valset_805.npz`(805) + dompdb
    `data/ligand_train/v14_valset_pdb`。
  - 指标：ce / cd / total（+ 组成相关项 v12_comp/gravy/pocket）。
- **生成式抽查**：小蛋白 + 1 长蛋白 / 1 RNA 代表，native+n2/p2 n30 → H2(dev≤2)、native 臂带电
  保留率（删减 proxy）。
- 报告：`ablation/report/2026-09-05_ablation_prot.md` / `_lig.md`（FULL vs 各 off 的相对变化表 +
  排序结论 + "under-train 压平"检查）；对比图数据 `ablation/figure/`。

## 6. GPU / 纪律
- GPU6(cuda:6) 主跑（2026-09-05 02:xx nvidia-smi 空）；单卡串行蛋白族→配体族。
- 不抢 0/1/2/3/4/5/7（他人在用）。GPU4 util 99% → 不并行。
- 每 run 独立 log；可 resume。每步落盘。过程 `session/2026-09-05_ablation_run.md`。
