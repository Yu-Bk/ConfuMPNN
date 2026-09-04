# v14 配体干净全链重跑 — 自动检查点日志（2026-09-04）

> 由 claude 会话自动维护：每个检查点更新本文件 + git 归档。
> 权威口径判据见 `analysis/report/2026-09-03_validation_standards.md`（coverage in/boundary/out；两口径 big-global + 小样本；H2 单臂 |dev|≤2）。

## 一、为什么重跑（背景，勿误读旧链）
- 旧 v14 链 ③ 采样基于旧 manifest（含 2E9R_X、无 5O60_E）→ 组成/统计/回折基于废弃验证集；
- ⑦⑧⑨ 曾瞬时 Permission denied + DONE 被无条件 touch = 假完成；
- 5O60_E（核糖体 RNA 结合代表，held-out）的 组成/H1/Tm-Sol 从未干净跑过。
- → 本链用**新 in-10 manifest（validation_manifest_v14_in.json：10 in 含 5O60_E、无 2E9R_X）** + boundary 1A65，从头一致重算。

## 二、运行参数
- 脚本：`/tmp/run_v14_ligand_validation_clean.sh`（207 行，每步产物检查，DONE 只在全链成功时写）
- 权重：`output/finetune_ligand_v14_rna/finetune_epoch050.pt`（v14 RNA/DNA+A1global，50ep）
- 训练集注记：`labels_v14_final.npz`（5371 域，RNA/DNA 7.7%）
- 输出：`output/*_clean`（DIAG/CAL/COMP/STATS/H3OUT/PROPKA_DIR + generalization_ligand_v14_clean + tm_sol_ligand_v14_clean）
- GPU：DIAG/SAMP/ESM = cuda:6（当时唯一空闲）；启动 PID 1581049，start 10:02:30
- 日志：`log/v14_ligand_validation_clean.stdout` + `log/v14_ligand_{diag,val_sample,esmfold,tm,stats,uncond_sample,tm_seqs,tm_uncond}_clean.log`

## 三、检查点进度（由自动检查更新）

| 步 | 阶段 | 产物判据 | 状态 | 关键数值 |
|---|---|---|---|---|
| ① | 配体诊断 slope | `output/v14_ligand_diag_response_clean.json` | 🔄 运行中 | — |
| ② | 校准表 | `output/charge_calibration_v14_ligand_clean.json` | ⏳ | — |
| ③ | 泛化采样 n50（in10×5） | `output/generalization_ligand_v14_clean/ligand/*/validation.json` ≥10 | ⏳ | — |
| ④ | 组成 | `output/v14_ligand_comp_clean.json` | ⏳ | — |
| ⑤ | ESMFold H1 + TM | plddt.csv≥50 + tm.csv≥50 | ⏳ | — |
| ⑥ | H2 统计 | `output/v14_ligand_gen_stats_clean.json` | ⏳ | — |
| ⑦ | PROPKA H4 | `output/propka_v14_ligand_clean/*.json` ≥6 | ⏳ | — |
| ⑧ | uncond + H3 | `output/h3_ligand_v14_clean.json` | ⏳ | — |
| ⑨ | Tm/Sol S2 | `output/tm_sol_ligand_v14_clean/tm_sol_summary.json` | ⏳ | — |
| ★ | boundary 1A65 | `.../ligand/1A65/validation.json` | ⏳ | — |
| 终 | DONE | `log/v14_ligand_validation_clean.DONE` | ⏳ | — |

## 四、检查点时间戳记录
> 归档游标：`archived_stage = 0`（每归档完成一个阶段就 +1；检查点据此判断"是否有新阶段完成、要不要 push"）

（仅在某阶段完成时追加一行：时间 / 阶段 / 关键数值 / 观测；阶段进行中不写不 push）
