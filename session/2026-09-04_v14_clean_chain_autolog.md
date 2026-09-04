# v14 配体干净全链重跑 — 自动检查点日志（2026-09-04）

> 由 claude 会话自动维护：每个检查点更新本文件 + git 归档。
> 权威口径判据见 `analysis/report/2026-09-03_validation_standards.md`（coverage in/boundary/out；两口径 big-global + 小样本；H2 单臂 |dev|≤2）。

## 一、为什么重跑（背景，勿误读旧链）
- 旧 v14 链 ③ 采样基于替换前的 manifest（缺 5O60_E）→ 组成/统计/回折基于废弃验证集；
- ⑦⑧⑨ 曾瞬时 Permission denied + DONE 被无条件 touch = 假完成；
- 5O60_E（核糖体 RNA 结合代表，held-out）的 组成/H1/Tm-Sol 从未干净跑过。
- → 本链用**新 in-10 manifest（validation_manifest_v14_in.json：10 in 含 5O60_E）** + boundary 1A65，从头一致重算。

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
| ① | 配体诊断 slope | `output/v14_ligand_diag_response_clean.json` | ✅ | valid n=10 slope 均值 **1.473**（6D2O1.63/1AS2 1.45/2FEO0.93/5CQH1.05/1CGE0.98/1BJ42.18/21KL_A1.74/5O60_E1.78/3MXB_A1.24/9DWG_L1.75）+ trainish 8 | 
| ② | 校准表 | `output/charge_calibration_v14_ligand_clean.json` | ✅ | global slope **1.492**/intercept −1.260；per_protein 18 条（10 valid + 8 trainish） |
| ③ | 泛化采样 n50（in10×5） | `output/generalization_ligand_v14_clean/ligand/*/validation.json` ≥10 | ✅ | **10/10** 蛋白完成（每蛋白 51 条 seqs = 50 gen + native ref） |
| ④ | 组成 | `output/v14_ligand_comp_clean.json` | ✅ | **10/10 蛋白**（倍率 0.43-0.69× native，删减捷径未根治；修 compare_comp_ligand 硬编码旧 manifest → 加 `--manifest` 后补全） |
| ⑤ | ESMFold H1 + TM | plddt.csv≥50 + tm.csv≥50 | ✅ | **plddt 50/50 + tm.csv 50/50**（GPU6，11:05→13:0x 完成） |
| ⑥ | H2 统计 | `output/v14_ligand_gen_stats_clean.json` | ✅ | 13:10 产出（25KB） |
| ⑦ | PROPKA H4 | `output/propka_v14_ligand_clean/*.json` ≥6 | ✅ | **6/6**（1BJ4/21KL_A/3MXB_A × native/n8） |
| ⑧ | uncond + H3 | `output/h3_ligand_v14_clean.json` | ✅ | uncond 10/10 fa；**H3 50/50（100%）**，in-10 全臂通过 |
| ⑨ | Tm/Sol S2 | `output/tm_sol_ligand_v14_clean/tm_sol_summary.json` | ✅ | **S2 = 0/50 明显恶化**（新 in-10 全 10；修 v12_2_ligand_tm_sol_summarize.py 硬编码旧 PDBS → `--manifest` 后重跑） |
| ★ | boundary 1A65 | `.../ligand/1A65/validation.json` | ✅ | native −26.85，5 臂 dev 2.1-3.2 欠冲（能力边界档） |
| 终 | DONE | `log/v14_ligand_validation_clean.DONE` | ✅ | 全链完成（resume PID 2030311） |

## 四、检查点时间戳记录
> 归档游标：`archived_stage = 4`（每归档完成一个阶段就 +1；检查点据此判断"是否有新阶段完成、要不要 push"）

（仅在某阶段完成时追加一行：时间 / 阶段 / 关键数值 / 观测；阶段进行中不写不 push）
- 10:57 ①完成：valid n=10 slope 均值 1.473（未校准响应增益，与 §4.2 旧值 1.49±0.40 一致）
- 10:57 ②完成：校准表 global slope 1.492 / intercept −1.260；per_protein 18 条
- 11:1x ③完成：泛化采样 10/10 蛋白（各 51 条 seqs = 50 gen + native）
- 11:1x ④⚠️：comp_clean 仅 5/10 蛋白（1AS2 0.46/2FEO 0.46/5CQH 0.43/1CGE 0.60/1BJ4 0.46）→ 部分，终局前需重跑 compare_comp_ligand 补全 6D2O/21KL_A/5O60_E/3MXB_A/9DWG_L；随后进入 ⑤ ESMFold（预计 3-5h）
- 13:0x ⑤完成：ESMFold **plddt 50/50 + tm.csv 50/50**（GPU6 满载，esmfold_score 进程结束）
- 13:10 ⑥完成：`v14_ligand_gen_stats_clean.json`（25KB）
- 13:1x ⑦完成：PROPKA **6/6**（1BJ4/21KL_A/3MXB_A × native/n8）
- 13:12 ⑧🚨崩溃：uncond 采样 10/10 完成，但 **H3 统计 FileNotFoundError `ref/1C6O_ref.pdb`** → 整链异常终止（未写 DONE）。根因=`code/tests/h3_charge_legality.py:55` 硬编码旧 PDBS 含已删二聚体 1C6O/1AZM/1AXW/1AG0、缺新成员 → 已删蛋白已不在新 manifest/ref
- 13:2x 🔧修复：h3 脚本加 `--manifest` 覆盖硬编码 PDBS（默认不变，向后兼容）；改 `for pdb in PDBS`→`pdbs`；语法过
- 13:31 ⑧-H3重跑：`--manifest validation_manifest_v14_in.json` → **50/50（100%）** 全臂通过，`output/h3_ligand_v14_clean.json` 写入
- 13:32 启动 resume（`/tmp/resume_v14_clean_tail.sh` PID 1934207）：只补 ⑨ Tm/Sol（CPU 1-3h）+ ★ 1A65（GPU6）+ DONE；⑤⑥⑦⑧ 产物齐全不重跑
- 13:34 ⑨🚨第二次崩：temberture FileNotFoundError `seqs/1AS2/arm_n2/seqs.fa` → 根因 = resume 用**相对路径**建 symlink（相对目标相对 symlink 自身目录解析→全 50 悬空）；改绝对路径（`$ROOT/output/...`）后悬空 0
- 14:38 resume 重跑（PID 2030311）：⑧-H3 幂等 50/50 → ⑨ temberture（CPU ~2h）→ protein-sol → 汇总
- ~17:0x ⑨完成 S2 0/50、★ 1A65 完成、**DONE 落盘 = clean 链全绿**
- 终局补跑：④ comp 修复（`compare_comp_ligand --manifest in-10`）→ **10/10**（倍率 0.43-0.69）；⑨ tm_sol 汇总修复（`--manifest in-10`）→ **新 in-10 全 10、S2 0/50**（初版错算在旧集合上）
- 权威报告：`analysis/report/2026-09-04_v14_clean_validation.md`（H2 45/50=90%、H1 50/50、H3 50/50、H4 caveat、S2 0/50、组成删减未根治、1A65 边界档）
