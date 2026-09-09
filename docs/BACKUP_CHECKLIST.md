# 手动保存/下载清单（BACKUP CHECKLIST，2026-09-09）

> 本文件是"git 之外还需你手动保存/下载"的**唯一清单**。git 内(代码/脚本/日志/报告/论文JSON)随 `git push` 已在 GitHub；此处只列**没进 git** 的。
> 归档唯一规则/去向见 `docs/MIGRATION_GIT_POLICY.md`；大文件 SHA 见 `data/SHA256SUMS.txt`。

## A. 模型权重 → GitHub Release（5 个，已暂存 `weights_release/`）
| 文件(在 weights_release/) | 大小 | 说明 |
|---|---|---|
| `condition_encoder_v12_2_last.pt` | 296K | 蛋白当前交付（MoMPNN） |
| `condition_encoder_v12_3_last.pt` | 296K | 蛋白长蛋白版 |
| `condition_encoder_v14_ligand_epoch050.pt` | 887K | 配体当前交付（LigandMPNN RNA） |
| `condition_encoder_v7_last.pt` / `v9_epoch030.pt` | 296K/887K | 历史 |
→ 上传到 GitHub `Yu-Bk/ConfuMPNN` Release（见 `weights_release/README.md`）。**状态：待你上传/核对。**

## B. 大文件 → 网盘/NAS 异地（tar 在 `~/data/nfs/IC/baokun_yu/ConfuMPNN_backup/` 与 `L11design/backup/`；**仍在本机同 NAS，需拷到网盘/异机**）
| tar | 大小 | 内容 |
|---|---|---|
| `ConfuMPNN_data_2026-09-05.tar.gz` | 3.7G | data/ 全量（CATH/配体/验证/7K00/各 labels） |
| `ConfuMPNN_output_heavy_2026-09-05.tar.gz` | 1.7G | finetune_* 权重 + generalization_* 采样/折叠 + tm_sol/propka/largen/fixbinding/ribosome |
| `ConfuMPNN_final_extend_2026-09-06.tar.gz` | 34M | 09-06 补充实验序列（exp_control/exp_pH/exp7_H1/exp2_comp） |
| `confumpnn_data_v1_20260819.tar.gz` (历史) | 2.7G | 早期 data v1 |
| `confumpnn_artifacts_v1_20260819.tar.gz` / `confumpnn_tools_temberture_v1_20260819.tar.gz` | 0.4G/0.5G | 早期产物 / TemBERTure 工具(无 git，必存) |
| `L11design/backup/L11_exp1_2026-09-09.tar.gz` | 9M | L11 Exp1（4 组×100 采样+折叠） |
| `L11design/backup/exp2_output_data_20260909_0434.tar.gz` | 55M | L11 Exp2 |
| `L11design/backup/L11_exp34_2026-09-09.tar.gz` | 62M | L11 Exp3+Exp4（标定重做全量） |
| `ConfuMPNN_output_remaining_2026-09-09.tar.gz` | ~14M | output 剩余小实验(ph_scan/transfer/pocket/propka_prep 等) |

**状态：tar 已生成但都还在同一台机器 → 需你下载并传到网盘/异机（异地才算备份）。**

## C. 无需手动保存（可复现/在 git/可再生）
- 全部代码、脚本、日志、报告、论文 JSON：已在 git（GitHub）。
- 外部源码（LigandMPNN/MoMPNN/foundry/protein_sol_mcp/TemBERTure 前四个可再 clone；TemBERTure 见 B 包）。
- conda 环境：按 `docs/SETUP_NEW_MACHINE.md` 重建；ESMFold 权重首次运行自动下。
- data 重建脚本在 git；`ablation/data/*.npz`（小）在 git。

## D. 说明
- `output/` **已全量覆盖**：heavy(09-05, finetune_*/generalization_*/tm_sol/propka/ribosome/largen/fixbinding) + final_extend(09-06, exp_control/exp_pH/exp7_H1/exp2_comp) + remaining(09-09, ph_scan/transfer/pocket/paper_gap/propka_prep 等) + L11(exp1/exp2/exp34)。根 `output/*.json`=论文关键数字已在 git。无需另打全量 5.4G 包。

## 登记（传完在 `docs/MIGRATION_GIT_POLICY.md` 底表填写日期/位置）
