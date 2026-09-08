# Session — L11 Exp1: native 电荷下蛋白 vs 配体模式设计能力（2026-09-09）

## 设计（用户 2026-09-09 更正后）
- 目标：增强 L11(uL11)–RNA 结合；所有设计固定链 I 残基 3,5,9,34,35,89,124,131,134,135（native）。
- **更正**：模式绑定输入，不做跨输入交叉。4 组 × n=100：
  | grp | 模式 | 权重 | 输入 | 条件 | pH / target |
  |-----|------|------|------|------|-------------|
  | prot_C1 | 蛋白 MoMPNN+v12.2 | mompnn_...ckpt + finetune_v12_2/finetune_epoch030.pt | L11.pdb | C1 | 7.4 / +6.87 |
  | prot_C2 | 蛋白 MoMPNN+v12.2 | 同上 | L11.pdb | C2 | 8.0 / +6.59 |
  | lig_C1 | 配体 LigandMPNN+v14 | ligandmpnn_32_010_25.pt + finetune_ligand_v14_rna/finetune_epoch050.pt | L11_RNA.pdb | C1 | 7.4 / +6.87 |
  | lig_C2 | 配体 LigandMPNN+v14 | 同上 | L11_RNA.pdb | C2 | 8.0 / +6.59 |
- 校准：L11 表外 → global（蛋白 `output/charge_calibration_v12_2.json`；配体 `output/charge_calibration_v14_ligand_clean.json`）；另跑 raw(不校准) n=30/组。
- 结论口径：同 native target 下蛋白 vs 配体模式差异 + pH7.4 vs pH8 条件差异。

## 已核实（CPU 侧，2026-09-09 01:0x）
- native 链 I（两输入 PDB 一致，141 aa，编号 1..141）：长度一致；固定位 AA 3=K 5=Q 9=K 34=I 35=M 89=S 124=M 131=T 134=S 135=M。
- native 净电荷（HH）：pH7.4 = +6.874 ≈ Q6.87；pH8.0 = +6.585 ≈ Q6.59。
- L11.pdb：纯蛋白链 I；L11_RNA.pdb：链 I 蛋白 + 链 A RNA(A/C/G/U, 2304 ATOM)。
- 解析确认：protein 模式只取链 I（RNA 归入 other_atoms，featurize 忽略）；ligand 模式把链 A RNA 作配体上下文（2304 atoms）。
- run_guided 固定位机制有效：dry-run 生成序列 10 位固定均 = native（fixed_ok=True）。
- env 冒烟：confumpnn-esmfold 有 esm 2.0.0（esmfold_3B_v1.pt 缓存）；confumpnn-temberture 可加载（native Tm≈51.8）；protein-sol perl 管线可跑（native %sol≈94.6）。

## 执行状态
- 2026-09-09 01:09 启动后台：`exp1_cpu_worker.sh`（CPU 采样 4 组 main + 4 raw diag + postprocess）与 `exp1_gpu_fold_worker.sh`（轮询空卡后 ESMFold+TM+Tm+Sol）。
- GPU 全 1-7 99%（他人训练）；GPU0 0% util 但 142/143GB 被他人占用。采样先走 CPU（线程上限 48），ESMFold 等空卡。

## 产物位置
- manifest: `L11design/test/exp1_groups.json`
- 采样/序列: `L11design/output/exp1/<grp>/seqs.fa` + `summary.json` + `seqs_clean.fa` + `meta.json`；raw diag: `output/exp1/rawdiag/<grp>/`
- 折叠/打分: `L11design/data/exp1/<grp>/{folds,plddt.csv,tm.csv,seqs.fa.tm.csv,seqs.fa-protein_sol_prediction.txt,merged.csv}`
- 统计: `L11design/data/exp1/summary_stats.json`（含 4 组 mean±std + native 参考 + pairwise Wilcoxon）
- 脚本: `L11design/test/exp1_{groups.json,run_sampling.sh,postprocess.py,run_fold.sh,run_cpu_scores.sh,analyze.py,exp1_master.sh,exp1_cpu_worker.sh,exp1_gpu_fold_worker.sh}`
- 报告正文（markdown）由本子代理最终回复全文返回（工具护栏禁止子代理直接写 report.md）；主会话据 `summary_stats.json` + 返回文本落盘 `L11design/test/report_exp1.md` 后 push。

## 完成状态（2026-09-09 03:00）
- 4 组 × n100 采样完成（校准 global）＋ 4 组 raw(n30) 诊断完成；postprocess：固定位 400/400 = native（0 错配）。
- ESMFold/TM/Tm/Sol 全链完成（每 data/exp1/<grp> 101 折叠 + plddt/tm/tmCsv/sol 全 OK）。
- 分析完成：`summary_stats.json`。要点见最终返回。

## 执行中修的两个 bug（记录备查）
1. `exp1_run_sampling.sh`：rawdiag 子目录日志路径不存在 → 改用扁平日志名 `sample_${sub//\//_}.log`。
2. `exp1_gpu_fold_worker.sh`：变量名 `GROUPS` 与 bash 特殊数组（用户 gid，此处 4021）冲突 → 更名 `GRP_LIST`；旧 worker 曾卡在 `waiting data for 4021` 被 kill 重启。
