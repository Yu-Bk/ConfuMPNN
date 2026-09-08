# L11 Exp1 报告 — native 电荷下蛋白 vs 配体模式设计能力（2026-09-09）

> 实验范围（2026-09-09 用户更正）：模式绑定输入，2 模式 × 2 pH-电荷条件 = 4 组 × n=100。
> 全部设计固定链 I 残基 3,5,9,34,35,89,124,131,134,135（native 保留，保结合界面）。

## 摘要（结论先行）
native 电荷目标下（L11 表外蛋白，用 global 校准）：
1. 电荷控制不达标：校准后均值 vs 目标 prot_C1 +8.57(6.87,Δ+1.7)、prot_C2 +10.53(6.59,Δ+3.9)、lig_C1 +10.19(Δ+3.3)、lig_C2 +11.93(Δ+5.3)；raw 更高(prot 9.4–10.5, lig 12.4–15.2)。→ global 校准对高 pI L11(native pI≈10.1) 回拉有限。
2. 结构保持蛋白模式优于配体模式：同条件 Wilcoxon TM p<0.001、RMSD p≤0.0005；但两者均低于 native 自洽基线(native ESMFold TM 0.79 / RMSD 2.67 / pLDDT 76.9 vs 设计 54–56)。
3. 可溶性配体模式显著更高：%sol≈95 vs 蛋白≈79（p<10⁻²⁶；与 v14 组成删减倾向一致）。
4. pH 差异主要反映在电荷：两模式 pH8 组电荷显著更高(p≈0.02–0.024)，尽管 pH8 native 目标更低(6.59<6.87)；结构/稳定/可溶指标 pH 间基本不显著。回收率两模式≈27–28%。
5. 固定位 100% 生效：400/400 设计序列 10 固定位 = native（0 错配）。
6. Tm 两模式≈54–55°C（native 52.9），未恶化。

## 1. 设置
- 蛋白模式：MoMPNN mompnn_temberture_tm_esm_6_4_4_b01.ckpt + v12.2 编码器 finetune_epoch030.pt；输入 L11.pdb（纯蛋白链 I）。
- 配体模式：LigandMPNN ligandmpnn_v_32_010_25.pt + v14 RNA finetune_epoch050.pt；输入 L11_RNA.pdb（链 I + RNA）。
- 条件：C1=pH7.4/Q+6.87；C2=pH8.0/Q+6.59（native HH）。
- 校准：global（蛋白 charge_calibration_v12_2.json；配体 charge_calibration_v14_ligand_clean.json）；raw=同条件 --calibrate off n=30/组。
- 固定位：--fixed_residues "I3 I5 I9 I34 I35 I89 I124 I131 I134 I135"；n=100/组，种子 111/222/333/444。

| grp | 模式 | 输入 | pH/target |
|-----|------|------|-----------|
| prot_C1 | 蛋白 v12.2 | L11.pdb | 7.4/+6.87 |
| prot_C2 | 蛋白 v12.2 | L11.pdb | 8.0/+6.59 |
| lig_C1 | 配体 v14 | L11_RNA.pdb | 7.4/+6.87 |
| lig_C2 | 配体 v14 | L11_RNA.pdb | 8.0/+6.59 |

## 2. 固定位校验
native 链 I(141aa) 固定位 AA：3K 5Q 9K 34I 35M 89S 124M 131T 134S 135M；native 电荷 pH7.4=+6.874、pH8=+6.585。400 条设计固定位错配 0 条（meta.json fixed_mismatch 全 0）。

## 3. 管线
net_charge@组pH + 固定位比对 + 全 141 位 identity 回收率 → ESMFold pLDDT → US-align TM/RMSD vs L11.pdb 链 I → TemBERTure Tm → protein-sol %sol。
逐序列 merged.csv：data/exp1/<grp>/merged.csv；统计 summary_stats.json。

## 4. 结果表（mean±std, n=100；native 参考单值）
| grp | 电荷@pH | pLDDT | TM-score | RMSD(Å) | Tm(°C) | %sol | 回收率% | raw电荷 |
|-----|---------|-------|----------|---------|--------|------|---------|---------|
| prot_C1 | 8.57±4.71 | 55.9±5.0 | 0.663±0.077 | 3.36±0.47 | 55.5±6.7 | 78.7±8.9 | 27.9±2.0 | 9.38±4.92 |
| prot_C2 | 10.53±4.54 | 55.0±6.7 | 0.652±0.105 | 3.38±0.52 | 55.4±6.7 | 79.8±8.6 | 27.3±2.5 | 10.54±4.63 |
| lig_C1 | 10.19±4.04 | 53.9±7.5 | 0.607±0.113 | 3.62±0.59 | 54.5±1.9 | 95.0±5.9 | 27.1±2.1 | 12.39±4.02 |
| lig_C2 | 11.93±4.54 | 54.8±5.2 | 0.626±0.074 | 3.55±0.41 | 54.5±1.9 | 95.8±5.3 | 27.2±2.3 | 15.16±4.74 |
| native | 6.87 | 76.9 | 0.794 | 2.67 | 52.9 | 94.6 | — | — |

## 5. 蛋白 vs 配体模式差异（同条件 Mann-Whitney U）
- C1: 电荷 8.57vs10.19 p=0.032*；TM 0.663vs0.607 p<0.001***；RMSD 3.36vs3.62 p<0.001***；Tm 55.5vs54.5 p=0.09；%sol 78.7vs95.0 p<0.001***；回收率 27.9vs27.1 p=0.011*；pLDDT ns。
- C2: 电荷 10.53vs11.93 p=0.12 ns；TM 0.652vs0.626 p<0.001***；RMSD 3.38vs3.55 p<0.001***；Tm 55.4vs54.5 p=0.015*；%sol 79.8vs95.8 p<0.001***；回收率 ns；pLDDT ns。
结论：配体模式(v14)电荷过冲更大、回折骨架保持更弱、但预测可溶性显著更高；蛋白模式(v12.2)骨架保真更好。

## 6. pH7.4 vs pH8（同模式内）
- 蛋白: 电荷 8.57vs10.53 p=0.022*；其余 p>0.4 ns（回收率 27.9vs27.3 p=0.027*，幅度<1%）。
- 配体: 电荷 10.19vs11.93 p=0.024*；其余 p>0.7 ns。
结论：pH8 电荷过冲更明显（目标更低反而更高），电荷可控性随 pH 上移变差；结构/稳定/可溶对 pH 不敏感。

## 7. raw vs 校准（global）
prot_C1 9.38→8.57；prot_C2 10.54→10.53；lig_C1 12.39→10.19；lig_C2 15.16→11.93。
global 校准对配体拉回较明显、对蛋白几乎无效；校准后仍高于 target +1.7~+5.3 → 表外蛋白电荷达标需小样本现场标定(build_calibration_small.py)。

## 8. 局限
L11 高 pI 碱性表外困难样本，电荷散布大(std≈4–4.7)；ESMFold pLDDT 54–56 明显低于 native 77，但 TM≈0.63–0.66/RMSD≈3.4–3.6Å 说明仍能回折到 native 骨架附近；回收率 27–28% 含 10 强制固定位；配体 %sol 高与 v14 组成删减一致，需结合组成/GRAVY/口袋判据；统计为描述性+Mann-Whitney(未多重校正)。

## 9. 产物
output/exp1/<grp>/{seqs.fa,summary.json,seqs_clean.fa,meta.json}；output/exp1/rawdiag/<grp>/；
data/exp1/<grp>/{folds,plddt.csv,tm.csv,seqs.fa.tm.csv,seqs.fa-protein_sol_prediction.txt,merged.csv}；
data/exp1/summary_stats.json；test/{exp1_groups.json,exp1_*.sh/py,session_exp1.md,report_exp1.md}。
