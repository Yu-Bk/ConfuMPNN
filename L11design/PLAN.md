# L11 蛋白设计实验计划（2026-09-09）

> 目标：增强 L11（uL11）与 RNA 的结合；**所有设计固定链 I 残基 3,5,9,34,35,89,124,131,134,135（10 位，native）**以保结合界面。
> 模式：蛋白模式（MoMPNN+v12.2）；配体模式（LigandMPNN RNA 上下文+v14）。native 电荷（HH）：pH7.4→Q+6.87；pH8→Q+6.59。
> 目录约定：设计序列 → `output/`；每条序列折叠/打分数据 → `data/`；脚本/分析报告 → `test/`；大文件打包 → `backup/`（不 push）。input/ 与 PLAN 与 test 报告 push；output/data/backup 仅本地+backup tar。
> 输入：`input/L11.pdb`（纯蛋白，链 I）；`input/L11_RNA.pdb`（含 RNA 配体，链 I 蛋白 + RNA）。

## 固定位点（两种模式一律）
`--fixed_residues "I3 I5 I9 I34 I35 I89 I124 I131 I134 I135"`（链 I，native 保留）。

## 实验一：native 电荷下两模式两输入设计能力（2 模式 × 2 输入 × 2 pH-电荷，n=100/组 = 8 组×100）
- 条件：C1 = pH7.4 / Q+6.87；C2 = pH8 / Q+6.59。
- 组合（4 组/条件）：
  1. 蛋白模式 × L11.pdb（蛋白-only）
  2. 蛋白模式 × L11_RNA.pdb（取其链 I，RNA 忽略 → 检验"输入含配体文件对蛋白模式是否有影响"）
  3. 配体模式 × L11_RNA.pdb（链 I + RNA 上下文）
  4. 配体模式 × L11.pdb（无 RNA 配体 → 配体模式无配体对照）
- 比较：净电荷@该pH、回收、Tm、RMSD(回折vs native 链 I)、Sol、pLDDT；判定"同模式不同输入 / 同输入不同模式"是否有影响。
- 产物：seqs.fa+csv → output/exp1/；逐序列折叠+指标 → data/exp1/；分析 → test/report_exp1.md。

## 实验二：多目标电荷可设计性（pH7.4 × 4 target × 2 模式 × n=300）
- target ∈ {+6.87(native), +8, +10, +12}；pH7.4。
- 每 (target, 模式) 300 条 = 4×2×300=2400 条。
- 指标：电荷偏差/达标、回收、Tm、RMSD、Sol、pLDDT → 看 ConfuMPNN 是否设计出合理、电荷可控的蛋白。
- 产物：output/exp2/（seqs）；data/exp2/（折叠+打分）；test/report_exp2.md。

## 打分/折叠管线（每条序列）
1. net_charge@pH（`net_charge`）+ 固定位校验（I3…135 = native）。
2. ESMFold 回折（confumpnn-esmfold）→ pLDDT。
3. TM & RMSD vs 参考 = `input/L11.pdb` 链 I（`tm_score.py --folds --ref --out`）。
4. Tm（confumpnn-temberture）、Sol（protein_sol_mcp）。
- 校准：L11 不在校准表 → 用 global 校准（蛋白 `charge_calibration_v12_2.json` global；配体 `charge_calibration_v14_ligand_clean.json` global）；记录 raw 与校准后。

## GPU/时序
- 采样+回折量大（~3200 条 ESMFold）；GPU 先做 CPU 准备，再在空卡跑（暂 GPU 全 99% 占用，等空或轮询）；两子代理各领一卡（Exp1=GPU2、Exp2=GPU6，若空）。
- 回折可分批 resume；完成打包 `backup/`。
