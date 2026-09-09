# L11 蛋白设计实验计划（2026-09-09）

> 目标：增强 L11（uL11）与 RNA 的结合；**所有设计固定链 I 残基 3,5,9,34,35,89,124,131,134,135（10 位，native）**以保结合界面。
> 模式：蛋白模式（MoMPNN+v12.2）；配体模式（LigandMPNN RNA 上下文+v14）。native 电荷（HH）：pH7.4→Q+6.87；pH8→Q+6.59。
> 目录约定：设计序列 → `output/`；每条序列折叠/打分数据 → `data/`；脚本/分析报告 → `test/`；大文件打包 → `backup/`（不 push）。input/ 与 PLAN 与 test 报告 push；output/data/backup 仅本地+backup tar。
> 输入：`input/L11.pdb`（纯蛋白，链 I）；`input/L11_RNA.pdb`（含 RNA 配体，链 I 蛋白 + RNA）。

## 固定位点（两种模式一律）
`--fixed_residues "I3 I5 I9 I34 I35 I89 I124 I131 I134 I135"`（链 I，native 保留）。

## 实验一：native 电荷下两模式设计能力（2026-09-09 更正：模式绑定输入，2 模式 × 2 pH-电荷，n=100/组 = 4 组×100）
> 蛋白模式只用 `L11.pdb`；配体模式只用 `L11_RNA.pdb`（不做跨输入交叉）。
- 条件：C1 = pH7.4 / Q+6.87；C2 = pH8 / Q+6.59。
- 4 组：
  1. 蛋白模式(MoMPNN+v12.2) × L11.pdb × C1
  2. 蛋白模式 × L11.pdb × C2
  3. 配体模式(LigandMPNN+v14) × L11_RNA.pdb × C1
  4. 配体模式 × L11_RNA.pdb × C2
- 比较：净电荷@该pH、回收、Tm、RMSD(回折 vs native 链 I)、Sol、pLDDT；结论 = 同 native target 下**蛋白 vs 配体模式**差异 + **pH7.4 vs pH8** 条件差异。
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
- 采样+回折量大（Exp1 400 + Exp2 2400 条 ESMFold）；GPU 先做 CPU 准备，再在空卡跑（暂 GPU 全 99% 占用，等空或轮询）。
- 回折可分批 resume；完成打包 `backup/`。

## 复现 / 统计 / 校验补充（2026-09-09）
- **种子**：每组固定 seed（`seed=42 起 + 组号`），可复现。
- **native 参考与编号**：native 序列 = `L11.pdb` 链 I（protein_only 与 RNA 文件的链 I 一致）；RMSD/TM/回收都以它为参考；先核对两文件的链 I 残基编号/长度一致，并确认固定位 3,5,9,34,35,89,124,131,134,135 落在链 I 且都 = native（报告里列这 10 位 native AA）。
- **统计**：Exp1 比较（模式间 / pH7.4-vs-8）样本小（每指标 n=100 均值），给 mean±SD + 差异方向 + Wilson/自助 CI，不下强显著性结论（n=4 组×100）；Exp2 给 target 递增的电荷-性质趋势。
- **校准**：L11 表外 → global（蛋白 v12_2 表、配体 v14 clean 表），报告同时给 raw 电荷分布。
- **后续（湿实验候选，另起）**：uL11 已定为湿实验蛋白；Exp2 高正电 target(+8~+12) 序列作为结合增强候选池，候选挑选（电荷+回收+折叠+稳定多判据）待本组完成后按用户指示做。

---

## 实验三（规划，2026-09-09）：L11 现场小样本标定精度补充（Bsmall）——待执行
> 目的：验证"表外高 pI 蛋白用现场小样本标定后，电荷命中是否明显提升"（Exp1/2 仅用 global 校准，过冲 +2~+5）。
> **方法澄清**：标定样本 = 在 L11 骨架上**新采的探针批**（建议 5 target ≈ native−2, native, native+2, +6, +10 × 每档 n10 = 50 条），拟合 L11 自身 响应 slope/intercept；**不是用 Exp1 里"设计得好"的序列**。"设计得好"的序列用于湿实验候选挑选，不用于标定。
> 范围：蛋白模式（×L11.pdb）与配体模式（×L11_RNA.pdb），各拟合后对 target {+6.87, +8, +10, +12}(pH7.4) 重采样（n 可 100-300，与 Exp2 同构以便对比）。
> 判据：标定后 mean dev / |dev|≤2 命中率 vs Exp2（global）提升幅度；给"治增益不治散布"的定量结论。
> 目录（与 Exp1/2 分开子文件夹）：采样 `output/exp3/<mode>/`；折叠/打分 `data/exp3/<mode>/`；报告 `test/report_exp3.md`；脚本 `test/exp3_*.py`；大件打包 `backup/L11_exp3_*.tar.gz`。
