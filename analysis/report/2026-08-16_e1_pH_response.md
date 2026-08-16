# E1 对照实验（第一阶段）：pH 电荷响应对比

> 日期：2026-08-16　|　对应 `PROJECT_EXTEND.md` Stage E1
> 结论：**MoMPNN 的 pH 电荷响应显著优于原版 LigandMPNN**（目标电荷命中更精准、无系统偏差）。

## 一、实验设计

- 输入：`code/input/1BC8.pdb`（93 残基，native 净电荷 +8.90 @ pH7.4）
- 模型：**MoMPNN**（`mompnn_temberture_tm_esm_6_4_4_b01.ckpt`，多目标 DPO 训练）vs **原版 LigandMPNN**（`ligandmpnn_v_32_010_25.pt`）
- 引导：`run_guided.py`，同一套 logit bias（charge_lookahead，strength=0.5）+ default 结构过滤器
- 条件矩阵（5 组 × 2 模型 = 10 次运行）：pH 7.4 × target {−8, 0, +8}；pH {4.0, 10.0} × target 0
- 每条件 10 条序列，seed=111
- 原始数据：`code/output/compare/*/summary.json`（已 gitignore）

## 二、结果

| 条件 (pH, target) | 原版 LigandMPNN (mean±std) | 偏差 | **MoMPNN (mean±std)** | 偏差 |
|------|------|------|------|------|
| 7.4, −8 | −7.32 ± 0.96 | **+0.68** | **−7.99 ± 0.90** | **+0.01** |
| 7.4, 0 | +0.37 ± 0.80 | +0.37 | **−0.00 ± 0.77** | −0.00 |
| 7.4, +8 | +8.19 ± 0.86 | +0.19 | **+8.10 ± 0.68** | +0.10 |
| 4.0, 0 | +0.42 ± 0.83 | +0.42 | **+0.01 ± 0.68** | +0.01 |
| 10.0, 0 | +0.43 ± 0.63 | +0.43 | **−0.06 ± 0.78** | −0.06 |

## 三、分析

1. **MoMPNN 在全部 5 组条件下目标电荷偏差 ≤ 0.10**，几乎完美命中；原版 LigandMPNN 有 **系统性 +0.2~+0.7 的正电荷偏差**。
2. 两模型用**完全相同**的 logit bias 引导机制，差异只能来自**模型本身的序列先验分布**：原版 LigandMPNN 在 pH 环境（特别是碱性 AA 丰富时）先验偏向正电荷残基；MoMPNN（DPO 训练后）先验更中性，更"听指挥"。
3. 在 pH 4.0→10.0 变化下（target=0），两模型都能稳定控制净电荷（σ≈0.7~0.8），MoMPNN 均值更贴 0。
4. **对项目的意义**：MoMPNN 作为生成器，pH 电荷控制精度更高——这意味着在「电荷达标率」（可用率指标之一）上 MoMPNN 天然占优，为 E1 第二阶段的 pLDDT/溶解度/热稳对比提供了干净的基础（排除了电荷未达标带来的混淆）。

## 四、待办

- 第二阶段：对生成的序列做三目标打分（ESMFold pLDDT / Protein-Sol / TemBERTure），验证 MoMPNN 的多目标优势（任务进行中，ESMFold 权重首次下载中）。

## 附

- 复现：`bash code/tests/e1_pH_batch.sh`（或逐条 `run_guided.py`）；汇总 `python code/tests/e1_summarize.py`
