# 对比统计 exp5 — Wilcoxon 配对检验（计划 2026-09-06，数据到位后执行）

> 归属：`compare/`。依赖：exp1/exp2 数据（`output/exp_control_{prot,lig}/`）。

## 配对与检验
- **配对 A vs B**（裸 backbone vs 条件，同蛋白同臂）：逐蛋白对 mean dev 与 达标率配对 Wilcoxon signed-rank（双侧），报告 p 与效应量。
- **配对 C vs B**（bias-only vs encoder，exp2）：同上，逐蛋白配对。
- 分层：per-mode（蛋白/配体）分别；并按 电荷臂温和(native/n2/p2) vs 极端(n8/p8) 子分组。
- 小样本警示：配对 n=3 蛋白时 Wilcoxon 功效极低——会一并给"描述性 + 逐蛋白表"，仅当样本足够（含多蛋白扩展）才下显著性结论；CI 用 `code/tests/hitrate_ci.py`。

## 输出
`output/wilcoxon_exp15.json` + `compare/report_2026-09-06_exp15_stats.md`；脚本 `code/tests/wilcoxon_exp15.py`（复用 scipy.stats.wilcoxon）。
