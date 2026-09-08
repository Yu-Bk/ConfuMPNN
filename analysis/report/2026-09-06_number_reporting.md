# 数值出表规范 + 主指标带 CI 汇总（2026-09-06）

> 目的：论文/报告数字**有效位与样本量匹配 + 命中率一律附 CI**，杜绝"45/50 写成 0.900000"式假精确。
> 规则落地：后续所有 `compare/report_*`、`ablation/report_*`、`analysis/report_*` 与正文表格均按本规范出。

## 1 规则
**A. 命中率 / 占比（k/n 类，如 H2、三达标、达标率）**
- 一律写成 `k/n (p%，Wilson95%CI lo-hi%)`；示例：`45/50 (90.0%, 78.6-95.7%)`。
- **小数位由 n 定**：n≥30 → 主值 1 位百分小数；n<30 → 主值整数% 并给 CI（CI 1 位）；**不写 >1 位**。
- n≤10 时 CI 很宽（如 0/5 [0-43%]）→ 正文注明"小样本、只作定性"。
- 精确法可用 Clopper-Pearson；实现 `code/tests/hitrate_ci.py`（wilson_ci / clopper_pearson_ci）。

**B. 均值类（recovery、mean dev、TM、slope、保留率/倍率）**
- 底层浮点全精度保留在 `output/*.json`；表格显示位数由 SE≈sd/√N 定，默认：recovery/保留率 **2-3 位**、dev **2 位**、slope **3 位**；若 SE≥0.01 的量最多 2 位。
- 不做 6 位小数；除非指标定义与文献逐字一致且 N 巨大（≥10⁴ 级）。

**C. 出表清单**
- 出每张主表时：注明 N（每条/每臂序列数）与口径（per-protein/global/现场标定、是否含校准），并在表头或脚注给样本量。

## 2 主指标带 CI 汇总（正文/摘要候选）
| 指标 | k/n | 主值 | Wilson 95%CI | 数据源(全精度) |
|---|---|---|---|---|
| v14-clean H2（in-10 5臂总） | 45/50 | 90.0% | 78.6-95.7% | `output/v14_ligand_gen_stats_clean.json` |
| 1BJ4 蛋白 表内→小样本后 | 4/5 | 80% | 37.6-96.4% | `output/exp_control_prot_v2/` |
| 2FEO 配体 clean | 0/5 | 0% | 0.0-43.4% | 同上 clean |
| v13-in10 H2 总 | 32/50 | 64.0% | 50.1-75.9% | `output/v13_ligand_gen_stats_in10.json` |
| 7K00 核糖体 native H2 | 26/46 | 56.5% | 42.2-69.8% | `output/ribosome_7k00/summary_7k00.json` |
| exp1-v2 蛋白碱 1LYZ（Bsmall） | 5/5 | 100% | 56.6-100% | `output/exp_control_prot_v2/1LYZ/` |
| exp7b 蛋白 T3 跨网格 H2 | 117/135 | 86.7% | 79.9-91.4% | `output/exp_pH2_prot/_summary.json` |
| exp7b 配体 T3 | 119/135 | 88.1% | 81.6-92.6% | `output/exp_pH2_lig/_summary.json` |
| Task3 三达标（单条） | 520/10000 | 5.2% | 4.8-5.7% | `output/largen_v14_summary.json` |
| 蛋白 v12.2 小样本现场标定 | —（见 08-31 报告） | 74% | 参考 CI 工具 | `output/...v12_2` |

> 均值类示例（recovery，n≈50/臂）：v14-clean native recovery 0.32-0.54（2 位；JSON 全精度）。slope：v14 global 1.492（3 位）。

## 3 用法
```bash
python code/tests/hitrate_ci.py            # 关键率打表
# 均值类：用源 JSON 的 float（勿二次 round 出 6 位）
```
