# 报告 exp3 — 命中率的置信区间（CI）（2026-09-06 完成）

> 目的：过去所有命中率都是**点估计**；给每个"n 条采样命中 k"的命中率补 **95% 置信区间**（Wilson + Clopper-Pearson 精确法），避免小 n 误导。
> 脚本：`code/tests/hitrate_ci.py`（可导入 `wilson_ci(k,n)` / `clopper_pearson_ci(k,n)`）。
> 汇总数据：`output/hitrate_ci_summary.json`。

## 关键结果（n≈50 口径）
| 口径 | 命中率 | Wilson 95%CI | Clopper-Pearson |
|---|---|---|---|
| v14-clean H2（45/50） | 90.0% | [78.6, 95.7]% | [78.2, 96.7]% |
| per-protein 5/5 | 100% | [56.6, 100]% | [47.8, 100]% |
| 2FEO 0/5 | 0% | [0, 43.4]% | [0, 52.2]% |
| v13-in10 H2（32/50） | 64.0% | [50.1, 75.9]% | [49.2, 77.1]% |
| 7K00 native H2（26/46） | 56.5% | [42.2, 69.8]% | [41.1, 71.1]% |
| Task3 三达标单条（≈5.2%） | 5.2% | [4.8, 5.7]% | [4.8, 5.7]% |

## 用法
- 论文所有"X/N 命中率"建议统一给 Wilson CI；n≤10 用精确法。
- 后续实验报告（exp1/2/5/7）命中率一律带 CI。

## 数据/图路径
- 汇总 JSON：`output/hitrate_ci_summary.json`（作图数据）
- 脚本：`code/tests/hitrate_ci.py`
