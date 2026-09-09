# design_bench — 批量条件设计 + 打分流水线（可复用，2026-09-09 建）

> 位置说明：放 `code/tools/`（子目录）而非 `code/tests/`——`tests/` 是**验证/打分手稿仓库**；
> `tools/` 是可复用任务工具（已有 `pocket_protect/` 先例）。本项目真实模型工具是
> `code/run_guided.py`（生成）+ `code/tests/{esmfold_score,tm_score,temberture_score}.py`
> + `protein_sol_mcp`（打分）。**用模型的人不需要本目录**——本目录提供的是"一次跑多条件×多模式并出指标表/统计"的**任务胶水**。

## 干什么 / 怎么用（不用自己写脚本）
典型需求 = 对某蛋白，按"模式 × (pH,target) 条件网格"各生成 n 条、固定若干残基、逐条算 电荷/回收/TM/RMSD/pLDDT/Tm/Sol，并汇总对比。
1. 生成（含固定位/校准）→ 直接 `code/run_guided.py`（`--pH --target_charge --fixed_residues "I3 ..." --num_samples --calibrate auto|global --calibration_file`）。
2. 逐条打分 → `code/tests/esmfold_score.py`（pLDDT/回折）、`tm_score.py --folds --ref --out`（TM/RMSD vs 参考骨架）、`temberture_score.py`（Tm）、`protein_sol_mcp/scripts/protein_sol_predict.py`（%sol）。
3. 净电荷/回收/固定位校验 → `net_charge` + 与参考序列逐位比对（见 `code/tests/` 或本项目分析脚本）。
4. 统计/出表 → `wilcoxon`/`mannwhitney` + `code/tests/hitrate_ci.py`（CI）。

## 现场小样本标定（表外蛋白精度开关，可复用）
- **它是对"使用"的一个可控开关**：`run_guided.py --calibrate {auto,global,off} --calibration_file <json>`。
  `auto`：pdb 在表的 `per_protein` 里就用它的 slope/intercept，否则回退 `global`；`off`/`--no_calibration` 关掉。
- **但"拟合出那条 slope"是前置一步**：对表外蛋白采探针批（**native ±[8,4,0,4,8] 5 档 × 每档 n_per=10 = 50 条**，见 `code/tests/build_calibration_small.py` 同法；勿加大 n_per 到 20）拟合成 json 的 `per_protein[蛋白]`，再 `--calibrate auto --calibration_file 该json` 即"用上"。
- 归纳：**无"采样时自动自拟合"开关**；是"先建表(offline) → 用表(开关 auto)".

## 本目录脚本（L11 案例的活拷贝，供推广）
- L11 实际用例脚本：`L11design/test/exp*.py/sh`（组循环/采样/分批折叠调度/打分汇总/统计）；它们多数可直接换蛋白名/组表复用。
- 待推广通用化的通用件（如需要，我可从 `L11design/test/` 提炼成参数化版放此目录）：
  - `design_group_runner`（组网格 → run_guided 逐组采样）
  - `fold_scheduler`（ESMFold 分批/GPU 轮询，可 resume）
  - `score_aggregator`（合并 plddt/tm/tm/seqtm/sol → merged.csv）
  - `per_protein_small_cal`（单蛋白探针 → per_protein json）
