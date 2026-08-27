# 会话概要：v3 方案 P0 代码改造（2026-08-27）

> 前置：v9 阶段节点（2026-08-19 暂停训练）→ 论文导向 v3 方案落盘 `index/PROJECT_LOCAL.md`（2026-08-27），v10 演进中。
> 本轮执行 v3 §8 P0 阶段（代码/环境改造，1-2 天），全部完成并验证。

## 做了什么

### P0-1：pH-only 自动补全（D1/A9）— `run_guided.py`
- 问题（v3 P2）：训练恒 flag=1，推理 flag=0 从未见过 → 行为不可预测
- 改法：`--target_charge` 未给出时自动补全 `target=native_charge@pH`（落在训练分布内）；`--no_auto_target_charge` 关闭回到旧 flag=0 对照（A9）
- 自动补全的 target 跳过校准（native 电荷本就在分布内，无过冲可补偿）
- summary.json 新增 `auto_target` 字段
- 冒烟：1BC8 自动补全 +8.90 命中（生成 +8.66±0.95）；v9 条件注入 cond_vec=[7.4,1.0,-1.71] flag=1 正常

### P0-2：RMSD 联报（D6）— 3 个统计脚本
- `generalization_stats.py` / `transfer_stats.py` / `ph_scan_stats.py` 增加 `rmsd_median`（US-align 已输出 RMSD）
- H1b 辅助指标（TM 判拓扑、RMSD 看局部贴合，按域报告）

### P0-3：PROPKA 物理复核（P5/H4）— 新 `tests/propka_charge_check.py`
- PROPKA3 微环境修正 pKa → 重算 Q_phys（物理修正电荷）
- 对照 Q_design（游离 pKa），判据 H4：|Q_phys 均值 − target| ≤ 2.0
- 支持单条 PDB 或目录（臂）
- 验证：1BC8 Q_design +8.90 vs Q_phys +9.17，H4 PASS（Δ0.27）
- **踩坑**：① propka3 本版本不支持 `--output`，.pka 写到 cwd → cd 临时目录运行；② HH 电荷公式方向易写反（酸性去质子化分数=1/(1+10^(pKa-pH))，碱性质子化分数=1/(1+10^(pH-pKa))），已修并用物理校验兜底

### P0-4：fractional SASA（D3/D10）— 新 `src/sasa.py`
- freesasa `relativeTotal` 直接给出 fractional SASA（残基 SASA/参考值，即 Gly-X-Gly 标准口径）
- 输出 `surface_mask`（fracSASA ≥ θ 表面资格，默认 θ=0.25），供 v10 B（L_add）与 D10 SASA 旁路
- 验证：1BC8 93 残基 48 表面位点；1AZM（配体）258 残基正常
- 新依赖：freesasa 2.2.1（pip 装入 confumpnn）

### P0-5：pH 自适应带电集合（D4-③）— `structure_aware_filter.py` + `charge_lookahead.py`
- 新增 `pH_adaptive_charged_aa(pH)`：质子化分数 ≥ 0.5 纳入弱带电残基
  - His(pKa 6.0)：pH≤6 → 算正电
  - Cys(pKa 8.3)：pH≥8.3 → 算负电
  - Tyr(pKa 10.1)：pH≥10.1 → 算负电
- `compute_bias(seq_int, pH=None)`：pH=None 向后兼容强电荷 K/R/D/E
- `make_dynamic_callback` 透传工作 pH 给过滤器
- 踩坑：info 键名 `neg_over` vs 变量 `over_neg` 笔误，已修

### 回归
- `test_all.py` 新增 9 项 pH 自适应测试 → **45/45 全过**

## 提交
- `f8618a0`：feat v3 P0 代码改造（10 文件，+815 行）
- 本轮补充 session 记录 + 根目录 `logical_chain.md`（文件规范）/ `require..md`（早期任务清单，均已完成的记录）纳入 git

## 下一步（v3 §8 P1）
缺口实验：C1–C4、C6 对照 + PROPKA 复核（H4）+ AF2 子集 + 统计脚本（预计 3–5 天）
