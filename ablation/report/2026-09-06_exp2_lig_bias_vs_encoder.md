# 消融 exp2（配体模式）— condition_embedding 去留：bias-only vs 条件编码器（2026-09-06）

> 归属：ablation/（受控消融）。计划：ablation/plan_exp2_bias_vs_encoder.md。session：session/2026-09-06_exp_lig_control.md。
> 数据文件均在 `output/exp_control_lig/` 下逐表注明。蛋白模式对照见 ablation/report/2026-09-06_exp2_prot_bias_vs_encoder.md；exp1（裸 backbone 对照）见 compare/report_2026-09-06_exp1_lig_barebackbone.md。
> 运行：GPU=cuda:6；日志 log/exp_lig_control.full.log（route C 扩展臂 log/exp_lig_control.Cext.log）；不 git push。

## 1 科学问题
隔离"学到的 ConditionEncoder"相对"简单电荷引导 logit bias（guided_sampler + charge_lookahead）"的增益：
- route B_cal：v14 条件编码器 + per-protein 校准（`output/charge_calibration_v14_ligand_clean.json`）——代表 encoder 的操作形态（exp1 已示直接注入有过冲，须校准）。
- route C：**去掉 encoder**，仅 guided_sampler + ChargeLookahead logit bias（strength=0.5，无结构过滤器=纯电荷 bias，不校准）。
- 同 seed 配对（n=1000/臂，seed 偏移同一套）；已确认 guided_sampler + ChargeLookahead 对 **ligand_mpnn + 配体上下文可跑通**（无需最小 bias 注入实现）。

## 2 设置
- 蛋白/臂同 exp1（5O60_E 好 / 1CGE 中 / 2FEO 难），5 臂 native/n2/p2/n8/p8（spec 规范臂 native/n2/p2；n8/p8 为扩展，以补全 exp1 对照与 exp2 极值观察）。
- 数据：`output/exp_control_lig/{pdb}/routeC/arm_{native,n2,p2,n8,p8}/sequences.json`（C，n=1000）+ `routeB_cal/arm_*/`（B，n=1000）。
- 统计：`output/exp_control_lig/summary.json`；`_report_tables.md` 表3/表4。

## 3 结果：encoder(B_cal) vs bias(C) 每臂（n=1000）
数据：`output/exp_control_lig/{pdb}/routeB_cal/arm_*` 与 `routeC/arm_*`。

### 5O60_E（native_q=+11.18）
| 臂(target) | Bcal mean_q(dev) | Bcal hit≤2 | C mean_q(dev) | C hit≤2 | Bcal−C hit | Bcal rec | C rec | Bcal cr | C cr |
|---|---|---|---|---|---|---|---|---|---|
| native(+11) | +8.88(−2.1) | 0.292 | +12.81(+1.8) | 0.586 | −0.294 | 0.388 | 0.433 | 0.54 | 1.49 |
| n2(+9) | +7.08(−1.9) | 0.314 | +10.89(+1.9) | 0.582 | −0.268 | 0.385 | 0.438 | 0.51 | 1.46 |
| p2(+13) | +10.93(−2.1) | 0.296 | +14.74(+1.7) | 0.620 | −0.324 | 0.392 | 0.429 | 0.59 | 1.52 |
| n8(+3) | +2.52(−0.5) | 0.344 | +5.05(+2.1) | 0.521 | −0.177 | 0.375 | 0.447 | 0.45 | 1.40 |
| p8(+19) | +17.71(−1.3) | 0.313 | +20.48(+1.5) | 0.692 | −0.379 | 0.402 | 0.416 | 0.77 | 1.60 |

### 1CGE（native_q=−11.66）
| 臂(target) | Bcal mean_q(dev) | Bcal hit≤2 | C mean_q(dev) | C hit≤2 | Bcal−C hit | Bcal rec | C rec | Bcal cr | C cr |
|---|---|---|---|---|---|---|---|---|---|
| native(−12) | −11.54(+0.5) | 0.375 | −12.95(−1.0) | 0.866 | −0.491 | 0.534 | 0.597 | 0.61 | 1.17 |
| n2(−14) | −13.97(+0.0) | 0.365 | −14.92(−0.9) | 0.865 | −0.500 | 0.534 | 0.590 | 0.70 | 1.22 |
| p2(−10) | −9.11(+0.9) | 0.395 | −11.08(−1.1) | 0.844 | −0.449 | 0.533 | 0.604 | 0.53 | 1.12 |
| n8(−20) | −20.69(−0.7) | 0.339 | −20.69(−0.7) | 0.906 | −0.567 | 0.525 | 0.568 | 0.94 | 1.38 |
| p8(−4) | −3.31(+0.7) | 0.451 | −5.37(−1.4) | 0.777 | −0.326 | 0.526 | 0.611 | 0.36 | 1.00 |

### 2FEO（native_q=−6.88）
| 臂(target) | Bcal mean_q(dev) | Bcal hit≤2 | C mean_q(dev) | C hit≤2 | Bcal−C hit | Bcal rec | C rec | Bcal cr | C cr |
|---|---|---|---|---|---|---|---|---|---|
| native(−7) | −4.68(+2.3) | 0.322 | −7.42(−0.4) | 0.902 | −0.580 | 0.322 | 0.385 | 0.47 | 1.34 |
| n2(−9) | −6.79(+2.2) | 0.322 | −9.32(−0.3) | 0.904 | −0.582 | 0.323 | 0.383 | 0.49 | 1.37 |
| p2(−5) | −2.64(+2.4) | 0.287 | −5.53(−0.5) | 0.910 | −0.623 | 0.321 | 0.386 | 0.46 | 1.32 |
| n8(−15) | −13.16(+1.8) | 0.289 | −15.00(+0.0) | 0.922 | −0.633 | 0.328 | 0.373 | 0.67 | 1.45 |
| p8(+1) | +3.31(+2.3) | 0.303 | +0.12(−0.9) | 0.878 | −0.575 | 0.320 | 0.386 | 0.59 | 1.31 |

### 聚合（3 蛋白 × 5 臂，n=1000/臂）
数据：`output/exp_control_lig/summary.json` aggregate / `_report_tables.md` 表4。

| route | per-seq 命中≤2 | mean\|dev\| | 均值口径 H2 | mean_rec | 带电残基倍率 cr |
|---|---|---|---|---|---|
| B_cal（encoder，v14 校准） | 0.334 | 1.44 | 9/15 | 0.414 | 0.58 |
| C（bias-only，lookahead） | 0.785 | 1.07 | 14/15 | 0.470 | 1.34 |
| B_cal − C | **−0.451** | +0.37 | −5 臂 | −0.056 | −0.77 |

## 4 native 回收（exp2 关注：bias 贴靶是否牺牲"保持 native"）
native 臂（charge 回收 = native 臂 per-seq |dev|≤2；序列回收 = rec）：

| 蛋白 | B_cal 序列 rec(native 臂) | C 序列 rec(native 臂) | B_cal native 电荷命中 | C native 电荷命中 |
|---|---|---|---|---|
| 5O60_E | 0.388 | 0.433 | 0.292 | 0.586 |
| 1CGE | 0.534 | 0.597 | 0.375 | 0.866 |
| 2FEO | 0.322 | 0.385 | 0.322 | 0.902 |

→ C 在 native 臂的序列一致率与电荷命中均**不低于** B_cal。

## 5 结论
1. **逐序列电荷命中上，简单 logit-bias 全面且显著强于 learned ConditionEncoder**（配体模式同蛋白模式）：per-seq |dev|≤2 命中 B_cal=0.29–0.45 vs C=0.52–0.92，聚合 B_cal−C=−0.45。机制差异与蛋白模式一致：bias 是**推理侧逐步贪心修正**（每步用已生成电荷前瞻把"每一条"推向 target，逐序列 std≈1–3）；encoder 是**训练学到的全局 soft-prompt 条件**，只把分布均值搬到 target 附近（弥散大，std≈3–5）。
2. bias 对**表内校准失真的难蛋白 2FEO 也 5/5（均值口径）**、对 5O60_E 的 4/5（n8 dev +2.1 差一档），无需校准——因为 bias 不经过 encoder 的响应增益。
3. **但 bias 的贴靶有组成代价**：route C 各臂带电残基总数相对 native **膨胀**（cr 1.00–1.60，正电/负电同向增，5O60_E neg_ratio 高达 1.58–1.59）——靠堆带电残基把电荷推向 target；而 encoder（B_cal）延续其"删减带电残基"路径（cr 0.36–0.94）。两种机制都偏离 native 组成，但方向相反。
4. native 序列回收：C（0.43–0.60）≥ B_cal（0.32–0.53），说明 bias 的"贴靶"未以简单序列保持率为代价（折叠/组成合理性/Tm/Sol/H3/H4 等超出本 bundle 范围，未测）。
5. **因此，在"电荷命中"单一指标上，配体模式 encoder 相对纯 bias 没有增益而是落后约 45 个百分点**。与蛋白模式结论一致。ConditionEncoder 的潜在价值需在电荷之外维度论证（pH 感知/可学习条件响应、非电荷物理量、结构先验与组成的整体影响、负电外推等），本报告只做电荷控制口径。

## 6 数据文件索引（全部相对 ConfuMPNN 根）
- `output/exp_control_lig/summary.json`：全量统计（含 routeC 5 臂、gains、聚合）
- `output/exp_control_lig/_report_tables.md`：本报告表 3/表 4 Markdown
- `output/exp_control_lig/{5O60_E,1CGE,2FEO}/routeC/arm_{native,n2,p2,n8,p8}/sequences.json`、`seqs.fa`
- `output/exp_control_lig/{5O60_E,1CGE,2FEO}/routeB_cal/arm_{native,n2,p2,n8,p8}/sequences.json`、`seqs.fa`
- 运行日志：`log/exp_lig_control.full.log`、`log/exp_lig_control.Cext.log`
