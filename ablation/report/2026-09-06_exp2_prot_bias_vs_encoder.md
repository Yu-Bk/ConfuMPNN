# 消融 exp2（蛋白模式）— condition_embedding 去留：bias-only vs 条件编码器（2026-09-06）

> 归属：ablation/（受控消融）。计划：ablation/plan_exp2_bias_vs_encoder.md。session：session/2026-09-06_exp_prot_control.md。
> 数据文件均在 output/exp_control_prot/ 下逐表注明。配体 bundle 另见 ablation/report/2026-09-06_exp2_lig_bias_vs_encoder.md。

## 1 科学问题
隔离"学到的 ConditionEncoder"相对"简单电荷引导 logit bias（guided_sampler/charge_lookahead）"的增益：
route B（encoder 条件注入，v12.2 + per-protein 校准） vs route C（去 encoder，仅 charge-lookahead logit bias，无校准，strength=0.5），同 seed 配对。

## 2 设置
- 蛋白/臂同 exp1（1AZM 好 / 1AS2 中 / 1BJ4 难 × native/n2/p2/n8/p8）。
- route B：n=1000/臂；route C：n=300/臂（bias 为逐步电荷前瞻，见 code/src/charge_lookahead.py）。
- 已确认 guided_sampler + ChargeLookahead 对 MoMPNN backbone 可跑通（无需最小 bias 注入实现）。
- C 用直接 target（不校准）；B 用 v12.2 per-protein 校准（run_guided auto）。
- 数据：output/exp_control_prot/<pdb>/route_{B,C}/arm_*/per_seq.jsonl。

## 3 结果：encoder(B) vs bias(C) 每臂
| 蛋白 | 臂 | B dev_of_mean | C dev_of_mean | B seq命中≤2 | C seq命中≤2 | B rec | C rec | B−C seq命中 |
|---|---|---|---|---|---|---|---|---|
| 1AZM | native | 0.21 | 0.31 | 0.400 | 0.903 | 0.448 | 0.510 | −0.503 |
| 1AZM | n2 | 0.14 | 0.31 | 0.401 | 0.917 | 0.448 | 0.510 | −0.516 |
| 1AZM | p2 | 0.11 | 0.35 | 0.390 | 0.910 | 0.445 | 0.511 | −0.520 |
| 1AZM | n8 | 0.84 | 0.17 | 0.358 | 0.857 | 0.440 | 0.501 | −0.499 |
| 1AZM | p8 | 1.11 | 0.38 | 0.319 | 0.897 | 0.431 | 0.501 | −0.578 |
| 1AS2 | native | 1.47 | 0.24 | 0.291 | 0.887 | 0.341 | 0.402 | −0.596 |
| 1AS2 | n2 | 1.51 | 0.16 | 0.275 | 0.880 | 0.341 | 0.401 | −0.605 |
| 1AS2 | p2 | 0.92 | 0.11 | 0.291 | 0.890 | 0.342 | 0.402 | −0.599 |
| 1AS2 | n8 | 1.44 | 0.28 | 0.271 | 0.833 | 0.337 | 0.397 | −0.562 |
| 1AS2 | p8 | 0.32 | 0.11 | 0.283 | 0.913 | 0.340 | 0.400 | −0.630 |
| 1BJ4 | native | 4.58 | 0.01 | 0.188 | 0.873 | 0.407 | 0.454 | −0.685 |
| 1BJ4 | n2 | 5.41 | 0.10 | 0.184 | 0.880 | 0.407 | 0.454 | −0.696 |
| 1BJ4 | p2 | 4.33 | 0.12 | 0.215 | 0.937 | 0.407 | 0.454 | −0.722 |
| 1BJ4 | n8 | 5.63 | 0.21 | 0.148 | 0.877 | 0.405 | 0.452 | −0.729 |
| 1BJ4 | p8 | 3.35 | 0.05 | 0.203 | 0.853 | 0.407 | 0.452 | −0.650 |

汇总（数据 output/exp_control_prot/_analysis_summary.json）：

| 蛋白 | B mean口径 H2 | C mean口径 H2 | B per-seq 总命中 | C per-seq 总命中 | B std范围 | C std范围 |
|---|---|---|---|---|---|---|
| 1AZM | 5/5 | 5/5 | 0.374 | 0.897 | 3.7–4.5 | 1.2–1.3 |
| 1AS2 | 5/5 | 5/5 | 0.282 | 0.881 | 5.4–5.5 | 1.1–1.3 |
| 1BJ4 | 0/5 | 5/5 | 0.188 | 0.884 | 5.8–6.2 | 1.1–1.3 |

## 4 结论
1. **在逐序列电荷命中上，简单 logit-bias 全面且显著强于 learned ConditionEncoder**：per-seq |dev|≤2 命中 B=0.19–0.40 vs C=0.83–0.94（每臂 B−C≈−0.50~−0.73）。
2. 机制差异清晰：bias 是**推理侧逐步贪心修正**——每解码一步用已生成电荷前瞻，把"每一条序列"都推向 target，故单序列电荷 std≈1.1–1.3、几乎条条贴靶；encoder 是**训练学到的全局 soft-prompt 条件**——只把整个分布的平均电荷搬向 target（mean口径命中好），但分布弥散大（std≈4–6）。
3. bias 对**表内校准失真蛋白 1BJ4 也直接 5/5**（无需校准），因为它不经过 encoder 的响应增益；而 encoder 路线需正确校准才能命中。
4. native 回收率：C(0.40–0.51) 不低于 B(0.34–0.45)，说明 bias 的"贴靶"未以简单序列保持率为代价（折叠/组成/Tm 等超出本 bundle 范围，未测）。
5. **因此，在"电荷命中"这一单指标上，encoder 相对纯 bias 没有增益而是落后**。ConditionEncoder 的潜在价值须在电荷之外的维度论证：pH 感知/可学习条件响应、非电荷物理量（局部电荷分布、pI、表面）、以及对结构先验的整体影响（本报告只做电荷控制口径）。

## 5 数据文件索引
- output/exp_control_prot/_analysis_summary.json；output/exp_control_prot/_report_tables.md（表5/表6）
- output/exp_control_prot/{1AZM,1AS2,1BJ4}/route_B/arm_*/per_seq.jsonl（B，n=1000）
- output/exp_control_prot/{1AZM,1AS2,1BJ4}/route_C/arm_*/per_seq.jsonl（C，n=300）
