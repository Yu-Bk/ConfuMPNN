# 对比实验 exp1（蛋白模式）— 裸 backbone 无条件 vs ConfuMPNN 条件生成（2026-09-06）

> 归属：compare/（方法对比）。计划：compare/plan_exp1_barebackbone_control.md。session：session/2026-09-06_exp_prot_control.md。
> 本报告每张表均注明对应数据文件完整路径（output/exp_control_prot/...）便于作图。配体 bundle 另见 compare/report_2026-09-06_exp1_lig_barebackbone.md。
> 运行：GPU=cuda:2；日志 log/exp_control_prot_{A,B}.log、log/exp_prot_chain.log。

## 1 科学问题
量化"条件电荷控制（v12.2 编码器 + 校准）相对裸 backbone（MoMPNN 无条件重设计）多控制了多少"：
净增益 = 条件化 5 臂达标率 − 裸 backbone 落在对应天然电荷区的占比。

## 2 受控设置
- backbone：MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt
- 条件模型：backbone + output/finetune_v12_2/finetune_epoch030.pt（v12.2 ConditionEncoder）
- 电荷校准（route B，run_guided 同款 auto）：output/charge_calibration_v12_2.json（表内 per-protein）
- pH=7.4；net_charge(seq,7.4)；温度 0.3；bias strength 0.5（exp2）
- 固定 seed：seed(p,arm,k)=424242+p*1e6+arm*1e4+k；route A 用 native 臂 seed → 与 B native 逐条配对可比
- n：route A=1000/蛋白；route B=1000/臂；route C=300/臂；Bsmall=1000/臂（1BJ4）

## 3 测试蛋白选择（好/中/坏）
数据源：output/generalization_v12_2_calib/protein/<pdb>/validation.json；analysis/report/2026-08-31_v12_2_diag.md §七/§八/§十。

| 蛋白 | L | cat | native_q@7.4 | per-protein H2 | no-leak(global) | recovery | 角色 | 理由 |
|---|---|---|---|---|---|---|---|---|
| 1AZM | 258 | small_mol(单体) | -1.71 | 5/5 | 4/5 | 0.45 | 好 | 全臂稳、回收最高、近中性 |
| 1AS2 | 312 | RNA(单体) | -2.69 | 5/5 | 2/5 | 0.34 | 中 | 回收最低、响应增益 2.09 需强校准、no-leak 掉 2/5 |
| 1BJ4 | 470 | long(单体) | +0.42 | 0/5 | 1/5 | 0.41 | 难 | 长蛋白、表内 slope 2.49 高方差、D/K 0.59× |

## 4 route A：裸 backbone 净电荷分布（天然基线）
数据：output/exp_control_prot/<pdb>/route_A/per_seq.jsonl（n=1000）；统计 output/exp_control_prot/_analysis_summary.json。

| 蛋白 | mean_q±std | native±1 | native±2 | n2±1 | n2±2 | p2±1 | p2±2 | n8±1 | n8±2 | p8±1 | p8±2 | 平均 recovery |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1AZM | -4.35±3.87 | 0.151 | 0.322 | 0.208 | 0.384 | 0.119 | 0.232 | 0.056 | 0.133 | 0.011 | 0.026 | 0.517 |
| 1AS2 | -0.96±5.33 | 0.130 | 0.267 | 0.118 | 0.219 | 0.161 | 0.303 | 0.020 | 0.046 | 0.076 | 0.169 | 0.405 |
| 1BJ4 | -6.07±5.94 | 0.064 | 0.143 | 0.099 | 0.189 | 0.063 | 0.122 | 0.141 | 0.253 | 0.008 | 0.023 | 0.456 |

要点：裸 backbone 无引导时天然产出 2–38% 落在任一 ±2 电荷区（主要落在分布中心附近）；p8(+8) 背景最低（0.017–0.169），n8/native 背景中等。裸分布中心蛋白特异（1AZM/1BJ4 明显偏负，1AS2 近中性）。

## 5 route B：v12.2 条件生成 + 增益
数据：output/exp_control_prot/<pdb>/route_B/arm_{native,n2,p2,n8,p8}/per_seq.jsonl（逐臂 arm_summary.json）。均值口径 dev=|mean_q−target|；H2 命中=dev_of_mean≤2；per-seq 命中=单序列 |q−target|≤2 占比；增益=B per-seq 命中 − A 同区±2 占比（round 定心）。

| 蛋白 | 臂 | target | B mean_q | B dev_of_mean | B seq命中≤2 | A 区±2 | 增益 | B rec |
|---|---|---|---|---|---|---|---|---|
| 1AZM | native | -2 | -2.21 | 0.21 | 0.400 | 0.322 | +0.078 | 0.448 |
| 1AZM | n2 | -4 | -4.14 | 0.14 | 0.401 | 0.384 | +0.017 | 0.448 |
| 1AZM | p2 | 0 | -0.11 | 0.11 | 0.390 | 0.232 | +0.158 | 0.445 |
| 1AZM | n8 | -10 | -10.84 | 0.84 | 0.358 | 0.133 | +0.225 | 0.440 |
| 1AZM | p8 | +6 | +4.89 | 1.11 | 0.319 | 0.026 | +0.293 | 0.431 |
| 1AS2 | native | -3 | -1.53 | 1.47 | 0.291 | 0.267 | +0.024 | 0.341 |
| 1AS2 | n2 | -5 | -3.48 | 1.51 | 0.275 | 0.219 | +0.056 | 0.341 |
| 1AS2 | p2 | -1 | -0.08 | 0.92 | 0.291 | 0.303 | -0.012 | 0.342 |
| 1AS2 | n8 | -11 | -9.56 | 1.44 | 0.271 | 0.046 | +0.225 | 0.337 |
| 1AS2 | p8 | +5 | +4.68 | 0.32 | 0.283 | 0.169 | +0.114 | 0.340 |
| 1BJ4 | native | 0 | -4.58 | 4.58 | 0.188 | 0.143 | +0.045 | 0.407 |
| 1BJ4 | n2 | -2 | -7.41 | 5.41 | 0.184 | 0.189 | -0.005 | 0.407 |
| 1BJ4 | p2 | +2 | -2.33 | 4.33 | 0.215 | 0.122 | +0.093 | 0.407 |
| 1BJ4 | n8 | -8 | -13.63 | 5.63 | 0.148 | 0.253 | -0.105 | 0.405 |
| 1BJ4 | p8 | +8 | +4.65 | 3.35 | 0.203 | 0.023 | +0.180 | 0.407 |

汇总（output/exp_control_prot/_analysis_summary.json）：

| 蛋白 | route A native±2 | route B mean口径 H2 | route B per-seq 总命中 |
|---|---|---|---|
| 1AZM | 0.322 | 5/5 | 0.374 |
| 1AS2 | 0.267 | 5/5 | 0.282 |
| 1BJ4 | 0.143 | 0/5 | 0.188 |

## 6 附加组 Bsmall（1BJ4 现场小样本标定）
触发条件：route B 全臂 mean口径 miss。现场 5 target×n10=50 条标定 slope=2.42/inter=+1.73（LOOCV 1.13）。
数据：output/exp_control_prot/1BJ4/route_Bsmall/arm_*/per_seq.jsonl；参数 output/exp_control_prot/1BJ4/route_Bsmall/1BJ4_smallcal.json。

| 臂 | target | Bsmall mean_q | dev_of_mean | seq命中≤2 | rec |
|---|---|---|---|---|---|
| native | 0 | -0.46 | 0.46 | 0.269 | 0.408 |
| n2 | -2 | -2.93 | 0.93 | 0.258 | 0.408 |
| p2 | +2 | +1.60 | 0.40 | 0.275 | 0.408 |
| n8 | -8 | -10.25 | 2.25 | 0.255 | 0.406 |
| p8 | +8 | +8.88 | 0.88 | 0.248 | 0.408 |

→ mean口径 H2 4/5（仅 n8 差一档）。小样本标定救回 1BJ4 的均值口径，但 per-seq 仍 ~0.26（encoder 弥散未变）。

## 7 结论
1. 条件化相对裸 backbone 的"控制"主要是**把分布均值精确搬到 target**：好/中蛋白（1AZM/1AS2）在 per-protein 校准下 5 臂 mean口径全命中（dev_of_mean≤1.5），而裸 backbone 分布中心只能落在蛋白固有电荷处。
2. 每序列命中增益在**远离 native 的极端臂最大**（n8/p8：+0.11~+0.29）；近 native/native 臂裸分布本已覆盖较多，增益小（+0.02~+0.16）甚至为负（1AS2 p2 −0.012）。
3. 但 encoder 的**单序列电荷弥散大（std≈4–6）**：即使均值命中，per-seq |dev|≤2 也仅 27–40%（好/中蛋白）。"条件化"不等于"每条序列都贴靶"。
4. 1BJ4 揭示表内 per-protein 校准失真的实际代价：直接生成 0/5 且 n8 比裸 backbone 更差（增益 −0.105）；现场小样本标定可救回均值口径 4/5。这印证"校准表蛋白特异，陌生/高方差蛋白应先小样本标定"的使用指南。
5. 跨 3 蛋白均值口径 H2：B=10/15（1AZM/1AS2 好、1BJ4 差）；per-seq 总命中 B=0.281。

## 8 数据文件索引
- output/exp_control_prot/_analysis_summary.json：全量统计（route A/B/Bsmall/C 逐臂）
- output/exp_control_prot/_report_tables.md：各表 Markdown（作图数据）
- output/exp_control_prot/{1AZM,1AS2,1BJ4}/route_{A,B,Bsmall}/per_seq.jsonl（每行 {seed,seq,charge,target,dev,recovery}）
- output/exp_control_prot/{1AZM,1AS2,1BJ4}/route_{A,B,Bsmall}/seqs.fa
- output/exp_control_prot/{pdb}/meta.json、output/exp_control_prot/meta.json
