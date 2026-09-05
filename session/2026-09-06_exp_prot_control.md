# Session — 蛋白模式对照实验 exp1(裸backbone) + exp2(bias-only)（2026-09-06）

> 归属：`compare/` exp1 裸 backbone 对照（报告 `compare/report_2026-09-06_exp1_prot_barebackbone.md`）+ `ablation/` exp2 bias-vs-encoder（报告 `ablation/report/2026-09-06_exp2_prot_bias_vs_encoder.md`）。
> 数据：`output/exp_control_prot/`。GPU=cuda:2。不 git push（主会话统一归档）。
> 本 session 记录执行过程/决策/数值；报告为结论版。

## 1 目的
量化"条件电荷控制（v12.2 编码器）相对 ①裸 backbone（MoMPNN 无条件） ②简单电荷引导 logit bias"多控制了多少（%达标 vs 天然电荷分布占比）。

## 2 受控设置
- backbone（蛋白模式）：`MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt`
- 条件模型 = backbone + v12.2 `output/finetune_v12_2/finetune_epoch030.pt`（condition_encoder_last 等价，memory 定稿用 epoch030）
- 校准表（route B 默认 auto）= `output/charge_calibration_v12_2.json`（表内 per-protein）
- pH=7.4，温度 0.3，`net_charge(seq,7.4)`；bias strength=0.5（run_guided Phase1 默认）
- seed 体系：`seed(p_idx,arm_idx,k)=SEED_BASE + p_idx*1e6 + arm_idx*1e4 + k`，SEED_BASE=424242；route A 用 native 臂 seed → 与 route B native 逐条配对可比；route C 与 B 同臂同 seed 配对。
- 产物断点续跑：per_seq.jsonl 追加，fasta/summary 由完整 jsonl 重建。

## 3 测试蛋白选择（好/中/坏，选择依据 = 既有 v12.2 H2/recovery 数据）

数据源 = `output/generalization_v12_2_calib/protein/<pdb>/validation.json`（per-protein 校准口径，5 臂 mean-dev≤2 记命中）+ `analysis/report/2026-08-31_v12_2_diag.md` §七/§八/§十 + no-leak(global)。

| 蛋白 | 长度 | cat | native_q@7.4 | v12.2 per-protein H2 | no-leak global | recovery | 选定角色 |
|---|---|---|---|---|---|---|---|
| 1AZM | 258 | small_mol（单体）| −1.71 | 5/5（各臂 dev 0.3–1.4）| 4/5 | 0.45（monomer 中最高）| **好** |
| 1AS2 | 312 | RNA（单体）| −2.69 | 5/5 | 2/5 | 0.34（monomer 中最低）| **中** |
| 1BJ4 | 470 | long（单体）| +0.42 | **0/5**（dev 2.8–5.4）| 1/5 | 0.41 | **坏/难** |

- 三蛋白均为单链单体天然蛋白（data/validation_pdbs 内 ATOM 仅 chain A）；已在蛋白模式泛化集中测过。
- **好（1AZM）**：per-protein 全臂达标、recovery 最高、native 电荷近中性；no-leak 也保持 4/5 → 电荷控制最稳。
- **中（1AS2）**：per-protein 达标但 recovery 最低(0.34)、未校准响应增益高(uncal slope≈2.1，需强校准)、no-leak 掉到 2/5 → 校准敏感，中等难度。
- **难（1BJ4，L470）**：per-protein 校准 0/5（v12.2 表 slope 2.488 高方差拟合失真）、组成 D/K 仅 0.59×、长蛋白响应弯曲 → 直接 route B 预期差，需附加小样本现场标定（v12.2 small 口径实证可救回 5/5）。

> 注：1AG0/1C6O/1AXW 为同源二聚体（parse 会合并链），非"单体天然蛋白"，不作本次蛋白组测试蛋白（配体 bundle 单独处理）。

## 4 实验设计
- **route A 裸 backbone**：MoMPNN 无条件重设计 n=1000/蛋白 → 逐条 pH7.4 净电荷，统计落在 native(±1)/n2/p2/n8/p8(±1 与 ±2 容差) 区占比。
- **route B 条件生成**：v12.2 编码器 + 校准(auto per-protein)，5 臂 native/n2/p2/n8/p8 × n=1000；per-arm mean-dev、mean口径 H2(≤2)、per-seq 命中效率(|dev|≤2)、native recovery。
- **route C bias-only**：去 encoder，guided_sampler + ChargeLookahead 电荷 logit bias(strength 0.5)，5 臂 × n=300（exp2 计划 native/n2/p2 起步，本次扩到 5 臂以便完整对比）；per-arm dev/H2/native recovery。
- **route Bsmall（附加，仅当 B 某臂命中差）**：现场小样本标定（5 offset±[8,4,0,4,8]×n10=50 条拟合 slope/intercept）→ 重采样 5 臂 n=1000。预期 1BJ4 需要。
- 增益表 = route B per-seq 命中效率 − route A 对应电荷区占比（±2 容差口径为主，±1 附注）。

## 5 执行过程（时间线）
- 01:5x 启动采样。首轮 A/B/C 三进程并发 → GPU2 利用率 99% 但总吞吐反降（单进程 25% util，2 seq/s → 并发后 3 进程总 ~1.2 seq/s）→ **改为串行**。route C 曾因同 shell conda activate 被 `&` 后台化失效、误用系统 python（无 prody）崩溃 → 已单独重启。
- 01:57 清理 route A/C 部分产物，route B 单独运行。
- B 预计 ~3h（1AZM≈42m / 1AS2≈58m / 1BJ4≈78m，单进程吞吐估算）。

## 6 结果摘要（采样完成，2026-09-06 07:37）

> 数据：`output/exp_control_prot/_analysis_summary.json`（全量统计）、`_report_tables.md`（报告表）。逐序列 `output/exp_control_prot/<pdb>/route_{A,B,C,Bsmall}/.../per_seq.jsonl`。

### 6.1 route A（裸 backbone 无条件，n=1000/蛋白）天然电荷分布（round 定心，±2 容差区内占比）
| 蛋白 | native±2 | n2±2 | p2±2 | n8±2 | p8±2 | A 平均电荷±std |
|---|---|---|---|---|---|---|
| 1AZM | 0.322 | 0.384 | 0.232 | 0.133 | 0.026 | −4.35±3.87 |
| 1AS2 | 0.267 | 0.219 | 0.303 | 0.046 | 0.169 | −0.96±5.33 |
| 1BJ4 | 0.143 | 0.189 | 0.122 | 0.253 | 0.023 | −6.07±5.94 |
→ 裸 backbone 无引导时天然就产出 2–38% 落在任意 ±2 电荷区；其分布中心蛋白特异（1AZM/1BJ4 偏负、1AS2 近中性）。

### 6.2 route B（v12.2 条件生成，per-protein 校准 auto）每臂 mean-口径 H2 与 per-seq 命中
| 蛋白 | mean口径 H2 | dev_of_mean 范围 | per-seq 总命中 | 极端臂(n8/p8) 相对 A ±2 增益 |
|---|---|---|---|---|
| 1AZM（好）| 5/5 | 0.11–1.11 | 0.374 | +0.225/+0.293 |
| 1AS2（中）| 5/5 | 0.32–1.51 | 0.282 | +0.225/+0.114 |
| 1BJ4（难）| **0/5** | 3.35–5.63 | 0.188 | n8 −0.105（表内校准失真→比裸 backbone 更差）|
→ 条件化把**分布均值**精确搬移到 target（好/中蛋白 mean口径 5/5）；但 per-seq 紧度有限（std≈4–6 → 单序列 |dev|≤2 仅 27–40%）。增益集中于极端臂（+0.11~+0.29）；近 native 臂增益小甚至为负（裸分布本已覆盖）。
1BJ4 因 v12.2 表内 per-protein slope 2.488 高方差失真，route B 直接全臂 miss → **触发附加小样本组**。

### 6.3 route Bsmall（1BJ4 附加，现场 50 条标定 slope 2.422/inter +1.73）
mean口径 H2 **4/5**（native/n2/p2/p8 过，n8 dev 2.25 差一档）；per-seq 0.248–0.275。→ 小样本标定救回均值口径大部分，但不如既有 v12.2 diag 表 5/5（本次 n8 略过冲；标定采样种子差异）。

### 6.4 route C（bias-only 电荷 lookahead，无校准，n=300/臂）
| 蛋白 | mean口径 H2 | dev_of_mean 范围 | per-seq 总命中 | rec |
|---|---|---|---|---|
| 1AZM | 5/5 | 0.17–0.38 | 0.897 | 0.50–0.51 |
| 1AS2 | 5/5 | 0.11–0.28 | 0.881 | 0.40 |
| 1BJ4 | **5/5** | 0.01–0.21 | 0.884 | 0.45 |
→ bias 把**每条序列**都推向 target（std≈1.1–1.3），per-seq 命中 0.83–0.94，甚至 1BJ4 直接 5/5 无需校准；native 回收率 0.40–0.51（不低于 encoder）。即"简单 logit-bias 推理侧逐步修正"在电荷命中紧度上**全面强于** learned ConditionEncoder（per-seq B 0.15–0.40 vs C 0.83–0.94）。

### 6.5 一句话结论
- exp1：条件化相对裸 backbone 的**控制增益主要体现为"分布均值可控"**（mean口径 5/5），且对远离 native 的电荷臂每序列增益最大（+0.1~+0.3）；但每序列命中受限于 encoder 的分布弥散（std≈4–6）。
- exp2：**encoder 相对纯 bias 在"逐序列电荷命中"上并无增益，反而显著落后**（−0.50~−0.73/seq）。bias 是"逐步贪心修正"，逐序列紧；encoder 是"全局条件"，只控均值、分布散。encoder 的存在价值需从电荷之外（pH 感知、整体先验/折叠、可学习响应）另行论证。
- 附带：1BJ4 的表内 per-protein 校准在本实验直接生成下全臂 miss 且 n8 比裸 backbone 更差——校准表蛋白特异失真的实际代价；现场小样本标定可救（4/5）。

## 7 产物
- 采样/分析脚本：`code/tests/exp_prot_control/sample_control.py`、`analyze_control.py`、`_smoke_timing.py`
- 数据：`output/exp_control_prot/<pdb>/route_{A,B,Bsmall,C}/...`
- 报告：`compare/report_2026-09-06_exp1_prot_barebackbone.md`、`ablation/report/2026-09-06_exp2_prot_bias_vs_encoder.md`
- 日志：`log/exp_control_prot_{A,B,C,Bsmall}.log`
