# 对比实验 exp1（配体模式）— 裸 backbone 无条件 vs ConfuMPNN 条件生成（2026-09-06）

> 归属：compare/（方法对比）。计划：compare/plan_exp1_barebackbone_control.md。session：session/2026-09-06_exp_lig_control.md。
> 本报告每张表均注明对应数据文件完整路径（`output/exp_control_lig/...`）便于作图。蛋白模式对照见 compare/report_2026-09-06_exp1_prot_barebackbone.md；exp2（bias-only 消融）见 ablation/report/2026-09-06_exp2_lig_bias_vs_encoder.md。
> 运行：GPU=cuda:6；日志 log/exp_lig_control.full.log；不 git push。

## 1 科学问题
量化配体模式下「条件电荷控制（v14 ConditionEncoder + LigandMPNN backbone）」相对「裸 LigandMPNN 无条件重设计」**多控制了多少净电荷**：
- route A：裸 backbone 无条件 n=1000/蛋白，天然净电荷分布 = 基线。
- route B：v14 编码器条件注入 5 臂（native/n2/p2/n8/p8，target=round(native_q)+Δ）n=1000/臂，**直接生成（未校准）**。
- route B_cal：因诊断发现 B 直接生成系统性过冲/欠冲（"某臂 B 很差→补现场标定附加组"），另跑 per-protein 校准附加组 n=1000/臂。
- 净增益 = B 各臂命中率 − A 落在对应天然电荷区的占比。

## 2 受控设置
- backbone：`LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt`（atom_context=25，配体上下文）。
- 条件模型 = backbone + v14 条件编码器 `output/finetune_ligand_v14_rna/finetune_epoch050.pt`。
- pH=7.4；净电荷 = `net_charge(seq,7.4)`（含 N/C 端）；温度 0.3；LigandMPNN 配体原子上下文照常（protein 链 + 附近 RNA/配体）。
- 固定 seed 体系：同蛋白同臂 A/B/Bcal 用同一批 seed 偏移（`seed_base(p,k)=3000+蛋白序位×100000+k`；逐序列 `torch.manual_seed`），每序列解码顺序逐条可复现。
- n：route A = 1000/蛋白；route B = 1000/臂；route B_cal = 1000/臂。
- 校准表（B_cal）：`output/charge_calibration_v14_ligand_clean.json` per-protein（5O60_E s=1.78 o=−0.705 / 1CGE s=0.983 o=−2.349 / 2FEO s=0.935 o=−5.082）。

## 3 测试蛋白选择（好/中/坏；选择依据 = v14 clean 链 H2/回收/slope）
数据源：`analysis/report/2026-09-04_v14_clean_validation.md`；清单 `data/validation_pdbs/validation_manifest_v14_in.json`。

| 蛋白 | 档位 | L | native_q@7.4 | v14 clean H2 | recovery | diag slope | 角色理由 |
|---|---|---|---|---|---|---|---|
| 5O60_E | 好（RNA 结合，rRNA） | 209 | +11.18 | 5/5 | 0.39 | 1.78 | 天然正电 RNA 结合蛋白（"好"必须 RNA 结合蛋白），正电外推区挑战大 |
| 1CGE | 中（金属 CA+ZN） | 162 | −11.66 | 5/5 | 0.54 | 0.98 | 近单位响应、回收最高，作"中"参照 |
| 2FEO | 难（DC 核苷酸） | 221 | −6.88 | 0/5 | 0.32 | 0.93 | clean 验证唯一全臂失败（高电荷方差），作"难" |

## 4 route A：裸 LigandMPNN 无条件净电荷分布（天然基线）
数据：`output/exp_control_lig/{5O60_E,1CGE,2FEO}/routeA/sequences.json` + `seqs.fa`；统计 `output/exp_control_lig/summary.json` + `_report_tables.md` 表1/1b。

| 蛋白 | native_q | A mean_q±std | A rec | A native±1 | A native±2 | A p5–p95 |
|---|---|---|---|---|---|---|
| 5O60_E | +11.18 | +26.72±3.39 | 0.448 | 0.000 | 0.000 | [21.0, 32.0] |
| 1CGE | −11.66 | −12.30±2.40 | 0.620 | 0.307 | 0.600 | [−15.8, −8.7] |
| 2FEO | −6.88 | −10.78±3.40 | 0.395 | 0.106 | 0.255 | [−16.1, −4.9] |

各电荷窗 A±1 / A±2 占比（`_report_tables.md` 表1b）：

| 蛋白 | 窗 | native | n2 | p2 | n8 | p8 |
|---|---|---|---|---|---|---|
| 5O60_E | A±1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.022 |
| 5O60_E | A±2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.048 |
| 1CGE | A±1 | 0.333 | 0.213 | 0.230 | 0.001 | 0.001 |
| 1CGE | A±2 | 0.600 | 0.456 | 0.458 | 0.012 | 0.003 |
| 2FEO | A±1 | 0.122 | 0.214 | 0.055 | 0.096 | 0.001 |
| 2FEO | A±2 | 0.255 | 0.410 | 0.129 | 0.207 | 0.005 |

**要点**：裸 backbone 无引导时，天然电荷分布**蛋白特异**且常远离 native 电荷——5O60_E（RNA 结合）裸设计强烈正偏（mean +26.7，native +11.2，native±1 命中 0）；1CGE 裸设计恰好在 native 附近（native±1 = 30.7%）；2FEO 偏负（−10.8，native −6.9）。因此"条件控制"的基线高度因蛋白而异。

## 5 route B：v14 条件生成（直接/未校准）+ route B_cal（现场标定附加组）
数据：`output/exp_control_lig/{pdb}/routeB/arm_{native,n2,p2,n8,p8}/sequences.json` + `seqs.fa`；`routeB_cal/arm_*/`；统计 `_report_tables.md` 表2。
指标：`mean_q(dev)` = 均值与名义 target 偏差；`hit≤2` = 逐序列 |q−target|≤2 占比；A±1 为裸基线；rec = 与 native 序列一致率。

| 蛋白 | 臂(target) | A±1 基线 | B_direct mean_q(dev) | B_direct hit≤2 | B_cal mean_q(dev) | B_cal hit≤2 |
|---|---|---|---|---|---|---|
| 5O60_E | native(+11) | 0.000 | +17.57(+6.6) | 0.13 | +8.88(−2.1) | 0.29 |
| 5O60_E | n2(+9) | 0.000 | +13.53(+4.5) | 0.22 | +7.08(−1.9) | 0.31 |
| 5O60_E | p2(+13) | 0.000 | +21.55(+8.6) | 0.08 | +10.93(−2.1) | 0.30 |
| 5O60_E | n8(+3) | 0.000 | +3.66(+0.7) | 0.34 | +2.52(−0.5) | 0.34 |
| 5O60_E | p8(+19) | 0.022 | +29.28(+10.3) | 0.05 | +17.71(−1.3) | 0.31 |
| 1CGE | native(−12) | 0.333 | −14.11(−2.1) | 0.34 | −11.54(+0.5) | 0.38 |
| 1CGE | n2(−14) | 0.213 | −16.43(−2.4) | 0.31 | −13.97(+0.0) | 0.36 |
| 1CGE | p2(−10) | 0.230 | −11.76(−1.8) | 0.37 | −9.11(+0.9) | 0.40 |
| 1CGE | n8(−20) | 0.001 | −22.69(−2.7) | 0.28 | −20.69(−0.7) | 0.34 |
| 1CGE | p8(−4) | 0.001 | −5.21(−1.2) | 0.42 | −3.31(+0.7) | 0.45 |
| 2FEO | native(−7) | 0.122 | −9.66(−2.7) | 0.29 | −4.68(+2.3) | 0.32 |
| 2FEO | n2(−9) | 0.214 | −11.53(−2.5) | 0.27 | −6.79(+2.2) | 0.32 |
| 2FEO | p2(−5) | 0.055 | −7.60(−2.6) | 0.29 | −2.64(+2.4) | 0.29 |
| 2FEO | n8(−15) | 0.096 | −17.64(−2.6) | 0.27 | −13.16(+1.8) | 0.29 |
| 2FEO | p8(+1) | 0.001 | −1.80(−2.8) | 0.31 | +3.31(+2.3) | 0.30 |

### 5.1 为什么需要 B_cal（附加组）
- 5O60_E（slope 1.78）：B 直接正臂严重过冲（native/p2/p8 dev +6.6/+8.6/+10.3），均值口径全 miss。
- 1CGE/2FEO（slope≈0.93–0.98 但 intercept 负偏）：B 直接各臂系统性负偏（dev −1.2~−2.8），均值口径大多 miss。
- B_cal（per-protein 校准后注入）把均值拉到 dev≈±2 以内（5O60_E n2/n8/p8、1CGE 全臂；2FEO 残差仍 +2.2~+2.4）。

## 6 exp1 增益汇总（route B − route A 基线）
数据：`output/exp_control_lig/summary.json` gains；`_report_tables.md` 表5。

按臂（3 蛋白均值，命中定义 |dev|≤2；基线 A±1 为 spec 口径）：

| route | native | n2 | p2 | n8 | p8 | 均值 |
|---|---|---|---|---|---|---|
| B_direct − A±1 | +0.104 | +0.126 | +0.154 | +0.265 | +0.252 | +0.180 |
| B_cal − A±1 | +0.178 | +0.191 | +0.231 | +0.292 | +0.348 | +0.248 |

（若基线取 A±2 同窗口径，B_cal 增益 ≈ +0.07~+0.30，见 summary.json gains 的 B_hit_minus_A_pm2。）

跨 3 蛋白 × 5 臂聚合（n=1000/臂）：`_report_tables.md` 表4。

| route | per-seq 命中≤2 均值 | mean\|dev\| | 均值口径 H2 达标 | mean_rec | 带电残基倍率 cr |
|---|---|---|---|---|---|
| route A（裸） | native±1 均值 13.8% | — | — | 0.49 | ~1.14 |
| route B_direct | 0.266 | 3.60 | 3/15 | 0.418 | 0.69 |
| route B_cal | 0.334 | 1.44 | 9/15 | 0.414 | 0.58 |

**要点**：
1. 条件化（encoder）相对裸 backbone 的"控制"在**极端臂增益最大**（n8/p8 +0.17~+0.35），近 native/native 臂增益小（5O60_E native 臂因裸分布命中≈0 增益 0.13–0.18；1CGE native 臂裸基线已 33%，增益仅 +0.04）。
2. **直接注入有过冲增益**（5O60_E 正臂 +6~+10 dev），需要校准；校准后均值口径 H2 从 3/15 升到 9/15。2FEO 即便校准均值口径仍只 1/5（残差 +2.3，与 v14 clean 的 0/5 一致——高方差蛋白）。
3. 即使均值命中，**encoder 逐序列弥散大（std≈3–5）**：per-seq |dev|≤2 命中仅 0.27–0.45。"条件化"把分布均值搬到 target 附近，但不是每条都贴靶。
4. 组成：encoder 各臂带电残基总数相对 native 普遍 <1（cr 0.36–1.00），延续 v14 已知的"删减带电残基"路径。

## 7 结论（exp1）
1. 配体模式下 v14 条件电荷控制相对裸 LigandMPNN 提供了**真实但中等**的净增益：极端臂（±8）每序列命中约 +0.25（B_cal +0.29~+0.35），近 native 臂增益被裸 backbone 固有电荷分布稀释。
2. 增益实现前提是**正确的推理侧校准**——直接注入因编码器响应增益（尤其高增益蛋白 5O60_E slope 1.78）可能不比裸 backbone 好多少（5O60_E p8 直接 hit 0.05 仅比裸 A 0.048 高 0.002）。这再次印证项目"校准三口径/表外先小样本标定"的使用规范。
3. 与蛋白模式 exp1 对照（compare/report_2026-09-06_exp1_prot_barebackbone.md）：两模式结论同构——裸 backbone 天然分布中心蛋白特异、条件化增益集中在远离 native 的臂、encoder 均值可控但逐序列弥散大。

## 8 数据文件索引（全部相对 ConfuMPNN 根）
- `output/exp_control_lig/summary.json`：全量统计（route A/B/Bcal/C 逐臂 + gains + 聚合）
- `output/exp_control_lig/_report_tables.md`：本报告各表 Markdown（作图数据源）
- `output/exp_control_lig/{5O60_E,1CGE,2FEO}/native.json`
- `output/exp_control_lig/{5O60_E,1CGE,2FEO}/routeA/sequences.json`、`seqs.fa`
- `output/exp_control_lig/{5O60_E,1CGE,2FEO}/routeB/arm_{native,n2,p2,n8,p8}/sequences.json`、`seqs.fa`
- `output/exp_control_lig/{5O60_E,1CGE,2FEO}/routeB_cal/arm_{native,n2,p2,n8,p8}/sequences.json`、`seqs.fa`
- 运行日志：`log/exp_lig_control.full.log`
