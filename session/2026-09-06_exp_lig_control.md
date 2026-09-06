# 2026-09-06 配体模式补充对照实验（exp1 裸 backbone + exp2 bias-only）— 执行日志

> 归属：`compare/plan_exp1_barebackbone_control.md` + `ablation/plan_exp2_bias_vs_encoder.md`（配体 bundle，GPU=cuda:6）。
> 报告落点（用户 09-06 协调口径）：exp1 → `compare/report_2026-09-06_exp1_lig_barebackbone.md`；exp2 小节 → `ablation/report/2026-09-06_exp2_lig_bias_vs_encoder.md`。数据 `output/exp_control_lig/`。不 git push（主会话统一归档）。

## 1 目标
量化配体模式下：
1. **exp1** 条件电荷控制相对**裸 LigandMPNN 无条件重设计**加了多少控制；
2. **exp2** 去掉 encoder、仅用**简单电荷 logit bias（guided_sampler + charge lookahead）**还剩多少控制。

## 2 受控设置（全部固定）
- backbone：`LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt`（atom_context=25，ligand_mpnn 类型，解析 model_type=auto 检测）。
- 条件模型 = backbone + v14 条件编码器 `output/finetune_ligand_v14_rna/finetune_epoch050.pt`。
- pH=7.4；净电荷 = `net_charge(seq, 7.4)`（含 N/C 端）；配体原子上下文照常（protein 链 + 附近 RNA/配体）。
- 温度 0.3（与既往 v14 验证一致）；seed_base = 3000 + 蛋白序位×100000；同蛋白同臂 A/B/C/Bcal 用同一批 seed 偏移（每序列 `torch.manual_seed(seed_base+k)`）。
- 采样均逐条 batch=1（conditioned_sample / GuidedSampler），保证 per-index seed 复现与配对可比。

## 3 测试蛋白（好/中/坏，选择依据 = v14 clean 链 H2/回收/slope）
| pdb | 档位 | L | native_q | v14 clean H2 | rec | diag slope | 依据 |
|---|---|---|---|---|---|---|---|
| 5O60_E | 好（RNA 结合，核糖体蛋白E，rRNA） | 209 | +11.18 | 5/5 | 0.39 | 1.78 | 天然正电 RNA 结合，符合"好必须 RNA 结合蛋白" |
| 1CGE | 中（金属 CA+ZN） | 162 | −11.66 | 5/5 | 0.54 | 0.98 | 近单位响应、回收最高，作"中" |
| 2FEO | 难（DC 核苷酸结合） | 221 | −6.88 | 0/5 | 0.32 | 0.93 | clean 验证唯一全臂失败的蛋白（电荷高方差），作"难" |

manifest 参照 `data/validation_pdbs/validation_manifest_v14_in.json`；pdb 用 `data/validation_pdbs/{pdb}.pdb`（含配体）。

## 4 route 与采样规模
- route A 裸无条件：3 蛋白 × n=1000（enc=None）。
- route B 条件直接生成：5 臂 native/n2/p2/n8/p8 × n=1000（target=round(native_q)+Δ，**未校准**直接注入）。
- route C bias-only：native/n2/p2 × n=1000，strength=0.5（ChargeLookahead，无结构过滤器——纯电荷 bias）。
- route B_cal 附加组：3 蛋白 × 5 臂 × n=1000，per-protein 校准表 `charge_calibration_v14_ligand_clean.json`（5O60_E s1.78 o−0.705 / 1CGE s0.983 o−2.349 / 2FEO s0.935 o−5.082）。原因：诊断显示 **route B 直接生成三蛋白系统性过冲/欠冲**（5O60 正偏 +5~+9；1CGE/2FEO 负偏 −3~−6），"某臂 B 很差→补小样本现场标定附加组"。

总计 ~4.2 万序列（A 3000 + B 15000 + C 9000 + Bcal 15000），LigandMPNN+RNA 上下文 ~0.33–0.48 s/seq，GPU6 独占，预计 ~4.5–5 h。

## 5 脚本
- `code/tests/exp_lig_control/run_exp_control_lig.py` — 采样（断点续跑：文件存在则跳过）。
- `code/tests/exp_lig_control/analyze_exp_control_lig.py` — 汇总 → `output/exp_control_lig/summary.json`。
- smoke（已跑）：`output/exp_control_lig/_smoke*`（n=4/50）验证管线与吞吐。

### smoke n=50 关键预兆（5O60_E, 未校准口径）
- route A 裸：mean q=+27.1，native±1 占比 0.000，p5–p95=[19.9, 32.1] → 裸 backbone 在 RNA 结合蛋白上强烈正偏（native +11.2）。
- route B 直接：mean dev native +6.7 / n2 +4.9 / p2 +8.8 / n8 +0.7 / p8 +9.3；per-seq hit≤2 0.04–0.38 → 正臂过冲严重、负臂(n8)尚可。
- route C bias(0.5)：native/n2/p2 mean dev +1.7/+1.9/+1.6，per-seq hit≤2 0.50–0.58 → **简单 bias 精确命中 target**，但带电残基总数倍率 cr≈1.45–1.49（以堆带电残基换电荷）。
- route B 组成 cr 0.46–1.12（延续"删减带电残基"路径）——与 v14 已知删除问题一致。
- route B_cal（校准）native/n8/p8 mean dev −1.7/−0.2/−0.2 → 校准把 encoder 拉回目标。

**故事雏形**：裸 backbone 无电荷约束（天然高正偏）；encoder 以"调整组成（减少带电残基）"实现电荷控制、直接注入有过冲增益需校准；简单 bias 精确命中但以 1.45× 带电残基为代价——电荷-组成 trade-off 三者不同。

## 6 执行时间线
- 01:52 smoke50 完成（~200s/450 seq）。
- 01:59 全量 nohup 启动（pid 1531125，`log/exp_lig_control.full.log`），GPU6。
- 05:08 5O60_E + 1CGE 完成（A/B/C/B_cal 全量）；07:01 2FEO 完成（"采样全部完成"）。
- 07:08–07:56 route C 扩展臂 n8/p8（pid 2108126，`log/exp_lig_control.Cext.log`）——补全 bias 在极值臂表现。
- 07:57 analyze 汇总 → `output/exp_control_lig/summary.json` + `_report_tables.md`。
- 报告落盘：exp1 → `compare/report_2026-09-06_exp1_lig_barebackbone.md`；exp2 → `ablation/report/2026-09-06_exp2_lig_bias_vs_encoder.md`。

## 7 结果速览（n=1000/臂；详见两报告）
| route | per-seq 命中≤2 | mean\|dev\| | 均值口径 H2 | rec | cr |
|---|---|---|---|---|---|
| A 裸（native±1 命中基线） | 13.8% | — | — | ~0.49 | ~1.14 |
| B 直接（encoder 未校准） | 0.266 | 3.60 | 3/15 | 0.418 | 0.69 |
| B_cal（encoder+per-protein 校准） | 0.334 | 1.44 | 9/15 | 0.414 | 0.58 |
| C（bias-only lookahead 0.5） | 0.785 | 1.07 | 14/15 | 0.470 | 1.34 |

exp1 增益（B−A±1，3 蛋白×5 臂均值）：B_direct +0.18；B_cal +0.25（极端臂 ±8 最大 +0.29~+0.35）。
exp2：B_cal − C per-seq 命中 = **−0.45**（C 全面领先）；C 以带电残基膨胀（cr 1.0–1.6）贴靶，encoder 以删减（cr 0.36–0.94）控制——两机制都偏离 native 组成、方向相反；C native 序列回收 ≥ B_cal。
结论（同蛋白模式）：配体模式在"电荷命中"单指标上 learned encoder 落后于简单 bias；encoder 价值须在电荷之外维度论证。

## 8 报告落点（数据文件路径见报告正文）
- exp1 图/表：`compare/report_2026-09-06_exp1_lig_barebackbone.md`
- exp2 小节：`ablation/report/2026-09-06_exp2_lig_bias_vs_encoder.md`
- 数据根：`output/exp_control_lig/`（seqs.fa + sequences.json + summary.json + _report_tables.md）
