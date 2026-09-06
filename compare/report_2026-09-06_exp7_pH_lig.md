# 对比实验 exp7（配体模式 v14/LigandMPNN）— 跨 pH(5/7.4/9) 敏感性（2026-09-06）

> 归属：`compare/`（条件敏感性对照）。计划：`compare/plan_exp7_pH_response.md`。session：`session/2026-09-06_exp57_run.md`。
> 蛋白模式对照：`compare/report_2026-09-06_exp7_pH_prot.md`。
> 机器可读/作图数据：`output/exp_pH_lig/_summary.json`、`output/exp_pH_lig/_report_tables.md`；逐序列 `output/exp_pH_lig/<pdb>/class{A,B}.json`。
> 运行：GPU=cuda:6；日志 `log/exp7_lig.full.log`；不 git push。

## 1 目的与设置
证 **v14 配体模式编码器是否对 pH 本身敏感**，并给出其 pH 边界。
- backbone：LigandMPNN（atom_context=25）+ v14 ConditionEncoder（`output/finetune_ligand_v14_rna/finetune_epoch050.pt`）。
- 净电荷 = `net_charge(seq, pH)`（pKa sigmoid，含 N/C 端）；温度 0.3；n=60/臂；同 seed 跨 pH 配对。

**class A｜换 pH 看臂**：target = round(native_q@pH)+Δ，注入用 **7.4 固定 per-protein 校准**
（`output/charge_calibration_v14_ligand_clean.json`：5O60_E s=1.78/o=−0.705；1CGE s=0.983/o=−2.349；
2FEO s=0.935/o=−5.082）。
**class B｜固定 target 换 pH**：固定 target（T_native74/T_zero），**不校准**直接注入，
看实测 q_own 与同 seed identity/组成随 pH 的变化方向。

## 2 蛋白代表（native pI 覆盖 碱/中性/酸）
数据：`output/exp_pH_lig/<pdb>/native.json`。

| 蛋白 | L | native_q@pH5 | @7.4 | @9 | round74 | 角色 |
|---|---|---|---|---|---|---|
| 5O60_E | 209 | +17.85 | +11.18 | +10.02 | +11 | 碱（RNA 结合核糖体蛋白） |
| 1CGE | 162 | −1.22 | −11.66 | −12.66 | −12 | 酸（金属 CA+ZN） |
| 2FEO | 221 | +2.39 | −6.88 | −8.36 | −7 | 酸（DC 核苷酸结合；v14 已知难蛋白） |

## 3 class A：跨 pH 电荷控制（pH 锚定 target）
逐臂表：`output/exp_pH_lig/_report_tables.md` classA 各蛋白段。

| pH | H2 达标 | mean dev_of_mean | mean per-seq≤2 | 逐臂 identity vs 7.4（均值） |
|---|---|---|---|---|
| 5.0 | 7/15 | 2.00 | 0.312 | 0.77 |
| 7.4 | 9/15 | 1.46 | 0.349 | — |
| 9.0 | 4/15 | 3.40 | 0.258 | 0.85 |

（3 蛋白 × 5 臂 = 15 臂/行。）

### 3.1 逐蛋白
| 蛋白 | pH5 H2 | pH5 dev 范围 | pH7.4 H2 | pH7.4 dev 范围 | pH9 H2 | pH9 dev 范围 |
|---|---|---|---|---|---|---|
| 5O60_E | 2/5 | 1.5–3.0 | 4/5 | 0.3–2.5 | 2/5 | 0.2–2.7 |
| 1CGE | 1/5 | 1.5–3.2 | 5/5 | 0.1–1.0 | 2/5 | 1.6–3.0 |
| 2FEO | 4/5 | 0.4–2.3 | 0/5 | 2.0–2.7 | 0/5 | 5.2–7.3 |

要点：
1. **pH 7.4 基线并非全过**（9/15）：2FEO 0/5（与 v14 clean 已知一致，响应弱/方差大），
   5O60_E native 臂 dev 2.46 差一档——这是既有边界。
2. **pH5 相对温和退化**（H2 7/15，mean dev 2.0，不似蛋白模式崩盘）。2FEO 在 pH5 反而最好
   （4/5）：其 native@5 ≈ +2 靠近零点，响应落入易控区。1CGE/5O60_E 在 pH5 各降到 1–2/5，
   dev ≈ 1.5–3.2（低 pH 转正、正 target 未完全追到）。
3. **pH9 是配体模式最差区**（H2 4/15，mean dev 3.4）：2FEO 完全失守（负 target 达不到，
   如 native target −8 只到 −2.8，p8 target 0 反而 +7.3 正冲），1CGE 降到 2/5（负 target 偏正
   欠冲，dev≈3.0），5O60_E 2/5。
4. 序列随 pH 变（class A 同臂）：pH5 vs 7.4 identity ≈ 0.77（1CGE 0.83 较稳；2FEO/5O60_E
   0.70–0.75），pH9 vs 7.4 ≈ 0.85。

结论（class A）：**v14 配体模式的电荷控制也随 pH 漂移**，但形状与蛋白模式相反——
蛋白模式 pH5 崩盘、pH9 尚可；配体模式 pH5 相对温和、**pH9 最差**（强负 target 欠负冲/正冲）。
两模式的甜点都在 pH7.4。pH9 对负电/中性 target 需单独标定。

## 4 class B：固定 target 换 pH — 验真 pH 敏感
逐表：`output/exp_pH_lig/_report_tables.md` classB 各蛋白段。

### 4.1 跨蛋白聚合
| target | pH5 mean_q | pH7.4 mean_q | pH9 mean_q | id(pH5 vs7.4) | id(pH9 vs7.4) |
|---|---|---|---|---|---|
| T_native74（target −12~+11） | −3.67 | −2.46 | −0.41 | 0.771 | 0.852 |
| T_zero（=0） | −3.15 | −2.08 | +0.02 | 0.780 | 0.850 |

固定 target、仅换 pH：跨蛋白 mean q_own 漂移 ~3–7 个电荷（幅度小于蛋白模式），同 seed
一致率 0.77–0.85 → 序列随 pH 改变 15–23%。

### 4.2 方向性例子（表 `_report_tables.md` classB 各蛋白）
- **1CGE T_native74 target −12**：q_own pH5 −18.4 → pH9 −11.4；D/E 33.2→18.1（低 pH 加负电残基、
  高 pH 减负电残基，物理方向正确但低 pH 下过冲到 −18）。
- **2FEO T_zero target 0**：q_own pH5 −3.5 → pH9 +1.0；D/E 28.2→16.6、K/R 18.0→20.0（pH9 加正
  电减负电保持近 0）。
- **5O60_E T_native74 target +11**：q_own 几乎恒定（pH5 +16.2 / pH9 +17.0），但序列变化
  id(pH5)=0.746、id(pH9)=0.822，K/R 25.4→33.0（随 pH 升高加更多正电残基以把实测电荷维持在
  ~+17）——**用序列变化补偿 pH 对电荷测量值的影响**，是明确的 pH-组成响应。

### 4.3 判定
**pH 敏感为真阳性**：固定 target 下仅改 pH → 生成序列与带电组成系统变化（identity
0.77–0.85），且方向物理一致（低 pH 更依赖负电残基、高 pH 加正电残基补偿）。但同样**不是
"恒 target"**：固定标称 target 时实测 q_own 仍随 pH 移动（≤~7 个电荷），需按 pH 重标定。

## 5 结论（配体模式）
1. v14 编码器对 pH 本身敏感（固定电荷 target、换 pH 会改序列与组成）。
2. 控制边界与蛋白模式相反：**pH9 最差**（2FEO 全失、mean dev 3.4），pH5 温和退化，
   pH7.4 甜点（但也受 2FEO 基线弱限制）。
3. 与蛋白模式一致的含义：极端 pH 下不能直接套 7.4 校准；v14 在强负 target + 高 pH 组合
   明显欠负冲/正冲。

## 6 数据源索引（相对根）
- 逐序列与元信息：`output/exp_pH_lig/{5O60_E,1CGE,2FEO}/native.json`、`classA.json`、`classB.json`
- 逐臂汇总：`output/exp_pH_lig/_summary.json`；人读表：`output/exp_pH_lig/_report_tables.md`
- 校准：`output/charge_calibration_v14_ligand_clean.json`
- 日志：`log/exp7_lig.full.log`
