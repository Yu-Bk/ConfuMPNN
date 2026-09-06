# 对比实验 exp7（蛋白模式 v12.2/MoMPNN）— 跨 pH(5/7.4/9) 敏感性（2026-09-06）

> 归属：`compare/`（条件敏感性对照）。计划：`compare/plan_exp7_pH_response.md`。session：`session/2026-09-06_exp57_run.md`。
> 配体模式对照：`compare/report_2026-09-06_exp7_pH_lig.md`。
> 机器可读/作图数据：`output/exp_pH_prot/_summary.json`、`output/exp_pH_prot/_report_tables.md`；逐序列 `output/exp_pH_prot/<pdb>/class{A,B}.json`。
> 运行：GPU=cuda:6；日志 `log/exp7_prot.full.log`；不 git push。

## 1 目的与设置
证 **v12.2 编码器是否对 pH 本身敏感（不只是对电荷数字）**，并给出 pH 环境的控制边界。
- backbone：MoMPNN + v12.2 ConditionEncoder（`output/finetune_v12_2/finetune_epoch030.pt`）。
- 净电荷 = `net_charge(seq, pH)`（pKa sigmoid，含 N/C 端）。温度 0.3；n=60/臂。
- 同 seed 跨 pH 配对（同一 seed 序列逐条可比）。

**class A｜换 pH 看臂**：target = round(native_q@pH)+Δ（pH 锚定，native/n2/p2/n8/p8）；
注入用 **7.4 固定校准**（1AZM/1AS2 = v12.2 表 per-protein；1BJ4 = exp1 route_Bsmall 现场标定
`output/exp_control_prot/1BJ4/route_Bsmall/1BJ4_smallcal.json` slope 2.422/inter 1.729；
3MXB_A = 本次现场小样本标定 `output/exp_pH_prot/3MXB_A/smallcal74.json` slope 1.475/inter −1.361）。
判据：dev_of_mean=|mean_q−target|；H2=dev≤2；per-seq≤2=命中占比。

**class B｜固定 target 换 pH**：固定 target（T_native74 = round(native_q@7.4)；T_zero=0），
pH 5/7.4/9 **不校准**直接注入。量 **实测净电荷 q_own=net_charge(seq,本pH)**、组成、以及
相对 pH7.4 的同 seed 逐序列一致率（identity）。

## 2 蛋白代表（native pI 覆盖 碱/中性/酸）
数据：`output/exp_pH_prot/<pdb>/native.json`。

| 蛋白 | L | native_q@pH5 | @7.4 | @9 | round74 | 角色 |
|---|---|---|---|---|---|---|
| 1AZM | 258 | +11.19 | −1.71 | −4.13 | −2 | 中性 |
| 1AS2 | 312 | +8.66 | −2.69 | −10.59 | −3 | 酸 |
| 1BJ4 | 470 | +21.24 | +0.42 | −9.78 | 0 | 中性偏碱（长） |
| 3MXB_A | 153 | +11.48 | +7.95 | +6.14 | +8 | 碱（正电 native） |

注意：同一 native 序列的净电荷随 pH 变化极大（如 1BJ4 从 pH5 的 +21 到 pH9 的 −10），
故 class A 的 target 本身随 pH 移动——这正是测试点。

## 3 class A：跨 pH 电荷控制（pH 锚定 target）
逐臂表：`output/exp_pH_prot/_report_tables.md` classA 各蛋白段。

| pH | H2 达标 | mean dev_of_mean | mean per-seq≤2 | 逐臂 identity vs 7.4（均值） |
|---|---|---|---|---|
| 5.0 | **0/20** | 7.44 | 0.131 | 0.78 |
| 7.4 | 19/20 | 1.06 | 0.292 | — |
| 9.0 | 13/20 | 2.70 | 0.237 | 0.84 |

（4 蛋白 × 5 臂 = 20 臂/行；仅 1BJ4 p8@7.4 dev 3.18 未达 H2，与 exp1 一致。）

### 3.1 逐蛋白
| 蛋白 | pH5 H2 | pH5 dev 范围 | pH7.4 H2 | pH7.4 dev 范围 | pH9 H2 | pH9 dev 范围 |
|---|---|---|---|---|---|---|
| 1AZM | 0/5 | 4.1–9.9 | 5/5 | 0.2–1.2 | 5/5 | 0.6–1.1 |
| 1AS2 | 0/5 | 5.0–7.6 | 5/5 | 0.3–1.8 | 0/5 | 5.5–8.1 |
| 1BJ4 | 0/5 | 10.1–12.9 | 5/5 | 0.3–3.2 | 5/5 | 0.8–1.7 |
| 3MXB_A | 0/5 | 2.7–5.3 | 5/5 | 0.0–1.8 | 3/5 | 0.1–3.3 |

要点：
1. **pH 7.4 控制完整**（H2 19/20，mean dev≈1），是既有验证基线。
2. **pH 5 全面失效**：4 蛋白 0/20 达 H2，mean dev 7.4。根因是低 pH 下 native 电荷大幅转正
   （1AZM/1AS2/1BJ4/3MXB_A 的 native@5 ≈ +9~+21），target 变得很正；v12.2 编码器在这些
   强正 target 下**系统性欠冲**（如 1BJ4 native target +21 只生成 mean +8.3；1AZM target +11
   只 +3.6）→ 生成的正电残基不足。
3. **pH 9 部分失效**：主要在酸蛋白 1AS2（target −11~−19，生成 mean 只到 −4~−13，欠负冲，
   H2 0/5）与碱蛋白 3MXB_A 的 p2/p8（dev≈3.3）。中性/偏碱长蛋白 1BJ4 在 pH9 全过（−10~−18
   均达）。
4. **序列随 pH 变**：class A 同臂下 pH5 序列相对 7.4 有 ~22% 位置不同（identity 0.78），
   pH9 ~16% 不同（0.84）——即同电荷臂也非同一序列，编码器确实在 pH 维度上改了序列，
   但改得**不彻底**（部分补偿，欠冲）。

结论（class A）：**v12.2 蛋白模式对电荷的控制不是 pH 不变的**。7.4 最优；pH9 在除强负
target 的酸蛋白外仍可；**pH5 是明确边界**（native 转正 → 强正 target → 编码器欠冲），
所有测试臂 H2 失败。若要在 pH5 环境使用，必须按该 pH 重新做响应标定（而非套用 7.4 斜率）。

## 4 class B：固定 target 换 pH — 验真 pH 敏感
逐表：`output/exp_pH_prot/_report_tables.md` classB 各蛋白段；数据 `output/exp_pH_prot/<pdb>/classB.json`。

### 4.1 跨蛋白聚合
| target | pH5 mean_q | pH7.4 mean_q | pH9 mean_q | id(pH5 vs7.4) | id(pH9 vs7.4) |
|---|---|---|---|---|---|
| T_native74（4 蛋白 target −3~+8） | **−6.05** | +0.29 | **+3.21** | 0.796 | 0.856 |
| T_zero（=0） | **−7.26** | −1.48 | **+0.86** | 0.792 | 0.862 |

固定 target、仅换 pH：跨蛋白 mean q_own 从 pH5 到 pH9 漂移 **~7–9 个电荷**，
同 seed 序列一致率只有 **0.79–0.86**（即 14–21% 位置随 pH 改变）——**序列确实随 pH 而变**，
模型不是"只看 target 数字、忽略 pH"。

### 4.2 方向：组成如何随 pH 变（例：1AS2 T_native74 target −3，表 `_report_tables.md`）
| pH | mean_q | n_pos(K+R) | n_neg(D+E) | id vs 7.4 |
|---|---|---|---|---|
| 5.0 | −13.0 | 25.0 | **47.8** | 0.762 |
| 7.4 | −6.9 | 23.0 | 30.0 | — |
| 9.0 | −1.0 | 30.2 | 29.1 | 0.845 |

低 pH 下模型放入**更多负电残基 D/E**（pH5 47.8 vs pH9 29.1）但仍把实测 q_own 推到更负
（−13），即对"负 target 应加 D/E"作了物理方向正确的强响应但**过冲**；高 pH 反向减少 D/E 并增
加 K/R。这证明 pH 维度在驱动序列/组成，而非仅重复 7.4 的电荷映射。

### 4.3 判定
**pH 敏感是真阳性**：固定电荷 target 时，仅改 pH 会系统性改变生成序列（identity 0.79–0.86）
与带电组成，且实测净电荷随 pH 大幅漂移。**但该 pH 响应不是"恒 target"的**——在固定标称
target 下，模型的实测净电荷随 pH 移动 ±7~9，说明其 pH-电荷映射未按 pH 校准到位（尤其 pH5）。

## 5 极端 pH 抽查（H1 结构保持，ESMFold TM）
对最可能破坏折叠的极端条件各抽 8 条回折（ESMFold, confumpnn-esmfold）并 US-align 到 native 骨架：
- 1AS2 @pH9（酸蛋白强负电；native arm target −11，n8 arm target −19）：
  **TM 0.90–0.96**（16/16 > 0.7），mean pLDDT 78–82。
- 3MXB_A @pH5（碱蛋白强正电；native arm target +11，p8 arm target +19）：
  **TM 0.87–0.97**（16/16 > 0.7），mean pLDDT 82–84。
→ 极端电荷重设计（±19~−19）未见折叠破坏；数据路径
  `output/exp7_H1/prot/{1AS2,3MXB_A}/{pH9.0_n8,pH9.0_native,pH5.0_p8,pH5.0_native}_plddt.csv`、
  `tm_*.csv`。

**H4（PROPKA 微环境电荷复核，同一批回折结构）**：Q_phys（PROPKA 修正 pKa 重算净电荷）与
Q_design（net_charge 设计口径）一致（差 <1.5），说明序列电荷**物理自洽**；H4 判定基本跟随
class A 是否达靶——1AS2 pH9 native/n8（target −11/−19，均欠负冲）FAIL（dev 8.2/4.0）、
3MXB_A pH5 p8（target +19 欠正冲）FAIL（dev 4.3）、而 3MXB_A pH5 native（达靶）**PASS**
（dev 0.59）。即未出现"设计电荷合理但 PROPKA 物理上不可能"的额外破坏。JSON：
`output/exp7_H1/prot/{1AS2,3MXB_A}/h4_*.json`。

## 6 结论（蛋白模式）
1. **v12.2 编码器对 pH 本身敏感**（非只对电荷）：class B 固定 target 下 pH 改变 → 序列变化
   14–21%、组成随 pH 系统移动。
2. 但电荷控制**随 pH 边界显著**：class A 在 pH5（target 强正）0/20 H2、mean dev 7.4；
   pH9 对强负 target 的酸蛋白也欠冲（0/5）；7.4 仍是控制甜点区。
3. 含义：**pH5 环境不能用 7.4 校准斜率**；需 pH 现场标定（响应曲线在低 pH 下斜率/饱和区不同），
   且强正 target（如 pH5 的 native）超出 v12.2 可控区。折叠（H1 抽查）未被破坏。

## 7 数据源索引（相对根）
- 逐序列与元信息：`output/exp_pH_prot/{1AZM,1AS2,1BJ4,3MXB_A}/native.json`、`classA.json`、`classB.json`
- 逐臂汇总：`output/exp_pH_prot/_summary.json`；人读表：`output/exp_pH_prot/_report_tables.md`
- 现场标定：`output/exp_pH_prot/3MXB_A/smallcal74.json`；1BJ4 复用 `output/exp_control_prot/1BJ4/route_Bsmall/1BJ4_smallcal.json`
- H1 抽查：`output/exp7_H1/prot/1AS2/{pH9.0_n8_plddt.csv,folds_pH9n8,tm_pH9n8.csv}`、`3MXB_A/...`
- 日志：`log/exp7_prot.full.log`
