# 实验报告 — v14 配体结合残基 fix 后重生成（Task2，2026-09-04）

> **状态**：完成。fix 把**结合口袋带电残基 100% 保留**（0.46×→1.0×），但删减**不转移**到非结合区
> （口袋外表面 0.60→0.61、核心 0.39→0.37，几乎不动）→ 全局删减（surface/core 0.4-0.6×）是
> 模型固有"删带电残基总数"倾向，**fix 只堵结合区一个出口，非根治**。电荷命中代价大：H2
> **90%→36%**（mean |dev| 1.0→4.0），正电 native 的核酸/金属结合蛋白过冲最重。H3 聚集违规
> **全部 50 臂略升**（mean viol 0.139→0.172，+3.3pp）但仍在基线+5pp 内（50/50 PASS）。
> **结论：fix=保留功能位点的有效手段（不保折叠不测，本任务不判），但作为"治删减"无效；作"治电荷"有害（需重标定）。**

---

## 一、目的

v14 配体（RNA/DNA+A1global）条件生成有"删带电残基捷径"（native 臂 D/E+K/R 删到 0.43-0.69×，
删减偏口袋）。本任务把**配体结合残基固定为 native**（Cα-配体重原子 ≤8Å）后重新条件生成，
量化：① 删减捷径被抑制多少；② 删减是否转移到非结合区；③ 电荷命中(H2)/电荷聚集(H3) 代价。
只测电荷+删减+聚集，不做 ESMFold/Tm/Sol。

## 二、方法

- **模型**：编码器 `output/finetune_ligand_v14_rna/finetune_epoch050.pt`；骨架
  `LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt`（atom_context=25）。
- **校准**：`output/charge_calibration_v14_ligand_clean.json`（in-10 全在 per-protein 表内；
  表外回退 global——本批无需）。target=(round(native_charge)+Δ−intercept)/slope，同 validate_generalization。
- **结合残基定义**：parse_PDB 的 Y（HETATM/核酸，非水非氢）重原子 与蛋白 Cα 距离 ≤8Å。
  固定位点数量见表 1（6.2%-54%）。**固定机制**：`protein_dict["chain_mask"]=0` →
  `guided_sampler` 原生自回归强制 native（无需改共享采样核心）。验证：固定位 mismatch=0。
- **采样**：新脚本 `code/tests/ligand_v9/sample_fixbinding.py`；in-10 manifest × 5 臂
  (native/n2/p2/n8/p8) × n=40，seed_base=2000，T=0.3，pH7.4，cuda:6。
  产物 `output/fixbinding_v14/`（ligand/<pdb>/pH7.4/arm_*/seqs.fa + validation.json + fixed/*_fixed.json）。
- **对比基线（unfix）**：`output/generalization_ligand_v14_clean/ligand`（同蛋白同臂，n=50）。
- **指标**：mean net charge & dev、H2(|dev|≤2.0)；D/E+K/R 计数总量+3 区（pocket/surface/core，
  surface=frac_sasa≥0.25 非口袋，core=frac_sasa<0.25 非口袋；freesasa 按残基号对齐 parse_PDB）；
  H3=structure_aware_filter 4 规则事后违规率（`code/tests/h3_charge_legality.py`，同 unfix 方法）。
- **分析脚本**：`code/tests/ligand_v9/analyze_fixbinding.py` → `output/v14_fixbinding_summary.json`；
  H3 JSON `output/h3_ligand_v14_fixbinding.json`。
- 注：fix 与 unfix 采样数 n 不同（40 vs 50），mean 指标可比；H2 为单臂均值判据，样本量差异不影响结论。

## 三、结果

### 3.1 结合残基固定位数量 & H2（电荷命中，全 5 臂）

表 1：H2 命中数（5 臂中命中几臂，|mean−target|≤2.0）。native_charge@pH7.4；nfix=固定残基数。

| 蛋白 | L | nfix(fix%) | native charge | D/E+K/R native | **unfix H2** | **fix H2** |
|---|--:|--:|--:|--:|:--:|:--:|
| 6D2O | 209 | 18 (8.6%) | −6.22 | 42 | 5/5 | **5/5** |
| 1AS2 | 312 | 40 (12.8%) | −2.69 | 84 | 5/5 | **5/5** |
| 1BJ4 | 470 | 29 (6.2%) | +0.42 | 105 | 5/5 | **5/5** |
| 2FEO | 221 | 40 (18.1%) | −6.88 | 57 | 0/5 | 0/5 |
| 5CQH | 183 | 55 (30.1%) | −5.53 | 43 | 5/5 | 3/5 |
| 1CGE | 162 | 53 (32.7%) | −11.66 | 36 | 5/5 | 0/5 |
| 21KL_A | 237 | 128 (54.0%) | +10.02 | 62 | 5/5 | 0/5 |
| 5O60_E | 209 | 112 (53.6%) | +11.18 | 51 | 5/5 | 0/5 |
| 3MXB_A | 153 | 73 (47.7%) | +7.94 | 38 | 5/5 | 0/5 |
| 9DWG_L | 323 | 43 (13.3%) | +3.98 | 98 | 5/5 | 0/5 |
| **总计** | | | | | **45/50 (90%)** | **18/50 (36%)** |

分臂汇总：unfix native/n2/p2/n8/p8 = 9/9/9/9/9；fix = 4/4/3/4/3 → 代价在各臂均匀（不只在极端臂）。
mean |dev|：unfix 1.0-1.1 → fix native 4.1 / n2 4.3 / p2 4.0 / n8 5.1 / p8 3.7。

### 3.2 删减（native 臂 D/E+K/R 生成均值/native 倍率，分区）

表 2：native 臂生成带电残基总数倍率，按总量与 3 区（1.0 = 完全保留）。

| 蛋白 | 总量 unfix→fix | 口袋 unfix→fix | 口袋外表面 unfix→fix | 核心 unfix→fix |
|---|:--:|:--:|:--:|:--:|
| 6D2O | 0.56→0.60 | 0.58→**1.00** | 0.57→0.60 | 0.44→0.47 |
| 1AS2 | 0.46→0.55 | 0.30→**1.00** | 0.52→0.51 | 0.30→0.33 |
| 2FEO | 0.46→0.64 | 0.26→**1.00** | 0.56→0.58 | 0.17→0.17 |
| 5CQH | 0.43→0.66 | 0.32→**1.00** | 0.55→0.57 | 0.28→0.26 |
| 1CGE | 0.60→0.82 | 0.44→**1.00** | 0.76→0.77 | 0.43→0.43 |
| 1BJ4 | 0.46→0.47 | 0.49→**1.00** | 0.48→0.49 | 0.38→0.33 |
| 21KL_A | 0.61→0.80 | 0.57→**1.00** | 0.68→0.62 | 0.54→0.57 |
| 5O60_E | 0.56→0.78 | 0.58→**1.00** | 0.59→0.58 | 0.15→0.21 |
| 3MXB_A | 0.69→0.90 | 0.60→**1.00** | 0.83→0.88 | 0.70→0.47 |
| 9DWG_L | 0.47→0.52 | 0.48→**1.00** | 0.47→0.46 | 0.49→0.48 |
| **均值** | **0.53→0.675** | **0.461→1.00** | **0.601→0.607** | **0.388→0.372** |

要点：
- **口袋删减被 fix 构造性消除**（固定位即 native）。总量倍率提升 +0.15，全部来自口袋恢复。
- **删减未转移到非结合区**：口袋外表面 0.601→0.607、核心 0.388→0.372（几乎不动），
  3MXB_A 核心反而删更多（0.70→0.47，其结合面大、可调残基少），无系统性"转移"。
- 因此**删减捷径的主体是非结合区的全局删减**（表面 ~0.60×、核心 ~0.39×），fix 结合区只堵了一个出口。
  与 v12.2 pocket_fix 结论一致：删减是全局性"删带电残基总数"行为。

### 3.3 H3 电荷聚集（structure_aware_filter 4 规则，viol_rate/残基）

表 3：50 臂全部 viol_rate fix > unfix；全部仍在 native_ref/uncond 基线 +5pp 内。

| 指标 | unfix | fix |
|---|:--:|:--:|
| 平均 viol_rate | 0.1390 | 0.1720 |
| Δ(fix−unfix) 均值 | | **+0.0330**（50/50 臂升高；1CGE p8 最大 +0.072） |
| PASS（≤基线+5pp） | 50/50 | **50/50** |

解读：fix 保留口袋带电残基后，模型为达电荷目标被迫在**非结合区更密集**地放置电荷
（或更少删除聚集区域残基），使 4 规则（charge_cluster/salt_bridge/core/same_sign_cluster）
违规率整体上升 +3.3pp，但仍低于 native 结构与无条件基线的 +5pp 上限，**未产生物理不可能的电荷布局**。

## 四、结论

1. **fix 显著抑制"结合区删减"这个出口**：口袋带电残基保留率 0.46×→1.0×（构造性）；
   但**删减不转移到非结合区**——表面/核心倍率 fix 前后不变，说明删减捷径主因是模型**全局**
   删带电残基总数，结合区 fix 只消除了其中与口袋相关的部分（总量 0.53→0.68）。
2. **电荷命中代价大**：H2 90%→36%，mean |dev| ~1→4。正电 native 的核酸/金属结合蛋白
   （5O60_E/21KL_A/3MXB_A/9DWG_L/5CQH/1CGE，结合口袋大且含大量必须保留的带电残基）失效最重；
   无配体配体电荷基线蛋白 6D2O/1AS2/1BJ4（fix% 6-13%）电荷控制基本保留。
   **方向性证据**：删减捷径本质=“删带电残基”，fix 拿走结合区这个杠杆后模型缺乏替代机制
   （几乎不做“反号替换”）→ 正电 native 蛋白（5O60_E/21KL_A/3MXB_A，RNA/DNA 结合）负向臂
   （n8/n2）无法下调，mean 恒定偏高 +10（5O60_E 各臂 +19~+25）；高负电金属结合 1CGE 正向上调
   同样失灵（p8 仍 −9.4）。此外 fix 改变了响应曲线，**unfix 拟合的校准表对 fix 后响应失配**
   （与 v12.2 pocket_fix 2FEO 现象一致）——两者叠加导致 H2 崩。
3. **H3 略升但仍在阈值内**：平均 viol_rate +3.3pp、50/50 臂都在基线+5pp 内——fix 使电荷
   布局更拥挤但不物理不合理。
4. 工程/论文含义：fix 结合位点是**保住功能关键带电残基的有效手段**（对"删减偏口袋"叙述给出
   一个受控对照），但**不是根治全局删减的机制**；若实际以 fix 口袋为生成约束，电荷需**重新现场标定**，
   否则 H2 不成立（本任务沿用 unfix 校准以做同口径对比，代价即 fix 的 H2 偏低）。

## 五、产物
- `output/fixbinding_v14/`（seqs.fa + validation.json + fixed/*_fixed.json）
- `output/v14_fixbinding_summary.json`（逐蛋白逐臂 fix/unfix 组成+H2）
- `output/h3_ligand_v14_fixbinding.json`（H3 事后统计）
- 脚本：`code/tests/ligand_v9/sample_fixbinding.py`、`code/tests/ligand_v9/analyze_fixbinding.py`
- 过程：`session/2026-09-04_task2_fixbinding.md`
