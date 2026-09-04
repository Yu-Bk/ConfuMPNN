# v14 配体干净测试集链 — 权威验证报告（2026-09-04）

> 本报告为 **v14 配体模式最终权威测试集结果**（论文引用本文，勿引用 2026-09-02 旧 v14 主链产物）。
> 口径依据：`analysis/report/2026-09-03_validation_standards.md` §4.2；数据 = 全链产物 `output/*_clean`。
> 版本：`finetune_ligand_v14_rna/finetune_epoch050.pt`（RNA/DNA 扩充 414 域 + A1 global，50ep）。
> 测试集 = **in-10 manifest**（`validation_manifest_v14_in.json`：6D2O/1AS2/2FEO/5CQH/1CGE/1BJ4/21KL_A/5O60_E/3MXB_A/9DWG_L，全部 coverage=in、无泄露）+ boundary 1A65 单列。
> 深负/深正代表蛋白（2E9R_X 等）不在此列，见 `analysis/report/2026-09-03_long_neg_charge_limitation.md`。

---

## 〇、执行方式与修复记录（为何本文是干净的）

全链 `/tmp/run_v14_ligand_validation_clean.sh` 从头一致重算（新 in-10 含 5O60_E）。过程中发现并修复 **3 处脚本硬编码旧蛋白清单 bug**（均已入库，同类根因 = 代码写死 v12.2 时代旧 manifest）：
1. `h3_charge_legality.py` 硬编码 PDBS 含已删二聚体 → 加 `--manifest` 覆盖；⑧ H3 由崩溃 → 50/50。
2. `compare_comp_ligand.py` 硬编码旧 `validation_manifest.json` → 加 `--manifest`；④ 组成由 5/10 → **10/10**。
3. `v12_2_ligand_tm_sol_summarize.py` 硬编码旧 PDBS → 加 `--manifest`；⑨ Tm/Sol 由缺 5 个新成员 → **新 in-10 全 10**。

---

## 一、电荷响应与校准

| 项 | 数值 |
|---|---|
| 诊断 slope（valid 10 均值，未校准） | **1.473**（6D2O 1.63 / 1AS2 1.45 / 2FEO 0.93 / 5CQH 1.05 / 1CGE 0.98 / 1BJ4 2.18 / 21KL_A 1.74 / 5O60_E 1.78 / 3MXB_A 1.24 / 9DWG_L 1.75）+ trainish 8 |
| 校准表 global | slope **1.492** / intercept **−1.260**（`charge_calibration_v14_ligand_clean.json`，18 蛋白 per_protein） |

判据参考：valid 区内 slope ∈ [0.9,1.15] 为理想。未校准均值 1.473 → **存在响应过冲增益**（长/大电荷蛋白 1BJ4/21KL_A/5O60_E/9DWG_L 明显 >1.3），校准表用于推理侧校正。

## 二、组成分析（④，10/10 蛋白，native 臂 target=native 电荷）

**D/E+K/R 带电残基总数 native→生成倍率：**

| 蛋白 | 倍率 | | 蛋白 | 倍率 |
|---|---|---|---|---|
| 6D2O | **0.56** | | 1BJ4 | **0.46** |
| 1AS2 | **0.46** | | 21KL_A | **0.61** |
| 2FEO | **0.46** | | 5O60_E | **0.56** |
| 5CQH | **0.43** | | 3MXB_A | **0.69** |
| 1CGE | 0.60 | | 9DWG_L | 0.47 |

**⚠️ 结论：系统性删减带电残基仍未根治（0.43-0.69×，10/10 全部低于 native）**——即使 target=native 自身电荷，"删减捷径"依然存在：模型用较少带电残基凑出目标净电荷。与 v12.2/v13 观察一致（组成删减是跨版本未决问题，见决策 D，待 v12.4/组成删减方案裁决）。

## 三、H2 电荷命中（⑤采样→⑥统计，per-protein 表内校准口径）

**totals：H2 45/50（90%）**，H1 TM≥0.7 **50/50（100%）**（`v14_ligand_gen_stats_clean.json`）。

| 蛋白 | H2 | TM≥0.7 | native rec | gravy |
|---|---|---|---|---|
| 6D2O | 5/5 | 5/5 | 0.38 | −0.45 |
| 1AS2 | 5/5 | 5/5 | 0.35 | −0.22 |
| 2FEO | **0/5** | 5/5 | 0.32 | 0.04 |
| 5CQH | 5/5 | 5/5 | 0.44 | −0.40 |
| 1CGE | 5/5 | 5/5 | 0.54 | −0.51 |
| 1BJ4 | 5/5 | 5/5 | 0.43 | −0.25 |
| 21KL_A | 5/5 | 5/5 | 0.32 | −0.58 |
| 5O60_E | 5/5 | 5/5 | 0.39 | −0.30 |
| 3MXB_A | 5/5 | 5/5 | 0.49 | −0.52 |
| 9DWG_L | 5/5 | 5/5 | 0.37 | −0.37 |

- 9/10 蛋白全臂命中；**唯一失败 2FEO（0/5）**：其响应 slope 0.93 正常、TM 全过，但 5 个目标臂 mean charge 偏差均 >2。2FEO 为 DC 结合（221aa），native 电荷近 0、偏差集中在生成方差大/目标区间过近，属该蛋白特例（此前 v12.2 蛋白模式也记录 2FEO 高方差需小样本标定）。逐臂表见 `output/v14_ligand_gen_stats_clean.json`。
- 电荷控制整体：**温和 native±2 与多数 ±8 可靠**；已知边界（长×大电荷）另见 capability-limits 文档。

## 四、H1 折叠 + H3 电荷聚集合法性 + H4 物理真实性

- **H1（ESMFold→TM，n50 每蛋白）**：plddt 50/50、tm 50/50 → **TM≥0.7 命中 50/50（100%）**，监督与电荷控制未伤折叠。
- **H3（电荷聚集/非法聚集，10 蛋白 ×5 臂 vs native/uncond 基线 +5pp）**：**50/50（100%）**。
- **H4（PROPKA 物理电荷 vs 设计 target，1BJ4/21KL_A/3MXB_A × native/n8）**：

| 臂 | target | q_design 均值 | q_phys(PROPKA) 均值 | mean\|q_phys−target\| |
|---|---|---|---|---|
| 1BJ4 native | 0.42 | 0.22 | 0.96 | 4.98 |
| 1BJ4 n8 | −8.0 | −6.97 | −6.35 | 4.99 |
| 21KL_A native | 10.02 | 8.65 | 8.89 | 5.31 |
| 21KL_A n8 | 2.0 | 0.49 | 0.87 | 4.70 |
| 3MXB_A native | 7.94 | 8.61 | 8.92 | 3.52 |
| 3MXB_A n8 | 0.0 | 1.41 | 1.68 | 3.91 |

  **解读**：模型自计数电荷（q_design）贴近 target（方向/量级正确），但 PROPKA 物理电荷与 target 偏离 **3.5-5.3** —— 无系统性翻车（无符号反转、无折叠失败），但存在对简单 pI 计数的部分依赖（His/末端/pKa 差异由 PROPKA 揭示），量级上物理可实现性**中等偏可**，作 caveat 报备而非判据级失败。

## 五、Tm/Sol（⑨，S2 判据 vs 无条件基线）

**S2 明显恶化臂数 = 0/50**（ΔTm < −5 或 Δ%sol < −10 为恶化；`tm_sol_summary.json`，新 in-10 全 10）。电荷工程化设计**未引入热稳定性/溶解度系统恶化**。

## 六、Boundary 1A65（单列，global 校准回退，不进判据）

native_charge **−26.85**（深负长蛋白，native 电荷落在训练 q 分布 ~2.8% 分位 + L504 超训练 max500）。

| 臂 | target | mean | dev |
|---|---|---|---|
| native | −26.9 | −24.4 | 2.6 |
| n2 | −28.9 | −25.8 | 3.2 |
| p2 | −24.9 | −22.7 | 2.3 |
| n8 | −34.9 | −32.0 | 3.0 |
| p8 | −18.9 | −16.9 | 2.1 |

符合"长×深负可设计性有限"档（欠冲方向，见 `2026-09-03_long_neg_charge_limitation.md`）。

---

## 七、结论

| 判据 | 结果 | 状态 |
|---|---|---|
| 校准后 H2（in-10，per-protein） | 45/50（90%） | ✅ 达标（9/10 蛋白全臂，2FEO 特例 0/5） |
| H1 折叠 TM≥0.7 | 50/50 | ✅ |
| H3 电荷聚集合法性 | 50/50 | ✅ |
| H4 PROPKA 物理真实性 | 偏差 3.5-5.3，无翻车 | ⚠️ caveat |
| Tm/Sol S2 恶化 | 0/50 | ✅ |
| 组成（带电残基总数） | 0.43-0.69× native | ❌ **删减捷径未根治（跨版本未决）** |
| boundary 1A65 | dev 2.1-3.2 欠冲 | ⚠️ 能力边界档 |

**v14 干净测试集链条全面收敛**：电荷命中 90%、折叠/合法性/稳定性全绿；**唯一未决硬伤 = 组成系统性删减**（0.43-0.69×，全 10 蛋白），与既往 v12.2/v13 一致，属需用户决策 D 的未解决问题（v12.4 / 组成删减监督 / 论文口径）。

---

*产物：`output/*_clean`（DIAG/CAL/COMP/STATS/H3OUT/PROPKA/tm_sol + generalization_ligand_v14_clean）；逐阶段日志 `log/v14_ligand_validation_clean.*`；检查点日志 `session/2026-09-04_v14_clean_chain_autolog.md`。*
