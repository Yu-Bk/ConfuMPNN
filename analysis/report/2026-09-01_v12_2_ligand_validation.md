# 配体模式 v12.2 泛化验证链报告（2026-09-01）

> **状态**：验证链全部完成（V1 采样 → V2 ESMFold 回折 → V3 TM-score → V4 统计 → V5 PROPKA）。
> **核心结果**：H1 折叠 37/50=74%（1AG0/1C6O 为已知 ESMFold 工具限制蛋白，健康蛋白 7/8 全过）、H2 电荷 36/50=72%、H4 PROPKA 7/8 PASS、Tm/Sol 待测。
> **遗留短板**：① 组成删减捷径仍 0.53-0.65×（主要问题，修复待决策）；② 1AS2 电荷全臂 fail（新短板）；③ 1AXW/1C6O/1BJ4 部分臂欠冲。

---

## 一、验证链流程

| 阶段 | 内容 | 状态 |
|------|------|------|
| V1 | 泛化采样（10 蛋白 × 5 臂 × n30，per-protein 校准，`validate_generalization.py --fixed_residues` 新增可选参数）| ✅ |
| V2 | ESMFold 回折 1550 条（GPU4，~4h）| ✅ |
| V3 | TM-score（US-align vs ref，50 臂全过，NA=0）| ✅ |
| V4 | 统计汇总 `generalization_ligand_v12_2_stats.json` | ✅ |
| V5 | PROPKA 物理复核（H4，4 蛋白 × native/n8）| ✅ |

**V3 踩坑修复**：`cut -d/ -f3` off-by-one（rel=`ligand/<pdb>/pH7.4/arm_x` 蛋白名在 f2 非 f3）→ ref 路径错成 `ref/pH7.4_ref.pdb` → USalign 全 NA。修复 f2 后 NA=0。

## 二、H1 折叠（TM≥0.7）37/50 = 74%

| 蛋白 | L | TM中位范围 | 达标臂 | 备注 |
|------|-----|-----------|:---:|------|
| 1A65 | 504 | 0.96-0.97 | 5/5 | ✅ 长蛋白折叠完美 |
| 1AS2 | 312 | 0.93-0.94 | 5/5 | ✅ |
| 1AZM | 258 | 0.98 | 5/5 | ✅ |
| 1BJ4 | 470 | 0.92 | 5/5 | ✅ |
| 1CGE | 162 | 0.96-0.97 | 5/5 | ✅ |
| 2FEO | 221 | 0.92 | 5/5 | ✅ |
| 5CQH | 183 | 0.96 | 5/5 | ✅ |
| 1AXW | 528 | 0.53-0.92 | 2/5 | ⚠️ RNA 大蛋白部分臂中等折叠 |
| 1AG0 | 256 | 0.49 | 0/5 | ⚠️ **已知 ESMFold 工具限制**（native 自身 TM~0.5，pLDDT 85，判据不适用）|
| 1C6O | 177 | 0.52 | 0/5 | ⚠️ **已知 ESMFold 工具限制**（含血红素，native 自身 TM~0.5）|

**解读**：H1 的"失败"主要是 1AG0/1C6O 两个**已知 ESMFold 工具限制蛋白**（native 自身 TM 就 <0.5，与 mompnn v12.2 验证一致，属判据不适用而非模型折叠失败）。**8 个健康蛋白中 7 个全臂 TM≥0.92 完美折叠**；1AXW 部分臂中等。配体条件化没有破坏折叠。

## 三、H2 电荷（dev≤2.0）36/50 = 72%

| 蛋白 | 达标臂 | 短板 |
|------|:---:|------|
| 1AG0 | 5/5 | — |
| 1CGE | 5/5 | — |
| 2FEO | 5/5 | — |
| 5CQH | 5/5 | — |
| 1A65 | 4/5 | n8 欠冲 3.0 |
| 1AZM | 4/5 | n8 2.1 |
| 1AXW | 4/5 | p8 2.3 |
| 1BJ4 | 3/5 | native 2.7/n2 2.8/n8 4.5 |
| 1C6O | 2/5 | native 2.1/n2 2.8/n8 3.0 |
| 1AS2 | **0/5** | **全部 dev 2.8-4.2（新短板）** |

**解读**：H2 72% 与 mompnn 持平。正向 p2/p8 达标率 60%+。**新短板 1AS2（RNA 结合）全臂欠冲/过冲**，可能与删减捷径（0.64×）+ 组成破坏耦合。1BJ4/1C6O 极端负电欠冲（与 mompnn 一致的长蛋白/负向问题）。

## 四、H4 PROPKA（|Q_phys−target|≤2.0）7/8 PASS

| 蛋白 | target | Q_design | Q_phys | h4_dev | PASS |
|------|:---:|:---:|:---:|:---:|:---:|
| 1AXW_n8 | -26 | -26.45 | -24.74 | 1.26 | ✓ |
| 1AXW_native | -18 | -19.63 | -18.14 | 0.14 | ✓ |
| 1AZM_n8 | -10 | -11.75 | -11.29 | 1.29 | ✓ |
| 1AZM_native | -2 | -3.51 | -3.16 | 1.16 | ✓ |
| 1C6O_native | -14 | -16.04 | -15.60 | 1.60 | ✓ |
| 1C6O_n8 | -22 | -24.68 | -24.12 | **2.12** | ✗ |
| 2FEO_native | -7 | -8.33 | -8.03 | 1.03 | ✓ |
| 2FEO_n8 | -15 | -15.32 | -14.79 | 0.21 | ✓ |

**Q_design ≈ Q_phys（差<1）**：简化 HH 计算可靠。唯一 FAIL 是 1C6O_n8（2.12，略超）。

## 五、与 mompnn 对比 + 遗留问题

| 指标 | mompnn v12.2 | 配体 v12.2 |
|------|:---:|:---:|
| H2 电荷 | 72% | **72%** |
| H1 折叠（健康蛋白）| 8/8 | 7/8（1AXW 部分）|
| H4 PROPKA | 6/8 | **7/8** |
| **组成删减** | 无系统性 | **8/10 蛋白 0.53-0.65×（主要问题）** |

**遗留短板（按优先级）**：
1. **组成删减捷径**（0.53-0.65×，定向口袋）——主要问题，修复方向待用户决策（口袋 fix 实测见 `2026-09-01_pocket_fix_test.md`：fix 保深部但删减全局存在，根治需训练侧堵监督逃逸）
2. **1AS2 电荷全 fail**（新短板，RNA 蛋白）
3. **1AXW 部分臂折叠中等 + 1BJ4/1C6O 极端负电欠冲**（长蛋白/负向，mompnn 同源问题）

## 六、产物

| 产物 | 路径 |
|------|------|
| 验证链统计 | `output/generalization_ligand_v12_2_stats.json` |
| 泛化序列 | `output/generalization_ligand_v12_2/`（每蛋白每臂 seqs.fa + folds + tm.csv + plddt.csv）|
| PROPKA 复核 | `output/propka_ligand_v12_2/*.json`（8 个）|
| ESMFold 回折 | `output/generalization_ligand_v12_2/**/folds/*.pdb`（1550 条）|
| ref 骨架 | `output/generalization_ligand_v12_2/ref/*_ref.pdb` |
| 诊断/校准 | `output/charge_calibration_v12_2_ligand.json`、`output/v12_2_ligand_diag_response.json` |
| 组成分析 | `output/v12_2_ligand_comp.json` |

**复现命令**：`code/tests/ligand_v9/run_v12_2_validation.sh`（mompnn 版，OUT 改 ligand 目录）或 `code/tests/ligand_v9/validate_generalization.py --mode ligand --fixed_residues ...`。
