# 2026-09-04 v13 配体 in-10 全链权威对照（v13-in10 vs v14-clean）

## 目的与协议
v13（`output/finetune_ligand_v13/finetune_epoch030.pt`，**训练无 RNA/DNA 数据、A1 pocket(keep) 非 global**）在
**v14-clean 同一套测试协议**上完整重跑，产出与 v14-clean 逐蛋白同集同协议可比的权威结果，**取代旧 v13 旧测试集数字**作为对照。
旧产物 `output/generalization_ligand_v13` 与旧 json（`v13_ligand_*.json`）**保留不动、仅归档**。

- in-10 manifest：`data/validation_pdbs/validation_manifest_v14_in.json`（10 蛋白，含 RNA/DNA 4 成员）
- boundary manifest：`data/validation_pdbs/validation_manifest_v14_boundary.json`（1A65，单列，global 校准回退，不进判据）
- 校准：v13 自建（diag 18 蛋白 → global slope=1.285 intercept=6.825 n=216 + per_protein 18），**未借 v14 cal**
- 采样 n=50/臂，pH 7.4，5 臂 native/n2/p2/n8/p8，per-protein 校准（表内 per-protein、表外回退 global）
- 回折 ESMFold（esmfold env）+ TM-score（US-align）、H3 合法性、PROPKA H4、Tm（temBERTure）/Sol（protein-sol）
- 全程驱动：`code/tests/ligand_v9/run_v13_in10_chain.sh`（逐阶段产物检查、可 resume）
- 产物全新命名 `*_in10`，与 v14 `*_clean` 及旧 v13 隔离

**OOD 说明**：RNA/DNA 成员（5O60_E/21KL_A/9DWG_L/3MXB_A）与核苷酸配体成员（1AS2/2FEO/5CQH）对 v13
训练域（纯蛋白-有机小分子配体）均属 **out-of-domain**；其偏弱表现应作为「RNA/DNA 数据扩充收益」论据。

## 结果总览
| 判据 | v13-in10 | v14-clean | 说明 |
|---|---|---|---|
| H2 电荷控制（逐臂 dev≤2，50 臂） | **32/50 (64%)** | 45/50 (90%) | v13 明显偏弱 |
| H1 折叠 TM≥0.7（50 臂） | **50/50 (100%)** | 50/50 (100%) | 监督未伤折叠 |
| H3 带电残基物理合法性 | **48/50 (96%)** | 50/50 (100%) | 仅 5O60_E/3MXB_A 的 n8 臂失败（均 RNA/DNA）|
| Tm/Sol S2 恶化臂（vs 各自 uncond） | **11/50** | 0/50 | v13 部分蛋白条件序列显著失稳 |
| PROPKA H4（物理电荷 vs target） | 1BJ4/21KL_A 偏负 | 更贴近 target | 见 §4 |

## 1) H2 电荷控制逐蛋白（每蛋白 5 臂中命中数 + native 臂 dev）
| 蛋白 | cat | L | OOD | v13 H2 | v14 H2 | v13 dev(nat) | v14 dev(nat) |
|---|---|---|---|---|---|---|---|
| 6D2O | small_mol | 209 | - | 5/5 | 5/5 | 0.83 | 1.09 |
| 1CGE | metal | 162 | - | 5/5 | 5/5 | 0.15 | 0.75 |
| 2FEO | nucleotide | 221 | nuc-lig | **5/5** | 0/5 | 1.45 | 3.45 |
| 5CQH | nucleotide | 183 | nuc-lig | 5/5 | 5/5 | 0.03 | 1.67 |
| 1AS2 | nucleotide | 312 | nuc-lig | 1/5 | 5/5 | 2.68 | 0.62 |
| 1BJ4 | long | 470 | - | **0/5** | 5/5 | 2.56 | 0.21 |
| 21KL_A | RNA | 237 | RNA/DNA | 1/5 | 5/5 | 3.53 | 1.37 |
| 5O60_E | RNA | 209 | RNA/DNA | 4/5 | 5/5 | 1.08 | 0.97 |
| 3MXB_A | DNA | 153 | RNA/DNA | 5/5 | 5/5 | 0.10 | 0.63 |
| 9DWG_L | DNA | 323 | RNA/DNA | 1/5 | 5/5 | 2.66 | 0.03 |
| **合计** | | | | **32/50 (64%)** | **45/50 (90%)** | | |

观察：
- v13 短板高度集中：**1BJ4（长蛋白 470）、1AS2（GDP 核苷酸配体 312）、21KL_A、9DWG_L** → 均只命中 0-1/5。
- **互补性**：v13 恰在 v14 唯一短板 2FEO 上 5/5（v14 0/5，v14 native dev 3.45），说明两模型对中等/短链的控制差异不是单调优劣。
- RNA/DNA 4 成员中 **21KL_A、9DWG_L 明显偏弱**（1/5），5O60_E(4/5)、3MXB_A(5/5) 尚可 → OOD 惩罚不均匀，与序列长度/核酸配体亲和面相关。

## 2) 组成（native 臂生成 D/E+K/R 总数 ÷ native，<1 即删减；删减比例 ≈ 1−ratio）
| 蛋白 | v13 ratio | v13 删减 | v14 ratio | v14 删减 |
|---|---|---|---|---|
| 6D2O | 0.71 | 0.29 | 0.56 | 0.44 |
| 1AS2 | 0.70 | 0.30 | 0.46 | 0.54 |
| 2FEO | 0.56 | 0.44 | 0.46 | 0.54 |
| 5CQH | 0.57 | 0.43 | 0.43 | 0.57 |
| 1CGE | 0.69 | 0.31 | 0.60 | 0.40 |
| 1BJ4 | 0.61 | 0.39 | 0.46 | 0.54 |
| 21KL_A | **0.96** | 0.04 | 0.61 | 0.39 |
| 5O60_E | **0.93** | 0.07 | 0.56 | 0.44 |
| 3MXB_A | **0.99** | 0.01 | 0.69 | 0.31 |
| 9DWG_L | 0.50 | 0.50 | 0.47 | 0.53 |

观察：v13 在 RNA/DNA 蛋白上几乎**不删减**带电残基（21KL_A/5O60_E/3MXB_A ratio≈0.93-0.99，明显优于 v14 的 0.56-0.69），
但在 9DWG_L 上删减最重（0.50）。v14 总体删减更重（0.43-0.69）但电荷控制更准——删减与电荷命中在此数据上呈此消彼长。

## 3) H3（带电残基物理合法性）与 H1 折叠（native 臂）
| 蛋白 | v13 H3 | v14 H3 | v13 tm_nat | v14 tm_nat | v13 plddt_nat | v14 plddt_nat |
|---|---|---|---|---|---|---|
| 6D2O | 5/5 | 5/5 | 0.949 | 0.951 | 83.6 | 83.7 |
| 1AS2 | 5/5 | 5/5 | 0.940 | 0.933 | 81.7 | 81.2 |
| 2FEO | 5/5 | 5/5 | 0.920 | 0.904 | 77.9 | 76.6 |
| 5CQH | 5/5 | 5/5 | 0.967 | 0.964 | 84.8 | 84.1 |
| 1CGE | 5/5 | 5/5 | 0.969 | 0.969 | 86.4 | 86.3 |
| 1BJ4 | 5/5 | 5/5 | 0.925 | 0.923 | 77.9 | 77.8 |
| 21KL_A | 5/5 | 5/5 | 0.858 | 0.830 | 74.9 | 76.7 |
| 5O60_E | **4/5**(n8) | 5/5 | 0.908 | 0.927 | 73.1 | 72.8 |
| 3MXB_A | **4/5**(n8) | 5/5 | 0.964 | 0.965 | 85.3 | 84.7 |
| 9DWG_L | 5/5 | 5/5 | 0.923 | 0.916 | 86.2 | 83.9 |

- H1：两版 **50/50 TM≥0.7**，native 臂 tm_med v13 0.858-0.969 ≈ v14 0.830-0.969；pLDDT 相当。**v13 电荷微调未伤折叠**。
- H3：v13 仅 RNA/DNA 两蛋白（5O60_E、3MXB_A）的 **n8（更负）臂**出现核心/成簇带电残基违规；其余全过。

## 4) PROPKA H4（PROPKA 物理净电荷 q_phys vs 设计 target；dev=mean_q_phys−target）
| 蛋白/臂 | v13 dev | v14 dev |
|---|---|---|
| 1BJ4_native (t=0.42) | −1.92 | +0.54 |
| 1BJ4_n8 (t=−8) | −2.24 | +1.65 |
| 21KL_A_native (t=10.0) | −2.99 | −1.13 |
| 21KL_A_n8 (t=2.0) | **−5.00** | −1.13 |
| 3MXB_A_native (t=7.9) | +0.33 | +0.98 |
| 3MXB_A_n8 (t=0.0) | +0.89 | +1.68 |

观察：v13 在 1BJ4/21KL_A 上 **PROPKA 实测比 target 更负 2-5 个单位**（尤其 21KL_A n8 dev −5），
v14 各臂都在 ±1.7 内。3MXB_A 两版都接近 target。→ v13 在部分蛋白上「设计电荷」与「物理可实现电荷」偏离更大。

## 5) Tm/Sol S2（各条件臂 vs 各自无条件基线的 Tm/％sol 明显恶化：ΔTm<−5 或 Δ%sol<−10）
| | v13 | v14 |
|---|---|---|
| 恶化臂数 | **11/50** | 0/50 |
| 恶化集中蛋白/臂 | 1AS2 全 5 臂（ΔTm≈−6~−7）；5CQH native/n2/p2/n8（ΔTm≈−7~−11）；2FEO n2/n8（ΔTm≈−5~−6） | 无 |

观察：v13 的电荷条件在一些核苷酸配体蛋白（1AS2/5CQH/2FEO）上使生成序列相对自身无条件基线显著失稳；
v14 则无此现象。提示 v13 电荷监督在 OOD 配体蛋白上以热稳为代价。

## 6) boundary 1A65（单列，global 校准回退，不进判据）
| 臂 | target | v13 mean (dev) | v14 mean (dev) |
|---|---|---|---|
| native | −27 | −18.0 (8.98) | −24.4 (2.60) |
| n2 | −29 | −19.3 (9.72) | −25.8 (3.21) |
| p2 | −25 | −16.5 (8.46) | −22.7 (2.34) |
| n8 | −35 | −24.8 (10.21) | −32.0 (3.03) |
| p8 | −19 | −12.7 (6.35) | −16.9 (2.14) |

v13 在长蛋白极端负电区（nativeQ=−26.9）**严重欠冲**（最多到约 −25，达不到 −30 以下），v14 明显更接近 target。
与 H2 上 1BJ4（长蛋白）欠冲一致 → v13 对「长序列 + 极端负电」的控制是核心短板。

## 结论
1. v13 在 v14-clean 同协议下的**权威对照数字**：H2 **32/50 (64%)**、H1 TM **50/50**、H3 **48/50**、S2 **11/50**、boundary 1A65 严重欠冲。
2. v13 相对 v14-clean 的差距主要在**电荷控制 H2**（尤其 1BJ4/1AS2/21KL_A/9DWG_L），且**以热稳为代价**（S2 11/50 vs 0/50）。
3. **RNA/DNA 数据扩充收益论据成立**：v13 无 RNA/DNA 训练，21KL_A/9DWG_L H2 仅 1/5、5O60_E/3MXB_A n8 臂 H3 违规；
   同时核苷酸配体蛋白（1AS2/5CQH/2FEO）在 v13 上出现 S2 热稳恶化，v14（RNA/DNA 扩充后）全部消除。
4. **折叠稳健性未被破坏**：即使电荷控制差，v13 生成序列 TM≥0.7 仍 50/50、pLDDT 与 v14 相当。
5. 旧 v13 结果（`output/generalization_ligand_v13/` 及 `output/v13_ligand_*.json`）**保留未动、仅归档**；本文数字以 `*_in10` 为准。

## 产物清单
- 驱动脚本：`code/tests/ligand_v9/run_v13_in10_chain.sh`
- diag：`output/v13_ligand_diag_response_in10.json`
- 校准：`output/charge_calibration_v13_ligand_in10.json`（global slope 1.285, per_protein 18）
- 采样/折叠：`output/generalization_ligand_v13_in10/`（10 蛋白 × 5 臂 n50 + 1A65）
- 组成：`output/v13_ligand_comp_in10.json`
- H2/H1 统计：`output/v13_ligand_gen_stats_in10.json`
- H3：`output/h3_ligand_v13_in10.json`
- PROPKA：`output/propka_v13_ligand_in10/`（6 json）
- Tm/Sol：`output/tm_sol_ligand_v13_in10/`（seqs/uncond/ref_native/tm_sol_summary.json）
- DONE：`log/v13_ligand_in10_chain.DONE`；日志 `log/v13_*_in10.log`
- 过程：`session/2026-09-04_v13_in10_chain.md`
