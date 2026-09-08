# 论文图目录 / 计划（主文档，2026-09-05 更新）

> 规则：本文件是**全部要生成图的唯一目录**；新增想法一律追加到本文件并按编号登记；`figure/`
> 目录保存所有实际生成的图（脚本+成品）。每张图给出：**编号｜标题｜内容要点｜数据源｜状态**。
> 术语（电荷臂 native/n2/p2/n8/p8、H2/H3/H1、删减保留率等）见
> `analysis/report/2026-09-04_paper_subconclusions.md` 术语表。

---

## A. 数据与分布（Fig 1–3）
- **Fig 1｜训练/验证/测试集酸碱性分布**：训练 vs 验证(805) vs 测试(in-10) 的 native 电荷@pH7.4 分布（按类型分层），说明验证/测试严格按训练比例、无偏差。
  - 数据：`data/cath/*.npz`、`data/ligand_train/labels_v14_final.npz`、`labels_v14_valset_805.npz`、in-10 manifest。
- **Fig 2｜序列长度分布**：同上三集 L 分布（含 RNA/DNA、长/深负尾）。
- **Fig 3｜配体类型构成**：训练 5371 里 small_mol/metal/nucleotide/RNA-DNA 占比 vs 805 vs in-10（堆积条）。数据同上。

## B. 方法总览 / 版本更迭（Fig 4–6）
- **Fig 4｜蛋白模式版本更迭表/曲线**：v7→v9→v10/11→v12/v12.1→v12.2→v12.3 的机制/数据/指标变化（Task A 输出 `analysis/report/2026-09-05_protein_history_vs_ligand_deletion.md`）。
- **Fig 5｜配体模式版本更迭**：v9→v12.2-ligand→v13→v14 的改动与指标（Task B 输出 `2026-09-05_ligand_history_v13_v14.md`）。
- **Fig 6｜删减根因机制示意**：蛋白 vs 配体"为什么 v12.2 轻删而 v14 重删"的概念图（含配体疏水先验/深口袋盲区/删减换电荷命中的代价互换）。

## C. 与 baseline 对照（Fig 7–9）
- **Fig 7｜MoMPNN / LigandMPNN / ConfuMPNN 原生生成对照**：同结构，三者在无电荷条件(native)下的 recovery；配体模式对标 LigandMPNN。
- **Fig 8｜电荷条件 vs 无条件的生成差异**：加条件 vs 不加（占位=训练均值）的序列差异/电荷差。
- **Fig 9｜Tm/Sol 稳定性对照**：设计 vs native vs 无条件基线（S2 判据），证明电荷工程不伤热稳/溶解。
  - 数据：`output/tm_sol_ligand_v14_clean/tm_sol_summary.json`、v13 in-10、v12.2 蛋白系列。

## D. 电荷控制主结果（Fig 10–14）
- **Fig 10｜校准表 + 校准前后**：global/per-protein 校准表（如 global slope 1.492 样式）+ 校准前后响应曲线。
  - 数据：`output/charge_calibration_v14_ligand_clean.json`、diag response `v14_ligand_diag_response_clean.json`。
- **Fig 11｜H2 电荷命中（逐蛋白 × 5 臂）**：in-10 每臂 dev、命中/未命中热图或点图。
  - 数据：`output/v14_ligand_gen_stats_clean.json`。
- **Fig 12｜H2 两口径对比**：big-global vs per-protein vs 小样本现场标定（三口径柱）。
- **Fig 13｜per-epoch train-vs-val 曲线**：三版（v12.2/v12.3/v14）每轮 train/val 的 total loss + 平均电荷偏差（证明拟合+无过拟合）。
  - 数据：`output/val_loss_curve_trainval_plot.json`（+ `val_loss_curve_{v12_2,v12_3,v14_ligand}.json`）。
- **Fig 14｜recovery / 折叠保持**：native 回收率 vs 电荷臂；ESMFold TM 分布（H1）。

## E. 删减（局限）表征（Fig 15–18，本工作特色）
- **Fig 15｜删减定位（三区）**：in-10 逐蛋白 pocket/surface/core 带电保留率（条/箱），展示删减遍布三区。
  - 数据：`output/v14_deletion_location.json`（Task1 报告）。
- **Fig 16｜分电荷档删减**：native / ±2 / n8 / p8 的 DEKR 保留率 + D/E vs K/R 拆分（温和照删、极端不对称）。
  - 数据：`output/v14_deletion_arm.json`（Task1 补充）。
- **Fig 17｜fix 结合残基对照**：fix vs unfix 的 pocket/surface/core 保留率 + H2 掉落（图：左边组成、右边电荷）。
  - 数据：`output/v14_fixbinding_summary.json`（Task2）。
- **Fig 18｜大样本"三达标"存在率**：随 n 的达标条数曲线 + Pareto（dev vs 删减）；标出 40% 系统性零臂。
  - 数据：`output/largen_v14_summary.json`、`largen_v14/<pdb>_arm_*/stats.json`（Task3）。

## F. v13 vs v14 同协议对照（Fig 19–21）
- **Fig 19｜v13-in10 vs v14-clean H2 逐蛋白**：成对点图（x=v13,y=v14，对角线上 v14 好；2FEO 反超点标出）。
- **Fig 20｜组成删减 v13 vs v14**：每蛋白保留率配对（v13 每蛋白更轻）。
- **Fig 21｜RNA/DNA OOD 消除**：21KL_A/5O60_E/3MXB_A/9DWG_L 在 v13(vs v14) 的 H2/删减/H3 变化（数据扩充收益）。
  - 数据：Task B 报告 + `output/generalization_ligand_v13_in10/`、`v14_clean/`。

## G. 应用层验证（Fig 22–24，待做）
- **Fig 22｜RNA 结合核蛋白可设计性（native & 温和 ±2）**：目标蛋白集逐蛋白 H2/dev/删减（见 §计划 RNA 核蛋白）。
- **Fig 23｜核糖体蛋白 pI 微调验证**：先算每蛋白 pI → native 采样 10 条 → native±1 采样 10 → native±3 采样 10（环境微调 → 泛化）。
- **Fig 24｜实验/湿实验对照（数据待上传）**：真实表达验证。

## H. 消融 / 对比实验（Fig 25–27，规划见下）
- **Fig 25｜关键模块消融**：λ_target 表面锚 / A1-global vs keep / ph_aware_filter / 课程学习 vs decouple 的开关对照（可复用 v12.2/v12.3/v13/v14 + 少量重训）。
- **Fig 26｜校准 + fix + 大样本三种"补救"对比**：分别/组合对 H2 与组成的收益。
- **Fig 27｜ESMFold 与 AF3 取样对比**（若算力允许）。

---

## 还需做的消融/对比实验（论文缺口，待批准执行）
1. **蛋白模式模块消融矩阵**（复用已训版 + 少量小重训）：v12.2 的 λ_target 表面电荷锚是否必要（对照无锚版，看表面组成是否回到删减）；λ_keep 序列保持是否必要。
2. **配体 A1 keep vs global 的直接量化**：v13(A1 keep+pocket) vs v14(A1 global) 已在不同轮次/数据下，若要干净归因需同数据同轮次各训一版（受控小实验，GPU ~半日/版）——决定"global 是否真比 keep 好"。
3. **删减"根治方向"的对照**（仅作 mechanism 佐证，不 claim 已解决）：在 v14 上加"带电总数保真/反删减"监督训一版，看能否降删减而不伤 H2（若做，作为 limitation 的边界探针）。
4. **RNA 结合核蛋白 native/温和可设计性**（Fig 22-23，见下节）。

## RNA 结合核蛋白可设计性测试（Fig 22/23）— 待确认方案
- **目标**：测 RNA 结合（含核糖体/核内）蛋白在 **native 与温和(native±2)** 条件下的可设计性（H2/dev + 删减 + 折叠）；暂不做极端。
- **待定**：目标集规模与构成（多少条、是否需外部 RCSB 下载、是否 coverage-in 去重 vs 训练 414 RNA/DNA）；协议（n、pH7.4、per-protein 校准）。
- 确认后再采样（GPU6 现空）。

---

## I. 补充对照与机制实验（2026-09-06 追加，报告在 compare//ablation/，数据路径可作图）
| 图 | 内容 | 数据文件（作图源） | 状态 |
|---|---|---|---|
| Fig I-1 | **裸 backbone vs 条件生成增益**（三类均衡集 v2，酸/中/碱；v1 偏酸子集并排） | `output/exp_control_{prot,lig}_v2/_report_tables.md`；报告 `compare/report_2026-09-06_exp1_{prot,lig}_barebackbone_v2.md` | ✅ 待画 |
| Fig I-2 | **bias-only vs encoder** 逐序列命中/mean dev/回收 | `output/exp_control_{prot,lig}/_report_tables.md`；`ablation/report/2026-09-06_exp2_{prot,lig}_bias_vs_encoder.md` | ✅ 待画 |
| Fig I-3 | **组成分解**：加/删/同号删·膨胀/对侧置换 占比 | `output/exp2_comp_decomposition_tables.md`；`ablation/report/2026-09-06_exp2b_comp_decomposition.md` | ✅ 待画 |
| Fig I-4 | **pH 包络 T1原始/T2-7.4外推/T3逐pH标定** + 校准外推误差 | `figure/exp7b_{prot}_*_pHgrid.png`(已生成) + `output/exp_pH2_{prot,lig}/_report_tables.md` | 🟡 蛋白/配体各 3-4 张已出 |
| Fig I-5 | **命中率 95%CI**（Wilson/精确） | `output/hitrate_ci_summary.json` | ✅ 待画 |
| Fig I-6 | **Wilcoxon 配对**（A-B, B-C；v13-vs-v14） | `output/wilcoxon_exp15.json`；`compare/report_2026-09-06_exp5_wilcoxon.md` | ✅ 待画 |

## J. 文章板块清单（manuscript 映射，2026-09-06）— 论文框架唯一正本见 `paper/OUTLINE.md`（本表为其轻量映射）
> 把补充实验分配到论文板块；每个板块给"结论 + 主报告 + 图"。

1. **Intro/Method 背景**：pH-感知电荷条件化（LigandMPNN/MoMPNN 首例）、删减捷径现象。
2. **Results·基线对照**：Fig I-1/7 —— 条件化增益集中在极端电荷臂、近 native 靠裸分布覆盖（exp1 v2 三类均衡 + v1）。
3. **Results·机制（关键双刃）**：Fig I-2/I-3 —— 逐序列命中 bias>encoder，但 encoder=同号对称删减 vs bias=同号对称膨胀、对侧置换极少 → 引出"需组成保真/置换监督"。
4. **Results·pH 敏感性**：Fig I-4/7 —— pH 非边界、逐 pH 现场标定后 87-88% 可达；7.4 外推才是假边界；极端 pH 抽查 H1/H3/H4 不破坏折叠。
5. **Results·校准与统计**：Fig I-5/6 + 校准三口径 —— CI + Wilcoxon 配对。
6. **Results·应用验证**：7K00 核糖体 RNA 蛋白 native/±2 可设计性（对照 in-10 90%）；RNA 结合蛋白图。
7. **Discussion**：global bias 局限（2025 论点 + 删减/膨胀）+ encoder 差异化价值（learned 全局 pH 条件）→ 下一步"encoder+bias+置换监督"。
8. **Limitation**：组成删减未愈、逐序列弥散、bias 膨胀物理代价未测（需 Tm/Sol/H1 补验）、小样本校准依赖。
