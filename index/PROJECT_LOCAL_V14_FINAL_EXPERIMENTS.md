# 收尾实验详细计划（2026-09-05 用户批准，先计划后执行）

> 目标：①受控减预算消融（蛋白 v12.2 配方 + 配体 v14 配方都做）→ 论文 Fig25 模块贡献；②E. coli
> 核糖体 RNA 结合蛋白可设计性测试（native + 温和 ±2，不做极端）→ 论文 Fig22/23 应用验证。
> 关联：图目录 `figure/plan_01.md`；背景结论 `analysis/report/2026-09-05_{protein_history_vs_ligand_deletion,ligand_history_v13_v14}.md`。

---

## 计划 1｜受控减预算消融（蛋白 + 配体）

### 1.1 目的与原则
- 目的：对**最终配方**（蛋白 v12.2 / 配体 v14）给出各监督模块的单点 off 贡献（相对排序），不是绝对 SOTA。
- 铁律：同族内**同一数据子集、同轮次、同 seed、同 eval 集**；只允许"目标模块 off vs 全开"一处差异。
- 预算：数据 25%（分层抽，配体保 RNA/DNA≈7.7% 比例）；轮次 ≈ 正式 1/3（蛋白 v12.2 30→**10**，配体 v14 50→**16**）。判定可行（消融=相对），但 absolute 值不与正式版比；若 25%/1/3 差异被 under-train 压平，先小蛋白试点调预算（≤20%/≤1/4）。

### 1.2 模块开关矩阵（每族 6 runs）
| run | 蛋白 v12.2 配方 | 配体 v14 配方 |
|---|---|---|
| FULL | 全开（对照） | 全开（对照） |
| −v12 组成 | v12_supervision/frac_floor/gravy OFF | 同 |
| −λ_target | surface_charge_target OFF | 同 |
| −A1 | —（蛋白无配体 A1 自然不启，跳过） | pocket_count(global) OFF |
| −ph_filter | pH-adaptive 结构惩罚 OFF | 同 |
| −seq_keep | sequence_keep(loss) OFF | 同 |

蛋白族最终 = **5 runs**（FULL、−v12组成、−λ_target、−ph_filter、−seq_keep）；配体族 = **6 runs**（加 −A1）。蛋白族的 A1 无意义列删去。

### 1.3 数据与训练
- 蛋白：从 v12.3 蛋白训练域（6580，含长蛋白）**分层抽 25%**（保 L/电荷分布）→ `labels_ablate_prot.npz`。
- 配体：从 `labels_v14_final.npz`(5371) 分层抽 25%（**保 RNA/DNA 7.7%**）→ `labels_ablate_lig.npz`。
- 基座/超参：完全照该族正式版命令（蛋白 MoMPNN + v12.2 全套 flag；配体 LigandMPNN + v14 flag、atom25）；只改 labels 子集 + epochs + out_dir。
- 每次训练落 `output/ablate_{prot,lig}/run_{tag}/` + log；共享同一 backbone 权重（不重训 backbone，只训 ConditionEncoder——与正式一致）。

### 1.4 评估（轻量、与全量可比口径）
- 每个 run 用**前向 val-loss 回放**（`val_loss_curve.py` 同法）在对应族验证集（蛋白 1199 / 配体 805）上给最终 epoch 的 ce/cd/total。
- 另在**小蛋白 + 1 长/1 RNA 代表**上做一发生成式抽查（native + ±2，n30）量 H2/删减（native 臂保留率），补模块对"真实设计"的影响。
- 输出 `analysis/report/2026-09-05_ablation_{prot,lig}.md` + `figure` 数据（柱状：FULL vs 各 off 的 cd/组成/H2 变化）。

### 1.5 GPU/顺序
- GPU6 为主（现空）；每次单跑；蛋白族 → 配体族顺序。GPU4（共享）可并行第二条减半总时。预计总 ~1-2 天（后台）。

---

## 计划 2｜E. coli 核糖体 RNA 结合蛋白可设计性测试（应用验证）

### 2.1 目标与范围
- 目标：E. coli 核糖体上拆出的核糖体蛋白，验证在 **native 与温和(native±2)** 下 v14 的可设计性（H2/dev + native 臂删减 + 折叠可选），**不做极端**。
- 理由：核糖体蛋白天然正电、RNA 结合界面上带电残基密集 → 最考验"电荷条件化 vs 删减"的应用类。

### 2.2 源结构选定（**待用户确认两项**）
- 待确认①：用户此前举例的**两个核糖体 PDB**具体代码（本地 rna_complex_raw 有 5O60/4YBB/4V4T/9RVC .cif）。已知 **4YBB=E. coli 70S**（title 证实），**但它已在 RNA/DNA 训练源内**（4V4T/9RVC/4YBB→414 训练域）→ 直接拆会与训练重复，需先做序列去重；**5O60 已作为 held-out（蛋白 E）在测试集**。
- 待确认②：若 4YBB 与训练重复不可用，是否允许下载**新的、高分辨率 E. coli 核糖体**（用户要求结构清晰，不用低分辨率；候选如 E. coli 70S 高分辨 7K00(~2.0Å)/6QI3 等，待我核实分辨率与 coverage 后列候选）。
- 备选：若无全新高分辨可用，则**只用本地中与训练序列不重复的核糖体蛋白链** + 5O60_E（held-out）补足。

### 2.3 拆分与建集（同训练集策略）
- 拆分策略=此前 RNA/DNA 训练集构建口径：从核糖体复合物拆单链蛋白 → 保留**结合 RNA 的蛋白链**（含 rRNA 接触残基/配体原子上下文，仿 15Å rRNA 处理）→ **序列精确去重** vs 训练 5371 + in-10 + 已建测试 → coverage 判定（相对 `labels_v14_final`）→ 每域 8 个 (pH, native@pH) 臂标签 → npz。
- 建 ~15-30 个核糖体蛋白（视去重后可得量）。

### 2.4 采样与指标
- v14 编码器 + per-protein 校准（或小样本现场标定）；native + n2/p2（温和）n≈50/臂。
- 指标：H2(dev≤2)、native 臂带电保留率（删减）、H3 聚集（可选）、H1 ESMFold（可选/抽样几条）。
- 输出 `output/ribosome_e_coli/` + `analysis/report/2026-09-05_ribosome_ecoli_design.md`；核糖体蛋白天然正电 → 重点看"负向温和(−2)是否触发删减"，与 Task1/2 结论衔接。

---

## 3 执行顺序与登记
1. 本计划先存 `index/`（本文件）+ `figure/plan_01.md` 已含 Fig22-23/25 占位。
2. 消融（计划1）**用户已批蛋白+配体都跑** → 保存本计划后即可启动（先蛋白族，配体族随后）。
3. 核糖体集（计划2）→ **先回应用户两个待确认项**，确认源结构后按 2.3-2.4 执行。
4. 每块完成：报告 + 归档 + 记忆更新。
