# 配体模式版本更迭史 + v13 vs v14 差异与改进面对照（Task B, 2026-09-05）

> 任务：梳理配体（LigandMPNN backbone）模式版本史 v9 → v12.2-ligand → v13 → v14，
> 重点量化 v13 与 v14 的数据/监督差异与改进面/未改进面。
> 纯文档/数据分析，未做任何采样。所有数字均来自既有报告/日志/产物（见文末产物路径）。
> 关键口径警示见 §0 —— v13 的「旧测试集数字」与 v14 的「in-10 数字」**不是同一测试集**。

---

## 0. 口径警示（先读，否则会误读对比）

配体模式测试集在两版之间**换了成员**，任何 v13↔v14 对比必须先分清口径：

| 口径 | 测试集成员 | 采样 | 校准 | 使用处 |
|---|---|---|---|---|
| **旧测试集**（v12.2-ligand/v13 原版验证） | 10 蛋白：1C6O/1AZM/1AS2/**1AXW**/2FEO/5CQH/1CGE/**1AG0**/1A65/1BJ4（含 3 个同源二聚体 1C6O/1AXW/1AG0、1AZM 后证实训练集泄漏） | v12.2-ligand n30 / v13 n50 | per-protein（各自建表） | `2026-09-01_v12_2_ligand_validation.md`、`2026-09-02_v13_ligand_validation.md` |
| **in-10**（v14 权威测试集） | 10 蛋白：6D2O/1AS2/2FEO/5CQH/1CGE/1BJ4/21KL_A/**5O60_E**/3MXB_A/9DWG_L（coverage=in、11 验证蛋白全 leak=False；删除 1C6O/1AXW/1AG0 二聚体、1AZM 泄漏、2E9R_X 移出） | n50 | per-protein（各自建表，表外回退 global） | `2026-09-04_v14_clean_validation.md` |
| **boundary** | 1A65（L504, native ≈ −26.9，深负长蛋白） | n50 | global 校准回退 | 同上 + `2026-09-03_long_neg_charge_limitation.md` |

- **v13-in10** = v13 模型在 **in-10 测试集 + v14-clean 同协议**上完整重跑的结果（`2026-09-04_v13_in10_validation.md`）。**v13↔v14 唯一严格同集同协议的对比 = v13-in10 vs v14-clean**。
- 因此：旧报告里「v13 H2 70%、S2 17/50、组成 0.55-0.69×」是**旧测试集**数字；v13 在 in-10 上重跑是 **64% / S2 11/50 / 组成 0.50-0.99×**。
- 「v13 70% vs v14 90%」**不是同集对比**（70% 在旧 10-蛋白集，90% 在 in-10）。真正的同协议提升是 **64% → 90%**。

---

## 1. 配体模式版本时间线

| 版本 | 训练起止 | 数据（域×8pH） | 监督/损失改动（相对上一版） | epoch（耗时） | 产物 | 该版关键结论 |
|---|---|---|---|---|---|---|
| **v9-ligand** | 2026-08-18（23:53 完成） | `data/ligand_train/labels.npz` **4957** 域（≈4972 PDB 文件，15 个不可解析/未入标签；构成 ≈ small_mol 4145 / metal 564 / 核苷酸辅因子 248） | v7 基线方法首次搬上 LigandMPNN backbone：`λ_c 0.5/λ_kl 0.05/λ_keep 0.5/charge_temp 0.5/perturb_prob 0.3/placeholder_prob 0.15`，**无 v12 组成监督、无 decouple、无 pocket** | 30（**111.5 min**） | `output/finetune_ligand_v9/finetune_epoch030.pt` | 修复「配体模式电荷失效」：1MBN dev 14.05→1.55；3 复合物 dev≤1.55。⚠️ 本版起即存在 **number_of_ligand_atoms=16 vs 权重 atom_context_num=25 不匹配**（v14 才修） |
| **v12.2-ligand**（迁移） | 2026-08-31 15:56 → 09-01 08:00 | 同 `labels.npz` 4957 | **迁移 v12.2 蛋白法**：`--decouple_absolute(-35,20)` + `--v12_supervision(frac_floor 0.5/gravy_margin 0.4/λ_v12 0.2)` + `--lambda_target 0.2`(surface 电荷锚) + `sasa_threshold 0.25` + `ph_aware_filter` + `structure_boost 1.5` | 30（**992.6 min ≈ 16.5 h**） | `output/finetune_ligand_v12_2/finetune_epoch030.pt` | H2 72%（旧集 n30）。**发现删减捷径**：8/10 蛋白带电残基总数删到 0.53-0.65×，空间定向配体口袋；根因 = frac_sasa 盲区监督逃逸 × 配体疏水先验 × v12 微调放大。1AS2 新短板 |
| **v13**（A1+A2 口袋保护） | 2026-09-01 19:41 → 09-02 04:40 | 同 `labels.npz` 4957 | 在 v12.2-ligand 上加 **A1 `pocket_count_loss`（keep 模式：仅护 pocket Cα-配体<8Å，floor 0.7/ceil 1.3/λ_pocket 0.2）+ A2**（`surface_charge_target_loss` extra_mask=surface∪pocket；三块互斥分区 core/pocket/surface，消双算）。v12 全套参数不动 | 30（**572 min ≈ 9.5 h**，末 3ep charge 仍缓降 → 未完全收敛） | `output/finetune_ligand_v13/finetune_epoch030.pt` | 旧集 H2 70%、H1 单链 35/35、校准 slope 1.00；但 **组成 8/10 仍 0.55-0.69×（A1 只护 pocket）**、**Tm/Sol S2 17/50 反恶化**（v12.2-ligand 9/50）。未达标 |
| **v14**（RNA/DNA 扩充 + A1 全局化） | 2026-09-02 19:23 → 09-03 08:57 | `labels_v14_final.npz` **5371** = 旧 4957 + **RNA/DNA 414（7.7%）**（DNA 155 含 hybrid 3 / 非核糖体 RNA 108 / 核糖体 RNA 148） | 三项改动：① **number_of_ligand_atoms 16→25 全脚本修正**（对齐权重 atom_context_num=25）；② **A1 全局化**（`--pocket_mode global`：计数锚 = surface∪pocket，绕开 frac_sasa 盲区，floor 0.8/ceil 1.3/λ_pocket 0.3 + normalize/min_abs_cap）；③ **epochs 30→50**（v13 30ep 未收敛 + RNA 新类型 + 25 原子首用）。**A2 保留；frac_floor/gravy_margin/λ_v12/λ_target 全未变** | 50（**832.8 min ≈ 13.9 h**，末轮 total 4.389 / charge 3.083） | `output/finetune_ligand_v14_rna/finetune_epoch050.pt` | in-10：H2 90%、H1/H3 50/50、S2 0/50；**唯一硬伤 = 组成删减 0.43-0.69× 全部 10 蛋白（比 v13 更深）**，见 §4 |

> v9-ligand 与 v12.2-ligand/v13 训练数据同为 `labels.npz`（4957 域）。v14 是唯一扩了数据的版本（+414 RNA/DNA）。
> 训练日志佐证：`log/v9_train.log`(111.5min)、`log/v12_2_ligand_train.log`(992.6min)、`log/v13_ligand_train.log`(572min)、`log/v14_ligand_train.log`(832.8min)。

### 1.1 v14 数据扩充细节（RNA/DNA 414 = 旧 4957 + 414）

- 主源核糖体 70S 拆链（4V4T 46 / 9RVC 44 / 4YBB 58 → 核糖体 RNA **148**），每条核糖体蛋白单独 PDB + 15Å rRNA 配体。
- 非核糖体 RNA **108**：RNase/tRNA 合成酶与修饰酶/衣壳蛋白/Pumilio/Dicer/CRISPR 等。
- DNA **155**（含 hybrid 3）：限制酶/重组酶/转座酶/DNA 修复酶/Y 家族聚合酶/解旋酶/甲基转移酶/转录因子/Argonaute 等。
- 序列精确去重（跨结构同源 + 与旧训练集去重）。最终 11 个验证蛋白全 leak=False。
- L 50-497（>300 有 26+），charge@7.4 碱性富集（核糖体蛋白天然正电，mean +8.7）。
- 中间版本：`labels_rna_v14.npz`(191) → `labels_rna_v14_sup.npz`(209) → `labels_rna_v14_sup2.npz`(414)；训练实际用最终 `labels_v14_final.npz`(5371)。首个 merged 版 `labels_v14_merged.npz`(5148) 已作废。

### 1.2 三次无效/被叫停的 v14 启动（均为过程，非最终产物）

1. `labels_v14_merged.npz`(5148) epoch1/2 → 用户暂停重构（1AZM 泄漏 → 换 6D2O；验证集重划）。
2. `labels_v14_final.npz`(5166，RNA/DNA 209) → 用户再暂停：数据池充足应补齐核酸短板（DNA 35→≥100）再训。
3. 最终 `labels_v14_final.npz`(5371，RNA/DNA 414) 50ep → **本轮产物**。

---

## 2. v13 vs v14 差异表（重点）

| 维度 | **v13**（2026-09-01/02） | **v14**（2026-09-02/03） | 是否变化 |
|---|---|---|---|
| **训练数据** | `labels.npz` 4957 域（全部旧配体：small_mol/metal/核苷酸辅因子） | `labels_v14_final.npz` 5371 域 = 旧 4957 + **RNA/DNA 414**（7.7%：DNA 155 / 非核糖 RNA 108 / 核糖 RNA 148） | ✅ 数据扩充 |
| **sup 集并入方式** | —（无新数据） | 新 RNA/DNA 拆链域整体重跑 `build_rna_v14_labels.py` → `labels_rna_v14_sup2.npz`(414) → **不 append**、与旧 `labels.npz` 合并（避免标签段错位） | ✅ |
| **A1 pocket_count_loss** | **keep 模式**：计数区 = pocket only（Cα-配体<8Å）；floor 0.7 / ceil 1.3 / λ_pocket 0.2；**未 normalize** | **global 模式**：计数区 = **charge_surf_mask = surface∪pocket**（绕开 frac_sasa 盲区）；floor **0.8** / ceil 1.3 / λ_pocket **0.3**；**normalize=True**（分数化、按 native 计数）+ min_abs_cap（N=0 死锁保护） | ✅ 全局化 + 调参 |
| **A2 surface_charge_target_loss（电荷锚）** | extra_mask=surface∪pocket（λ_target 0.2） | 保留，global 下同（extra_mask=charge_surf_mask） | 不变 |
| **三块互斥分区 core/pocket/surface** | 开（keep/global 都算） | 开 | 不变 |
| **frac_floor / gravy_margin / λ_v12** | 0.5 / 0.4 / 0.2 | 0.5 / 0.4 / 0.2 | 不变 |
| **λ_target（surface 电荷锚权重）** | 0.2 | 0.2 | 不变 |
| **sasa_threshold / ph_aware_filter / structure_boost** | 0.25 / on / 1.5 | 0.25 / on / 1.5 | 不变 |
| **charge_temp / perturb_prob / placeholder_prob / λ_c / λ_kl / λ_keep** | 0.5 / 0.3 / 0.15 / 0.5 / 0.05 / 0.5 | 同左 | 不变 |
| **decouple** | `--decouple_absolute -35..20` | 同左 | 不变 |
| **number_of_ligand_atoms** | **16**（与权重 atom_context_num=25 不匹配的 bug，v9 起存在） | **25**（train_finetune/run_guided/validate_generalization/transfer_validation/smoke_guided 等全脚本修正） | ✅ bug 修复 |
| **epochs** | 30（末 3ep charge 3.60 仍缓降，未完全收敛） | **50**（末轮 charge 3.08，收敛更好） | ✅ 30→50 |
| **backbone 权重** | `ligandmpnn_v_32_010_25.pt` | 同左 | 不变 |
| **checkpoint 产物** | `output/finetune_ligand_v13/finetune_epoch030.pt` | `output/finetune_ligand_v14_rna/finetune_epoch050.pt` | — |
| **末轮训练 loss** | total 4.914 / ce 1.549 / charge 3.602 / keep 0.835 | total 4.389 / ce 1.593 / charge 3.083 / keep 0.862 | 训练更深 |

一句话：**v14 = v13 方法骨架 + 三项改动（数据 +414 RNA/DNA、atom 16→25、A1 keep→global 并把 floor/λ 调高）+ 训练拉长到 50ep；v12 组成/GRAVY/λ_target 等全部参数原样保留**。

---

## 3. 改进面（量化）

### 3.1 同集同协议（in-10, n50, per-protein 校准）：v13-in10 vs v14-clean

| 判据 | **v13-in10** | **v14-clean** | 变化 | 达标？ |
|---|---|---|---|---|
| **H2 电荷命中** | **32/50 (64%)** | **45/50 (90%)** | **+26pp** | v14 ✅（9/10 蛋白全臂，仅 2FEO 0/5 特例） |
| H1 折叠 TM≥0.7 | 50/50 | 50/50 | 持平 | ✅ / ✅ |
| H3 聚集合法性 | 48/50 | 50/50 | +2 臂 | ✅ |
| **Tm/Sol S2 恶化臂** | **11/50** | **0/50** | **−11 臂** | v14 ✅ |
| H4 PROPKA（1BJ4/21KL_A/3MXB_A dev） | 1BJ4 −1.9~−2.2、21KL_A −3.0~**−5.0**、3MXB_A +0.3~+0.9 | 全部在 ±1.7 内 | 更贴近 target | v14 更好 |
| boundary 1A65（global 回退） | native dev **8.98**（各臂 6.35-10.21，最深只到 ≈−25） | native dev **2.6**（各臂 2.1-3.2） | **大幅改善** | v14 ⚠️边界档 |

### 3.2 H2 逐蛋白（in-10 同协议，每蛋白 5 臂中命中数）

| 蛋白 | cat | L | OOD(v13) | v13 | v14 | 变化 |
|---|---|---|---|---|---|---|
| 6D2O | small_mol | 209 | — | 5/5 | 5/5 | = |
| 1CGE | metal | 162 | — | 5/5 | 5/5 | = |
| **2FEO** | nucleotide | 221 | nuc-lig | **5/5** | **0/5** | **↘ 唯一倒退** |
| 5CQH | nucleotide | 183 | nuc-lig | 5/5 | 5/5 | = |
| **1AS2** | nucleotide | 312 | nuc-lig | 1/5 | 5/5 | ↗ +4 |
| **1BJ4** | long | 470 | — | 0/5 | 5/5 | ↗ +5 |
| **21KL_A** | RNA | 237 | RNA/DNA | 1/5 | 5/5 | ↗ +4 |
| 5O60_E | RNA | 209 | RNA/DNA | 4/5 | 5/5 | ↗ +1 |
| 3MXB_A | DNA | 153 | RNA/DNA | 5/5 | 5/5 | = |
| **9DWG_L** | DNA | 323 | RNA/DNA | 1/5 | 5/5 | ↗ +4 |

- **v14 的提升集中在**：1BJ4（长蛋白，v13 全臂欠冲）与 RNA/DNA OOD 成员 21KL_A/9DWG_L（1→5），加 1AS2（核苷酸配体 1→5）。这正是 RNA/DNA 数据扩充 + 更长训练的直接收益区。
- **v13 反超的唯一蛋白 = 2FEO**（v13 5/5，v14 0/5；v14 native dev 3.45）。两模型对中短链/核苷酸的控制差异不是单调优劣（互补性）。
- 旧集参考：v13 旧 10-蛋白集 H2 70%（35/50）——**与 90% 不可直接比**（不同测试集成员）；同协议 v13-in10 为 64%。

### 3.3 RNA/DNA out-of-domain（数据扩充收益的核心论据）

| 成员 | v13（无 RNA/DNA 训练） | v14 | 
|---|---|---|
| 21KL_A（RNA, L237） | H2 1/5（native dev 3.53） | H2 5/5（native dev 1.37） |
| 9DWG_L（DNA, L323） | H2 1/5（native dev 2.66） | H2 5/5（native dev 0.03） |
| 5O60_E（RNA, L209） | H2 4/5 + **H3 n8 违规** | H2 5/5 + H3 5/5 |
| 3MXB_A（DNA, L153） | H2 5/5 + **H3 n8 违规** | H2 5/5 + H3 5/5 |

→ v13 由于训练集无 RNA/DNA 真核酸链，对 21KL_A/9DWG_L 电荷控制基本失灵、且在「更负」n8 臂出现 H3 聚集违规；v14（扩充后）全部消除。注意 v13 的 native dev（2.66-3.53）反映其「没学会对这些蛋白做电荷编辑」——这同时解释了 §4 中 v13 在 RNA/DNA 上反而删得少的反直觉现象。

### 3.4 Tm/Sol S2（同协议 in-10）

- v13-in10 S2 = 11/50：集中在核苷酸配体蛋白 **1AS2 全 5 臂（ΔTm≈−6~−7）、5CQH 4 臂（ΔTm≈−7~−11）、2FEO 2 臂**——v13 的电荷条件在 OOD 配体蛋白上以热稳为代价。
- v14-clean S2 = **0/50**：电荷工程化不再牺牲热稳/溶解。
- 旧集参考：v13 旧集 S2 17/50（v12.2-ligand 9/50）。任务描述中「17/50→0/50」是**旧集 v13 → in-10 v14** 的跨集参考；严格同协议是 **11/50→0/50**。

---

## 4. 未改进面：组成删减（且 v14 比 v13 更深）

### 4.1 量化（native 臂 gen/native 带电残基总数倍率，in-10 协议）

| 蛋白 | v13-in10 | v14-clean | Δ | | 蛋白 | v13-in10 | v14-clean | Δ |
|---|---|---|---|---|---|---|---|---|
| 21KL_A | **0.96** | 0.61 | −0.35 | | 1AS2 | **0.70** | 0.46 | −0.24 |
| 5O60_E | **0.93** | 0.56 | −0.37 | | 1BJ4 | **0.61** | 0.46 | −0.15 |
| 3MXB_A | **0.99** | 0.69 | −0.30 | | 6D2O | **0.71** | 0.56 | −0.15 |
| 9DWG_L | **0.50** | 0.47 | −0.03 | | 1CGE | 0.69 | 0.60 | −0.09 |
| 2FEO | 0.56 | 0.46 | −0.10 | | 5CQH | 0.57 | 0.43 | −0.14 |

- **v14 在 in-10 全部 10 蛋白上都删得更重**（0.43-0.69×），无一例外。
- **共享 5 旧单体**（1AS2/2FEO/5CQH/1CGE/1BJ4）：v13 0.56-0.70× → v14 0.43-0.60×（每蛋白更深；报告《2026-09-04_v14_deletion_location.md》逐区表：v13→v14 pocket/surface/core 各区 retention 普遍 −0.1~−0.25）。
- **RNA/DNA 成员差距最大**：v13 几乎不删（21KL_A/5O60_E/3MXB_A = 0.93-0.99×，因这些蛋白对 v13 是 OOD、模型根本没学会做电荷编辑）；v14 训练见过 RNA/DNA 后，把「删带电残基调电荷」这套习惯也带到了这些蛋白上（0.56-0.69×）。
- 对比目标：组成判据是 0.7-1.3×（A1 设计 floor 0.8），**v14 全 10 蛋白远低于该下限** → 判据 ❌。

### 4.2 删减定位（v14，native 臂 n50 生成均值，10 蛋白聚合）

`output/v14_deletion_location.json` + 报告：

| 区 | native CHG | gen CHG | retention |
|---|---|---|---|
| pocket（配体 8Å 内） | 144 | 69.6 | **0.48** |
| surface（frac_sasa≥0.25 非 pocket） | 389 | 213.8 | **0.55** |
| core（其余） | 83 | 32.3 | **0.39** |

- 删减**跨 pocket/surface/core 全域分布**（不是只删某区）；SASA 分位 Q1 最深箱 retention 0.35、Q2 0.40 → **越埋藏越重删**。
- 判据特征（616 个 native 带电位点，重删 583 / 轻删 16）：frac_sasa 越低越重删（AUC 0.833）、距配体越近越重删（AUC 0.648）、同号电荷簇处更易删 → **删减是「删带电残基总数」的全局习惯，深埋残基（frac_sasa 盲区）尤其不受约束**。
- fix 结合残基（Task2, n40）实证：口袋带电保留 0.46→**1.00**，但口袋外表面 0.601→0.607、核心 0.388→0.372 几乎不动 → **删减不转移也不消失**，主体是非结合区的全局删减。

### 4.3 为何「A1-global」没治好删减（三点机制 + 跨版本证据）

1. **A1-global 是软监督（soft loss），被电荷目标压过**。`pocket_count_loss` 只是总损失里 λ_pocket=0.3 的一项（normalize 后每域 O(0.1-0.3)），与 CE + 电荷 loss（v14 末轮 charge 分量仍 3.08、加权后 ~1.54）竞争。模型在与电荷精确性冲突时**选择保住电荷精度而放弃计数 floor**——训练目标里没有一个硬约束能同时满足「净电荷 = target」与「带电残基总数 ≥ 0.8 native」，模型学会了取舍。实测即使 floor 覆盖的 surface∪pocket，surface retention 也仅 0.55、pocket 0.48，**远低于 floor 0.8** → 计数约束实际被突破。
2. **监督逃逸的「深埋核心」口子仍在**。v12 三损失与 A1-global 都只覆盖 frac_sasa≥0.25（surface）+ pocket（几何）；**frac_sasa<0.25 且非 pocket 的深埋残基**既不在 v12 表面监督内、也不在 A1-global 计数区（core 只锁 native 净电荷、不锁逐残基）。实测 core retention 0.39、最深 SASA 箱 0.35 —— 这部分「成对删」依然无损失惩罚。
3. **删减是微调强化出来的「组成习惯」，且是模型调电荷的主杠杆**。删一个 K 再删一个 D（成对删）净电荷不动、却能精确凑 target；模型几乎没学会「反号替换/新增带电残基」这条替代路径（fixbinding 报告：拿走结合区删减杠杆后，正电 native 的核酸蛋白负向臂完全无法下调，mean 恒定偏高 +10）。A1-global 只是「堵删」的下限，没有给模型一条「保总数还能调净电荷」的正向机制，所以模型仍走「删」的老路。
4. **跨版本证据：删减随数据/监督演化而加深**。v12.2-ligand 0.53-0.65× → v13（A1 keep 护 pocket）0.55-0.69× → v14（A1-global）0.43-0.69×。同一删减捷径在 v13/v14 两代稳定存在，v14 训练更深（50ep、RNA/DNA 新数据、更强电荷控制）后，**用更深删减换来了更高 H2（64%→90%）——「组成换电荷」的取舍是 v14 达成高命中率的一部分代价**（论文子结论 §四）。单看 H2 会高估 v14：电荷与组成必须合起来评估。

---

## 5. 结论

1. **v14 相对 v13 的改进（同集 in-10 口径）是真实且大的**：H2 64%→90%、H3 48/50→50/50、Tm/Sol S2 11/50→0/50、boundary 1A65 dev 8.98→2.6；H1 折叠两版均 50/50（监督没伤折叠）。收益集中在 RNA/DNA 数据扩充的直接受益蛋白（21KL_A/9DWG_L 1/5→5/5）+ 长蛋白 1BJ4（0/5→5/5）+ 核苷酸 1AS2（1/5→5/5）。**数据扩充 + atom 25 修复 + 50ep 确实带来了能力提升**。
2. **v14 未解决、反而加深的问题是组成删减**：10/10 蛋白 0.43-0.69×（v13-in10 0.50-0.99×，共享 5 蛋白 v13 0.56-0.70→v14 0.43-0.60）。根因 = A1-global 软监督压不过电荷目标 + 深埋核心 frac_sasa 盲区逃逸口仍在 + 删减是模型默认调电荷杠杆。这是跨 v12.2-ligand/v13/v14 三代未决的已知局限，需训练侧「既堵删又教替换/新增带电残基」或用户决策 D。
3. **唯一倒退点 = 2FEO**（H2 v13 5/5 → v14 0/5），两模型对中短链/核苷酸蛋白的控制非单调优劣，属需单独小样本现场标定的特例（v12.2 蛋白模式亦记录 2FEO 高方差）。
4. **对比口径教训**：引用 v13↔v14 任何数字前必须标注测试集（旧 10-集 vs in-10）。同模型 v13 在旧集 H2 70% 但 in-10 只有 64%——跨集直接比 70% vs 90% 会高估收益且掩盖「v13 在 in-10 上 1BJ4/21KL_A/9DWG_L 全败」的真实差距。

---

## 6. 关键产物路径

- 训练 ckpt：`output/finetune_ligand_v9/`、`output/finetune_ligand_v12_2/`、`output/finetune_ligand_v13/`、`output/finetune_ligand_v14_rna/finetune_epoch050.pt`
- 训练日志：`log/v9_train.log`、`log/v12_2_ligand_train.log`、`log/v13_ligand_train.log`、`log/v14_ligand_train.log`
- 数据标签：`data/ligand_train/labels.npz`(4957)、`labels_rna_v14_sup2.npz`(414)、`labels_v14_final.npz`(5371)
- 测试集 manifest：`data/validation_pdbs/validation_manifest_v14_in.json`、`validation_manifest_v14_boundary.json`
- 权威验证：`output/generalization_ligand_v13_in10/`、`output/generalization_ligand_v14_clean/`
- 组成：`output/v13_ligand_comp_in10.json`、`output/v14_ligand_comp.json`、`output/v14_deletion_arm.json`
- 校准：`output/charge_calibration_v13_ligand_in10.json`(global 1.285)、`output/charge_calibration_v14_ligand_clean.json`(global 1.492)
- Tm/Sol：`output/tm_sol_ligand_v13_in10/`、v14 clean 同批
- 源报告：`analysis/report/2026-09-01_v12_2_ligand_comp_analysis.md`、`2026-09-02_v13_ligand_validation.md`、`2026-09-04_v14_clean_validation.md`、`2026-09-04_v13_in10_validation.md`、`2026-09-03_long_neg_charge_limitation.md`、`2026-09-04_v14_deletion_location.md`、`2026-09-04_v14_fixbinding.md`、`2026-09-04_paper_subconclusions.md`
- 会话/设计：`session/2026-08-31_v12_2_ligand_migration.md`、`session/2026-09-01_v13_pocket_retrain.md`、`session/2026-09-02_v14_rna_data_a1_global.md`、`index/PROJECT_LOCAL_V12_2.md`（§7/§9）
