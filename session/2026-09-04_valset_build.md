# 验证集专项：轨 A（蛋白 v12.2/v12.3）+ 轨 B（配体 v14 外部未见）过程记录（2026-09-04）

> 执行：验证集专项子 agent。范围：CPU 数据/脚本。未占用任何 GPU（GPU6 跑 v14 clean ESMFold，禁碰）。
> 产物：轨 A = 最终验证集规格 + supplement npz + 两版本 valcurve 驱动脚本（待 GPU 空调度）；
>       轨 B = v14 训练分布 + 外部候选池盘点 + 配额方案 + v14 valcurve 驱动（等闸口=用户确认后再执行）。

---

## 轨 A：蛋白验证集（v12.2/v12.3）——沿用 hold-out + 少量补充长蛋白/深负

### A0. 核对事实（先读已有划分/工具，不重造）
- `labels_holdout_train.npz` = 1176 域（v12.2/v12.3 都真未见）；`labels_v12_2_train.npz` = 6710；
  `labels_v12_3_train.npz` **实际 6580 域**（= 6125 干净 S40 基 + 455 长；中间 7165 版剔除 585 个不可解析域，
  会话 `2026-09-02_v12_3_long_retrain.md` §修正）。coverage 判定用 `coverage_check.py` 口径。
- 覆盖核查工具/验证逻辑已存在：`coverage_check.py`（in≥100 / boundary 30-99 / out<30）、`validate_holdout.py`。
- 结构域 PDB：S40/dompdb_pdb（.pdb 子集，含全部 used+holdout）、S40/dompdb（34653 全量，无扩展名）。

### A1. 缺口量化（hold-out 相对 v12.3 训练分布）
| 维度 | HOLD-OUT 1176 | V12.3 TRAIN 6580 | 结论 |
|---|---|---|---|
| L≥400 | 32 (2.72%) | 590 (8.97%) | 缺（~3.3×） |
| L≥450 | 20 (1.70%) | 292 (4.44%) | 缺 |
| L≥500 | 10 (0.85%) | 150 (2.28%) | 缺 |
| q≤-20 | 8 (0.68%) | 129 (1.96%) | 缺（~2.9×） |
| L≥400 且 q≤-20 | 2 (0.17%) | 83 (1.26%) | 严重缺（深负长蛋白） |
| L≥400 且 q≤-10 | 13 | 280 | 缺 |
| L 均值 / q 均值 | 157.7 / +0.05 | 181.8 / −1.63 | 长度偏短、电荷偏正（455 长负电富集不在 hold-out） |
| 补充后最终验证集 L≥400 | **44 (3.69%)** | — | 仍低于训练 8.97%（残余缺口需下载） |

**结论：确缺长蛋白/深负代表性 → 从内部 S40 剩余长域少量补充（15 域），残余缺口量化报告（A4）。**

### A2. 内部可用补充域筛选（真未见）
排除集合 used = v12.2train(6710) ∪ v12.3train(6580) ∪ holdout(1176) ∪ 10 测试蛋白 PDB 前缀。
- ext_basic_dompdb（781）：103 个未入 used 的全部 parse_PDB 失败（属 585 不可解析类）→ **不可用**。
- S40/dompdb 全池 34653（fasta 726 个 L≥400）：真正未用且 parse 后 L≥400 的 = **12 个**；
  S40/dompdb_pdb 未用里另有深负(boundary) 3 个 → 合计 **15 个补充域**。
- 补充域 PDB 缺失 .pdb 的 12 个已拷入 `data/cath/S40/dompdb_valsupp/`（parse_PDB 需 .pdb 扩展名；已逐一验证 parse 通过）。

补充清单与 coverage（相对 v12.3 训练 6580）：
| id | L | native_q@7.4 | n_close | coverage | 备注 |
|---|---|---|---|---|---|
| 1gw5B00 | 533 | −13.25 | 68 | boundary | 最长 |
| 1cc1L00 | 487 | −3.33 | 122 | in | |
| 1nthA00 | 456 | −20.10 | 72 | boundary | **深负长蛋白** |
| 3opbA02 | 452 | −1.43 | 115 | in | |
| 2afaA00 | 408 | −15.83 | 131 | in | |
| 1u1jA01 | 403 | −5.07 | 158 | in | |
| 5e7qA01 | 402 | −8.90 | 178 | in | |
| 4i2zA03 | 402 | −7.68 | 161 | in | |
| 1d8wC00 | 401 | −13.99 | 157 | in | |
| 5xwiA01 | 400 | −7.23 | 155 | in | |
| 3nixB00 | 400 | −2.93 | 133 | in | |
| 3fwlA02 | 400 | +4.83 | 68 | boundary | |
| 2wiyA00 | 389 | −21.72 | 52 | boundary | 深负 |
| 7v7yA01 | 294 | −20.04 | 49 | boundary | 深负 |
| 2i0oA00 | 273 | −20.52 | 35 | boundary | 深负 |

- 全部序列与 v12.2/v12.3/holdout 训练序列精确去重通过；电荷落在 v12.3 训练 q 范围；与 1A65/1BJ4/13BB/1CDG 等测试蛋白无重复。
- 补充 npz（与 labels 同构，parse_PDB 序列/坐标口径）：`data/cath/labels_v12_3_valsupp_a.npz`（15 域 × 8 pH = 120 样本）。
- 构建脚本：`code/tests/build_valset_supp_a.py`。

### A3. 最终验证集规格
- **base = hold-out 1176（不变）**；**supplement = 15 域**（A2 清单）。
- 联合分布对照（见 A1 末两行）：L≥400 2.72%→3.69%、q≤-20 0.68%→0.92%；仍低于 v12.3 训练（8.97%/1.96%）。
- 联合评估时把 base 与 supp 都交给 valcurve 驱动（--labels + --supp_labels），supplement 全部纳入抽样（n_supp=15）。

### A4. 残余缺口 + 下载方案（报告用，先不下载）
- 内部 S40 真未见 L≥400 已尽（12）；hold-out 自带 32 个 L≥400（含深负长 2-3 个）→ long-stratum 可用 ~44 域，够首轮。
- 若要验证集 L≥400 占比对齐 v12.3 训练 8.97%（即需 +~66 个 L≥400 真未见域）或强化"深负长蛋白"（L≥400&q≤-20 目前仅 holdout 3 + 1nthA00 = 4），内部无源 → **需从 CATH/RCSB 下载**：
  - 建议量：若做 dedicated long-stratum，补 **10-20 个** L 400-650 & q≤-20 真未见域即可显著增强（目标 ~15）。
  - 来源方案：RCSB 检索（enzyme/oxidoreductase 等富含酸性残基大蛋白，或 CATH 非 S40 长域）→ 经 prody 拆链 + parse QC + 序列去重 + coverage in/boundary 复核；或从 CATH 全库非冗余序列找。成本：15 个 ~1-2h（下载+处理）。**待用户确认。**

### A5. 逐 epoch 验证拟合曲线驱动（脚本已写好，未跑 GPU）
- 脚本：`code/tests/valcurve_driver.py`（validate_holdout 同款逻辑；蛋白模式默认；支持补充域分层抽样；可选配体模式）。
- 运行命令（等 GPU6 空 + 主会话调度）：
  - v12.2：
    PYTHONPATH=code /home/baokun_yu/miniconda3/envs/confumpnn/bin/python code/tests/valcurve_driver.py --tag v12_2 --ckpt_dir output/finetune_v12_2 --start_epoch 1 --end_epoch 30 --epoch_step 2 --labels data/cath/labels_holdout_train.npz --dompdb data/cath/S40/dompdb_pdb --supp_labels data/cath/labels_v12_3_valsupp_a.npz --supp_dompdb data/cath/S40/dompdb_valsupp --n_base 10 --n_supp 15 --n_per 5 --seed 42 --weights MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt --device cuda:6 --out output/valcurve_v12_2.json
  - v12.3：同上换 --tag v12_3 --ckpt_dir output/finetune_v12_3 --end_epoch 40 --out output/valcurve_v12_3.json
  - 备注：base 的 1176 域在 dompdb_pdb/（含 3 个补充 deep-neg）；12 个长域补充在 dompdb_valsupp/ → --supp_dompdb 指 dompdb_valsupp。默认不校准（开环口径一致）；如要对齐已发表 hold-out 口径可加 --calibration_file（版本最终 global 表）。
- 输出：`output/valcurve_v12_2.json`、`output/valcurve_v12_3.json`，形如 {"epochs":{"1":{hit_rate/mean_dev/mean_recovery}, ...}}。

---

## 轨 B：配体 v14 外部验证集 —— 选数据后闸口交用户确认

### B1. v14 训练 5371 分布（类型按 all_pdb symlink 实际归类）
| 类型 | n | 占比 | L 中位 | L≥300 | q@7.4 均值 | q≤-20 | q>+10 |
|---|---|---|---|---|---|---|---|
| small_mol | 4145 | 77.2% | 296 | 2049 | −4.2 | 222 (5.4%) | 102 (2.5%) |
| metal | 564 | 10.5% | 256 | 233 | −5.3 | 36 (6.4%) | 5 (0.9%) |
| rna_pdbs（真 RNA/DNA 结合） | 414 | 7.7% | 199 | 113 | **+4.8** | 0 | 100 (24%) |
| rna（旧核苷酸辅因子） | 242 | 4.5% | 336 | 166 | −4.2 | 3 (1.2%) | 0 |
| dna（旧脱氧核苷酸辅因子） | 6 | 0.1% | 344 | 4 | −2.6 | 0 | 0 |
| 合计 | 5371 | 100% | 291 | 2565 | −3.64 | 261 | 207 |
- 长度范围 20-500，q 范围 −75.7..+43.8。**RNA/DNA 类型碱性富集**（q>+10 占 24%），small_mol/metal 负偏。

### B2. 外部未见候选池盘点
1. **RNA/DNA 真未见（本地可产，免下载）**：
   - 立即可用已 QC：`rna_pdbs/3ADB_A.pdb`(L250,q+12.0)、`3ADB_B.pdb`(L247,q+13.2)——序列不在 5371 训练、coverage boundary。
   - 大规模本地池：`rna_complex_raw/` 793 复合物中 **398 个未拆可用复合物**含 ~1100 条 L50-500 且 15Å 有核酸的蛋白链 → 会话 §8.1 已确认去重后 **939 未见单链**（RNA 232/DNA 676/hybrid 31 的池子按 §9 精选用掉 205 后仍富余）。需用 `split_nucleic_complex.py` 拆出 .pdb（CPU ~10-20 min / 十来个复合物）。
2. **small_mol / metal / 旧 nucleotide（本地≈0 未见）**：all_pdb 即训练+15 个额外；`small_mol`4155 仅比训练 4145 多 ~10。**必须从 RCSB 下载**（candidates.json 10000 / candidates2.json 4000 中未下载部分 5410+ 候选码），再按训练同款处理（单链+配体 Z）。

### B3. 建议外部验证集配额（类型×长度×电荷对齐训练，RNA/DNA 有意过采样）
目标总池 **~30**（够 valcurve n_dom 20-30 逐 epoch 用）：
| 类型 | 拟选 | L 分配 | 电荷区覆盖 |
|---|---|---|---|
| small_mol | 15 | 0-200:2 / 200-350:7 / 350-500:6 | 主打 −8..+10 主体区；含 1-2 个 ≤−15 |
| metal | 5 | 0-200:1 / 200-350:2 / 350-500:2 | −15..+3 主体；含 1 个 ≤−15 |
| nucleotide（旧辅因子型） | 3 | 200-350:2 / 350-500:1 | −10..+3 |
| RNA/DNA（新能力，过采样） | 7 | 0-200:3 / 200-350:2 / 350-500:2 | 主打 q>+3 碱性；含 1-2 个近中性 |
- 每个候选：parse_PDB QC 通过、序列与 5371 训练及 in-10 测试蛋白(6D2O/1AS2/2FEO/5CQH/1CGE/1BJ4/21KL_A/5O60_E/3MXB_A/9DWG_L + boundary 1A65)精确去重、coverage in/boundary（相对 v14 训练 5371）、8pH charge/pI 标签（build_rna_v14_labels.py / build_ligand_labels.py 同款）。

### B4. 缺口与成本（下载需用户批准）
- 本地可立即产生：RNA/DNA ~7（3ADB_A/B + 拆 ~10-15 个复合物挑 5 个）。
- 需下载补齐：small_mol/metal/nucleotide 约 **23 个通过候选** → 建议 fetch ~40-50 个 PDB 码（冗余应付 QC/去重/coverage 淘汰），成本约 **30-60 min（RCSB 下载，CPU）+ 后续拆链/QC/建标签 ~1h**。
- 来源：`fetch_ligand_pdbs.py --targets small_mol:N,metal:N` 从 candidates.json/candidates2.json 未下载码抽；确保与训练 4957 旧码 + all_pdb 不重复。

### B5. v14 逐 epoch 拟合曲线脚本（已就绪）
- 复用 `code/tests/valcurve_driver.py --mode ligand`（ligand_mpnn backbone `ligandmpnn_v_32_010_25.pt`，
  use_atom_context=True, num_ligand_atoms=25）。等闸口+候选 labels npz 就绪 + GPU 空后调度：
  PYTHONPATH=code /home/baokun_yu/miniconda3/envs/confumpnn/bin/python code/tests/valcurve_driver.py --tag v14_ligand --mode ligand --ckpt_dir output/finetune_ligand_v14_rna --start_epoch 1 --end_epoch 50 --epoch_step 3 --labels <候选 labels npz> --dompdb <候选单链+配体 PDB 目录> --n_base 20 --n_supp 0 --n_per 5 --seed 42 --weights LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt --device cuda:6 --out output/valcurve_v14_ligand.json

### B6. 闸口状态
- ⏸️ 以上为**方案与分布对照**；候选最终清单依赖 B4 的下载决策。**交主会话转用户确认后再执行拆链/下载/建标签/v14 valcurve。**

---

## 遵守的铁律
- 未占用任何 GPU（GPU6 v14 ESMFold 100% 占用中）；valcurve/采样一律"脚本+命令就绪"，等 GPU 空由主会话调度。
- 未改 test manifest（in-10）定义；未碰任何 *.pt 权重（只读 ckpt keys）；未动 v14 clean 链产物。
- 数据文件（npz/pdb/json 大清单）在 data/ 下不入 git；脚本在 code/tests/。

---

## 轨 A 追加：外部下载深负长蛋白补足（2026-09-04 用户批准后执行，CPU）

### A6. RCSB 下载 + 筛选
- RCSB 检索单体蛋白（1 protein polymer entity / 1 molecule / resolution≤2.5），分页 5000 条 ID，
  排除 used CATH PDB 码 7729 个后剩 4757；批量 FASTA 算 L/q → 命中 L400-650 & q@7.4≤-20 共 60 个；
  下载全部 60 个 PDB 到 `data/cath/ext_deepneg_raw/`；parse QC（parse_PDB）后 52 个通过 L/q；
  与训练(6710/6580/1176/suppA)序列精确去重剔除 5 个 → 余 47；10% 序列多样性贪心 → 25 个不同家族。
- **coverage（相对 v12.3 训练）**：in=0 / boundary=8 / out=17。深负长蛋白本质处于训练 q 尾，
  只能到 boundary；**取 8 个 boundary 入标准验证集**（L433-528，q −20.1..−26.8）：
  1hxa(L528,q−23.4,n36)、1lj8(492,−22.3,51)、1h9a(485,−21.9,54)、1c7i(483,−26.8,32)、
  1h71(455,−24.7,46)、1jd9(448,−22.5,61)、1b0i(447,−21.5,71)、1egn(433,−24.1,47)。
- **out 的 17 个**（更极端深负 q<−28 或 n_close<30，如 1lfw q−36.6/1ju3 q−34/1l7r q−35/1gal q−28.6/1avk L615/1idq L564）：
  不入标准集（coverage 原则），如需"深负长外推检验"可单独加跑；结构留在 `data/cath/ext_deepneg_raw/`。

### A7. 并入验证集
- suppB 标签 npz：`data/cath/labels_v12_3_valsupp_b.npz`（8 域×8pH，parse_PDB 口径）；
  8 个结构拷入 `data/cath/S40/dompdb_valsupp/`。
- 合并 suppA(15)+suppB(8) → **`data/cath/labels_v12_3_valsupp.npz`（23 域×8pH=184）**。
- 最终联合验证集分布（1176+23=1199）：
  | 指标 | hold-out 1176 | 最终 1199 | v12.3 训练 6580 |
  |---|---|---|---|
  | L≥400 | 32 (2.72%) | **52 (4.34%)** | 590 (8.97%) |
  | L≥450 | 20 (1.70%) | 29 (2.42%) | 292 (4.44%) |
  | q≤−20 | 8 (0.68%) | **20 (1.67%)** | 129 (1.96%) |
  | L≥400 & q≤−20（深负长） | 2 | **11 (0.92%)** | 83 (1.26%) |
  → 深负长覆盖 2→11，已接近训练比例；L≥400 仍低于训练（天然域分布长蛋白稀疏所致）。
- spec：`data/cath/v12_3_valsupp_spec.json`（已更新）。

### A8. valcurve 最终运行命令（仍不运行，GPU6 空后主会话调度）
- 用合并 supp（23 域，全纳入）+ base 少量；主命令：
  - v12.2：
    PYTHONPATH=code /home/baokun_yu/miniconda3/envs/confumpnn/bin/python code/tests/valcurve_driver.py --tag v12_2 --ckpt_dir output/finetune_v12_2 --start_epoch 1 --end_epoch 30 --epoch_step 2 --labels data/cath/labels_holdout_train.npz --dompdb data/cath/S40/dompdb_pdb --supp_labels data/cath/labels_v12_3_valsupp.npz --supp_dompdb data/cath/S40/dompdb_valsupp --n_base 7 --n_supp 23 --n_per 5 --seed 42 --weights MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt --device cuda:6 --out output/valcurve_v12_2.json
  - v12.3：同结构换 --tag v12_3 --ckpt_dir output/finetune_v12_3 --end_epoch 40 --out output/valcurve_v12_3.json
  - 备注：n_base=7+n_supp=23 → 30 域/epoch（long/深负重采样，供两版对比）；另可选 base-only 曲线
    （--n_base 25 --n_supp 0）看纯同分布拟合。默认不校准；产物 output/valcurve_v12_2.json / valcurve_v12_3.json。

---

## 轨 B 执行结果（2026-09-04 用户批准后，CPU，未跑 GPU）

### B7. RNA/DNA 本地候选（免下载）
- 从 rna_complex_raw 398 未拆可用复合物中选 24 个（排除测试蛋白/超大核糖体），
  split_nucleic_complex.py 拆出 208 条单链；parse QC 全过；与 5371 训练+in-10 测试序列去重后
  **196 条真未见**（`data/ligand_train/rna_pdbs_ext/`）。coverage：in 162/boundary 32/out 2。

### B8. small_mol/metal/nucleotide 下载候选
- 从 candidates.json(10000)+candidates2(4000) 未用码随机抽 150 个下载到
  `data/ligand_train/ext_smallmol_raw/`；parse QC（单链 + L≤500 + Y 配体）通过 50，序列去重后 41：
  small_mol 35 / metal 5 / rna(核苷酸辅因子) 1；coverage in 37/boundary 4。

### B9. 最终配额抽样（总 30 = small_mol 15 + metal 5 + nucleotide 3 + RNA/DNA 7）
- **候选最终清单**：`data/ligand_train/v14_ext_valset_final.json`（30 项，各带 type/L/q7.4/coverage/n_close/src）。
- **标签 npz**：`data/ligand_train/labels_v14_ext_valset.npz`（30 域×8pH=240；parse_PDB 口径）。
- **结构 PDB**：`data/ligand_train/v14_ext_valset_pdb/`（30 个单链+配体 Z）。
- coverage：in 24 / boundary 6（全部非 out）。与 5371 训练 + in-10 测试序列全部精确去重。
- 构建脚本：`code/tests/build_v14_ext_valset.py`。

### B10. 分布对照（type × L 箱；pool vs v14 训练）
| type | pool L<200 / 200-350 / ≥350 | 训练说明 |
|---|---|---|
| small_mol (15) | 4 / 10 / 1 | 训练 L 20-500 med296；pool 偏中段，轻首尾 |
| metal (5) | 3 / 2 / 0 | 训练 med256 |
| nucleotide (3) | 0 / 3 / 0 | 训练多为中长段（med336）→ 对齐 |
| RNA/DNA (7) | 1 / 2 / 4 | 训练 med199 偏短；pool **有意偏长**(L270-442×5) 检长链核酸结合新能力；仍含 1 小链(8G57_G L118 q+13) |
- q 范围：pool q −23.0..+13.1（med −3.5），覆盖训练主体区与一小部分深负(small_mol 1lzk q−23)；
  RNA/DNA 7 个中 3 正(q+9..+13) 反映 RNA 结合蛋白碱性。
- **说明**：RNA/DNA 采样按"检新能力"刻意过采样且偏长链，故 type 内长度分布不要求严格对齐训练；
  总量与类型占比（small_mol 主导）基本对齐。

### B11. v14 valcurve 驱动（待复核+GPU 空后调度）
- `code/tests/valcurve_driver.py --mode ligand` 已就绪（featurize CPU 验证过）。
- 运行命令（复核通过 + GPU6 空后由主会话调度）：
  PYTHONPATH=code /home/baokun_yu/miniconda3/envs/confumpnn/bin/python code/tests/valcurve_driver.py --tag v14_ext --mode ligand --ckpt_dir output/finetune_ligand_v14_rna --start_epoch 1 --end_epoch 50 --epoch_step 3 --labels data/ligand_train/labels_v14_ext_valset.npz --dompdb data/ligand_train/v14_ext_valset_pdb --n_base 30 --n_supp 0 --n_per 5 --seed 42 --weights LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt --device cuda:6 --out output/valcurve_v14_ligand.json

### B12. 闸口状态
- 候选清单 + 分布对照已交主会话转用户复核；**复核通过前不跑 v14 valcurve**（且需 GPU6 空）。

---

## 轨 B 规模校正（用户 2026-09-04）：目标 805 外部未见池（训练 15%）—— 阶段 1 可行性简报

> 30 个已建候选并入作种子，不浪费。以下为简报（**未获批准不进行全量下载**）。

### B13.1 训练真实类型构成（all_pdb symlink 口径，= 建训练集口径）
| 类型 | 域数 | 占比 | L 中位 | q@7.4 均值 | q≤−20 | q>+10 |
|---|---|---|---|---|---|---|
| small_mol | 4145 | 77.2% | 296 | −4.2 | 222 | 102 |
| metal | 564 | 10.5% | 256 | −5.3 | 36 | 5 |
| rna（旧核苷酸辅因子） | 242 | 4.5% | 336 | −4.2 | 3 | 0 |
| dna（旧脱氧核苷酸辅因子） | 6 | 0.1% | 344 | −2.6 | 0 | 0 |
| rna_pdbs（真 RNA/DNA 结合） | 414 | 7.7% | 199 | +4.8 | 0 | 100 |
| 合计 | 5371 | 100% | 291 | −3.64 | 261 | 207 |
（训练按 pdb 码去重 = 5131；L 范围 20-500）

### B13.2 805 池配额（按 15% 比例，取整）
| 类型 | 配额 | 计算 |
|---|---|---|
| small_mol | 621 | 4145/5371×805 |
| metal | 85 | 564/5371×805 |
| nucleotide（旧 rna+dna 辅因子） | 37 | 248/5371×805 |
| RNA/DNA（真核酸结合） | **62** | 414/5371×805 |
| 合计 | **805** | |

### B13.3 本地"已可用未见"盘点（parse QC 通过、序列去重、coverage in/boundary）
| 类型 | 已有可用 | 说明 |
|---|---|---|
| small_mol | 35（含 15 seed） | ext_smallmol_raw 150 下载中 pass（含 1lzk q−23 等深负代表） |
| metal | 5（全部 seed） | 1rxf/1qnx/1mq9/1pnd/1jc2 |
| nucleotide（旧辅因子） | 3（全部 seed） | 1e2f(ADP)/1hku(NAD)/1pz0(NAP) |
| RNA/DNA（真） | 196（含 7 seed） | rna_pdbs_ext（本地拆链），coverage in162/boundary32/out2，免下载 |
| 合计可用 | 239 | 覆盖 RNA/DNA 配额绰绰有余；small_mol/metal/nucleotide 严重不足 |

### B13.4 缺口 + 需下载量（实测 pass 率 41/150≈27%；下载 150 中 pass 类型 small_mol35/metal5/rna1）
| 类型 | 配额 | 已有 | 缺口 | 需下载(按 pass 率) |
|---|---|---|---|---|
| small_mol | 621 | 35 | 586 | ≈586/0.23≈**2550** |
| metal | 85 | 5 | 80 | 计入同上（随机池 pass 中 ~10-12% 金属） |
| nucleotide | 37 | 3 | 34 | 部分靠 small_mol pass 中辅因子结合蛋白重分类（NAD/ADP/FAD）补 |
| RNA/DNA | 62 | 196 | 0（盈余） | 0 |
- **下载量级**：还需从 6302 候选码下载约 **2400-2600 个 PDB**（加现有 150 ≈ 2700），磁盘 ~2.7GB。
- **预计 CPU 时长**：下载（12 并行）≈10-20 min；prody parse QC（串行 ~0.5-1s/个，可并行 4-8×）≈30-45 min；
  建标签/去重/coverage/贪心选 805 ≈10-15 min。合计约 **1-2 h CPU**（全程不碰 GPU）。
- ⚠️ 建议（降低下载量）：用 RCSB 单体搜索（pdbx_number_of_molecules=1，把 ~45% 多聚体跳过率消掉）+ 按金属/核苷酸**定向搜索**，可把下载量降到 ~1200-1500 且更好满足 metal/nucleotide 配额。默认仍按指令走 6302 候选池，定向搜索作为可选项请用户定。

### B13.5 预期 coverage（相对 labels_v14_final）
- 已见 pass 样本：in 37 / boundary 4 / out 0（37/41 in）。RNA/DNA 196：in 162/boundary 32/out 2。
- 预期 805 池 ~90% in、~8% boundary、<2% out（同源同分布），类型/电荷/长度成比例可满足。

### B13.6 阶段 2（批准后执行）
下载缺失类别（优先 6302 同源候选池）→ parse QC（单链+L≤500+Y 口径沿用）→ 对 5371 训练
+in-10+1A65 精确去重 → coverage → 贪心多样性按配额/分布选满 805 → labels npz
（domain_ids/seqs/coords/pH×8/charge/pI，`data/ligand_train/labels_v14_valset_805.npz`）+ spec json
+ 结构 `data/ligand_train/v14_valset_pdb/`（gitignored）。valcurve 不先跑。

---

## 轨 B 阶段 2 执行结果（2026-09-04，805 池构建完成）

### B14.1 执行过程
- 同源候选池(candidates+candidates2 未用 6302 码)下载 **4550 个**（ext_smallmol_raw，~2.7GB）；
- 为补"≥350 长蛋白 + 深负"做 RCSB 单体+配体定向：长 L350-500 579 + 深负(q≤-20) 65 → ext_longlig_raw（合并 619 唯一）；
- 全部 parse QC（单链 + L≤500 + Y 配体，沿用训练口径）；main 4550 解析 pass 900；topup 619 解析 pass 341；
  合并跨池按 id/序列去重 → combined pool 1175（in/boundary：small_mol 813/metal 166/nucleotide 146 + RNA/DNA 本地 194）；
- coverage 相对 labels_v14_final；只留 in/boundary；配额分层(L×q 8 带)选满 **805**。

### B14.2 最终 805 vs 训练分布
| type | n(配额) | Lmed pool/train | L% <200/200-350/>=350 pool | train | qmed pool/train | q≤−20 pool/train | q>10 pool/train |
|---|---|---|---|---|---|---|---|
| small_mol | 621/621 | 257/296 | 38/36/26 | 27/42/31 | −3.0/−3.5 | 15/222 | 7/102 |
| metal | 85/85 | 151/256 | 60/22/18 | 41/34/25 | −4.0/−4.2 | 1/36 | 2/5 |
| nucleotide | 37/37 | 323/336 | 14/43/43 | 8/46/46 | −4.5/−3.6 | 0/3 | 0/0 |
| RNA/DNA | 62/62 | 106/199 | 74/19/6 | 50/31/19 | +8.3/+5.0 | 0/0 | 24/100 |
coverage：**in 730 / boundary 75 / out 0**。805 全部精确序列去重（vs 5371 训练 + in-10 + 1A65 + 2E9R_X）。

### B14.3 残余卡点（池受限，如实报告，未擅自降标）
- **metal / RNA/DNA 长度偏短**：合并池 long(≥350) 可选仅 metal 18 / RNA-D 9（RNA/DNA 本地池 74%<150，多小锌指/核糖体样）；距训练 L 分布（25%/19% ≥350）约差 3/3-7 个。
- **small_mol/metal 深负(q≤−20) 少**：覆盖内深负候选池仅 15/1（深负天然落 coverage out）；目标 ~33/5，只到 15/1。
- **small_mol ≥350 已达池上限附近**（池 183 个，选 160）；再要更贴合训练(31%)需再下载更长 small_mol。
- 类型配额与总数 805 全达标；coverage 全 in/boundary；电荷中位数逐类接近训练。若要严格卡长度/深负尾，需追加：更长 metal/RNA-DNA（或再拆长核酸复合物）与定向强酸性蛋白，投入 ~0.5-1h 额外下载/拆链——**请主会话裁决是否追加**。

### B14.4 产物（gitignored，不入 git）
- labels：`data/ligand_train/labels_v14_valset_805.npz`（805 域 × 8pH = 6440 样本；domain_ids/seqs/coords(Cα)/pH/charge/pI）
- spec：`data/ligand_train/v14_valset_805_spec.json`（类型/分布/coverage/来源/重分类/去重日志）
- 结构：`data/ligand_train/v14_valset_pdb/`（805 个单链+配体）
- 轻量清单：`/tmp/v14_805_built.json`；脚本：code/tests/select_v14_805.py、build_v14_805_labels.py、assemble_v14_805.py

### B14.5 v14 per-epoch valcurve（等用户复核 + GPU6 空后调度，先不跑）
  PYTHONPATH=code /home/baokun_yu/miniconda3/envs/confumpnn/bin/python code/tests/valcurve_driver.py --tag v14_805 --mode ligand --ckpt_dir output/finetune_ligand_v14_rna --start_epoch 1 --end_epoch 50 --epoch_step 3 --labels data/ligand_train/labels_v14_valset_805.npz --dompdb data/ligand_train/v14_valset_pdb --n_base 30 --n_supp 0 --n_per 5 --seed 42 --weights LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt --device cuda:6 --out output/valcurve_v14_ligand.json
