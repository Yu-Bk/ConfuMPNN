# v12.3 蛋白模式扩长蛋白重训 + 验证集重构（2026-09-02 执行）

## 目标
v12.2 蛋白模式（MoMPNN）为当前最优（slope 1.00 / H2 72% / Tm-Sol 0/50），短板 = **长蛋白训练量不足**
（CATH 6710 训练域 L>400 仅 132 = 2%；验证 1BJ4/470、1A65/504 为长度 OOD，H2 失败）。
v12.3 = v12.2 基础上**补入 S40 长蛋白域重训**（不改 v12.2 超参，单一变量 = 加长蛋白数据）。

---

## A. 可行性评估

### A1. S40 长蛋白统计（`/tmp/cath_s40_stats.json`，解析 34653 个 dompdb 文件实测）
| 长度 | 域数 | CATH class 1/2/3 分布 |
|---|---|---|
| L>300 | 2488 | 597 / 436 / 1455 |
| **L>400** | **617** | 229 / 109 / 279 |
| L>450 | 311 | 124 / 57 / 130 |
| L>500 | 154 | 59 / 32 / 63 |
| L>600 | 44 | — |
| L>700 | 25 | — |
| max | **1202**（1y5lA02） | |

- S40 域**全部单链**（多链=0，天然规避同源二聚体）；含 HETATM 6746 个（多为水/离子，protein parse 忽略）；非标残基域仅 37 个（候选长域命中 4 个，已剔除）。
- **结论：长蛋白池充足**，可支撑训练补长。

### A2. 采样方案（保留原 6710 + 追加，最小改动）
- **v12.3 训练 = 原 v12.2 的 6710 域 + S40 中 L≥400 且未入 v12.2 train(6710)/holdout(1176) 的域**。
- 候选粗筛（CATH fasta L≥395，排除已用/holdout/验证 PDB code/4 个非标域）568 → parse 后 L≥400 **455 个**。
- 追加后：**7165 域**，L>400 = 581（**8.1%**，v12.2 为 2%），L>450 = 287，L>500 = 144，max 1202。
  - 硬上限 ~9.5%（S40 未用 L≥400 池仅 ~534），8.1% 是保留 6710+全收长候选的实际可达值，略低于建议 10% 但在论证范围（较 v12.2 提升 4 倍，覆盖 1A65/504、1BJ4/470 OOD 区 + 外推至 1200）。
- CATH class 比例保持：v12.2 1/2/3 = 26.5%/21.4%/52.1% → v12.3 = 26.0%/20.4%/49.9%（微偏 αβ 少，可接受）。
- 标签与 v12.2 完全同构：domain_ids/seqs/coords(Cα)/pH/charge/pI，每域 8 pH Uniform[4,10]。
- **不动 condition_defaults.yaml**（μ/σ 沿用 v12.2，保持推理归一化一致）。

### A3. 长蛋白引入风险排查结论
1. **SASA/frac_sasa 预解析**：`src/sasa.py` 用 freesasa+Bio.PDB，按残基线性；455 长域 SASA 一次性 ~10min，内存线性，无异常（v12.2 已有 L=950 域成功）。
2. **GPU 显存/batch**：train_finetune 为**逐域 step**（非 batch padding），单 step 只处理 1 域 × 8 pH → 峰值由最长域决定。**v12.2 已成功训练含 950 残基域**；dry-run 实测最长 1202 域 1-epoch 无 OOM（缓存 0.79GB/50 长域）。
3. **parse_PDB/特征化**：新增 455 域全部经 train_finetune 同款 parse_PDB+featurize 预检 **ok=455/fail=0**（关键：训练中 skip=0 → 无 pH/charge 段错位）。
4. **非蛋白残基**：S40 域基本纯蛋白；37 个含非标 CA 的域中 4 个落在长候选，已剔除。
5. **多链**：S40 域单链天然保证。

---

## B. 数据前处理
- 脚本：`code/tests/build_v12_3_augment.py`（新写，复用 build_labels.py 同款 parse_domain/net_charge/find_pI）。
- 产物：`data/cath/labels_v12_3_train.npz`（7165 域 × 8 pH = 57320 样本）。
- 校验：域数/长度直方图/L>400 占比/class 比例/pH range[4,10] 全部打印通过；新增域 parse 预检 455/455。
- 泄漏保护：候选排除 10 个现有验证蛋白 PDB code + E 新增 5 个（1cdg/13bb/1acc/1ayl/1bpm）+ holdout 域。

## C. 训练前检查 + dry-run
- 环境：GPU6（显存空 1.3GB/140GB，但有人小进程占 util，训练仍正常）；confumpnn python。
- dry-run：50 长域（含最长 1202）临时 labels → 1-epoch **0 NaN、0 崩溃**，checkpoint 正常（`output/finetune_v12_3_dryrun`）。
- v12.2 训练命令已从 `log/v12_2_train_mompnn.log` + `session/2026-08-29_v11_ablation.md §9.2` 精确重建并沿用。

## D. 训练启动（PID 1400836，GPU6）
```bash
PYTHONPATH=code nohup setsid ~/miniconda3/envs/confumpnn/bin/python code/train_finetune.py \
  --device cuda:6 --epochs 30 \
  --weights MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt \
  --labels data/cath/labels_v12_3_train.npz --dompdb data/cath/S40/dompdb \
  --lambda_c 0.5 --lambda_kl 0.05 --lambda_keep 0.5 \
  --charge_temp 0.5 --perturb_prob 0.3 --placeholder_prob 0.15 \
  --decouple_perturb --decouple_range 12.0 \
  --ph_aware_filter --structure_boost 1.5 \
  --v12_supervision --frac_floor 0.5 --gravy_margin 0.4 --lambda_v12 0.2 --lambda_target 0.2 --sasa_threshold 0.25 \
  --out_dir output/finetune_v12_3 \
  --log_file log/v12_3_train_mompnn.log --log_progress log/v12_3_train_mompnn_prog.json \
  > log/v12_3_train_mompnn.stdout 2>&1 &
```
- 回退保障：v12.2 权重 `output/finetune_v12_2/finetune_epoch030.pt` + `condition_encoder_last.pt` 已 cp 到 `/tmp/v12_2_backup/`（md5 一致），未改动原目录。
- 启动确认：进程存活、日志正常（7165 域确认）、GPU6 显存开始累积、无 NaN。
- 预计 ~30ep × ~12-14min + 预解析 ~20-30min ≈ **7-9h**（主 session cron 监控）。

## E. 验证集重构（不阻塞训练）
- `data/validation_pdbs/validation_manifest_v12_3.json`（新写）：
  - **删除同源二聚体**：1C6O/1AXW/1AG0（native 回折 TM~0.5 为 ESMFold 工具限制，H1 无法单链评估）。
  - **保留 7 单体**：1AZM(258)/1AS2(315)/2FEO(221)/5CQH(186)/1CGE(162)/1A65(504)/1BJ4(470)——1A65/1BJ4 保留作 v12.3 长蛋白改进检验。
  - **新增 2 未训练单链长蛋白**：13BB(560, 乙酰乳酸合酶, TPP)、1CDG(686, 环糊精糖基转移酶)——扩展长蛋白 OOD 长度覆盖（>504）。
  - 9 个蛋白全部单链（链 A）、不在 v12.3 训练集（S40 6710+455 域）、leak=False。
- 兼容性：validate_generalization.py 以 `items[].pdb/path` 读取，额外字段 L/note 经 `.get()` 访问不影响。

## 后续（主 session）
1. **训练监控**：cron/手动看 `log/v12_3_train_mompnn.log`（epoch 30/30、无 NaN、checkpoint 完整 `output/finetune_v12_3/`）。
2. **验证链**（v12.3 checkpoint 替换）：
   - 17 蛋白响应诊断（valid 区 slope ∈ [0.9,1.15]）→ 建/复用校准表（新表应重拟合，因分布变长）。
   - 组成分析（D/K vs native，防删减/过度添加）——**长蛋白组成是重点**。
   - 泛化验证：`run_v12_1_validation.sh` 换 `--manifest validation_manifest_v12_3.json` + v12.3 checkpoint（9 蛋白 × 5 臂 × n30）。
   - H1 折叠（ESMFold TM，长蛋白尤其看 1A65/1BJ4/13BB/1CDG）、H2 电荷、H4 PROPKA、Tm/Sol。
3. **对比口径**：v12.3 vs v12.2 长蛋白（1A65/1BJ4）H2 是否改进 = 本轮核心结论。

---

## 训练修正记录（重要，2026-09-02 第二次启动）

### 第一次启动失败：labels 含 585 个不可解析域 → 标签段错位
- **现象**：首次启动（PID 1400836，30ep）预解析中途 `跳过坏域 8ooyA03/8ow4B01/9antA00...`。
- **根因**：v12.2 的 labels_v12_2_train.npz（= labels_balanced_v7 85%）**尾部含 585 个外部碱性域**
  （ext_basic，非 S40，PDB 不在 `S40/dompdb` → prody parse 失败）。v12.2 训练实际跳过这 585 个
  坏域、只训 6125 个 S40 域——**空洞在数据末尾所以 v12.2 无错位**（log 证实：`共跳过 585 个坏域，实际训练 6125 域`）。
- **v12.3 为何致命**：把 455 个新增长域 append 在含 585 空洞的 base 之后 → 空洞使**全部 455 新增域
  pH/charge 标签整体错位**（train_finetune 按"成功域计数"取标签段，skip 后续全错位）。
- **修复**：从 `labels_v12_3_train.npz` 剔除全部 585 个不可解析域（v12.2 log skip 名单）
  + 新增 455 不动 → **6580 域**（6125 干净 base S40 + 455 长域）。全量预检 `ok=6580/fail=0`。
  - L>400 = 581（**8.8%**）、L>450 = 287、L>500 = 144、max 1202；class 1/2/3 = 25.7/20.2/50.6%。
- **教训固化**：labels 只应含可 parse 域（train_finetune 的 pH/charge 段索引按成功域计数，**任何 skip 都会让其后所有域标签错位**）；v12.2 实际训练集 = 6125 域而非 6710（585 ext 从未被训）。

### epoch 决策（coordinator/用户 2026-09-02）
- 用户定 **40-50 epoch**；执行取 **40**。
- 理由：① 数据 6580 域/epoch（vs v12.2 实际 6125），长蛋白新类型 455 域占 6.9%（新分布补入，
  ConditionEncoder 需更多轮适应长蛋白响应）；② 40ep×~19min≈13h 在预算内；③ 记录 epoch30/40 收敛
  对比——若 30→40 仍明显下降说明新数据需更多轮，plateau 则与 v12.2(30ep 已收敛，log 末 3 ep
  total 4.211→4.199→4.198) 对照公平（均收敛态）。

### 第二次启动（当前，PID 1526600）
- 命令同 §D 但 `--epochs 40`；labels = 干净 6580 版。
- 确认：进程存活、预解析完成（缓存 ~29.3GB，无 skip）、**epoch 1/40 完成无 NaN**：
  total=4.9237 ce=1.8516 charge=3.6367 kl=0.1055 keep=0.8058（cd self/mild/extreme=3.44/3.61/4.43）。
- **速度**：epoch1 elapsed 19.4min（GPU6 被他人进程占 util 99% 拖慢 ~1.8×；v12.2 无竞争时 ~10.9min）。
  40ep 预计 ~13h + 预解析 15min ≈ 13.3h。若后续 GPU6 竞争缓解会更快。

---

## 验证链执行记录（2026-09-03 续）

### ① 诊断完成 + ② 校准表
- 16 蛋白诊断（7 trainish + 9 valid）→ `output/v12_3_diag_response.json`。
- **未校准 valid slope 均值 1.496±0.40（v12.2: 1.562）**——共同 7 单体全降（1BJ4 2.49→2.05、1AS2 2.09→1.83、2FEO 1.48→1.18…），补长蛋白+40ep 改善响应增益。
- 校准表 `output/charge_calibration_v12_3.json`：global slope 1.471 / intercept −5.935（192 点）+ per_protein 16。报告 `analysis/report/2026-09-02_v12_3_diag.md`。

### ③ 泛化 V1（per-protein 校准）H2 = 49%（22/45）⚠️ 低于 v12.2 72%
- **1BJ4（长蛋白 470）v12.2[0/5] → v12.3[5/5] 治愈**（校准后 dev ≤1.5）——长蛋白治疗核心证据。
- **1A65 v12.2[5/5] → v12.3[1/5] 恶化**（native dev +5.4 欠冲偏正）；2FEO 0/5（同 v12.2）；新增 13BB 0/5、1CDG 0/5。
- 1AS2/1BJ4/1CGE 全过；1AZM 3/5、5CQH 4/5 轻微退化。
- **根因（诊断）**：长蛋白（1A65/2FEO/13BB/1CDG/1BJ4）未校准响应为 **S 形弯曲**（-34~-25 过冲负、-15~-5 陡、0 附近平），
  全靶线性拟合 slope 被极端区拉高 → per-protein 校准把温和 target 反推到响应低斜率区 → 欠冲偏正。
  v12.2 的 1A65 命中是因 intercept 不同恰好反推到陡区（碰巧）；v12.3 intercept −9.6 → native −27 反推内部 −12.2（平区）→ 欠冲。
- **判定**：v12.3 模型长蛋白未校准响应确实改善（1BJ4 slope 2.49→2.05 且校准后全过）；H2 偏低主要是**线性校准表对 S 弯曲响应的固有局限**（非模型退化）。
- 进行中：global 校准口径重采样（`output/generalization_v12_3_global`，GPU3）对比，判断最优校准口径。

### 响应弯曲根因分析 + H2 三档（2026-09-03）
- **curvature 分析**（`output/v12_3_response_curvature_analysis.json` + `analysis/report/2026-09-02_v12_3_curvature_analysis.md`）：
  **v12.3 全部 14 蛋白响应更 S 弯曲**（bend 全面升 Δ+0.4~1.0、RMSE 除 1BJ4 外全升、intercept 更负）。
  唯一例外 1BJ4（RMSE 3.00→2.86 更线性、intercept +5.2→−2.7 居中）→ 解释其校准命中 0/5→5/5。
  机理：455 长蛋白（负电富集）+ 40ep → ConditionEncoder 极端区响应增强、中区偏弱 → 非线性更强 → 线性校准失效。
- **H2 三档**：per-protein 22/45=49%；global 18/45=40%；去掉最弯 3 valid（1AS2/1A65/2FEO）per 16/30=53%、global 9/30=30%。
  → H2 偏低是普遍性（非少数蛋白），v12.3 多数蛋白更弯。
- **决策记录（coordinator 2026-09-03）**：A 方向批准——H1/ESMFold+V3-V5+H3+Tm/Sol 全部跑完；电荷 H2 如实记录
  禁止二次/分段校准救 H2；最终交付判断由 coordinator 转用户（候选：交付 v12.3/回退 v12.2/深入诊断/混合）。

### ④ H3 + ⑤ Tm/Sol 执行（2026-09-03 续）
- **H3 电荷聚集合法性：45/45 PASS（100%）**——v12.3 条件化无非法电荷布局（per-protein 口径生成序列）。
  脚本 `code/tests/h3_charge_legality_v12_3.py`（PDBS 改 v12.3 9 单体），产物 `output/h3_protein_v12_3.json`。
- **Tm（TemBERTure）**：gen 54 arm（45 gen arm + 9 native_ref）+ uncond 9，后台跑（input 用 --dirs-file 真实目录，
  glob 不 follow symlink 的坑已避开）。预计 ~1h。
- **Sol（Protein-Sol）**：63 文件全完成（45 gen arm + 9 native_ref + 9 uncond）。
- **H1/ESMFold（③ V2）**：GPU6 跑 7/45，长蛋白每 arm ~10-20min，预计总 6-8h → V3-TM/V4-stats/V5-PROPKA 自动。

### ⑤ Tm/Sol 完成（S2 0/45 无恶化，2026-09-03）
- Protein-Sol 63/63、TemBERTure 63/63（45 gen arm + 9 native_ref + 9 uncond）全齐。
- 汇总 `code/tests/v12_3_tm_sol_summarize.py` → `output/tm_sol_v12_3/tm_sol_summary.json`。
- **S2 判据（vs 无条件基线，ΔTm<-5 或 Δ%sol<-10 算恶化）：0/45**——与 v12.2 (0/50) 一致，电荷条件化未牺牲热稳定/溶解性。
- 长蛋白：1A65 native Tm 67.2(Δu+6.8)/%sol 49.9(+6.9)；1BJ4 71.2(-0.5)/43.7(-0.5)；13BB 69.7(-2.5)/46.5(+1.7)；1CDG 62.0(-0.3)/37.9(-0.1)——全部无恶化。
- ⚠️ 踩坑：TemBERTure glob 不 follow symlink → 用 --dirs-file 真实目录；Tm seqs symlink 需绝对路径（相对少一级会断）。

### v12.3 小样本现场标定（用户方法论决策后，2026-09-03）
- 用户剔除 per-protein 表内校准（乐观口径），保留 global + 小样本现场标定两口径（v12.2 基准 global 44% / 小样本 74%）。
- `code/tests/build_calibration_small.py`（v12.3 ckpt，manifest v12_3，n_per=10，5 target native±[8,4,0,4,8] = 50 条/蛋白）
  → `output/charge_calibration_v12_3_small.json`（9 蛋白 per slope，global 兜底 = charge_calibration_v12_3 global 1.471）。
- **2 个 unreliable（LOOCV>3，回退 global）**：2FEO（LOOCV 3.47）、1A65（3.45）。
- 小样本局部 slope（vs 全靶 diag slope）：1A65 1.09 vs 1.43（局部 native 区更线性！）、1CGE 0.94、1AZM 0.95；
  1BJ4 2.07、13BB 2.11、1CDG 2.49（长蛋白高 slope 但 LOOCV 稳）。
- 泛化重采样（小样本表 auto）GPU3 → `output/generalization_v12_3_calib_small/`（9×5×n30，仅电荷 H2，不重跑 ESMFold）。
