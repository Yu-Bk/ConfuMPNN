# v12.2 配体迁移（#147）— 执行笔记（2026-08-31 启动）

## 目标
把 v12.2 电荷控制方法迁移到 LigandMPNN 配体模式：恢复配体模式电荷控制（v9 时代 1MBN dev 14.05 的修复），并加 v12 组成监督 + λ_target。

## 训练命令（已启动 PID 3404131，GPU5）
```bash
PYTHONPATH=code nohup ~/miniconda3/envs/confumpnn/bin/python code/train_finetune.py \
  --ligand --weights LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt \
  --labels data/ligand_train/labels.npz --dompdb data/ligand_train/all_pdb \
  --out_dir output/finetune_ligand_v12_2 --device cuda:5 --epochs 30 \
  --decouple_absolute --decouple_abs_lo=-35 --decouple_abs_hi=20 \
  --v12_supervision --frac_floor 0.5 --gravy_margin 0.4 --lambda_v12 0.2 \
  --lambda_target 0.2 --sasa_threshold 0.25 \
  --ph_aware_filter --structure_boost 1.5 \
  --charge_temp 0.5 --perturb_prob 0.3 --placeholder_prob 0.15 \
  --lambda_c 0.5 --lambda_kl 0.05 --lambda_keep 0.5 \
  --log_file log/v12_2_ligand_train.log
```
- backbone 确认：model_type=ligand_mpnn, 2.62M 参数（冒烟打印"MoMPNN"是文案 bug，实际正确）
- 冒烟通过：epoch1 ce=1.34 charge=7.9 keep=0.51 无 NaN
- 数据：4957 域 × 8 pH，SASA 预解析预计 ~1.2h

## 迁移前置检查（全部通过）
- ✅ symlink 0 dangling（data/ligand_train/all_pdb 4972 文件）
- ✅ SASA 冒烟：1JCG/1OE7/1PM7 无 NaN
- ✅ 路径解析：pdb_path = dompdb/did（did=文件名）
- ✅ 负值参数用 = 形式（--decouple_abs_lo=-35）

## 近几天实验记录教训（迁移必须规避）
| 教训 | 来源 | 处理 |
|---|---|---|
| 1GTV 残缺 PDB → SASA NaN → 全权重 NaN | 08-27 | SASA 冒烟通过；训练监控检测 NaN |
| dangling symlink → 训练崩溃 | 08-27 | 已清理 0 dangling |
| argparse 负值 | 08-31 | 用 = 形式 |
| conda run 并发锁死 | 08-16 | 直接 env python |
| 校准信息泄漏 | 08-31 | 泛化 per-protein + 补无泄露/小样本口径 |
| 1A65 响应弯曲→小样本失效 | 08-31 | LOOCV + 回退 global（已实现）|
| propka3 不在 PATH | 08-30 | export PATH |
| TemBERTure 联网失败 | 08-29 | HF_HUB_OFFLINE=1 |

## 后续阶段
1. **训练完成判断**：epoch 30/30 + 无 NaN + checkpoint 完整（monitor 文件 log/v12_2_ligand_train.monitor）
2. **配体响应诊断**：`index/v10_repair/v10_diag_response_curve.py --backbone ligand_mpnn`（17 蛋白，valid 区内 slope 判据 [0.9,1.15]）
3. **建配体校准表**：`index/v10_repair/build_calibration.py` → charge_calibration_v12_2_ligand.json
4. **泛化验证**：`validate_generalization.py --mode both --calibrate auto --calibration_file 配体表`，10 蛋白×5臂×n30，H1/H2/H3/H4/组成/GRAVY/Tm/Sol
5. **迁移复验**：1MBN/4DFR/1FQG（核心：1MBN dev 14.05 → ≤2）
6. 汇总报告 + git push

## 判据（对齐 v12.2）
- 配体 valid 区内 slope 校准后 ∈ [0.9,1.15]
- H2 dev ≤ 2.0、H1 TM≥0.7、H4 |Q_phys−target|≤2.0
- Tm/Sol 相对无条件基线无恶化（S2）
- 1MBN dev ≤ 2.0（v9 核心修复目标）

## 训练速度诊断（2026-08-31 15:00）
- **单 epoch 实测**：500 域 3.2min → 单域 0.38s → 4956 域 ~31min/epoch（含首 epoch 预解析额外开销 ~40min）
- **30 epoch 预计 ~15-16h**（比 v10 配体 5.5min/epoch 慢 5.7 倍）——v12 监督（组成/GRAVY/λ_target 逐样本前向）+ 配体大蛋白（L≤500 + 16 原子上下文）固有开销
- **不是 bug**：50域 7s / 200域 38s / 500域 3.2min 线性扩展正常；4956 域 23GB 缓存复用跨 epoch
- **重启**：首次训练（PID 3404131）epoch1 未完成即因过慢误判杀掉（无 checkpoint 损失）；重启 PID 3521281
- 监控：20min 间隔，log/v12_2_ligand_train.monitor；40min 后台检查 /tmp/ligand_check_40m3.out

---

## 执行记录（2026-09-01 续）

### ✅ 训练完成（PID 3521281）
- epoch 30/30（08:00:21），总耗时 **992.6min ≈ 16.5h**，无 NaN/OOM
- 末轮 loss：total 4.6196 / charge 3.5385 / cd self 3.204 / mild 3.109 / extreme 4.586
- checkpoint：`output/finetune_ligand_v12_2/`（epoch001–030 + condition_encoder_last.pt）
- 监控 `log/v12_2_ligand_monitor.sh` 每 30min 自查，4 类事件触发唤醒（训练期间误报 1 次：freesasa "Error: Radius" 被 grep `error:` 匹配，已收紧为只匹配 Traceback/RuntimeError/OOM/loss=nan）

### 🔧 配体响应诊断三个适配（无配体 CATH 域不可用）
1. **argparse 负值**：`--targets` 以负号开头需 `=` 形式（`--targets=-34,...`）——migration note 已有此教训，实操再踩，已用
2. **prody 无扩展名坑**：`data/cath/S40/dompdb/<name>`（无 .pdb）prody 2.4.1 按 mmCIF 解析失败（"mmCIF file contained no atoms"）。**正确路径 = `data/cath/S40/dompdb_pdb/<name>.pdb`**（带 .pdb 副本目录，v12.2 校准日志佐证）
3. **配体模式必须有配体原子**：CATH 域纯蛋白链无配体 → `get_nearest_neighbours` 对空配体张量 `L2_AB_nn[:,0]` IndexError 崩溃（validate_generalization.py 之前也踩过此坑）。**trainish 侧改用配体训练域 `data/ligand_train/all_pdb/`**（101M/102L/103L/105M/106M/107L/111M，均含配体 135-209 HETATM）；valid 10 蛋白全部带配体（HEM/AZM/GDP/UMP/DC/CA+ZN/CU/NAG/PLP）可直接用
- 诊断命令（PID 后台 `beutrifrm`）：`v10_diag_response_curve.py --backbone ligand_mpnn --cond_encoder finetune_ligand_v12_2/finetune_epoch030.pt --weights ligandmpnn_v_32_010_25.pt --pdb-list log/v12_2_ligand_trainish.list --manifest validation_manifest.json --targets=-34,...,18 --include_native --n 20` → `output/v12_2_ligand_diag_response.json`

### ⏳ 待办
- 诊断完成后：valid 区内 slope 判据 [0.9,1.15] → 建配体校准表 → 泛化验证 → 迁移复验（1MBN dev≤2）

### ✅ 适配已固化为共享模块 `index/v10_repair/_adapters.py`（2026-09-01，用户要求）
三个坑已做成代码修复，以后配体诊断/校准不再报错，不影响 protein_mpnn（mompnn）：
| 坑 | 修复函数 | 效果 |
|---|---|---|
| argparse 负值（`--targets -34,...` 被误判） | `fix_negative_targets()` | parse_args 前自动转 `--targets=-34,...` |
| prody 无扩展名按 mmCIF 误判 | `resolve_pdb_path()` | 无后缀自动补 .pdb（含 dompdb→dompdb_pdb 目录映射）|
| 配体模式无配体原子 IndexError | `safe_featurize()` | 捕获 IndexError/KeyError 跳过并提示，不崩溃 |
- 已验证：单元测试（3 helper）+ 无配体蛋白配体模式返回 None + protein_mpnn 正常
- 接入点：`v10_diag_response_curve.py`（import _adapters；parse_args 用 fix_negative_targets；pdbs 收集用 resolve_pdb_path；featurize 用 safe_featurize）
- 后续 build_calibration*.py 跑配体时接入同样 helper

### ⚠️ 配体组成分析发现删减捷径（2026-09-01，未根治）
- 泛化验证 native 臂 D/K vs native 倍率：1C6O 1.01 / 1A65 1.32 健康，**其余 8 蛋白 0.53-0.65 系统性删减**
  （1AZM 0.53 / 2FEO 0.54 / 5CQH 0.54 / 1BJ4 0.56 / 1AXW 0.59 / 1AG0 0.60 / 1AS2 0.64 / 1CGE 0.65）
- 对比 mompnn v12.2（0.59-1.45，无删减/过度添加捷径）：**配体模式带电残基删减未根治**
- 解释：净电荷可由更少残基实现（正负对消删减 + 微调），故 H2 72% 达标但组成不健康
- 与 v11 删减捷径（无差别删带电残基）同型；λ_target/frac_floor 在配体模式未完全约束
- 产物：output/v12_2_ligand_comp.json；脚本 code/tests/ligand_v9/compare_comp_ligand.py

### 🔬 删减的空间分布：配体上下文定向删减口袋带电残基（2026-09-01，用户假设验证）
**配体相互作用位点（口袋，配体 8Å 内）的带电残基被删得更狠**（7/8 删减蛋白）：
| 蛋白 | 配体口袋倍率 | 配体全序列 | mompnn 口袋 | mompnn 全序列 |
|------|:---:|:---:|:---:|:---:|
| 2FEO | **0.23**(13→3.0) | 0.54 | 0.47 | 0.73 |
| 1AS2 | **0.39**(11→4.3) | 0.64 | 0.53 | 0.60 |
| 1BJ4 | **0.40**(3→1.2) | 0.56 | 0.52 | 0.62 |
| 1AXW | **0.41**(26→10.5) | 0.59 | 0.68 | 0.64 |
| 1CGE | 0.54 | 0.65 | 0.65 | 0.76 |
| 1AG0 | 0.50 | 0.60 | 0.48 | 0.59 |
| 5CQH | 0.46 | 0.54 | 0.43 | 0.60 |
| 1AZM | 0.57 | 0.53 | 0.87 | 0.49 |

**结论**：
1. **配体上下文定向删减口袋带电残基**：mompnn（无配体）删减全局均匀（口袋≈非口袋），配体模式口袋系统性更狠 → 配体上下文把口袋"理解"为疏水区（LigandMPNN 配体蛋白训练集口袋偏疏水），带电残基从配体结合位点被排除
2. **配体相互作用位点被破坏**：2FEO DNA 结合口袋带电残基 13→3，配体氢键/盐桥残基丢失
3. 这是**配体模式特有**（mompnn 无此定向删减），v9 时代通病，v12.2 迁移未根治，且 λ_target/frac_floor 按"整体表面"打分被模型的"口袋疏水"策略绕过

### 🔬 监督逃逸机制坐实（2026-09-01，SASA 分析）
- 配体口袋残基 **60-75% frac_sasa < 0.25**（不算"表面"）→ surface_composition_loss 不覆盖深部口袋
- **深部口袋带电残基删减最狠**：2FEO 10→1.5(0.15)、1AS2 4→1.7(0.42)、1CGE 7→2.9(0.41)、5CQH 4→2.3(0.57)
- 机制 = **监督逃逸（深部口袋 frac_sasa<0.25 不受惩罚）× 配体先验（LigandMPNN 学"口袋疏水"，知道口袋位置定向删）**
- mompnn 同样有逃逸但无配体先验（不知道口袋位置）→ 删减全局均匀
- 修复方向（决策点）：v12_losses 加"口袋带电残基保护"监督（配体 8Å 内带电残基 ≥ native×frac_floor）

### 🔬 原始 LigandMPNN 测试：删减是微调放大，非模型固有通病（2026-09-01）
**原始 LigandMPNN（未微调，无条件生成）口袋带电残基：0.78-0.90（温和删 10-22%），全序列 0.87-1.30（不删甚至增加！）**
| 蛋白 | 原始口袋 | v12.2配体口袋 | 原始全带 | v12.2全带 |
|------|:---:|:---:|:---:|:---:|
| 2FEO | 0.78 | **0.23** | 1.19 | 0.54 |
| 1AS2 | 0.90 | **0.39** | 1.24 | 0.64 |
| 1AXW | 0.88 | **0.41** | 0.92 | 0.59 |
| 1BJ4 | 0.90 | **0.40** | 1.27 | 0.56 |
| 1CGE | 0.90 | 0.54 | 0.87 | 0.65 |
| 1C6O | 1.69 | 1.57 | 1.30 | 1.01 |

**结论（修正之前"v9 通病"判断）**：
1. 原始 LigandMPNN 只有**温和**疏水口袋倾向（删 10-22%），**无全局删减**（全序列带电残基甚至偏好增加）→ 系统性删减不是模型固有
2. **v12 微调放大**：口袋删减 0.78-0.90→0.23-0.54（放大 2-3 倍），并新增全局删减（→0.53-0.65）
3. 机制：v12 训练教会模型"用删带电残基来匹配电荷 target"这一策略；配体模式因监督逃逸（深部口袋 frac_sasa<0.25）+ 配体疏水先验，把删减集中/放大在口袋。mompnn 同样学"删残基调电荷"但表面监督无逃逸（删减均匀被拦）→ 无系统性删减
4. 修复方向强化：需"口袋带电残基保护"监督 + 核心区组成约束（堵监督逃逸）

### 🎯 设计决策：删减捷径修复方向收敛（2026-09-01 用户定调）
**用户对"修复方向"的关键修正——不做复杂学习约束，用人工 fix 覆盖确定性知识**：
1. **核心不整体 fix**：本模型适用小蛋白、可设计空间小；核心关键残基/结构域=设计前人工调研 fix（不是模型学）
2. **q_core 不退役**：不能保证设计前找全所有关键核心（尤其人工设计新蛋白）；MoMPNN 已训好，不做多余改动
3. **强相互作用筛选 = 人工结构分析**（PLIP/结构调研），模型侧只做**口袋范围定义防删减过度**
4. **工具 = 额外补充脚本**，单独目录 `code/tools/pocket_protect/`，不混训练脚本

**关键问答（用户问：只 fix 部分口袋残基，加/删电荷会改 pocket 吗？）**：
- **会**。分两种情况：
  - 口袋"表面"残基（frac_sasa≥0.25，约 25-40%）：参与电荷调节——加电荷臂可被换带电残基、删电荷臂被温和删
  - 口袋"深部"残基（frac_sasa<0.25，约 60-75%）：**完全不受监督，成对删逃逸，删得最狠**（2FEO 深部 10→1.5）
- **结论**：fix 部分残基 ≠ 口袋被保护。删减捷径清的是"深部口袋其他带电残基"。口袋范围 fix 列表作兜底仍必要。

### 🛠️ 口袋范围定义工具落地（commit d4c239b）
**`code/tools/pocket_protect/define_pocket.py`**——设计前定义配体口袋范围，输出删减风险预警：
- 输入：带配体 PDB（如 `data/validation_pdbs/2FEO.pdb`）
- 输出（`output/pocket_protect/<name>/`）：
  - `pocket_table.txt/.json`：逐残基分类（Cα-配体距离 / 骨架接触 / frac_sasa / 表面或深部 / 是否带电 / 保护建议）
  - `pocket_fix.txt`：**深部带电残基建议 fix 列表**，可直接 `--fixed_residues "$(cat 文件)"`
  - `contact_residues.txt`：强接触残基（骨架原子近似，人工 PLIP 交叉验证参考）
- 分级建议：人工fix(强接触<4.5Å) / 建议fix(深部带电) / 可选fix(表面带电) / 无需
- **强接触定义**（`--contact-cutoff` 默认 4.5Å）：残基**任一骨架原子（N/Cα/C/O）到配体任一原子最近距离 <4.5Å** → 标强接触。4.5Å=重原子非键接触上界（氢键 2.7-3.3 / 盐桥 3-4 / 范德华 3.5-4.5）。强接触 ⊆ 口袋（Cα 是骨架原子之一）。⚠️ 局限：① 只含骨架原子不含侧链（parse_PDB 无侧链）→ 侧链介导的相互作用（多数盐桥/氢键）**漏检**；② 无方向性判据 → 是"距离接近"粗筛，真实功能接触需 PLIP/人工确认
- **生成时保护，不动模型**：复用 run_guided 原生 `--fixed_residues`（chain_mask=0 强制 native）
- 10 蛋白汇总：
  | 蛋白 | L | 口袋 | 深部带电(建议fix) | 表面带电(可选) |
  |------|-----|------|------|------|
  | 1C6O | 177 | 86 | 4 | 4 |
  | 1AZM | 258 | 25 | 1 | 0 |
  | 1AS2 | 312 | 40 | 1 | 4 |
  | 1AXW | 528 | 119 | 10 | 13 |
  | 2FEO | 221 | 40 | 8 | 2 |
  | 5CQH | 183 | 55 | 3 | 8 |
  | 1CGE | 162 | 53 | 3 | 5 |
  | 1AG0 | 256 | 44 | 2 | 2 |
  | 1A65 | 504 | 49 | 1 | 0 |
  | 1BJ4 | 470 | 29 | 1 | 1 |
- **实证（交叉验证）**：34 个建议 fix 位点中 **27 个实际被删（命中率 79%）**——深部带电残基≈删减捷径实际删减位点（2FEO 8/7、1AXW 10/9、1AZM/1AS2/1AG0/1A65/1BJ4 全中、1C6O 4/2、1CGE 3/1）
- **踩坑修复**：① 2FEO 蛋白残基号从 3 起（R_idx 非 1 基，不能用 `int(resname)−1` 反查索引，用 R_idx 映射）；② 1AS2 parse_PDB 跳过 sasa 认为标准的 236-238（原子不全）→ resid 交集对齐兜底（丢弃 sasa 多出残基、保持 PDB 顺序）

**后续可选**：把 `pocket_fix.txt` 与用户人工 fix 列表合并喂给 `run_guided --fixed_residues`，在验证链复跑中对比"有/无口袋 fix"的组成改善。

### 🧪 口袋 fix 实测（2026-09-01，commit c059e27）
**实测"加口袋 fix"在生成流程的效果**（validate_generalization.py 加可选 `--fixed_residues`，2FEO/1AXW/1C6O native 臂 n30，对比无 fix 泛化）：
| 蛋白 | 口径 | 全序列倍率 | 口袋删减 | charge dev | rec_pkt | 深部fix位点 |
|------|------|:---:|:---:|:---:|:---:|:---:|
| 1C6O | 无fix | 1.01 | +57%增 | 2.1 | 0.569 | 4→2.0 (0.5×) |
| 1C6O | 有fix | 1.13 | +81%增 | 0.1 | 0.602 | 4→4.0 (1.0×) |
| 1AXW | 无fix | 0.59 | −59%删 | 1.7 | 0.441 | 10→2.0 (0.2×) |
| 1AXW | 有fix | 0.66 | −26%删 | 0.7 | 0.516 | 10→10.0 (1.0×) |
| 2FEO | 无fix | 0.54 | −77%删 | 1.4 | 0.337 | 10→1.4 (0.14×) |
| 2FEO | 有fix | 0.68 | −12%删 | ⚠️4.0 | 0.551 | 10→10.0 (1.0×) |

**结论**：
1. **fix 100% 保住深部带电残基**（三蛋白深部位点全 1.0×）——直接目标完全达成
2. **口袋删减大幅缓解**（2FEO −77%→−12%、1AXW −59%→−26%），rec_pkt 提升
3. **但删减转移**：全序列仍 0.66-0.68×——fix 深部后模型改删**未覆盖区域**（表面带电 + 口袋外）→ fix=保功能残基，**非根治全局删减**（需训练侧组成监督）
4. **电荷副作用（2FEO）**：dev 1.4→4.0 恶化=fix 缩窄可调空间 + **校准表（无fix拟合）失配** → fix 口袋后应重新小样本标定
5. 1C6O 口袋带电增加（+57%增）→ 配体先验方向与配体类型相关（小分子 vs 核酸）

**工具完善**：① 修 level 判定——深部带电+强接触残基（2FEO A18/A132）曾不在默认 fix 列表，深部带电优先后 2FEO fix 8→10；② 对比脚本显示修正（>1 标"+%增"）；③ validate_generalization 加 `--fixed_residues`（默认 None 不影响 mompnn）。报告 `analysis/report/2026-09-01_pocket_fix_test.md`。
