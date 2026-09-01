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
