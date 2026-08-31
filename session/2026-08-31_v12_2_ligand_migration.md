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
