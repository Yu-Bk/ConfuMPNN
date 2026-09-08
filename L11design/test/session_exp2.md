# Session — L11 Exp2 多目标电荷可设计性（2026-09-09）

> 归属：`L11design/`。目标：pH7.4 下 target{+6.87(native), +8, +10, +12} × 模式{protein,ligand} × n=300 = 2400 条。
> 固定链 I 残基 `I3 I5 I9 I34 I35 I89 I124 I131 I134 I135`（native）。
> 模型：蛋白 MoMPNN+v12.2（global 校准）；配体 LigandMPNN(atom25)+v14（global 校准）。

## 输入/校验
- `input/L11.pdb` 链 I（141 aa，native_charge@7.4 = **+6.874**，计划 +6.87 ✓）。固定位 native：I3K I5Q I9K I34I I35M I89S I124M I131T I134S I135M。
- `input/L11_RNA.pdb` = 链 I 蛋白 + 链 A RNA(107 残基)；parse_PDB 只取蛋白 141 aa，RNA 作为配体上下文。
- 参考骨架 `input/L11_chainI.pdb`（exp2_prep.py 提取，供 TM/RMSD）。

## 协议
- `run_guided.py --pH 7.4 --target_charge <t> --fixed_residues "I3 ... I135" --num_samples 300 --calibrate global`
  - protein: `--weights MoMPNN/...mompnn_temberture_tm_esm_6_4_4_b01.ckpt --cond_encoder output/finetune_v12_2/finetune_epoch030.pt --calibration_file output/charge_calibration_v12_2.json`
  - ligand : `--weights LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt --cond_encoder output/finetune_ligand_v14_rna/finetune_epoch050.pt --calibration_file output/charge_calibration_v14_ligand_clean.json`
- 逐条：net_charge(seq,7.4) → dev/达标；固定位校验；ESMFold 回折(pLDDT)；TM&RMSD vs 链I；Tm(TemBERTure)；Sol(Protein-Sol)。

## GPU/CPU 决策（重要）
- 01:1x 检查：GPU0 显存占满(空闲827MiB) 但 util 0%；GPU1-7 util 97-99%（他人任务），**无空闲卡**。
- 采样改为 **CPU 并行**：OMP_NUM_THREADS=12（默认192线程反而 ~13x 更慢），8 组同时 → **3.1 min 全部完成**（远超预期，0.9 s/条）。
- ESMFold 仍需 GPU → fold_scheduler 后台轮询空闲卡（util<12% & free≥20000MiB）。

## 采样结果（charge@7.4，目标 vs 实测均值 ± std）
| 组 | 目标 | 校准后 target_eff | 实测均值 | std |
|---|---|---|---|---|
| protein_q6.87 | +6.87 | 6.34 | +9.36 | 4.62 |
| protein_q8 | +8.0 | 7.05 | +10.31 | 4.91 |
| protein_q10 | +10.0 | 8.32 | +13.64 | 5.03 |
| protein_q12 | +12.0 | 9.59 | +15.56 | 5.02 |
| ligand_q6.87 | +6.87 | 5.45 | +10.02 | 3.93 |
| ligand_q8 | +8.0 | 6.20 | +11.66 | 4.46 |
| ligand_q10 | +10.0 | 7.55 | +13.69 | 4.35 |
| ligand_q12 | +12.0 | 8.89 | +17.00 | 4.62 |
- 固定位全对（0/2400 失配）；整体回收 ~27%，非固定回收 ~22%。
- ⚠️ global 校准下**两模式普遍过冲**（低 target 组 dev≈+2.5~3.7，高 target 组更大）→ 是本次"可设计性"待报告结论（表外蛋白 global 校准不足）。

## 打分管线
- `exp2_cpu_scores.sh`：TemBERTure Tm + Protein-Sol（CPU，后台）。
- `exp2_fold_scheduler.py`：ESMFold（GPU，等空卡）。
- `exp2_tm_run.sh`：TM/RMSD（US-align，CPU，fold 后）。
- `exp2_analyze.py`：聚合 → `test/report_exp2.md`。

## 产物
- seqs/csv → `output/exp2/<mode>_q<t>/`；折叠+指标 → `data/exp2/`；报告 `test/report_exp2.md`。

## 执行结果（完成）
- **采样**：CPU 并行 8 组 × 300 全部完成（~3 min；OMP_NUM_THREADS=12 是关键，默认 192 线程反慢 ~13×）。
- **Tm/Sol**：TemBERTure + Protein-Sol 全部 2400+native 完成。
- **ESMFold**：GPU 分窗完成全部 9 目录（native+8 组；GPU1/GPU7 时隙），pLDDT 全部。
- **TM/RMSD**：US-align 全部完成（native TM 0.794 对照）。
- 报告 `test/report_exp2.md`；逐序列 `data/exp2/*/metrics.csv`；组级 `data/exp2/_per_group_summary.csv`。

### 最终结果速览（target 6.87→12）
| mode | 电荷均值范围 | dev均值 | 达标率(|dev|≤2) | pLDDT | TM | Tm(°C) | Sol(%) |
|---|---|---|---|---|---|---|---|
| protein | +9.36 → +15.56 | +2.3~+3.6 | 29→23% | ~55 | 0.66 | 55→59 | 79→82 |
| ligand | +10.02 → +17.00 | +3.2~+5.0 | 28→18% | ~54-56 | 0.60-0.63 | ~54 | 94-95 |
- 响应斜率 蛋白 1.27 / 配体 1.32（>1 → 过冲）；固定位 2400/2400 全对；回收 ~27%。

### 结论一句话
方向可控、单调、界面固定位保留、结构未崩（TM 0.6+，pLDDT~55）；但表外 L11 用 global 校准每 target 过冲 2-5 个电荷 → 命中率仅 18-29%，精确控制需现场小样本标定。
