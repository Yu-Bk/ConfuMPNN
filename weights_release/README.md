# ConfuMPNN 自训编码器权重（GitHub Release 分发）

ConfuMPNN 微调训练的**交付权重**（被 `.gitignore` 排除，不入 git，经 GitHub Release / NAS 分发；唯一归档规则见 `docs/MIGRATION_GIT_POLICY.md`）。

## 附件（2026-09-05：补入当前交付 v12.2/v12.3/v14）

| 文件 | 大小 | backbone | 适用场景 | 说明 |
|------|------|----------|---------|------|
| `condition_encoder_v12_2_last.pt` | 296K | MoMPNN | 无配体/小蛋白 | **★蛋白当前交付**（校准后 slope 1.00、H2 72%+小样本 74%） |
| `condition_encoder_v12_3_last.pt` | 296K | MoMPNN | 同上+长/深负蛋白 | 长蛋白/深负外推增强（覆盖内略退，按需选用） |
| `condition_encoder_v14_ligand_epoch050.pt` | 887K | LigandMPNN | 配体口袋/大蛋白/RNA-DNA | **★配体当前交付**（clean 链 H2 90%；组成删减 0.43-0.69× 为已知局限） |
| `condition_encoder_v7_last.pt` | 296K | MoMPNN | （历史） | 早期蛋白版 |
| `condition_encoder_v9_epoch030.pt` | 887K | LigandMPNN | （历史） | 早期配体版 |

> ⚠️ **用前先校准**（三口径：per-protein 表内 / 表外小样本现场标定 ~50 条 / global 40-44%），见 `analysis/report/2026-08-31_v12_2_summary.md`。删减局限见 `analysis/report/2026-09-04_paper_subconclusions.md`。

## 校验
```bash
sha256sum -c SHA256SUMS.txt
```
SHA 见 `SHA256SUMS.txt`（同目录）。

## 用法（仓库根）
```bash
gh release download <tag> --pattern "condition_encoder*.pt" -D weights_release/
sha256sum -c weights_release/SHA256SUMS.txt

# 蛋白模式（v12.2）
python code/run_guided.py --pdb code/input/1BC8.pdb --pH 7.4 --target_charge 0 \
  --cond_encoder weights_release/condition_encoder_v12_2_last.pt \
  --weights MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt

# 配体模式（v14）
python code/run_guided.py --pdb data/validation_pdbs/1AZM.pdb --pH 7.4 --target_charge 0 \
  --cond_encoder weights_release/condition_encoder_v14_ligand_epoch050.pt \
  --weights LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt
```

## 训练来源
- v12.2：`output/finetune_v12_2/`（MoMPNN，30ep，λ_target 表面电荷锚等）
- v12.3：`output/finetune_v12_3/`（MoMPNN，40ep，+455 长 CATH 域）
- v14：`output/finetune_ligand_v14_rna/finetune_epoch050.pt`（LigandMPNN，50ep，RNA/DNA 414 + A1 全局）
- v7/v9：见历史版（`output/finetune_v7|ligand_v9/`）
