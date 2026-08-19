# ConfuMPNN 自训编码器权重（GitHub Release preview1.0.0）

ConfuMPNN 微调训练的**最终交付权重**（被 `.gitignore` 排除，不在仓库内，通过 Release 分发）。

## 附件

| 文件 | 大小 | backbone | 适用场景 | 电荷边界 |
|------|------|----------|---------|---------|
| `condition_encoder_v7_last.pt` | 296K | MoMPNN（纯骨架）| 无配体 / 小蛋白（L≤300）| 负电强（−8 可靠）、正电弱（+8 过冲）|
| `condition_encoder_v9_epoch030.pt` | 887K | LigandMPNN（配体上下文）| 配体口袋 / 大蛋白 | 正电强（+8 可靠）、负电弱（−8 欠冲）|

> 完整电荷边界与根因：`analysis/report/2026-08-18_model_charge_limits.md` §8。

## 校验

```bash
sha256sum -c SHA256SUMS.txt
```

| 文件 | SHA256 |
|------|--------|
| condition_encoder_v7_last.pt | `58aca0f5cf3887ca41d5d2760ed7f775622f45d6f6ab5c0dd469dc3b23d4e4a1` |
| condition_encoder_v9_epoch030.pt | `8ab1548fe314a2c7e8d774f219a25ef179c7538f42983d6d66ced4c6e3b2a578` |

## 用法

```bash
gh release download preview1.0.0 --pattern "condition_encoder*.pt" -D code/weights/
sha256sum -c code/weights/SHA256SUMS.txt

# v7（无配体/小蛋白）
python run_guided.py --pdb input/1BC8.pdb --pH 7.4 --target_charge 0 \
  --cond_encoder code/weights/condition_encoder_v7_last.pt

# v9（配体模式）
python run_guided.py --pdb data/validation_pdbs/1AZM.pdb --pH 7.4 --target_charge 0 \
  --cond_encoder code/weights/condition_encoder_v9_epoch030.pt \
  --weights LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt
```

## 训练来源

- v7：`output/finetune_v7/condition_encoder_last.pt`（30 epoch，CATH 7,886 域 + 外部碱性，课程学习 2.0→8.0）
- v9：`output/finetune_ligand_v9/finetune_epoch030.pt`（30 epoch，配体复合物 4,972 × 8pH）
