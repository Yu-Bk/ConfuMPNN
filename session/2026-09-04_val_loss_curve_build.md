# per-epoch 前向验证损失回放——构建记录（2026-09-04）

## 目的与产物
为已训三版（v12.2 蛋白 / v12.3 蛋白 / v14 配体）实现 per-epoch **确定性 no-grad 前向**
验证损失回放（train-loss vs val-loss 论文曲线数据）。不采样、不生成、无扰动、无 placeholder。

改动文件：
- 新增 `code/tests/val_loss_curve.py`（CLI 工具，回放主体）
- 新增 `code/tests/val_replay_configs.md`（三版训练配置 + self-arm 总损失装配判定依据 + 口径验证）
- 本 session 记录。**未改 `train_finetune.py`；未动任何权重/数据；未 push。**

## 口径设计
- 验证集每域用 npz 那 8 个 (pH, charge) 臂 = native 在 8 pH 自身电荷（自洽锚）→ 全为 self 臂。
- 逐域前向 = 训练同款：`featurize → build_domain(seed+i) → backbone.encode → 一次性 uncond logits/
  seq_anchor → enc(cond 8×[pH,charge]) → inject_prompt → decoder_forward`。
- 复用 `train_finetune.decoder_forward/build_domain/kl_anchor_loss/load_backbone` + src 同款损失，
  保证与训练 self-arm 分支数值一致。

### self-arm 总损失装配判定（train_finetune.py 行号）
| 项 | 行号 | self 臂 | 权重 |
|---|---|---|---|
| CE | 689-691 | 是 | 1.0 |
| charge_deviation | 696-714 | 是 | λ_c=0.5 |
| KL | 717-718 | 是 | λ_kl=0.05 |
| SeqKeep | 722-729 | 是（只在非扰动臂→全算） | λ_keep=0.5 |
| v10 B add | 735-750 | **否**（只扰动臂） | — |
| v10 C struct | 755-771 | 是（boost=1.0） | 0.05 硬编码 |
| v12 comp+gravy | 776-799 | 是 | λ_v12=0.2 |
| v12_ct | 804-830 | 是 | λ_target=0.2 |
| A1 pocket | 838-860 | 是 | λ_pocket（按版） |
| total | 862-872 | 拼装 | — |

## 三版配置表（详细见 code/tests/val_replay_configs.md）
| 版本 | 命令来源(log) | mode/backbone | λ_c/kl/keep | temp | v12(floor/grav/λv12/λtgt) | ph | pocket |
|---|---|---|---|---|---|---|---|
| v12_2 | v12_2_train_mompnn | protein/MoMPNN | .5/.05/.5 | 0.5 | .5/.4/.2/.2 | ✓boost1.5 | keep **λ0**（原版无） |
| v12_3 | v12_3_train_mompnn | protein/MoMPNN | .5/.05/.5 | 0.5 | 同 | ✓ | keep 0.7/1.3/λ0.2 |
| v14 | v14_ligand_train_stdout | ligand/LigandMPNN(atom25) | .5/.05/.5 | 0.5 | 同 | ✓ | global 0.8/1.3/λ0.3 |

protein 无配体 → pocket_mask 全 0 → A1 项≈0；v12_ct 监督 mask = surface（与训练一致）。

## 冒烟结果（CPU，均短）
- v12_2 蛋白 val 2 域：ep30 ce=2.245 cd=3.752 rec=0.300 total=5.848；ep1/10/30 cd 7.63→6.03→3.75（曲线有意义、无 NaN）。
- v12_3 蛋白 val 2 域：ep40 ce=2.243 cd=3.894 rec=0.298 total=5.974。
- v14 配体 val 2 域：ep50 ce=2.494 cd=2.124 rec=0.180 total=5.023（pocket=0.106 非零；v12_comp/gravy/struct≈0 符合 self 预期）。
- **口径实证**：v12.2 训练前 40 域 ep30 回放 → ce=1.935 / self-cd=2.305 / kl=0.196，
  对照训练 log ep30（全训练集）ce=1.949 / `[cd self] 2.233` / kl=0.201 → 吻合（40/6710 采样噪声级）→ 判定口径一致。

## 每版全量运行命令（GPU 空闲后调度；先用 nvidia-smi 确认）
```
# v12.2 蛋白（hold-out 1176 + supp 23）
python code/tests/val_loss_curve.py --tag v12_2 --ckpt_dir output/finetune_v12_2 \
  --start_epoch 1 --end_epoch 30 --epoch_step 1 \
  --labels data/cath/labels_holdout_train.npz --dompdb data/cath/S40/dompdb_pdb \
  --supp_labels data/cath/labels_v12_3_valsupp.npz --supp_dompdb data/cath/S40/dompdb_valsupp \
  --device cuda:6 --out output/val_loss_curve_v12_2.json

# v12.3 蛋白（同验证集）
python code/tests/val_loss_curve.py --tag v12_3 --ckpt_dir output/finetune_v12_3 \
  --start_epoch 1 --end_epoch 40 --epoch_step 1 \
  --labels data/cath/labels_holdout_train.npz --dompdb data/cath/S40/dompdb_pdb \
  --supp_labels data/cath/labels_v12_3_valsupp.npz --supp_dompdb data/cath/S40/dompdb_valsupp \
  --device cuda:6 --out output/val_loss_curve_v12_3.json

# v14 配体（805 域）
python code/tests/val_loss_curve.py --tag v14_ligand --ckpt_dir output/finetune_ligand_v14_rna \
  --start_epoch 1 --end_epoch 50 --epoch_step 1 \
  --labels data/ligand_train/labels_v14_valset_805.npz --dompdb data/ligand_train/v14_valset_pdb \
  --device cuda:6 --out output/val_loss_curve_v14_ligand.json
```
- 预解析（parse+SASA+encode）只做一次，各 epoch 复用；建议首次用 `--n_dom` 试跑。
- 逐 epoch 输出 JSON：`{"epochs":{"ep":{ce,cd,rec,kl,keep,v12_comp,v12_gravy,v12_ct,struct,pocket,total,n_dom,n_arm}},"meta":{...}}`。
