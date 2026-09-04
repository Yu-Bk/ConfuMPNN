# per-epoch 前向验证损失回放——三版配置记录

> 配套脚本：`code/tests/val_loss_curve.py`。本文档记录三个已训版本
> （v12.2 蛋白 / v12.3 蛋白 / v14 配体）在训练时**实际使用的 flag/λ/温度/mode/backbone**，
> 以及回放 self-arm 总损失装配的判定依据（引用 `code/train_finetune.py` 行号）。
> 脚本 `VAL_TAGS` 内默认值与本文档一一对应，命令行可覆盖。

## 0. 判定总则：回放走"self-arm（非扰动、非占位）分支"

验证集每个结构域 npz 里的 8 个 (pH, charge) 臂 = 该 native 序列在 8 个 pH 下的自身电荷
（PROPKA 滴定，"自洽/原生锚"臂）→ 全部视为 **self 臂**（`mask_p=False, mask_ph=False`）。

对照 `train_finetune.py` 损失块（~655-900），self 臂上**实际施加**的项与权重：

| train_finetune 行号 | 项 | 是否计入 self 臂 | 权重 |
|---|---|---|---|
| 691 / 689-690 | CE(→native) | 是（全 batch） | 1.0 |
| 696-714 | charge_deviation（逐 arm，target=native@pH，temperature=charge_temp） | 是 | λ_c |
| 717-718 | KL 锚（条件化‖无条件，全 batch） | 是 | λ_kl |
| 722-729 | SeqKeep（**只在非扰动臂施加** → self 臂全施加） | 是 | λ_keep |
| 735-750 | v10 B 表面添加监督 L_add（**只在扰动臂施加**） | **否**（self 不计） | — |
| 755-771 | v10 C pH 自适应结构惩罚（self 臂 boost=1.0） | 是 | 0.05（代码硬编码） |
| 776-799 | v12 组成+GRAVY（只跳占位 → self 全算） | 是 | λ_v12 |
| 804-830 | v12.2 表面电荷目标（只跳占位 → self 全算） | 是 | λ_target |
| 838-860 | A1 双向计数（只跳占位 → self 全算） | 是 | λ_pocket（若 >0） |
| 862-872 | total 装配 | — | — |

**总损失装配（self 臂）**：
```
total = ce + λ_c·cd + λ_kl·kl + λ_keep·keep
      + [ph_aware_filter]  0.05·struct
      + [v12_supervision]  λ_v12·(v12_comp + v12_gravy)
      + [lambda_target>0]  λ_target·v12_ct
      + [pocket 且 λ_pocket>0]  λ_pocket·v12_pocket
```
`add_supervision`（v10 B）在 self 臂上不施加；三个目标版本训练时也未开该 flag → 回放不计。

> 说明：训练日志的 `keep` 是「按 B=8 平均、扰动臂置 0」的数值；回放全为 self 臂 → keep
> 逐臂全算，量级会高于训练日志同字段（这是 self 口径的固有差异，非错误）。cd 同理与
> 日志 `[cd self=…]` 分组口径一致。

---

## 1. v12.2 蛋白本体（MoMPNN）

- 训练命令权威来源：`log/v12_2_train_mompnn.log` / `.stdout`（2026-08-31 训练，325.8min）。
- 数据：`data/cath/labels_v12_2_train.npz`（6710 域 × 8 pH）。backbone = MoMPNN
  `MoMPNN/mompnn_paper_checkpoints/mompnn_temberture_tm_esm_6_4_4_b01.ckpt`，protein 模式。

日志头（核对无误）：
```
epochs=30 lr=0.001 λ_c=0.5 λ_kl=0.05 λ_keep=0.5 perturb_prob=0.3 perturb_scale=4.0
placeholder_prob=0.15 charge_temp=0.5
[v10 A] decouple_perturb：target 与 native 无关 Uniform[-12.0,12.0]
[v12] 组成双计数(floor=0.5)+GRAVY(margin=0.4) λ_v12=0.2 λ_target=0.2 SASA θ=0.25
[v10 C] pH 自适应结构惩罚：boost=1.5
（无 [A1] 行 → 该版训练代码尚无 pocket 损失 → λ_pocket 等效 0）
```

回放配置（`--tag v12_2`）：
| 项 | 值 |
|---|---|
| mode / weights | protein / MoMPNN |
| charge_temp | 0.5 |
| λ_c / λ_kl / λ_keep | 0.5 / 0.05 / 0.5 |
| v12_supervision | True（frac_floor=0.5, gravy_margin=0.4, λ_v12=0.2, λ_target=0.2, sasa_threshold=0.25） |
| ph_aware_filter | True（boost 训练 1.5；self 臂 boost=1.0） |
| pocket_mode / λ_pocket | keep / **0.0**（该版训练无 pocket 项） |

epochs_default = 30。

---

## 2. v12.3 蛋白（MoMPNN）

- 训练命令权威来源：`log/v12_3_train_mompnn.log` / `.stdout`（2026-09-02，776.9min）。
- 数据：`data/cath/labels_v12_3_train.npz`（6580 域 × 8 pH）。backbone = MoMPNN，protein 模式。

日志头（核对无误）：
```
epochs=40 λ_c=0.5 λ_kl=0.05 λ_keep=0.5 perturb_prob=0.3 perturb_scale=4.0
placeholder_prob=0.15 charge_temp=0.5
[v10 A] decouple_perturb Uniform[-12.0,12.0]
[v12] floor=0.5 gravy_margin=0.4 λ_v12=0.2 λ_target=0.2 SASA θ=0.25
[v10 C] ph_aware boost=1.5
[A1 keep] 计数锚仅 pocket：floor=0.7 ceil=1.3 λ_pocket=0.2 cutoff=8.0
```

回放配置（`--tag v12_3`）：
| 项 | 值 |
|---|---|
| mode / weights | protein / MoMPNN |
| charge_temp | 0.5 |
| λ_c / λ_kl / λ_keep | 0.5 / 0.05 / 0.5 |
| v12_supervision | True（同上） |
| ph_aware_filter | True |
| pocket_mode / floor / ceil / λ_pocket | keep / 0.7 / 1.3 / **0.2** |

> ⚠️ v12.3 是 protein 模式，CATH 域无配体 → `pocket_mask` 全 0 → `pocket_count_loss` 区为 0，
> 该项数值恒 ≈ 0（但不影响其余项；`v12_ct` 的监督 mask = surface∪pocket = surface，与训练一致）。

epochs_default = 40。

---

## 3. v14 配体（LigandMPNN，RNA/DNA 扩充 + A1 全局化）

- 训练命令权威来源：`log/v14_ligand_train.log` / `v14_ligand_train_stdout.log`
  （2026-09-02 启动；832.8min 收尾 `output/finetune_ligand_v14_rna`）。
  session 记录 `session/2026-09-02_v14_rna_data_a1_global.md` §五命令摘要一致。
- 数据：`data/ligand_train/labels_v14_final.npz`（stdout，5371 域）或 `labels_v14_merged.npz`
  （.log，5148 域）——两次重训 flag 相同，只数据规模不同；本回放以 checkpoint 为准，训练 flag 不变。
- backbone = LigandMPNN `LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt`，ligand 模式，
  `num_ligand_atoms = 25`（配体原子数 bug 修复后统一口径，记忆 2026-09-01）。

日志头（核对无误）：
```
epochs=50 λ_c=0.5 λ_kl=0.05 λ_keep=0.5 perturb_prob=0.3 perturb_scale=4.0
placeholder_prob=0.15 charge_temp=0.5
[v11 A-fix] 绝对 target：Uniform[-35.0,20.0]
[v12] floor=0.5 gravy_margin=0.4 λ_v12=0.2 λ_target=0.2 SASA θ=0.25
[v10 C] ph_aware boost=1.5
[A1 global] 计数锚 surface∪pocket：floor=0.8 ceil=1.3 λ_pocket=0.3 cutoff=8.0
```

回放配置（`--tag v14_ligand`）：
| 项 | 值 |
|---|---|
| mode / weights | ligand / LigandMPNN v_32_010_25.pt |
| num_ligand_atoms | 25 |
| charge_temp | 0.5 |
| λ_c / λ_kl / λ_keep | 0.5 / 0.05 / 0.5 |
| v12_supervision | True（同上） |
| ph_aware_filter | True |
| pocket_mode / floor / ceil / λ_pocket | global / 0.8 / 1.3 / **0.3** |

epochs_default = 50。

---

## 4. 建议验证集（运行 `--labels/--dompdb` 时给出）

- **v12.2 / v12.3（蛋白）**：base `data/cath/labels_holdout_train.npz`（1176 真未见）
  + `data/cath/S40/dompdb_pdb`；补充 `data/cath/labels_v12_3_valsupp.npz`（23 长/深负）
  + `data/cath/S40/dompdb_valsupp`（见 `session/2026-09-04_valset_build.md` §A）。
- **v14（配体）**：`data/ligand_train/labels_v14_valset_805.npz` + `data/ligand_train/v14_valset_pdb`
  （805 域）；外部未见候选仍在构建（同 session §B），就绪后替换。

---

## 5. 口径验证（数值对照训练 log）

- 脚本 `val_loss_curve.py` 直接 import `train_finetune` 的 `decoder_forward/build_domain/
  kl_anchor_loss/load_backbone` 与 src 同款损失函数，逐域前向路径与训练 self-arm 分支完全一致。
- 实证：在 **v12.2 训练前 40 域**上以 `--tag v12_2 --epoch_list 30` 回放，结果 vs 训练日志
  epoch 30（全训练集）：
  | 指标 | 本回放(前 40 域) | 训练 log epoch30 |
  |---|---|---|
  | ce | 1.9352 | 1.9491 |
  | cd（self 口径） | 2.3053 | `[cd self] 2.233` |
  | kl | 0.1958 | 0.2013 |
  → 逐项吻合（40/6710 采样噪声量级），判定口径一致。
